#!/usr/bin/env python3
"""Build a Route-B Levanter token cache for the 130M optimizer sweep.

The original Marin experiment tokenized the full DCLM mixture (about 4.1T
tokens) and then trained on a prefix of the globally shuffled mixture.  Route B
does not download that full cache.  Instead it samples raw shards, tokenizes a
small, deterministic subset, and lays the resulting tokens out as 640 mixture
blocks.  Each block contains the same DCLM/StarCoder/ProofPile sequence counts
that Levanter's block-deterministic mixture would use at block size 2048.

The generated TreeCache is intentionally one document per block.  Levanter's
TokenSeqDataset flattens a TreeCache and slices it at the model sequence length,
so document boundaries are immaterial for causal-LM training.  Keeping one
large document per block makes writing the 21.5 GB cache much faster.

This is a statistically controlled Route-B approximation, not a bit-exact
replacement for the original globally shuffled 4.1T-token cache.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np


SEQ_LEN = 4096
CHUNK_BYTES = SEQ_LEN * np.dtype(np.int32).itemsize
ORIGINAL_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
DEFAULT_TOKENIZER = "marin-community/marin-tokenizer"
DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# These are the source weights used by experiments/dclm/tokenize_dclm.py.
MIX_WEIGHTS = {
    "dclm_baseline": 3.8,
    "starcoderdata": 0.25,
    "proofpile_2": 0.055,
}


@dataclasses.dataclass(frozen=True)
class SourceSpec:
    name: str
    repo_id: str
    revision: str
    text_field: str
    suffixes: tuple[str, ...]
    group_depth: int
    default_shards: int
    bytes_per_token: float
    min_file_bytes: int
    description: str


SOURCE_SPECS: dict[str, SourceSpec] = {
    "dclm_baseline": SourceSpec(
        name="dclm_baseline",
        repo_id="mlfoundations/dclm-baseline-1.0",
        revision="a3b142c183aebe5af344955ae20836eb34dcf69b",
        text_field="text",
        suffixes=(".jsonl.zst", ".jsonl.gz", ".jsonl"),
        group_depth=1,
        default_shards=41,
        bytes_per_token=1.80,
        min_file_bytes=5_000_000,
        description="DCLM baseline (4.1T-token pool)",
    ),
    "starcoderdata": SourceSpec(
        name="starcoderdata",
        repo_id="bigcode/starcoderdata",
        revision="9fc30b578cedaec69e47302df72cf00feed7c8c4",
        text_field="content",
        suffixes=(".parquet",),
        group_depth=1,
        default_shards=16,
        bytes_per_token=1.24,
        min_file_bytes=256_000,
        description="StarCoderData (language directories)",
    ),
    "proofpile_2": SourceSpec(
        name="proofpile_2",
        repo_id="EleutherAI/proof-pile-2",
        revision="901a9273a770e9d4138c5ddd91802f9c5c6cdc4b",
        text_field="text",
        suffixes=(".jsonl.zst", ".jsonl.gz", ".jsonl"),
        group_depth=1,
        default_shards=6,
        bytes_per_token=0.93,
        min_file_bytes=5_000_000,
        description="Proof-Pile-2 (arXiv/OpenWebMath/AlgebraicStack)",
    ),
}

PALOMA_SPEC = SourceSpec(
    name="paloma_c4_en",
    repo_id="allenai/paloma",
    revision="65cd6fc59dba021b21db414fa5e8d7765ffbe5e6",
    text_field="text",
    suffixes=(".jsonl.gz", ".jsonl"),
    group_depth=0,
    default_shards=6,
    bytes_per_token=1.0,
    min_file_bytes=1,
    description="Paloma c4_en validation",
)


def eprint(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def fmt_count(n: int | float) -> str:
    n = float(n)
    if abs(n) >= 1e12:
        return f"{n / 1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.2f}K"
    return f"{n:.0f}"


def fmt_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def atomic_json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_np_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.npy", dir=path.parent)
    try:
        # np.save() appends ".npy" when given a path without that suffix, which
        # would leave the temporary file empty and atomically publish zero
        # bytes.  Passing a file object avoids that implicit rename.
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def block_counts(weights: dict[str, float], block_size: int) -> dict[str, int]:
    """Mirror Levanter MixtureDataset._compute_expected_counts_per_block()."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Mixture weights must sum to a positive number")
    expected = {name: weight / total * block_size for name, weight in weights.items() if weight > 0}
    counts = {name: int(value) for name, value in expected.items()}
    largest = max(expected, key=expected.get)
    counts[largest] += block_size - sum(counts.values())
    return counts


@dataclasses.dataclass(frozen=True)
class RepoFile:
    path: str
    size: int

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "RepoFile":
        return RepoFile(path=str(obj["path"]), size=int(obj.get("size") or 0))

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size}


def hf_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch_repo_listing(repo_id: str, revision: str, endpoint: str, token: str | None) -> list[RepoFile]:
    """Fetch the dataset file list and sizes from the HF API.

    The mirror API is used directly so that a giant repo does not get cloned.
    """
    import requests

    url = f"{endpoint.rstrip('/')}/api/datasets/{repo_id}"
    api_timeout = float(os.environ.get("HF_API_TIMEOUT", "180"))
    params: dict[str, str] = {"blobs": "true"}
    if revision:
        params["revision"] = revision
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.get(url, params=params, headers=hf_headers(token), timeout=(30, api_timeout))
            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"无法读取 {repo_id}（HTTP {response.status_code}）。该数据集可能需要先在 Hugging Face "
                    "接受许可并设置 HF_TOKEN；国内镜像不会绕过 gated 授权。"
                )
            response.raise_for_status()
            payload = response.json()
            siblings = payload.get("siblings") or []
            if not siblings:
                raise RuntimeError(f"HF API 没有返回 {repo_id} 的文件列表")
            return [
                RepoFile(path=str(item["rfilename"]), size=int(item.get("size") or 0))
                for item in siblings
                if item.get("rfilename")
            ]
        except Exception as exc:  # requests/JSON/network all get the same retry path
            last_error = exc
            if attempt == 5:
                break
            delay = min(20, 2**attempt)
            eprint(f"[warn] 读取 {repo_id} 文件列表失败（{exc}），{delay}s 后重试...")
            time.sleep(delay)
    raise RuntimeError(
        f"读取 {repo_id} 文件列表失败。请检查 HF_ENDPOINT={endpoint!r} 是否可访问，"
        f"以及 HF_TOKEN 是否有权限。最后一个错误: {last_error}"
    ) from last_error


def filter_source_files(spec: SourceSpec, files: Sequence[RepoFile]) -> list[RepoFile]:
    return [f for f in files if f.path.endswith(spec.suffixes) and f.size >= spec.min_file_bytes]


def select_source_files(
    spec: SourceSpec,
    files: Sequence[RepoFile],
    shard_count: int,
    seed: int,
) -> list[RepoFile]:
    """Pick files deterministically, apportioned by top-level group size.

    StarCoderData has one directory per language and Proof-Pile-2 has three
    very differently sized sub-corpora.  A plain round-robin can omit Python or
    over-represent AlgebraicStack.  We instead allocate shard slots in
    proportion to each group's total compressed bytes using the D'Hondt method,
    then sample files deterministically inside each group.  Group sizes are a
    good proxy for token counts for these repositories because files within a
    group have similar sizes and tokenization rates.
    """
    eligible = filter_source_files(spec, files)
    if not eligible:
        raise RuntimeError(f"{spec.name}: 没有找到后缀为 {spec.suffixes} 的文件")
    if shard_count <= 0:
        shard_count = len(eligible)
    shard_count = min(shard_count, len(eligible))

    rng = random.Random(stable_seed(seed, spec.name))
    groups: dict[str, list[RepoFile]] = defaultdict(list)
    for file in eligible:
        parts = file.path.split("/")
        group = "/".join(parts[: spec.group_depth]) if spec.group_depth else ""
        groups[group].append(file)
    for group_files in groups.values():
        group_files.sort(key=lambda f: f.path)
        rng.shuffle(group_files)
    group_order = sorted(groups)
    group_weights = {
        group: max(1, sum(max(0, file.size) for file in group_files))
        for group, group_files in groups.items()
    }
    allocated = {group: 0 for group in group_order}
    # D'Hondt apportionment: each additional slot goes to the group with the
    # largest weight / (slots + 1), capped at the number of files it contains.
    for _ in range(shard_count):
        eligible = [group for group in group_order if allocated[group] < len(groups[group])]
        if not eligible:
            break
        chosen = max(eligible, key=lambda group: group_weights[group] / (allocated[group] + 1))
        allocated[chosen] += 1

    selected: list[RepoFile] = []
    for group in group_order:
        selected.extend(groups[group][: allocated[group]])
    # The final order does not affect the mixture; shuffle it so that a future
    # change to the worker scheduling does not systematically process one group
    # first.
    rng.shuffle(selected)
    return selected


def prefix_for_source(spec: SourceSpec, args: argparse.Namespace) -> int:
    if spec.name == "dclm_baseline":
        return args.dclm_shards
    if spec.name == "starcoderdata":
        return args.starcoder_shards
    if spec.name == "proofpile_2":
        return args.proofpile_shards
    return spec.default_shards


def split_targets(total: int, count: int) -> list[int]:
    if count <= 0:
        return []
    base, remainder = divmod(total, count)
    return [base + (1 if i < remainder else 0) for i in range(count)]


class LevanterStyleTokenizer:
    """Tokenize like Levanter's BatchTokenizer(enforce_bos=True, enforce_eos=True).

    In particular this reproduces Levanter's string-level BOS/EOS insertion and
    its >10k-character workaround for fast tokenizers with a Replace/Sequence
    normalizer.  Using tokenizer.encode(text) directly is not equivalent.
    """

    def __init__(self, name: str, revision: str | None, token: str | None):
        try:
            from tokenizers import normalizers
            from transformers import AutoTokenizer, PreTrainedTokenizerFast
        except ImportError as exc:
            raise RuntimeError(
                "缺少 transformers/tokenizers。目标 Pod 上请先安装仓库依赖，例如 "
                "`cd repo && uv sync --extra cuda12`。"
            ) from exc

        self.name = name
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            revision=revision or None,
            token=token,
            trust_remote_code=True,
        )
        self.is_fast = isinstance(self.tokenizer, PreTrainedTokenizerFast)
        if self.tokenizer.bos_token_id is None:
            enforce_bos = False
        else:
            enforce_bos = True
        if self.tokenizer.eos_token_id is None:
            enforce_eos = False
        else:
            enforce_eos = True

        probe = self.tokenizer("hi there", return_attention_mask=False)["input_ids"]
        self.need_bos = enforce_bos and probe[0] != self.tokenizer.bos_token_id
        self.need_eos = enforce_eos and probe[-1] != self.tokenizer.eos_token_id
        self.bos_text = self.tokenizer.bos_token or self.tokenizer.convert_ids_to_tokens(self.tokenizer.bos_token_id)
        self.eos_text = self.tokenizer.eos_token or self.tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id)
        self.long_workaround = False
        if self.is_fast:
            normalizer = self.tokenizer.backend_tokenizer.normalizer
            self.long_workaround = isinstance(normalizer, (normalizers.Replace, normalizers.Sequence))

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer": self.name,
            "resolved_revision": self.revision,
            "vocab_size": self.vocab_size,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "append_bos": self.need_bos,
            "append_eos": self.need_eos,
            "long_sequence_workaround": self.long_workaround,
            "fast": self.is_fast,
        }

    def _prepare_text(self, text: str) -> str:
        if self.need_bos and self.bos_text:
            text = self.bos_text + " " + text
        if self.need_eos and self.eos_text:
            text = text + " " + self.eos_text
        return text

    def _split_long(self, text: str) -> list[str]:
        if not self.long_workaround or len(text) <= 10_000:
            return [text]
        pieces: list[str] = []
        while len(text) > 10_000:
            match = re.search(r"\s", text, 10_000)
            split = match.start() if match else len(text)
            if split <= 0:
                split = len(text)
            pieces.append(text[:split])
            text = text[split:]
        pieces.append(text)
        return pieces

    def encode_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        flattened: list[str] = []
        groups: list[tuple[int, int]] = []
        for text in texts:
            prepared = self._prepare_text(str(text))
            pieces = self._split_long(prepared)
            start = len(flattened)
            flattened.extend(pieces)
            groups.append((start, len(flattened)))
        if not flattened:
            return []
        try:
            encoded = self.tokenizer(flattened, return_attention_mask=False, verbose=False)["input_ids"]
        except TypeError:
            encoded = self.tokenizer(flattened, return_attention_mask=False)["input_ids"]
        out: list[np.ndarray] = []
        for start, end in groups:
            if start == end:
                out.append(np.empty((0,), dtype=np.int32))
            else:
                arrays = [np.asarray(encoded[i], dtype=np.int32) for i in range(start, end)]
                out.append(np.concatenate(arrays).astype(np.int32, copy=False))
        return out


def resolve_tokenizer_revision(name: str, revision: str | None, endpoint: str, token: str | None) -> str | None:
    if revision:
        return revision
    import requests

    try:
        response = requests.get(
            f"{endpoint.rstrip('/')}/api/models/{name}",
            headers=hf_headers(token),
            timeout=45,
        )
        response.raise_for_status()
        return response.json().get("sha")
    except Exception:
        return None


def iter_docs(path: Path, field: str) -> Iterator[str]:
    """Yield document text from jsonl/jsonl.gz/jsonl.zst/parquet."""
    suffix = path.name.lower()
    if suffix.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("读取 StarCoder parquet 需要 pyarrow") from exc
        parquet = pq.ParquetFile(path)
        if field not in parquet.schema.names:
            raise RuntimeError(f"{path} 缺少字段 {field!r}，实际字段: {parquet.schema.names}")
        for batch in parquet.iter_batches(batch_size=256, columns=[field]):
            for value in batch.column(0).to_pylist():
                if value:
                    yield str(value)
        return

    if suffix.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from _iter_json_lines(handle, field, path)
        return

    if suffix.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError:
            import fsspec

            with fsspec.open(str(path), "rt", compression="zstd", encoding="utf-8") as handle:
                yield from _iter_json_lines(handle, field, path)
            return
        with path.open("rb") as raw:
            reader = zstd.ZstdDecompressor().stream_reader(raw)
            with io.TextIOWrapper(reader, encoding="utf-8", errors="replace") as handle:
                yield from _iter_json_lines(handle, field, path)
        return

    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        yield from _iter_json_lines(handle, field, path)


def _iter_json_lines(handle: Iterable[str], field: str, path: Path) -> Iterator[str]:
    bad_lines = 0
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            value = obj.get(field)
            if value is None and field == "content":
                value = obj.get("text")
            if value is None and field == "text":
                value = obj.get("content")
            if value:
                yield str(value)
        except Exception:
            bad_lines += 1
            if bad_lines <= 3:
                eprint(f"[warn] {path}: 跳过无法解析的 JSON 行（前 3 条警告）")


def download_file(
    spec: SourceSpec,
    file: RepoFile,
    raw_root: Path,
    endpoint: str,
    token: str | None,
    retries: int,
    on_bytes: Callable[[int], None] | None = None,
) -> Path:
    """Download one hub file with resumable HTTP range requests.

    huggingface_hub's downloader is fine but does not expose a byte callback, so
    a large shard looks completely stalled in our progress output.  The direct
    resolve URL is the same transport used by the mirror and lets us report
    bytes as they arrive and resume a partial file after a disconnect.
    """
    import requests

    target = raw_root / spec.name / file.path
    if target.exists() and (file.size <= 0 or target.stat().st_size == file.size):
        if on_bytes is not None:
            on_bytes(target.stat().st_size)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if target.exists():
        if file.size > 0 and target.stat().st_size < file.size:
            if partial.exists() and partial.stat().st_size > target.stat().st_size:
                target.unlink()
            else:
                os.replace(target, partial)
        else:
            target.unlink()

    encoded_path = urllib.parse.quote(file.path, safe="/")
    url = f"{endpoint.rstrip('/')}/datasets/{spec.repo_id}/resolve/{spec.revision}/{encoded_path}"
    headers = {
        **hf_headers(token),
        "User-Agent": "marin-route-b-cache/1.0",
        "Accept-Encoding": "identity",
    }

    initial_partial = partial.stat().st_size if partial.exists() else 0
    if on_bytes is not None and initial_partial:
        on_bytes(initial_partial)

    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        downloaded = partial.stat().st_size if partial.exists() else 0
        request_headers = dict(headers)
        if downloaded:
            request_headers["Range"] = f"bytes={downloaded}-"
        try:
            with requests.get(
                url,
                params={"download": "true"},
                headers=request_headers,
                stream=True,
                timeout=(30, 180),
                allow_redirects=True,
            ) as response:
                if response.status_code in (401, 403):
                    raise RuntimeError(
                        f"无法下载 {spec.repo_id}/{file.path}（HTTP {response.status_code}）。"
                        "该数据集可能需要先在 Hugging Face 接受许可并设置 HF_TOKEN；"
                        "国内镜像不会绕过 gated 授权。"
                    )
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else min(120, 15 * (attempt + 1))
                    raise RuntimeError(f"镜像限流 HTTP 429，建议等待 {wait:.0f}s 后重试")
                if response.status_code == 416 and downloaded and file.size > 0 and downloaded == file.size:
                    os.replace(partial, target)
                    return target
                response.raise_for_status()

                restart = response.status_code == 200 and downloaded > 0
                if restart:
                    eprint(f"[warn] 服务器不支持 Range，重新下载 {spec.repo_id}/{file.path}")
                    if on_bytes is not None:
                        on_bytes(-downloaded)
                    downloaded = 0
                mode = "ab" if downloaded and response.status_code == 206 else "wb"
                if mode == "wb":
                    downloaded = 0
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if on_bytes is not None:
                            on_bytes(len(chunk))
                    handle.flush()
                    os.fsync(handle.fileno())

            if file.size > 0 and downloaded != file.size:
                snippet = partial.read_bytes()[:200].decode("utf-8", errors="replace")
                if "Access to dataset" in snippet or "Access to model" in snippet:
                    raise RuntimeError(
                        f"{spec.repo_id} 是 gated/受限数据集，当前 token 没有访问权限。"
                        "请先在 Hugging Face 接受许可并设置 HF_TOKEN；镜像不会绕过授权。"
                    )
                raise RuntimeError(
                    f"{spec.repo_id}/{file.path} 下载大小不匹配: "
                    f"{downloaded} != {file.size}。内容开头: {snippet!r}"
                )
            os.replace(partial, target)
            return target
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max(1, retries):
                break
            if "429" in str(exc):
                delay = min(120, 15 * (attempt + 1))
            else:
                delay = min(60, 2 ** (attempt + 1))
            eprint(f"[warn] 下载 {spec.repo_id}/{file.path} 失败（{exc}），{delay}s 后重试...")
            time.sleep(delay)
    raise RuntimeError(
        f"下载 {spec.repo_id}/{file.path} 失败。请检查 HF_ENDPOINT={endpoint!r}、"
        f"HF_TOKEN 以及 gated 数据集授权。最后一个错误: {last_error}"
    ) from last_error


@dataclasses.dataclass
class SourceState:
    source: str
    file_index: int = 0
    docs_consumed_in_file: int = 0
    chunks_written: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "SourceState":
        return SourceState(
            source=str(obj["source"]),
            file_index=int(obj.get("file_index", 0)),
            docs_consumed_in_file=int(obj.get("docs_consumed_in_file", 0)),
            chunks_written=int(obj.get("chunks_written", 0)),
        )


class SourceWorker:
    def __init__(
        self,
        spec: SourceSpec,
        files: Sequence[RepoFile],
        target_chunks: int,
        args: argparse.Namespace,
        work_dir: Path,
        reporter: "ProgressReporter",
        tokenizer: LevanterStyleTokenizer,
    ):
        self.spec = spec
        self.files = list(files)
        self.target_chunks = target_chunks
        self.args = args
        self.work_dir = work_dir
        self.reporter = reporter
        self.tokenizer = tokenizer
        self.raw_root = work_dir / "raw"
        self.bin_path = work_dir / "chunks" / f"{spec.name}.bin"
        self.tail_path = work_dir / "chunks" / f"{spec.name}.tail.npy"
        self.state_path = work_dir / "source_state" / f"{spec.name}.json"
        self.bin_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = SourceState(source=spec.name)
        self.tail = np.empty((0,), dtype=np.int32)
        self._bin_handle: Any = None
        self._last_checkpoint = 0.0

    def _load_state(self) -> None:
        if self.state_path.exists():
            self.state = SourceState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
        size = self.bin_path.stat().st_size if self.bin_path.exists() else 0
        expected = self.state.chunks_written * CHUNK_BYTES
        if size < expected:
            raise RuntimeError(
                f"{self.spec.name}: chunk 文件小于状态记录（{size} < {expected}）。"
                "请删除 work 目录后重新开始，或检查磁盘。"
            )
        if size > expected:
            eprint(
                f"[warn] {self.spec.name}: chunk 文件尾部有未提交数据，截断到 checkpoint "
                f"({expected} bytes)"
            )
            with self.bin_path.open("r+b") as handle:
                handle.truncate(expected)
        if self.state.chunks_written > self.target_chunks:
            raise RuntimeError(
                f"{self.spec.name}: 已有 {self.state.chunks_written} chunks，超过目标 {self.target_chunks}"
            )
        if self.state.chunks_written >= self.target_chunks:
            self.tail = np.empty((0,), dtype=np.int32)
            return
        if not self.tail_path.exists():
            return
        if self.tail_path.stat().st_size == 0:
            if self.state.docs_consumed_in_file == 0:
                eprint(
                    f"[warn] {self.spec.name}: tail 文件为空，但断点位于文件边界，"
                    "按空 tail 继续"
                )
                self.tail = np.empty((0,), dtype=np.int32)
                return
            raise RuntimeError(
                f"{self.spec.name}: tail 文件为空且断点在文件中间，无法保证 token 连续性。"
                "请删除该 source 的 state/chunks 后从文件边界重新开始。"
            )
        self.tail = np.load(self.tail_path)
        if self.tail.dtype != np.int32:
            self.tail = self.tail.astype(np.int32)

    def _save_state(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_checkpoint < self.args.checkpoint_seconds:
            return
        atomic_np_save(self.tail_path, self.tail)
        atomic_json_write(self.state_path, self.state.to_dict())
        self._last_checkpoint = now

    def _flush_tail(self) -> int:
        written = 0
        while self.tail.size >= SEQ_LEN and self.state.chunks_written < self.target_chunks:
            assert self._bin_handle is not None
            self._bin_handle.write(self.tail[:SEQ_LEN].tobytes())
            self.tail = self.tail[SEQ_LEN:].copy()
            self.state.chunks_written += 1
            written += 1
        if written:
            self.reporter.update(tokens_delta=written * SEQ_LEN, source=self.spec.name)
        return written

    def _consume_texts(self, texts: Sequence[str], file_goal: int) -> bool:
        """Consume a batch; return True if this file's quota was reached."""
        encoded = self.tokenizer.encode_batch(texts)
        for ids in encoded:
            if self.state.chunks_written >= self.target_chunks:
                return True
            if ids.size:
                if self.tail.size:
                    self.tail = np.concatenate((self.tail, ids))
                else:
                    self.tail = ids.astype(np.int32, copy=True)
            self.state.docs_consumed_in_file += 1
            self._flush_tail()
            if self.state.chunks_written >= file_goal:
                return True
        return False

    def run(self) -> dict[str, Any]:
        self._load_state()
        if self.state.chunks_written >= self.target_chunks:
            return {"source": self.spec.name, "chunks": self.state.chunks_written}

        file_targets = split_targets(self.target_chunks, len(self.files))
        # Cumulative goals matter: a tiny language/validation shard may not be
        # able to fill its nominal quota.  Later, larger shards then make up the
        # deficit instead of leaving the whole source short.
        cumulative_targets = np.cumsum(file_targets).tolist()
        self.bin_path.parent.mkdir(parents=True, exist_ok=True)
        with self.bin_path.open("ab", buffering=0) as bin_handle:
            self._bin_handle = bin_handle
            for file_index in range(self.state.file_index, len(self.files)):
                if self.state.chunks_written >= self.target_chunks:
                    break
                if file_index != self.state.file_index:
                    self.state.docs_consumed_in_file = 0
                self.state.file_index = file_index
                chunks_at_start = self.state.chunks_written
                file_goal = min(self.target_chunks, int(cumulative_targets[file_index]))
                if file_goal <= chunks_at_start:
                    self.state.file_index = file_index + 1
                    self.state.docs_consumed_in_file = 0
                    self._save_state(force=True)
                    continue

                file = self.files[file_index]
                raw_path = download_file(
                    self.spec,
                    file,
                    self.raw_root,
                    self.args.hf_endpoint,
                    self.args.hf_token,
                    self.args.download_retries,
                    on_bytes=self.reporter.add_bytes,
                )
                self._flush_tail()
                self._save_state(force=True)

                docs = iter_docs(raw_path, self.spec.text_field)
                for _ in range(self.state.docs_consumed_in_file):
                    try:
                        next(docs)
                    except StopIteration:
                        break

                batch: list[str] = []
                batch_chars = 0
                quota_reached = False
                for text in docs:
                    if self.state.chunks_written >= self.target_chunks:
                        quota_reached = True
                        break
                    batch.append(text)
                    batch_chars += len(text)
                    if len(batch) >= self.args.doc_batch_size or batch_chars >= self.args.max_batch_chars:
                        quota_reached = self._consume_texts(batch, file_goal)
                        batch = []
                        batch_chars = 0
                        self._save_state()
                        if quota_reached:
                            break
                if batch and not quota_reached:
                    quota_reached = self._consume_texts(batch, file_goal)
                self._save_state(force=True)

                self.state.file_index = file_index + 1
                self.state.docs_consumed_in_file = 0
                self._save_state(force=True)
                if not self.args.keep_raw:
                    try:
                        raw_path.unlink()
                    except FileNotFoundError:
                        pass

        self._bin_handle = None
        if self.state.chunks_written != self.target_chunks:
            raise RuntimeError(
                f"{self.spec.name}: 只生成 {self.state.chunks_written} / {self.target_chunks} chunks。"
                "说明选中的 shard 不够；请增大对应的 --*-shards。"
            )
        return {"source": self.spec.name, "chunks": self.state.chunks_written}


class ProgressReporter:
    def __init__(self, total_tokens: int, interval: float = 10.0):
        self.total_tokens = total_tokens
        self.interval = interval
        self.lock = threading.Lock()
        self.global_start = time.monotonic()
        self.stage = "setup"
        self.stage_start = self.global_start
        self.stage_done = 0
        self.source_done: dict[str, int] = defaultdict(int)
        self.raw_done = 0
        self.raw_total = 0
        self._last_print = 0.0
        self._last_stage_done = 0
        self._last_time = self.stage_start
        self._ema_rate = 0.0
        self._last_raw_done = 0
        self._last_raw_time = self.stage_start
        self._raw_ema_rate = 0.0

    def begin_stage(self, name: str, total: int | None = None) -> None:
        with self.lock:
            now = time.monotonic()
            self.stage = name
            self.stage_start = now
            self.stage_done = 0
            self.source_done.clear()
            if total is not None:
                self.total_tokens = total
            self._last_print = 0.0
            self._last_stage_done = 0
            self._last_time = now
            self._ema_rate = 0.0
            self._last_raw_done = self.raw_done
            self._last_raw_time = now
            self._raw_ema_rate = 0.0

    def update(self, tokens_delta: int = 0, source: str | None = None) -> None:
        with self.lock:
            if tokens_delta:
                self.stage_done += tokens_delta
                if source:
                    self.source_done[source] += tokens_delta

    def add_bytes(self, n: int) -> None:
        with self.lock:
            self.raw_done = max(0, self.raw_done + int(n))

    def set_raw_total(self, n: int) -> None:
        with self.lock:
            self.raw_total = max(0, int(n))

    def maybe_print(self, force: bool = False) -> None:
        with self.lock:
            now = time.monotonic()
            if not force and now - self._last_print < self.interval:
                return
            elapsed_stage = max(1e-9, now - self.stage_start)
            elapsed_total = max(1e-9, now - self.global_start)
            dt = max(1e-9, now - self._last_time)
            instantaneous = (self.stage_done - self._last_stage_done) / dt
            if self._ema_rate == 0:
                self._ema_rate = instantaneous
            else:
                self._ema_rate = 0.3 * instantaneous + 0.7 * self._ema_rate
            raw_dt = max(1e-9, now - self._last_raw_time)
            raw_instantaneous = (self.raw_done - self._last_raw_done) / raw_dt
            if self._raw_ema_rate == 0:
                self._raw_ema_rate = raw_instantaneous
            else:
                self._raw_ema_rate = 0.3 * raw_instantaneous + 0.7 * self._raw_ema_rate
            overall_rate = self.stage_done / elapsed_stage
            rate = self._ema_rate if self._ema_rate > 0 else overall_rate
            raw_rate = self._raw_ema_rate if self._raw_ema_rate > 0 else self.raw_done / elapsed_stage
            remaining = max(0, self.total_tokens - self.stage_done)
            eta = remaining / rate if rate > 0 else None
            pct = 100.0 * self.stage_done / self.total_tokens if self.total_tokens else 0.0
            raw_pct = 100.0 * self.raw_done / self.raw_total if self.raw_total else 0.0
            source_text = " ".join(
                f"{name}={fmt_count(value)}" for name, value in sorted(self.source_done.items())
            )
            line = (
                f"[{fmt_duration(elapsed_total)}] {self.stage} | "
                f"tokens {fmt_count(self.stage_done)}/{fmt_count(self.total_tokens)} ({pct:5.1f}%) | "
                f"{rate / 1e6:.2f}M tok/s | raw {fmt_bytes(self.raw_done)}/{fmt_bytes(self.raw_total)} "
                f"({raw_pct:5.1f}%, {fmt_bytes(raw_rate)}/s) | ETA {fmt_duration(eta)}"
            )
            if source_text:
                line += f" | {source_text}"
            print(line, flush=True)
            self._last_print = now
            self._last_stage_done = self.stage_done
            self._last_time = now
            self._last_raw_done = self.raw_done
            self._last_raw_time = now


def run_parallel_sources(
    workers: Sequence[SourceWorker],
    reporter: ProgressReporter,
    num_procs: int,
) -> None:
    if not workers:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(num_procs, len(workers)))) as executor:
        futures = {executor.submit(worker.run): worker.spec.name for worker in workers}
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=0.5)
            reporter.maybe_print()
            for future in done:
                name = futures[future]
                try:
                    future.result()
                except Exception:
                    for other in pending:
                        other.cancel()
                    eprint(f"[error] source worker {name} 失败", file=sys.stderr)
                    raise
    reporter.maybe_print(force=True)


def check_imports(for_write: bool) -> None:
    required = ["numpy", "requests"]
    if for_write:
        required += ["transformers", "tokenizers", "huggingface_hub", "pyarrow", "fsspec", "levanter", "zstandard"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "缺少 Python 依赖: "
            + ", ".join(missing)
            + "。请在目标 Pod 上安装仓库环境（例如 `cd repo && uv sync --extra cuda12`）。"
        )


def build_paloma_cache(
    spec: SourceSpec,
    files: Sequence[RepoFile],
    output_dir: Path,
    args: argparse.Namespace,
    tokenizer: LevanterStyleTokenizer,
    reporter: "ProgressReporter | None" = None,
) -> dict[str, Any]:
    from levanter.store.cache import SerialCacheWriter

    cache_dir = output_dir / "validation" / "paloma_c4_en" / "validation"
    ledger_path = cache_dir / "shard_ledger.json"
    if ledger_path.exists() and args.resume:
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            if ledger.get("is_finished"):
                print(f"[paloma] 已完成，跳过: {cache_dir}", flush=True)
                return {"cache_dir": str(cache_dir), "skipped": True}
        except Exception:
            pass

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32)}
    token_count = 0
    doc_count = 0
    print(f"[paloma] 构建 c4_en 验证 cache: {cache_dir}", flush=True)
    with SerialCacheWriter(str(cache_dir), exemplar) as writer:
        for file in files:
            raw_path = download_file(
                spec,
                file,
                output_dir / "work" / "raw",
                args.hf_endpoint,
                args.hf_token,
                args.download_retries,
                on_bytes=reporter.add_bytes if reporter is not None else None,
            )
            batch: list[str] = []
            for text in iter_docs(raw_path, spec.text_field):
                batch.append(text)
                if len(batch) >= args.doc_batch_size:
                    rows: list[dict[str, np.ndarray]] = []
                    for ids in tokenizer.encode_batch(batch):
                        if ids.size:
                            rows.append({"input_ids": ids})
                            token_count += int(ids.size)
                            doc_count += 1
                            if reporter is not None:
                                reporter.update(tokens_delta=int(ids.size), source="paloma")
                    if rows:
                        writer.write_batch(rows)
                    batch = []
                    if reporter is not None:
                        reporter.maybe_print()
            rows = []
            for ids in tokenizer.encode_batch(batch):
                if ids.size:
                    rows.append({"input_ids": ids})
                    token_count += int(ids.size)
                    doc_count += 1
                    if reporter is not None:
                        reporter.update(tokens_delta=int(ids.size), source="paloma")
            if rows:
                writer.write_batch(rows)
            if not args.keep_raw:
                try:
                    raw_path.unlink()
                except FileNotFoundError:
                    pass
            if reporter is not None:
                reporter.maybe_print(force=True)
    print(f"[paloma] 完成: {fmt_count(token_count)} tokens, {doc_count} docs", flush=True)
    return {"cache_dir": str(cache_dir), "tokens": token_count, "docs": doc_count}


def verify_tree_cache(cache_dir: Path, expected_tokens: int | None = None) -> dict[str, Any]:
    import asyncio

    from levanter.store.cache import TreeCache

    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32)}
    cache = TreeCache.load(str(cache_dir), exemplar)
    tokens = int(cache.store.tree["input_ids"].data_size)
    rows = int(len(cache))
    if expected_tokens is not None and tokens != expected_tokens:
        raise RuntimeError(f"{cache_dir}: token 数 {tokens} != 期望 {expected_tokens}")
    return {"cache_dir": str(cache_dir), "tokens": tokens, "docs": rows}


def write_train_cache(
    output_dir: Path,
    blocks: int,
    block_size: int,
    counts: dict[str, int],
    source_states: dict[str, SourceState],
    seed: int,
    reporter: ProgressReporter,
    overwrite: bool,
) -> dict[str, Any]:
    from levanter.store.cache import SerialCacheWriter

    final_dir = output_dir / "train"
    if final_dir.exists():
        if not overwrite:
            raise RuntimeError(f"{final_dir} 已存在但未完成；请加 --resume，或显式使用 --overwrite")
        shutil.rmtree(final_dir)
    tmp_dir = output_dir / f".train.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.parent.mkdir(parents=True, exist_ok=True)
    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32)}
    memmaps: dict[str, np.ndarray] = {}
    for name, state in source_states.items():
        path = output_dir / "work" / "chunks" / f"{name}.bin"
        expected_size = state.chunks_written * CHUNK_BYTES
        actual_size = path.stat().st_size if path.exists() else 0
        if actual_size != expected_size:
            raise RuntimeError(f"{path}: 大小 {actual_size} != 期望 {expected_size}")
        memmaps[name] = np.memmap(path, dtype=np.int32, mode="r", shape=(state.chunks_written, SEQ_LEN))

    cursor = {name: 0 for name in counts}
    # The original mixture shuffles every component independently before the
    # block mixture is formed. Do the same over the generated sequence pool so
    # that a block does not take a contiguous prefix of one source shard.
    source_permutations: dict[str, np.ndarray] = {}
    for name, state in source_states.items():
        if state.chunks_written <= 0:
            source_permutations[name] = np.empty((0,), dtype=np.int64)
            continue
        source_rng = np.random.default_rng(stable_seed(seed, f"assemble:{name}"))
        source_permutations[name] = source_rng.permutation(state.chunks_written)
    tokens_per_block = block_size * SEQ_LEN
    with SerialCacheWriter(str(tmp_dir), exemplar) as writer:
        for block_id in range(blocks):
            parts = []
            for name, count in counts.items():
                if count <= 0:
                    continue
                start = cursor[name]
                end = start + count
                array = memmaps[name]
                if end > array.shape[0]:
                    raise RuntimeError(f"{name}: block {block_id} 需要 chunk {start}:{end}，但只有 {array.shape[0]}")
                indices = source_permutations[name][start:end]
                parts.append(np.array(array[indices], dtype=np.int32, copy=True))
                cursor[name] = end
            if not parts:
                raise RuntimeError("空 block")
            block = np.concatenate(parts, axis=0)
            if block.shape != (block_size, SEQ_LEN):
                raise RuntimeError(f"block {block_id} 形状 {block.shape} != {(block_size, SEQ_LEN)}")
            rng = np.random.default_rng(np.random.SeedSequence([seed, block_id]))
            block = block[rng.permutation(block_size)]
            writer.write_batch([{"input_ids": block.reshape(-1)}])
            reporter.update(tokens_delta=tokens_per_block)
            reporter.maybe_print()
    for name, count in counts.items():
        if cursor[name] != source_states[name].chunks_written:
            raise RuntimeError(f"{name}: 组装使用了 {cursor[name]} chunks，状态记录 {source_states[name].chunks_written}")
    os.replace(tmp_dir, final_dir)
    return verify_tree_cache(final_dir, blocks * tokens_per_block)


def train_cache_is_complete(output_dir: Path, expected_tokens: int, manifest_path: Path) -> bool:
    ledger = output_dir / "train" / "shard_ledger.json"
    if not ledger.exists():
        return False
    try:
        ledger_obj = json.loads(ledger.read_text(encoding="utf-8"))
        # The work/run_config.json fingerprint has already been checked, so a
        # finished ledger is enough to avoid rebuilding a cache merely because
        # the process was interrupted between renaming train/ and writing the
        # top-level manifest. The cache is re-verified immediately afterwards.
        return bool(ledger_obj.get("is_finished"))
    except Exception:
        return False


def make_config_snippet(output_dir: Path, tokenizer: str) -> str:
    return f'''# Levanter 130M optimizer-sweep 数据配置片段
from levanter.data.text import LMMixtureDatasetConfig, UrlDatasetSourceConfig

cache_root = "{output_dir}"
route_b_mixture = LMMixtureDatasetConfig(
    configs={{
        "route_b_dclm_mix": UrlDatasetSourceConfig(
            train_urls=[], validation_urls=[], cache_dir=cache_root
        ),
        "paloma/c4_en": UrlDatasetSourceConfig(
            train_urls=[], validation_urls=[],
            cache_dir=f"{{cache_root}}/validation/paloma_c4_en",
        ),
    }},
    train_weights={{"route_b_dclm_mix": 1.0, "paloma/c4_en": 0.0}},
    tokenizer="{tokenizer}",
    shuffle=False,   # Route B 已在 cache 内完成 block 级混排
)
'''


def estimate_time(args: argparse.Namespace, raw_bytes: int, target_tokens: int) -> tuple[float, float, float]:
    workers = max(1, args.num_procs)
    download_s = raw_bytes / max(1.0, args.download_mbps * 1e6)
    tokenize_s = target_tokens / max(1.0, args.tokens_per_sec * workers)
    assemble_s = target_tokens / max(1.0, args.assembly_tokens_per_sec)
    return download_s, tokenize_s, assemble_s


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    fingerprint_obj = {
        "blocks": args.blocks,
        "block_size": args.block_size,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "weights": MIX_WEIGHTS,
        "dclm_shards": args.dclm_shards,
        "starcoder_shards": args.starcoder_shards,
        "proofpile_shards": args.proofpile_shards,
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_obj, sort_keys=True).encode()).hexdigest()
    counts = block_counts(MIX_WEIGHTS, args.block_size)
    target_tokens = args.blocks * args.block_size * args.seq_len
    source_files: dict[str, list[RepoFile]] = {}
    for name, spec in SOURCE_SPECS.items():
        if name not in counts:
            continue
        listing = fetch_repo_listing(
            spec.repo_id,
            spec.revision,
            args.hf_endpoint,
            args.hf_token,
        )
        source_files[name] = select_source_files(spec, listing, prefix_for_source(spec, args), args.seed)
    paloma_listing = fetch_repo_listing(
        PALOMA_SPEC.repo_id,
        PALOMA_SPEC.revision,
        args.hf_endpoint,
        args.hf_token,
    )
    paloma_files = [
        f for f in paloma_listing if f.path.startswith("c4_en/val/") and f.path.endswith(PALOMA_SPEC.suffixes)
    ]
    if not paloma_files:
        raise RuntimeError("Paloma 中没有找到 c4_en/val/val*.jsonl.gz")
    return {
        "fingerprint": fingerprint,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "blocks": args.blocks,
        "block_size": args.block_size,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "target_tokens": target_tokens,
        "block_counts": counts,
        "weights": MIX_WEIGHTS,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "hf_endpoint": args.hf_endpoint,
        "source_files": {name: [f.to_dict() for f in fs] for name, fs in source_files.items()},
        "paloma_files": [f.to_dict() for f in paloma_files],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--blocks", type=int, default=640)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default=os.environ.get("TOKENIZER", DEFAULT_TOKENIZER))
    parser.add_argument("--tokenizer-revision", default=os.environ.get("TOKENIZER_REVISION", ""))
    parser.add_argument("--hf-endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or None)
    parser.add_argument("--dclm-shards", type=int, default=int(os.environ.get("DCLM_SHARDS", "41")))
    parser.add_argument("--starcoder-shards", type=int, default=int(os.environ.get("STARCODER_SHARDS", "16")))
    parser.add_argument("--proofpile-shards", type=int, default=int(os.environ.get("PROOFPILE_SHARDS", "6")))
    parser.add_argument("--num-procs", type=int, default=int(os.environ.get("NUM_PROCS", "3")))
    parser.add_argument("--doc-batch-size", type=int, default=int(os.environ.get("DOC_BATCH_SIZE", "128")))
    parser.add_argument("--max-batch-chars", type=int, default=int(os.environ.get("MAX_BATCH_CHARS", "2000000")))
    parser.add_argument("--progress-seconds", type=float, default=float(os.environ.get("PROGRESS_SECONDS", "10")))
    parser.add_argument("--checkpoint-seconds", type=float, default=float(os.environ.get("CHECKPOINT_SECONDS", "20")))
    parser.add_argument("--download-retries", type=int, default=int(os.environ.get("DOWNLOAD_RETRIES", "6")))
    parser.add_argument("--download-mbps", type=float, default=float(os.environ.get("DOWNLOAD_MBPS", "80")))
    parser.add_argument("--tokens-per-sec", type=float, default=float(os.environ.get("TOKENS_PER_SEC", "1500000")))
    parser.add_argument(
        "--assembly-tokens-per-sec",
        type=float,
        default=float(os.environ.get("ASSEMBLY_TOKENS_PER_SEC", "8000000")),
    )
    parser.add_argument("--keep-raw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-chunks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-paloma", action="store_true", default=os.environ.get("SKIP_PALOMA", "0") == "1")
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "0") == "1")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=os.environ.get("RESUME", "1") != "0")
    parser.add_argument("--overwrite", action="store_true", default=os.environ.get("OVERWRITE", "0") == "1")
    parser.add_argument("--verify-only", action="store_true", default=False)
    args = parser.parse_args(argv)
    if args.seq_len != SEQ_LEN:
        raise SystemExit("当前脚本固定 seq_len=4096；请保持 --seq-len 4096")
    if args.block_size <= 0 or args.blocks <= 0:
        raise SystemExit("--blocks/--block-size 必须为正数")
    if args.tokenizer != ORIGINAL_TOKENIZER:
        eprint("")
        eprint("=" * 100)
        eprint("[重要] 你选择的 tokenizer 不是原始实验使用的 meta-llama/Meta-Llama-3.1-8B。")
        eprint("       这会改变 token 序列和最终 loss，无法与原始 W&B run 做 bit-exact 对比。")
        eprint("       若要最大可能复现原始实验，请设置 TOKENIZER=meta-llama/Meta-Llama-3.1-8B。")
        eprint("=" * 100)
        eprint("")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    output_dir = args.output_dir.expanduser().resolve()
    work_dir = output_dir / "work"
    config_path = work_dir / "run_config.json"
    manifest_path = output_dir / "manifest.json"
    target_tokens = args.blocks * args.block_size * args.seq_len
    counts = block_counts(MIX_WEIGHTS, args.block_size)

    print("=" * 100, flush=True)
    print("Route B cache builder", flush=True)
    print(f"  output:       {output_dir}", flush=True)
    print(f"  tokenizer:    {args.tokenizer} (revision={args.tokenizer_revision or 'auto'})", flush=True)
    print(f"  mixture:      {MIX_WEIGHTS}", flush=True)
    print(f"  block counts: {counts} (block_size={args.block_size}, seq_len={args.seq_len})", flush=True)
    print(
        f"  target:       {args.blocks} blocks = {fmt_count(target_tokens)} tokens; "
        f"cache ≈ {fmt_bytes(target_tokens * 4)}",
        flush=True,
    )
    print(f"  HF endpoint:  {args.hf_endpoint}", flush=True)
    print("=" * 100, flush=True)

    if args.verify_only:
        check_imports(for_write=True)
        result = verify_tree_cache(output_dir / "train", target_tokens)
        print(json.dumps(result, indent=2), flush=True)
        return 0

    if args.overwrite and work_dir.exists():
        shutil.rmtree(work_dir)
    if args.overwrite and (output_dir / "validation").exists():
        shutil.rmtree(output_dir / "validation")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the selected file list on resume.  This avoids depending on the
    # mirror's current repository state and makes an interrupted run restartable.
    if config_path.exists() and args.resume:
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        fresh_fingerprint = build_fingerprint(args)
        if run_config.get("fingerprint") != fresh_fingerprint:
            raise SystemExit(
                f"现有 work/run_config.json 的参数指纹不一致。若要重建，请加 --overwrite；"
                f"当前输出目录: {output_dir}"
            )
    else:
        if work_dir.exists() and any(work_dir.iterdir()) and not args.overwrite:
            raise SystemExit(
                f"{work_dir} 非空但没有可复用的 run_config.json。请加 --overwrite 或换输出目录。"
            )
        run_config = build_run_config(args)
        atomic_json_write(config_path, run_config)

    source_files = {
        name: [RepoFile.from_dict(item) for item in items]
        for name, items in run_config["source_files"].items()
    }
    paloma_files = [RepoFile.from_dict(item) for item in run_config["paloma_files"]]
    raw_bytes = sum(file.size for files in source_files.values() for file in files)
    if not args.skip_paloma:
        raw_bytes += sum(file.size for file in paloma_files)
    download_s, tokenize_s, assemble_s = estimate_time(args, raw_bytes, target_tokens)

    print(f"[estimate] selected raw shards: {fmt_bytes(raw_bytes)}", flush=True)
    print(f"[estimate] token cache: {fmt_bytes(target_tokens * 4)}", flush=True)
    print(
        f"[estimate] 下载 ~{fmt_duration(download_s)}，tokenize ~{fmt_duration(tokenize_s)}，"
        f"组装 ~{fmt_duration(assemble_s)}；实际通常落在 0.7x–2x 区间",
        flush=True,
    )
    print("[estimate] 目标机器: H100/5090 均可做训练；本脚本只做 CPU tokenize/cache 构建。", flush=True)

    if args.dry_run:
        check_imports(for_write=False)
        dry_tokenizer = None
        try:
            revision = resolve_tokenizer_revision(args.tokenizer, args.tokenizer_revision or None, args.hf_endpoint, args.hf_token)
            dry_tokenizer = LevanterStyleTokenizer(args.tokenizer, revision, args.hf_token)
            print(f"[dry-run] tokenizer OK: {dry_tokenizer.metadata}", flush=True)
        except Exception as exc:
            print(f"[dry-run] tokenizer 未验证（安装完整环境后会自动检查）: {exc}", flush=True)
        print("[dry-run] 不下载、不写 cache。", flush=True)
        print(json.dumps(_manifest_from_config(run_config, args, raw_bytes, None, None, dry_tokenizer), indent=2, ensure_ascii=False), flush=True)
        return 0

    check_imports(for_write=True)
    revision = resolve_tokenizer_revision(args.tokenizer, args.tokenizer_revision or None, args.hf_endpoint, args.hf_token)
    if args.resume and run_config.get("resolved_tokenizer_revision"):
        revision = run_config["resolved_tokenizer_revision"]
    tokenizer = LevanterStyleTokenizer(args.tokenizer, revision, args.hf_token)
    run_config["resolved_tokenizer_revision"] = revision
    run_config["tokenizer_metadata"] = tokenizer.metadata
    atomic_json_write(config_path, run_config)

    reporter = ProgressReporter(target_tokens, interval=args.progress_seconds)
    reporter.set_raw_total(raw_bytes)

    paloma_result: dict[str, Any] | None = None
    if not args.skip_paloma:
        paloma_raw_bytes = sum(file.size for file in paloma_files)
        # Paloma c4_en is small; this estimate only drives the progress display.
        reporter.begin_stage("paloma", max(1, int(paloma_raw_bytes * 4)))
        paloma_result = build_paloma_cache(PALOMA_SPEC, paloma_files, output_dir, args, tokenizer, reporter)

    train_complete = (
        args.resume and not args.overwrite and train_cache_is_complete(output_dir, target_tokens, manifest_path)
    )
    if train_complete:
        print("[train] 已完成且 manifest 匹配，跳过 tokenize/assembly。", flush=True)
    else:
        reporter.begin_stage("tokenize", target_tokens)
        workers = []
        source_states: dict[str, SourceState] = {}
        for name, spec in SOURCE_SPECS.items():
            source_chunks = counts.get(name, 0) * args.blocks
            state = SourceState(source=name)
            state_path = work_dir / "source_state" / f"{name}.json"
            if state_path.exists() and args.resume:
                state = SourceState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
            source_states[name] = state
            if source_chunks <= 0:
                continue
            workers.append(SourceWorker(spec, source_files[name], source_chunks, args, work_dir, reporter, tokenizer))
        # On resume, make the ETA reflect the work already committed to disk.
        for state in source_states.values():
            reporter.update(tokens_delta=state.chunks_written * SEQ_LEN, source=state.source)
            reporter.add_bytes(sum(file.size for file in source_files.get(state.source, [])[: state.file_index]))
        run_parallel_sources(workers, reporter, args.num_procs)
        for worker in workers:
            source_states[worker.spec.name] = worker.state
        for name, state in source_states.items():
            expected = counts.get(name, 0) * args.blocks
            if state.chunks_written != expected:
                raise RuntimeError(f"{name}: 状态 chunks {state.chunks_written} != {expected}")
        reporter.begin_stage("assemble", target_tokens)
        train_result = write_train_cache(
            output_dir,
            args.blocks,
            args.block_size,
            counts,
            source_states,
            args.seed,
            reporter,
            overwrite=True,
        )
        reporter.maybe_print(force=True)
        print(f"[train] 完成: {train_result}", flush=True)
        if not args.keep_chunks:
            for state in source_states.values():
                try:
                    (work_dir / "chunks" / f"{state.source}.bin").unlink()
                    (work_dir / "chunks" / f"{state.source}.tail.npy").unlink()
                except FileNotFoundError:
                    pass

    if not (output_dir / "train" / "shard_ledger.json").exists():
        raise RuntimeError("train cache 未生成成功")
    train_verify = verify_tree_cache(output_dir / "train", target_tokens)
    manifest = _manifest_from_config(run_config, args, raw_bytes, train_verify, paloma_result, tokenizer)
    atomic_json_write(manifest_path, manifest)
    (output_dir / "README.md").write_text(_render_readme(manifest), encoding="utf-8")
    print("=" * 100, flush=True)
    print(f"[done] actual tokens: {fmt_count(train_verify['tokens'])}", flush=True)
    print(f"[done] train cache:   {train_verify['cache_dir']}", flush=True)
    print(f"[done] total time:    {fmt_duration(time.monotonic() - reporter.global_start)}", flush=True)
    print(f"[done] manifest:      {manifest_path}", flush=True)
    print("=" * 100, flush=True)
    print(manifest["config_snippet"], flush=True)
    return 0


def build_fingerprint(args: argparse.Namespace) -> str:
    obj = {
        "blocks": args.blocks,
        "block_size": args.block_size,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "weights": MIX_WEIGHTS,
        "dclm_shards": args.dclm_shards,
        "starcoder_shards": args.starcoder_shards,
        "proofpile_shards": args.proofpile_shards,
    }
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _manifest_from_config(
    run_config: dict[str, Any],
    args: argparse.Namespace,
    raw_bytes: int,
    train_verify: dict[str, Any] | None,
    paloma_result: dict[str, Any] | None,
    tokenizer: LevanterStyleTokenizer | None,
) -> dict[str, Any]:
    target_tokens = args.blocks * args.block_size * args.seq_len
    source_tokens = {
        name: count * args.blocks * args.seq_len
        for name, count in run_config["block_counts"].items()
    }
    manifest = {
        "route": "B",
        "description": "Route-B sampled DCLM mixture; not the original 4.1T globally shuffled cache",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_tokens": target_tokens,
        "actual_tokens": train_verify["tokens"] if train_verify else None,
        "blocks": args.blocks,
        "block_size": args.block_size,
        "seq_len": args.seq_len,
        "block_counts": run_config["block_counts"],
        "source_tokens": source_tokens,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": run_config.get("resolved_tokenizer_revision") or args.tokenizer_revision or None,
        "tokenizer_metadata": tokenizer.metadata if tokenizer else run_config.get("tokenizer_metadata"),
        "original_tokenizer": ORIGINAL_TOKENIZER,
        "tokenizer_matches_original": args.tokenizer == ORIGINAL_TOKENIZER,
        "hf_endpoint": args.hf_endpoint,
        "raw_bytes_selected": raw_bytes,
        "source_files": run_config["source_files"],
        "paloma_files": run_config["paloma_files"],
        "paloma": paloma_result,
        "train_cache": train_verify,
        "config_snippet": make_config_snippet(Path(args.output_dir).expanduser().resolve(), args.tokenizer),
        "warnings": [
            "Route B samples raw shards; it does not reproduce the original global 4.1T-token permutation.",
            "The original experiment used meta-llama/Meta-Llama-3.1-8B; another tokenizer changes token IDs and loss.",
            "The cache is pre-mixed and must be trained with shuffle=False to preserve block order.",
        ],
    }
    return manifest


def _render_readme(manifest: dict[str, Any]) -> str:
    return f"""# Route B cache

This directory was generated by `scripts/route_b/build_route_b_cache.py`.

- train tokens: `{manifest['actual_tokens']}`
- target tokens: `{manifest['target_tokens']}`
- blocks: `{manifest['blocks']}` x `{manifest['block_size']}` sequences x `{manifest['seq_len']}` tokens
- tokenizer: `{manifest['tokenizer']}` @ `{manifest['tokenizer_revision']}`
- original tokenizer: `{manifest['original_tokenizer']}`
- tokenizer matches original: `{manifest['tokenizer_matches_original']}`
- selected raw bytes: `{manifest['raw_bytes_selected']}`

## Important

This is Route B: a sampled approximation of the original DCLM mixture, not the
original globally shuffled 4.1T-token cache. Train with `shuffle=False`; the
blocks are already mixed and shuffled internally.

## Training config

```python
{manifest['config_snippet']}
```

For the original tokenizer, rebuild the cache with
`TOKENIZER=meta-llama/Meta-Llama-3.1-8B` after accepting the gated model terms.
"""


if __name__ == "__main__":
    raise SystemExit(main())
