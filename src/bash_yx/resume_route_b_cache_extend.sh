#!/usr/bin/env bash
set -euo pipefail

# 用途：修复 Route-B DCLM shard 数量不足导致的 chunk 校验失败，并原地续跑。
# 背景：现有 work/run_config.json 选了 40 个 DCLM shard，实际只产出
#       1,204,513 / 1,214,080 chunks；本脚本保留已完成的 40 个 shard，
#       只追加新的 shard，然后调用常规入口继续 tokenize/assembly。
# 可变参数：
#   OUTPUT_DIR    输出目录，默认 dclm_mix_5.37B。
#   DCLM_SHARDS   修复后的 DCLM shard 数，默认 41。
#   PATCH_ONLY=1  只更新 run_config.json，不启动 builder，便于检查。
#   HF_TOKEN、HF_ENDPOINT、PIP_INDEX_URL、KEEP_RAW、NUM_PROCS 等
#   继续沿用 run_route_b_cache.sh 的环境变量约定。
# 使用：HF_TOKEN=... bash src/bash_yx/resume_route_b_cache_extend.sh

date

REPO_DIR="${REPO_DIR:-/mnt/afs/250010233/marin-kaiyue-optimizers}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/250010233/data/marin_route_b/dclm_mix_5.37B}"
DCLM_SHARDS="${DCLM_SHARDS:-41}"
PATCH_ONLY="${PATCH_ONLY:-0}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/250010233/conda}"
CONDA_ENV="${CONDA_ENV:-route-b-cache}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${CONDA_ROOT}/envs/${CONDA_ENV}}"

cd "${REPO_DIR}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${CONDA_ENV_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_ENV_PREFIX}/bin/python"
  else
    PYTHON="python3"
  fi
fi

: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${PIP_INDEX_URL:=https://mirrors.aliyun.com/pypi/simple}"
: "${KEEP_RAW:=1}"
export HF_ENDPOINT PIP_INDEX_URL KEEP_RAW
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

echo "[repair] output:      ${OUTPUT_DIR}"
echo "[repair] dclm shards: ${DCLM_SHARDS}"
echo "[repair] python:      ${PYTHON}"

"${PYTHON}" - "${REPO_DIR}" "${OUTPUT_DIR}" "${DCLM_SHARDS}" <<'PY'
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


repo_dir = Path(sys.argv[1]).expanduser().resolve()
output_dir = Path(sys.argv[2]).expanduser().resolve()
desired_shards = int(sys.argv[3])
config_path = output_dir / "work" / "run_config.json"
state_path = output_dir / "work" / "source_state" / "dclm_baseline.json"
chunk_path = output_dir / "work" / "chunks" / "dclm_baseline.bin"


def fail(message: str) -> None:
    print(f"[repair][error] {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


if not config_path.is_file():
    fail(f"找不到 {config_path}；这个入口只用于修复已经跑过 tokenize 的输出目录")
if not state_path.is_file():
    fail(f"找不到 {state_path}；缺少断点状态，不能安全续跑")
if desired_shards < 1:
    fail("DCLM_SHARDS 必须大于 0")

module_path = repo_dir / "scripts" / "route_b" / "build_route_b_cache.py"
module_spec = importlib.util.spec_from_file_location("route_b_cache_builder_repair", module_path)
if module_spec is None or module_spec.loader is None:
    fail(f"无法加载 {module_path}")
module = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = module
module_spec.loader.exec_module(module)

config = json.loads(config_path.read_text(encoding="utf-8"))
dclm_source = module.SOURCE_SPECS["dclm_baseline"]
entries = list(config["source_files"]["dclm_baseline"])
state = json.loads(state_path.read_text(encoding="utf-8"))
chunks_written = int(state.get("chunks_written", 0))
file_index = int(state.get("file_index", 0))
target_chunks = int(config["block_counts"]["dclm_baseline"]) * int(config["blocks"])

if chunks_written <= 0:
    fail("DCLM 断点 chunks_written 为 0；这不是本次可续跑的短缺口场景")
if file_index > len(entries):
    fail(f"断点 file_index={file_index} 超过已选文件数 {len(entries)}，拒绝修改配置")
if chunks_written >= target_chunks:
    print(f"[repair] DCLM 已满足目标：{chunks_written} / {target_chunks} chunks，无需补 shard")
    raise SystemExit(0)
if desired_shards < len(entries):
    fail(f"DCLM_SHARDS={desired_shards} 小于现有文件数 {len(entries)}；缩容会破坏断点")

print(f"[repair] 当前断点：{chunks_written} / {target_chunks} chunks，缺 {target_chunks - chunks_written}")

if desired_shards == len(entries):
    print(f"[repair] run_config 已经包含 {desired_shards} 个 DCLM shard，跳过配置更新")
else:
    if file_index != len(entries):
        fail(
            f"准备追加 shard 时，断点 file_index={file_index} 必须等于现有文件数 {len(entries)}"
        )
    listing = module.fetch_repo_listing(
        dclm_source.repo_id,
        dclm_source.revision,
        config.get("hf_endpoint") or os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        os.environ.get("HF_TOKEN"),
    )
    fresh = module.select_source_files(dclm_source, listing, desired_shards, int(config["seed"]))
    fresh_by_path = {item.path: item for item in fresh}
    old_paths = [str(item["path"]) for item in entries]
    old_path_set = set(old_paths)
    missing = [path for path in old_paths if path not in fresh_by_path]
    if missing:
        fail(
            "新 shard 数对应的文件集合不再包含已有文件，直接续跑会错位；"
            f"第一个缺失文件: {missing[0]}"
        )
    extra = [item for item in fresh if item.path not in old_path_set]
    expected_extra = desired_shards - len(entries)
    if len(extra) != expected_extra:
        fail(f"预计追加 {expected_extra} 个文件，实际算出 {len(extra)} 个")

    backup_path = config_path.with_name(f"run_config.json.before_dclm{desired_shards}.bak")
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)

    config["source_files"]["dclm_baseline"] = entries + [item.to_dict() for item in extra]
    fingerprint_args = argparse.Namespace(
        blocks=int(config["blocks"]),
        block_size=int(config["block_size"]),
        seq_len=int(config["seq_len"]),
        seed=int(config["seed"]),
        tokenizer=config["tokenizer"],
        tokenizer_revision=config.get("tokenizer_revision", ""),
        dclm_shards=desired_shards,
        starcoder_shards=len(config["source_files"]["starcoderdata"]),
        proofpile_shards=len(config["source_files"]["proofpile_2"]),
    )
    config["fingerprint"] = module.build_fingerprint(fingerprint_args)
    module.atomic_json_write(config_path, config)
    print(f"[repair] 已保留前 {len(entries)} 个 DCLM shard，并追加 {len(extra)} 个：")
    for item in extra:
        print(f"[repair]   + {item.size:>12,d}  {item.path}")
    print(f"[repair] 配置备份: {backup_path}")

expected_bin_bytes = chunks_written * module.CHUNK_BYTES
actual_bin_bytes = chunk_path.stat().st_size if chunk_path.is_file() else 0
if actual_bin_bytes < expected_bin_bytes:
    fail(f"{chunk_path} 小于断点记录：{actual_bin_bytes} < {expected_bin_bytes}")
print(f"[repair] chunk 文件检查通过：{actual_bin_bytes:,d} bytes")
print("[repair] run_config.json 已就绪，可以续跑")
PY

if [[ "${PATCH_ONLY}" == "1" ]]; then
  echo "[repair] PATCH_ONLY=1，仅更新配置，退出。"
  date
  exit 0
fi

export REPO_DIR OUTPUT_DIR DCLM_SHARDS
bash "${REPO_DIR}/src/bash_yx/run_route_b_cache.sh"

date
