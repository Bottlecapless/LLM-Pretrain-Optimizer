# Route B cache for the 130M optimizer sweep

This builder avoids downloading the original multi-terabyte tokenized DCLM
cache. It samples raw shards from the same DCLM mixture, tokenizes them locally,
and writes a Levanter `TreeCache` with the same block-deterministic mixture as
`experiments/dclm/tokenize_dclm.py`.

It is a controlled approximation of the original data pipeline, not a bit-exact
reconstruction of the unavailable globally shuffled 4.1T-token cache.

## Output

- `train/`: 640 blocks x 2048 sequences x 4096 tokens = **5,368,709,120 tokens**
  (20.0 GiB as int32).
- `validation/paloma_c4_en/validation/`: Paloma `c4_en` validation cache.
- `manifest.json`: selected shards, resolved revisions, token counts, and a
  ready-to-use Levanter data-config snippet.
- `work/`: resumable source state, `.part` downloads, and token fragments.

The 1x run uses the first 320 blocks (2,684,354,560 tokens); the 2x run uses all
640 blocks. Both use the same cache and must be configured with `shuffle=False`.

## Entry point

Run all commands from the repository root. The script uses a persistent conda
environment at `/mnt/afs/250010233/conda/envs/route-b-cache`, so dependencies
survive pod restarts. Prepare it once before the first data download:

```bash
REPO_DIR="$PWD" \
SETUP_ONLY=1 \
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
bash src/bash_yx/run_route_b_cache.sh
```

This creates the conda environment if needed and installs the pinned CPU-only
dependencies with `uv`. Subsequent runs reuse it and skip the install:

```bash
REPO_DIR="$PWD" \
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
HF_ENDPOINT=https://hf-mirror.com \
OUTPUT_DIR=/data/marin_route_b/dclm_mix_5.37B \
bash src/bash_yx/run_route_b_cache.sh
```

Set `CONDA_ENV` or `CONDA_ENV_PREFIX` to use a different persistent environment.
`SKIP_CONDA=1` remains available for a pre-activated Python environment. The
builder resumes from `$OUTPUT_DIR/work`; downloaded `.part` files, source state,
chunks, and `$OUTPUT_DIR/.hf_home` are kept on AFS. Set `KEEP_RAW=1` if you also
want to retain successfully processed raw shards across restarts.

Before downloading anything, run the dry run. It only lists shards and prints
the selected size/time estimate:

```bash
REPO_DIR="$PWD" DRY_RUN=1 \
OUTPUT_DIR=/tmp/marin_route_b_dry \
bash src/bash_yx/run_route_b_cache.sh
```

After a build, verify the cache without modifying it:

```bash
REPO_DIR="$PWD" VERIFY_ONLY=1 \
OUTPUT_DIR=/data/marin_route_b/dclm_mix_5.37B \
bash src/bash_yx/run_route_b_cache.sh
```

Use a different output directory for every parameter set. Reusing the same
directory with a different fingerprint intentionally fails unless `OVERWRITE=1`.

## Defaults and resources

- Sources: DCLM baseline + StarCoderData + Proof-Pile-2, weights `3.8 / 0.25 / 0.055`.
- Dataset revisions are pinned to full commits:
  - DCLM: `a3b142c183aebe5af344955ae20836eb34dcf69b`
  - StarCoderData: `9fc30b578cedaec69e47302df72cf00feed7c8c4`
  - Proof-Pile-2: `901a9273a770e9d4138c5ddd91802f9c5c6cdc4b`
  - Paloma: `65cd6fc59dba021b21db414fa5e8d7765ffbe5e6`
- Shards: `40 / 16 / 6`; dry run selects about **16.9 GiB** of raw shards.
- Final int32 cache: **20.0 GiB**.
- Peak disk: about **42 GiB** by default; about **60 GiB** with `KEEP_RAW=1`.
- At an assumed 80 MB/s mirror, 1.5M token/s per worker with 3 workers, and
  8M token/s assembly: download about 4 min, tokenize about 20 min, assembly
  about 11 min. The script prints actual speed and ETA every 10 seconds.
- `hf-mirror.com` is heavily rate-limited. HTTP 429 can make the wall-clock time
  several times longer; the downloader uses Range requests, `.part` resume, and
  exponential backoff.

The mirror entry point is simply `HF_ENDPOINT=https://hf-mirror.com`; large
files are fetched directly from that endpoint. It is a transport mirror and does
not bypass gated dataset access.

## Tokenizer

The original sweep used `meta-llama/Meta-Llama-3.1-8B`. The requested public
default is `marin-community/marin-tokenizer`, pinned by default to commit
`a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`. Its BOS/EOS IDs are 128000/128001,
matching the Llama 3 tokenizer interface.

For the closest possible reproduction after accepting the gated model terms:

```bash
TOKENIZER=meta-llama/Meta-Llama-3.1-8B HF_TOKEN=hf_xxx \
bash src/bash_yx/run_route_b_cache.sh
```

The script warns whenever the tokenizer is not the original model name.

## Paloma access

`allenai/paloma` is `gated: auto`. If the pod has no accepted license/token and
Paloma returns 401/403, first build the training cache with `SKIP_PALOMA=1`.
That leaves only the validation cache missing; the training cache itself is
unaffected.

## Training data config

Use the snippet emitted in `manifest.json`. At a high level:

```python
from levanter.data.text import LMMixtureDatasetConfig, UrlDatasetSourceConfig

cache_root = "/data/marin_route_b/dclm_mix_5.37B"
data = LMMixtureDatasetConfig(
    configs={
        "route_b_dclm_mix": UrlDatasetSourceConfig(
            train_urls=[], validation_urls=[], cache_dir=cache_root
        ),
        "paloma/c4_en": UrlDatasetSourceConfig(
            train_urls=[], validation_urls=[],
            cache_dir=f"{cache_root}/validation/paloma_c4_en",
        ),
    },
    train_weights={"route_b_dclm_mix": 1.0, "paloma/c4_en": 0.0},
    tokenizer="marin-community/marin-tokenizer",
    shuffle=False,
)
```

Do not set `shuffle=True`: Route B performs component-level and block-level
shuffling while writing the cache.

## Remaining non-bit-exact differences

- Raw shards are sampled; the original 4.1T globally shuffled token cache is not
  used.
- The default tokenizer is the requested public Marin tokenizer rather than the
  gated original Llama 3 tokenizer.
- The original sweep trained on TPU; H100/5090 numerics and batching differ.
- Only Paloma `c4_en` validation is built here; the original default validation
  set also included the other Paloma subsets.
- The sweep launcher and template need a GPU launch path; this script only
  prepares data.
