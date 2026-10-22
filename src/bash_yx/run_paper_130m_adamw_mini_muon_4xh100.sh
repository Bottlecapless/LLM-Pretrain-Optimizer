#!/usr/bin/env bash
set -euo pipefail

# 用途：在 4 张 H100 上复现 130M 模型的 6 个论文场景，并在 Route-B
#       dclm_mix_5.37B cache 上记录 Paloma C4/EN validation loss：
#       AdamW (1x/2x)、Adam-mini (1x/2x)、Muon (1x/2x)。
# 数据：ROUTE_B_CACHE_ROOT 指向带有 train/ 和 Paloma c4_en validation 的本地 cache。
# 资源：每个 run 占 1 张 H100；6 个 run 由 Ray 在 4 张卡上排队执行。
# 产物：checkpoints 与 scenario manifest 写入 MARIN_PREFIX；loss 默认写入
#       WANDB_ENTITY/WANDB_PROJECT 的 W&B run。
# 可变参数（环境变量）：
#   GPU_IDS=0,1,2,3          可用 GPU 编号；已有 CUDA_VISIBLE_DEVICES 时不覆盖
#   EXPECT_GPUS=4            启动前要求可见的 H100 数量
#   SCENARIOS=all            可填 adamw_1x,adamw_2x,adam_mini_1x,adam_mini_2x,muon_1x,muon_2x
#   MARIN_PREFIX=...         checkpoint / manifest 输出根目录
#   ROUTE_B_CACHE_ROOT=...   Route-B cache 根目录
#   WANDB_ENTITY=marin-community
#   WANDB_PROJECT=optimizer-scaling
#   WANDB_GROUP=paper130m-adamw-mini-muon
#   WANDB_MODE=online        online 时必须提供 WANDB_API_KEY（或已有 ~/.netrc）
#   DRY_RUN=0                设为 1 只打印展开后的 6 个配置，不提交训练
# 运行：bash src/bash_yx/run_paper_130m_adamw_mini_muon_4xh100.sh

date

REPO_DIR="${REPO_DIR:-/mnt/afs/250010233/marin-kaiyue-optimizers}"
cd "${REPO_DIR}"

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/250010233/conda}"
CONDA_BOOTSTRAP="${CONDA_BOOTSTRAP:-${CONDA_ROOT}/conda_bootstrap.sh}"
CONDA_ENV="${CONDA_ENV:-route-b-cache}"

if [[ ! -f "${CONDA_BOOTSTRAP}" ]]; then
  echo "[error] 找不到 conda bootstrap: ${CONDA_BOOTSTRAP}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONDA_BOOTSTRAP}"
conda activate "${CONDA_ENV}"
PYTHON="$(command -v python)"

: "${GPU_IDS:=0,1,2,3}"
: "${EXPECT_GPUS:=4}"
: "${SCENARIOS:=all}"
: "${MARIN_PREFIX:=/mnt/afs/250010233/data/marin_optimizer_paper/130m_adamw_mini_muon}"
: "${ROUTE_B_CACHE_ROOT:=/mnt/afs/250010233/data/marin_route_b/dclm_mix_5.37B}"
: "${ROUTE_B_TOKENIZER:=marin-community/marin-tokenizer}"
: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${HF_HOME:=${ROUTE_B_CACHE_ROOT}/.hf_home}"
: "${HF_HUB_CACHE:=${HF_HOME}/hub}"
: "${WANDB_ENTITY:=marin-community}"
: "${WANDB_PROJECT:=optimizer-scaling}"
: "${WANDB_GROUP:=paper130m-adamw-mini-muon}"
: "${WANDB_MODE:=online}"
: "${DRY_RUN:=0}"
: "${SKIP_WANDB_CHECK:=0}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

export MARIN_PREFIX
export ROUTE_B_CACHE_ROOT
export ROUTE_B_TOKENIZER
export HF_ENDPOINT
export HF_HOME
export HF_HUB_CACHE
export WANDB_ENTITY
export WANDB_PROJECT
export WANDB_GROUP
export WANDB_MODE
export SCENARIOS
export DRY_RUN
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ "${DRY_RUN}" != "1" && "${WANDB_MODE}" =~ ^(online|shared)$ && -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" && "${SKIP_WANDB_CHECK}" != "1" ]]; then
  echo "[error] WANDB_MODE=${WANDB_MODE}，但未找到 WANDB_API_KEY 或 ~/.netrc。" >&2
  echo "        请先 export WANDB_API_KEY=...；或使用 WANDB_MODE=offline。" >&2
  exit 2
fi

if [[ ! -d "${ROUTE_B_CACHE_ROOT}/train" ]]; then
  echo "[error] 找不到训练 cache: ${ROUTE_B_CACHE_ROOT}/train" >&2
  exit 2
fi

mkdir -p "${MARIN_PREFIX}"

echo "============================================================"
echo "130M optimizer paper reproduction"
echo "  PYTHON:              ${PYTHON}"
echo "  GPU_IDS:             ${CUDA_VISIBLE_DEVICES}"
echo "  EXPECT_GPUS:         ${EXPECT_GPUS}"
echo "  SCENARIOS:           ${SCENARIOS}"
echo "  ROUTE_B_CACHE_ROOT:  ${ROUTE_B_CACHE_ROOT}"
echo "  MARIN_PREFIX:        ${MARIN_PREFIX}"
echo "  W&B:                 ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_GROUP}, mode=${WANDB_MODE})"
echo "============================================================"
nvidia-smi -L || true

VISIBLE_GPUS="$("${PYTHON}" - <<'PY'
import jax

print(jax.device_count())
PY
)"
echo "[gpu] JAX devices visible: ${VISIBLE_GPUS}"
if [[ "${DRY_RUN}" != "1" ]] && (( VISIBLE_GPUS < EXPECT_GPUS )); then
  echo "[error] 需要 ${EXPECT_GPUS} 张可见 GPU，但 JAX 只看到 ${VISIBLE_GPUS} 张。" >&2
  exit 2
fi

"${PYTHON}" experiments/optimizer_sweep/run_paper_130m_adamw_mini_muon.py

date
