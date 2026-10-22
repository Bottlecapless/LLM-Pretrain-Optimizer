#!/usr/bin/env bash
set -euo pipefail

# 用途：在单独 Pod 上构建 130M optimizer sweep 的 Route-B 训练缓存。
# 数据：DCLM baseline + StarCoderData + Proof-Pile-2，按原始 3.8/0.25/0.055
#       权重混合；默认 640 个 block，每 block 2048 条 4096-token 序列。
# 产物：$OUTPUT_DIR/train（Levanter TreeCache）以及 Paloma c4_en validation cache。
# 可变参数：全部通过前置环境变量覆盖，常用项见下方。
# 环境：默认自动使用 AFS 上持久化的 conda 环境 route-b-cache（不存在则创建）；
#       SETUP_ONLY=1 只准备环境后退出；SKIP_CONDA=1 或显式 PYTHON 可绕过 conda。
# 重启：依赖在 AFS 的 conda 环境中；OUTPUT_DIR/work、.part、HF_HOME 均可续用。
# 路径：REPO_DIR 可覆盖仓库目录；输出默认写到 AFS 的 data/marin_route_b。
# 严格原始复现：TOKENIZER=meta-llama/Meta-Llama-3.1-8B（需要 gated 授权）。
# 运行：bash src/bash_yx/run_route_b_cache.sh

date

REPO_DIR="${REPO_DIR:-/mnt/afs/250010233/marin-kaiyue-optimizers}"
cd "${REPO_DIR}"

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/250010233/conda}"
CONDA_BOOTSTRAP="${CONDA_BOOTSTRAP:-${CONDA_ROOT}/conda_bootstrap.sh}"
CONDA_ENV="${CONDA_ENV:-route-b-cache}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${CONDA_ROOT}/envs/${CONDA_ENV}}"
SKIP_CONDA="${SKIP_CONDA:-0}"
SETUP_ONLY="${SETUP_ONLY:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CONDA_ROOT}/pip-cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${CONDA_ROOT}/uv-cache}"
REQUIREMENTS_FILE="${REPO_DIR}/scripts/route_b/requirements-cache.txt"

if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ "${SKIP_CONDA}" == "1" ]]; then
  PYTHON="python"
else
  if [[ ! -f "${CONDA_BOOTSTRAP}" ]]; then
    echo "[error] 找不到 ${CONDA_BOOTSTRAP}。请设置 SKIP_CONDA=1 并自行提供 PYTHON，" >&2
    echo "        或通过 CONDA_BOOTSTRAP=/path/to/conda_bootstrap.sh 指定。" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${CONDA_BOOTSTRAP}"
  if [[ ! -x "${CONDA_ENV_PREFIX}/bin/python" ]]; then
    echo "[setup] 创建持久 conda 环境: ${CONDA_ENV_PREFIX}"
    if [[ -e "${CONDA_ENV_PREFIX}" ]]; then
      case "${CONDA_ENV_PREFIX}" in
        "${CONDA_ROOT}/envs/"*)
          echo "[setup] 清理不完整环境: ${CONDA_ENV_PREFIX}"
          rm -rf -- "${CONDA_ENV_PREFIX}"
          ;;
        *)
          echo "[error] ${CONDA_ENV_PREFIX} 已存在但不是有效的 Python 环境，且不在 ${CONDA_ROOT}/envs 下。" >&2
          exit 2
          ;;
      esac
    fi
    CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
    if [[ -z "${CONDA_BIN}" ]]; then
      CONDA_BIN="${CONDA_ROOT}/miniconda3/bin/conda"
    fi
    # conda-forge 不需要接受 Anaconda 默认 channel 的 ToS，且只需下载约 40 MB。
    if ! "${CONDA_BIN}" create --prefix "${CONDA_ENV_PREFIX}" \
      -c conda-forge --override-channels python=3.10 pip -y; then
      echo "[error] conda 环境创建失败。可手动执行：" >&2
      echo "        ${CONDA_BIN} create --prefix ${CONDA_ENV_PREFIX} -c conda-forge --override-channels python=3.10 pip -y" >&2
      exit 2
    fi
  fi
  export CONDA_PREFIX="${CONDA_ENV_PREFIX}"
  export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"
  PYTHON="${CONDA_ENV_PREFIX}/bin/python"
fi

: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${HF_API_TIMEOUT:=180}"
: "${TOKENIZER:=marin-community/marin-tokenizer}"
: "${TOKENIZER_REVISION:=a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2}"
: "${OUTPUT_DIR:=/mnt/afs/250010233/data/marin_route_b/dclm_mix_5.37B}"
: "${HF_HOME:=${OUTPUT_DIR}/.hf_home}"
: "${HF_HUB_CACHE:=${HF_HOME}/hub}"
: "${BLOCKS:=640}"
: "${DCLM_SHARDS:=41}"
: "${STARCODER_SHARDS:=16}"
: "${PROOFPILE_SHARDS:=6}"
: "${NUM_PROCS:=3}"
: "${PROGRESS_SECONDS:=10}"
: "${DOWNLOAD_RETRIES:=6}"
: "${KEEP_RAW:=0}"
: "${KEEP_CHUNKS:=0}"
: "${SKIP_PALOMA:=0}"
: "${DRY_RUN:=0}"
: "${RESUME:=1}"
: "${OVERWRITE:=0}"
: "${VERIFY_ONLY:=0}"
: "${INSTALL_DEPS:=1}"

if [[ ! "${HF_ENDPOINT}" =~ ^https?://[A-Za-z0-9._-]+(:[0-9]+)?/?$ ]]; then
  echo "[error] HF_ENDPOINT 格式不正确: '${HF_ENDPOINT}'" >&2
  echo "        常见原因是行尾反斜杠前少了空格，导致下一行被拼进来。" >&2
  echo "        正确示例: HF_ENDPOINT=https://hf-mirror.com \\" >&2
  exit 2
fi

export HF_ENDPOINT
export HF_API_TIMEOUT
export HF_HOME
export HF_HUB_CACHE
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

deps_ready() {
  DRY_RUN="${DRY_RUN}" "${PYTHON}" - <<'PY' >/dev/null 2>&1
import importlib.util
import os
if os.environ.get("DRY_RUN") == "1":
    mods = ["numpy", "requests"]
else:
    mods = ["numpy", "requests", "transformers", "tokenizers", "huggingface_hub", "pyarrow", "fsspec", "zstandard", "levanter"]
raise SystemExit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)
PY
}

ensure_deps() {
  local env_prefix uv_bin stamp_file req_hash py_version stamp_value
  if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "[error] 找不到依赖清单: ${REQUIREMENTS_FILE}" >&2
    exit 2
  fi
  env_prefix="$("${PYTHON}" -c 'import sys; print(sys.prefix)')"
  uv_bin="${env_prefix}/bin/uv"
  stamp_file="${env_prefix}/.route_b_requirements.sha256"
  req_hash="$(sha256sum "${REQUIREMENTS_FILE}" | awk '{print $1}')"
  py_version="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  stamp_value="sha256=${req_hash} python=${py_version}"

  if deps_ready; then
    if [[ ! -f "${stamp_file}" || "$(cat "${stamp_file}")" != "${stamp_value}" ]]; then
      printf '%s\n' "${stamp_value}" > "${stamp_file}"
    fi
    return 0
  fi
  if [[ "${INSTALL_DEPS}" != "1" ]]; then
    echo "[error] 当前 Python 环境缺少 transformers/tokenizers/levanter 等依赖。" >&2
    echo "        可设置 INSTALL_DEPS=1 自动安装，或先运行 SETUP_ONLY=1 预热环境。" >&2
    exit 2
  fi
  mkdir -p "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}"
  local pip_args=(-r "${REQUIREMENTS_FILE}")
  if [[ -n "${PIP_INDEX_URL}" ]]; then
    pip_args+=(--index-url "${PIP_INDEX_URL}")
  fi
  local pip_env=(PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= PIP_CACHE_DIR="${PIP_CACHE_DIR}")

  echo "[setup] 当前环境缺少依赖，安装 CPU-only 缓存依赖..."
  if [[ ! -x "${uv_bin}" ]]; then
    echo "[setup] 安装 uv（仅首次）..."
    local uv_pip_args=(--disable-pip-version-check)
    if [[ -n "${PIP_INDEX_URL}" ]]; then
      uv_pip_args+=(--index-url "${PIP_INDEX_URL}")
    fi
    if ! env "${pip_env[@]}" "${PYTHON}" -m pip install "${uv_pip_args[@]}" uv; then
      echo "[warn] uv 安装失败，将直接使用 pip。" >&2
    fi
  fi

  if [[ -x "${uv_bin}" ]]; then
    echo "[setup] 使用 uv 解析并安装依赖..."
    local uv_args=(pip install --python "${PYTHON}")
    if [[ -n "${PIP_INDEX_URL}" ]]; then
      uv_args+=(--index-url "${PIP_INDEX_URL}")
    fi
    uv_args+=(-r "${REQUIREMENTS_FILE}")
    if ! env UV_CACHE_DIR="${UV_CACHE_DIR}" PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
      "${uv_bin}" "${uv_args[@]}"; then
      echo "[warn] uv 安装失败，回退到 pip。" >&2
      env "${pip_env[@]}" "${PYTHON}" -m pip install "${pip_args[@]}"
    fi
  else
    env "${pip_env[@]}" "${PYTHON}" -m pip install "${pip_args[@]}"
  fi

  if ! deps_ready; then
    echo "[error] 依赖安装完成后仍无法导入所需模块。" >&2
    exit 2
  fi
  printf '%s\n' "${stamp_value}" > "${stamp_file}"
  echo "[setup] 依赖已就绪，环境状态已记录到 ${stamp_file}"
}

ensure_deps
if [[ "${SETUP_ONLY}" == "1" ]]; then
  echo "[setup] 环境已准备好，退出（SETUP_ONLY=1）。"
  echo "[setup] conda env: ${CONDA_ENV_PREFIX}"
  echo "[setup] python:    ${PYTHON}"
  exit 0
fi

echo "============================================================"
echo "Route B cache builder"
echo "  CONDA_ENV:       ${CONDA_ENV}"
echo "  SKIP_CONDA:      ${SKIP_CONDA}"
echo "  PYTHON:          ${PYTHON}"
echo "  HF_ENDPOINT:     ${HF_ENDPOINT}"
echo "  HF_HOME:         ${HF_HOME}"
echo "  TOKENIZER:       ${TOKENIZER}"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  BLOCKS:          ${BLOCKS} (block=2048 seq, seq=4096 token)"
echo "  SHARDS:          dclm=${DCLM_SHARDS} star=${STARCODER_SHARDS} proof=${PROOFPILE_SHARDS}"
echo "  NUM_PROCS:       ${NUM_PROCS}"
echo "  PIP_INDEX_URL:   ${PIP_INDEX_URL:-<pip default>}"
echo "  KEEP_RAW:        ${KEEP_RAW}"
echo "  SKIP_PALOMA:     ${SKIP_PALOMA}"
echo "  DRY_RUN:         ${DRY_RUN}"
echo "============================================================"
echo "[disk] 默认峰值约 42 GiB；KEEP_RAW=1 时约 60 GiB。"
echo "[network] 数据走 ${HF_ENDPOINT}；镜像 429 时脚本会退避重试。"

ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --blocks "${BLOCKS}"
  --hf-endpoint "${HF_ENDPOINT}"
  --tokenizer "${TOKENIZER}"
  --tokenizer-revision "${TOKENIZER_REVISION}"
  --dclm-shards "${DCLM_SHARDS}"
  --starcoder-shards "${STARCODER_SHARDS}"
  --proofpile-shards "${PROOFPILE_SHARDS}"
  --num-procs "${NUM_PROCS}"
  --progress-seconds "${PROGRESS_SECONDS}"
  --download-retries "${DOWNLOAD_RETRIES}"
)

if [[ "${KEEP_RAW}" == "1" ]]; then
  ARGS+=(--keep-raw)
else
  ARGS+=(--no-keep-raw)
fi
if [[ "${KEEP_CHUNKS}" == "1" ]]; then
  ARGS+=(--keep-chunks)
else
  ARGS+=(--no-keep-chunks)
fi
if [[ "${SKIP_PALOMA}" == "1" ]]; then
  ARGS+=(--skip-paloma)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--no-resume)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${VERIFY_ONLY}" == "1" ]]; then
  ARGS+=(--verify-only)
fi

"${PYTHON}" -u scripts/route_b/build_route_b_cache.py "${ARGS[@]}"

date
