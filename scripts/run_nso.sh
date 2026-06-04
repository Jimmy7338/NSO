#!/usr/bin/env bash
# 激活 NSO 环境并运行 main.py（用法: bash scripts/run_nso.sh [main.py 参数...]）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
if [[ "${NSO_HABITAT_VERSION:-}" == "2" ]]; then
  ENV_NAME="${ENV_NAME:-nso_h2}"
else
  ENV_NAME="${ENV_NAME:-nso}"
fi
export ENV_NAME

if [[ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]]; then
  echo "未找到 conda，请先运行: bash scripts/install_server.sh" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "[环境] 使用 conda: $ENV_NAME ($(command -v python))"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Habitat 2：优先 third_party/habitat-lab，避免误用 env/habitat/habitat_api
if [[ "${NSO_HABITAT_VERSION:-}" == "2" ]]; then
  H2_LAB="$PROJECT_DIR/third_party/habitat-lab/habitat-lab"
  if [[ -d "$H2_LAB" ]]; then
    export PYTHONPATH="$H2_LAB:$PYTHONPATH"
  fi
  if ! python -c "import hydra; from habitat.config import read_write" 2>/dev/null; then
    echo "错误: 环境 $ENV_NAME 缺少 hydra，请: conda activate $ENV_NAME && bash scripts/install_habitat2.sh" >&2
    exit 1
  fi
fi

# habitat-sim 在服务器上须用 xvfb + NVIDIA 渲染；勿与 MobaXterm X11 转发混用（会 GLXBadContextTag）
USE_XVFB_GPU=0
if [[ -n "${NSO_USE_XVFB_GPU:-}" ]]; then
  USE_XVFB_GPU=1
elif [[ -z "${DISPLAY:-}" ]]; then
  USE_XVFB_GPU=1
fi

if [[ "$USE_XVFB_GPU" == 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
fi

# 实时可视化：NSO_X11_DISPLAY 已保存时仍用 TkAgg；纯 xvfb 无弹窗时用 Agg
if [[ "$USE_XVFB_GPU" == 1 ]] && [[ -z "${NSO_X11_DISPLAY:-}" ]]; then
  export MPLBACKEND="${MPLBACKEND:-Agg}"
elif [[ -n "${NSO_X11_DISPLAY:-}" ]] || [[ -n "${DISPLAY:-}" ]]; then
  export MPLBACKEND="${MPLBACKEND:-TkAgg}"
else
  export MPLBACKEND="${MPLBACKEND:-Agg}"
fi

# 无 CUDA 时回退 CPU；有 GPU 时不加 --no_cuda
EXTRA_ARGS=()
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[设备] PyTorch CUDA 可用: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
  echo "[设备] PyTorch 未检测到 CUDA，使用 CPU（可安装 cu117 版 torch）"
  EXTRA_ARGS+=(--no_cuda)
fi

if [[ "$USE_XVFB_GPU" == 1 ]] && command -v xvfb-run >/dev/null; then
  exec xvfb-run -a python main.py "${EXTRA_ARGS[@]}" "$@"
else
  # MobaXterm X11 转发：不要强制 NVIDIA GLX
  unset __GLX_VENDOR_LIBRARY_NAME
  exec python main.py "${EXTRA_ARGS[@]}" "$@"
fi
