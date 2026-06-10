#!/usr/bin/env bash
# 单独启动 NSO Live 查看器（可在第二个 MobaXterm 终端里先运行，再跑 run_nso_vis.sh）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-nso}"
LIVE_DIR="${NSO_LIVE_VIS_DIR:-/tmp/nso_vis_live}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "错误: 需要 MobaXterm X11（DISPLAY 未设置）" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "$LIVE_DIR"
unset MPLBACKEND
exec python "$SCRIPT_DIR/nso_live_viewer.py" "$LIVE_DIR"
