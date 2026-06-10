#!/usr/bin/env bash
# 从 RPN checkpoint 推断输入通道数（2 或 4）
set -euo pipefail
CKPT="${1:?用法: infer_rpn_channels.sh <model_best.reach>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
python3 - "$CKPT" <<'PY'
import sys
from env.habitat.reachability_utils import infer_rpn_in_channels
print(infer_rpn_in_channels(sys.argv[1]))
PY
