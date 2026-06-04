#!/usr/bin/env bash
# Habitat 2 环境自检
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
export ENV_NAME="${ENV_NAME:-nso_h2}"
export NSO_HABITAT_VERSION=2

if [[ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]]; then
  echo "未找到 conda: $CONDA_DIR" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/third_party/habitat-lab/habitat-lab:${PYTHONPATH:-}"

echo "python: $(command -v python)"
python -c "
import importlib.util, os
p = 'env/habitat2/_lab.py'
s = importlib.util.spec_from_file_location('l', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.setup_habitat2_lab()
import habitat
from habitat.config import read_write
import hydra
import torch
print('habitat-lab:', habitat.__file__)
print('hydra:', hydra.__version__)
print('torch cuda:', torch.cuda.is_available())
print('OK — 可运行: bash scripts/run_nso_h2_vis.sh')
"
