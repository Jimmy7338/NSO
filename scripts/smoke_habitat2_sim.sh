#!/usr/bin/env bash
# 仅测试 habitat-sim 0.2.4 + NSO 测试场景（不依赖 habitat-lab）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-nso_h2}"

# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export __GLX_VENDOR_LIBRARY_NAME=nvidia

exec xvfb-run -a python scripts/smoke_habitat2_sim.py
