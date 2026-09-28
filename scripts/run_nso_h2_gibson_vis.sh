#!/usr/bin/env bash
# Habitat 2 + Gibson PointNav + matplotlib 可视化
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$PROJECT_DIR/data/scene_datasets/gibson/Cantwell.glb" ]]; then
  echo "请先运行: bash scripts/setup_gibson_habitat.sh" >&2
  exit 1
fi

export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"

PRETRAINED="$PROJECT_DIR/pretrained_models"
LOAD_ARGS=()
if [[ -f "$PRETRAINED/model_best.global" ]]; then
  LOAD_ARGS=(
    --load_global "$PRETRAINED/model_best.global"
    --load_local "$PRETRAINED/model_best.local"
    --load_slam "$PRETRAINED/model_best.slam"
  )
fi

echo "Gibson PointNav val | 场景: data/scene_datasets/gibson/"

exec bash "$SCRIPT_DIR/run_nso_h2_vis.sh" \
  --task_config pointnav_gibson.yaml \
  --split val \
  --exp_name h2_gibson_vis \
  "${LOAD_ARGS[@]}" \
  "$@"
