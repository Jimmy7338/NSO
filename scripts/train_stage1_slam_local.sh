#!/usr/bin/env bash
# 阶段 1：Neural SLAM + 局部策略（监督/模仿），全局 PPO 关闭
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
export MPLBACKEND=Agg
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
mkdir -p "$RUN_ROOT"

PREV="${STAGE1_LOAD_DIR:-pretrained_models}"
LOAD_ARGS=(
  --load_slam "$PREV/model_best.slam"
  --load_global "$PREV/model_best.global"
  --load_local "$PREV/model_best.local"
)

echo "阶段 1 | SLAM+Local | split=${NSO_GIBSON_SPLIT:-val} | 输出: $RUN_ROOT/models/stage1_slam_local/"

exec bash "$SCRIPT_DIR/run_nso.sh" \
  --habitat_version 2 \
  --task_config pointnav_gibson.yaml \
  --split "${NSO_GIBSON_SPLIT:-val}" \
  --eval 0 \
  --train_slam 1 --train_local 1 --train_global 0 \
  --auto_gpu_config 1 \
  --num_episodes 1000000 \
  --max_episode_length 500 \
  --save_interval 1 \
  --save_periodic 50000 \
  --log_interval 10 \
  -na 0 -no 0 -v 0 --print_images 0 \
  -d "$RUN_ROOT/" \
  --exp_name stage1_slam_local \
  "${LOAD_ARGS[@]}" \
  "$@"
