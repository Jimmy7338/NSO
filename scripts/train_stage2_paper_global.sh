#!/usr/bin/env bash
# 阶段 2：论文式奖励 + 全局 PPO + 语义（需 ultralytics）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
export MPLBACKEND=Agg
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
mkdir -p "$RUN_ROOT"

STAGE2_DIR="$RUN_ROOT/models/stage2_paper_global"
if [[ -f "$STAGE2_DIR/model_best.slam" && -f "$STAGE2_DIR/model_best.local" ]]; then
  PREV="${STAGE2_LOAD_DIR:-$STAGE2_DIR}"
else
  PREV="${STAGE2_LOAD_DIR:-$RUN_ROOT/models/stage1_slam_local}"
fi
PRETRAINED="$SCRIPT_DIR/../pretrained_models"
GLOBAL_CKPT="$PREV/model_best.global"
[[ -f "$GLOBAL_CKPT" ]] || GLOBAL_CKPT="$PRETRAINED/model_best.global"
LOAD_ARGS=(
  --load_slam "$PREV/model_best.slam"
  --load_global "$GLOBAL_CKPT"
  --load_local "$PREV/model_best.local"
)

echo "阶段 2 | 论文奖励+Global+语义 | 加载: $PREV"

exec bash "$SCRIPT_DIR/run_nso.sh" \
  --habitat_version 2 \
  --task_config pointnav_gibson.yaml \
  --split "${NSO_GIBSON_SPLIT:-val}" \
  --eval 0 \
  --paper_rewards 1 \
  --use_structural_reward 1 \
  --use_intrinsic_goal_penalty 1 \
  --use_semantic \
  --semantic_conf_thresh 0.15 \
  --semantic_interval 1 \
  --train_slam 1 --train_local 1 --train_global 1 \
  --auto_gpu_config 1 \
  --num_episodes 1000000 \
  --max_episode_length 500 \
  --save_interval 1 \
  --save_periodic 50000 \
  --log_interval 10 \
  -na 0 -no 0 -v 0 --print_images 0 \
  -d "$RUN_ROOT/" \
  --exp_name stage2_paper_global \
  "${LOAD_ARGS[@]}" \
  "$@"
