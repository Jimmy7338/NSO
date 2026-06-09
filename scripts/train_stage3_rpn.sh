#!/usr/bin/env bash
# 阶段 3：在阶段 2 基础上训练 RPN 可达性头（FMM 监督 BCE）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
export MPLBACKEND=Agg
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
mkdir -p "$RUN_ROOT"

PREV="${STAGE3_LOAD_DIR:-$RUN_ROOT/models/stage2_paper_global}"
PRETRAINED="$SCRIPT_DIR/../pretrained_models"
GLOBAL_CKPT="$PREV/model_best.global"
[[ -f "$GLOBAL_CKPT" ]] || GLOBAL_CKPT="$PRETRAINED/model_best.global"
REACH="${STAGE3_REACH_LOAD:-0}"
LOAD_ARGS=(
  --load_slam "$PREV/model_best.slam"
  --load_global "$GLOBAL_CKPT"
  --load_local "$PREV/model_best.local"
)
if [[ -f "$PREV/model_best.reach" ]]; then
  REACH="$PREV/model_best.reach"
fi
if [[ "$REACH" != "0" && -f "$REACH" ]]; then
  LOAD_ARGS+=(--goal_reachability_model_path "$REACH")
fi

echo "阶段 3 | RPN | 加载: $PREV"

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
  --use_goal_reachability \
  --train_goal_reachability \
  --reachability_mask_alpha 2.0 \
  --goal_reachability_max_candidates 16 \
  --train_slam 0 --train_local 1 --train_global 1 \
  --auto_gpu_config 1 \
  --num_episodes 1000000 \
  --max_episode_length 500 \
  --save_interval 1 \
  --save_periodic 50000 \
  --log_interval 10 \
  -na 0 -no 0 -v 0 --print_images 0 \
  -d "$RUN_ROOT/" \
  --exp_name stage3_rpn \
  "${LOAD_ARGS[@]}" \
  "$@"
