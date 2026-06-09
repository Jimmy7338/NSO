#!/usr/bin/env bash
# 论文方法完整训练：语义+结构奖励 + RPN 可达性 + 可选回环/SSC
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
mkdir -p "$RUN_ROOT"

PRETRAINED="$PROJECT_DIR/pretrained_models"
LOAD_ARGS=()
if [[ -f "$PRETRAINED/model_best.global" ]]; then
  LOAD_ARGS=(
    --load_global "$PRETRAINED/model_best.global"
    --load_local "$PRETRAINED/model_best.local"
    --load_slam "$PRETRAINED/model_best.slam"
  )
fi

echo "论文模式训练 | 输出: $RUN_ROOT/models/paper_h2/"

exec bash "$SCRIPT_DIR/run_nso.sh" \
  --habitat_version 2 \
  --task_config pointnav_gibson.yaml \
  --split "${NSO_GIBSON_SPLIT:-val}" \
  --eval 0 \
  --paper_rewards 1 \
  --use_structural_reward 1 \
  --use_intrinsic_goal_penalty 1 \
  --use_semantic \
  --semantic_use_all_classes \
  --semantic_conf_thresh 0.15 \
  --semantic_interval 1 \
  --use_goal_reachability \
  --train_goal_reachability \
  --reachability_mask_alpha 2.0 \
  --goal_reachability_max_candidates 16 \
  --use_loop_detection \
  --loop_pose_correction 1 \
  --loop_interval 100 \
  --use_ssc_completion \
  --ssc_update_interval 10 \
  --train_global 1 --train_local 1 --train_slam 1 \
  -na 0 -no 0 -v 0 --print_images 0 \
  --auto_gpu_config 1 \
  --num_episodes 1000000 \
  --max_episode_length 500 \
  -d "$RUN_ROOT/" \
  --exp_name paper_h2 \
  "${LOAD_ARGS[@]}" \
  "$@"
