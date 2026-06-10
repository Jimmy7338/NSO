#!/usr/bin/env bash
# 阶段 4：SSC 语义补全 + NetVLAD 回环 + 位姿校正
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
export MPLBACKEND=Agg
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
mkdir -p "$RUN_ROOT"

PREV="${STAGE4_LOAD_DIR:-$RUN_ROOT/models/stage3_rpn}"
STAGE2="${STAGE4_SLAM_FALLBACK_DIR:-$RUN_ROOT/models/stage2_paper_global}"
PRETRAINED="$SCRIPT_DIR/../pretrained_models"

# 阶段 3 train_slam=0 时通常不会写出 model_best.slam，按优先级回退
SLAM_CKPT="$PREV/model_best.slam"
if [[ ! -f "$SLAM_CKPT" ]]; then
  if [[ -f "$STAGE2/model_best.slam" ]]; then
    SLAM_CKPT="$STAGE2/model_best.slam"
    echo "警告: $PREV 无 model_best.slam，回退 stage2: $SLAM_CKPT"
  elif [[ -f "$PRETRAINED/model_best.slam" ]]; then
    SLAM_CKPT="$PRETRAINED/model_best.slam"
    echo "警告: 回退 pretrained: $SLAM_CKPT"
  else
    echo "错误: 找不到 model_best.slam（已查 stage3/stage2/pretrained）" >&2
    exit 1
  fi
fi

GLOBAL_CKPT="$PREV/model_best.global"
[[ -f "$GLOBAL_CKPT" ]] || GLOBAL_CKPT="$PRETRAINED/model_best.global"

LOCAL_CKPT="$PREV/model_best.local"
if [[ ! -f "$LOCAL_CKPT" && -f "$STAGE2/model_best.local" ]]; then
  LOCAL_CKPT="$STAGE2/model_best.local"
  echo "警告: $PREV 无 model_best.local，回退 stage2: $LOCAL_CKPT"
fi

REACH="${STAGE4_REACH_LOAD:-0}"
LOAD_ARGS=(
  --load_slam "$SLAM_CKPT"
  --load_global "$GLOBAL_CKPT"
  --load_local "$LOCAL_CKPT"
)
if [[ -f "$PREV/model_best.reach" ]]; then
  REACH="$PREV/model_best.reach"
fi
if [[ "$REACH" != "0" && -f "$REACH" ]]; then
  LOAD_ARGS+=(--goal_reachability_model_path "$REACH")
fi

echo "阶段 4 | SSC+回环 | 加载: $PREV"

exec bash "$SCRIPT_DIR/run_nso.sh" \
  --habitat_version 2 \
  --task_config pointnav_gibson.yaml \
  --split "${NSO_GIBSON_SPLIT:-val}" \
  --eval 0 \
  --paper_rewards 1 \
  --use_structural_reward 1 \
  --use_intrinsic_goal_penalty 1 \
  --use_semantic \
  --use_goal_reachability \
  --train_goal_reachability \
  --reachability_mask_alpha 2.0 \
  --use_ssc_completion \
  --ssc_update_interval 10 \
  --use_loop_detection \
  --loop_pose_correction 1 \
  --loop_interval 100 \
  --train_slam 0 --train_local 0 --train_global 1 \
  --auto_gpu_config 1 \
  --num_episodes 1000000 \
  --max_episode_length 500 \
  --save_interval 1 \
  --save_periodic 50000 \
  --log_interval 10 \
  -na 0 -no 0 -v 0 --print_images 0 \
  -d "$RUN_ROOT/" \
  --exp_name stage4_ssc_loop \
  "${LOAD_ARGS[@]}" \
  "$@"
