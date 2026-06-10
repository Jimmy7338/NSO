#!/usr/bin/env bash
# 阶段 4：SSC 语义补全 + NetVLAD 回环 + 位姿校正
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
export MPLBACKEND=Agg
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
mkdir -p "$RUN_ROOT"

STAGE3="${STAGE3_LOAD_DIR:-$RUN_ROOT/models/stage3_rpn}"
STAGE4="${STAGE4_LOAD_DIR:-$RUN_ROOT/models/stage4_ssc_loop}"
STAGE2="${STAGE4_SLAM_FALLBACK_DIR:-$RUN_ROOT/models/stage2_paper_global}"
PRETRAINED="$SCRIPT_DIR/../pretrained_models"

# STAGE4_RESUME=1：从 stage4 目录恢复 global/reach，slam/local 仍用 stage3
RESUME="${STAGE4_RESUME:-0}"
if [[ "$RESUME" == "1" || -f "$STAGE4/model_best.reach" ]]; then
  WEIGHT_SRC="$STAGE3"
  GLOBAL_CKPT="$STAGE4/model_best.global"
  [[ -f "$GLOBAL_CKPT" ]] || GLOBAL_CKPT="$STAGE3/model_best.global"
  REACH="$STAGE4/model_best.reach"
  if [[ ! -f "$REACH" ]]; then
    REACH="$STAGE3/model_best.reach"
  fi
  echo "阶段4 恢复训练 | global/reach 来自 stage4，骨干来自 stage3"
else
  WEIGHT_SRC="$STAGE3"
  GLOBAL_CKPT="$STAGE3/model_best.global"
  REACH="$STAGE3/model_best.reach"
  echo "阶段 4 | SSC+回环 | 加载: $STAGE3"
fi

[[ -f "$GLOBAL_CKPT" ]] || GLOBAL_CKPT="$PRETRAINED/model_best.global"

SLAM_CKPT="$WEIGHT_SRC/model_best.slam"
if [[ ! -f "$SLAM_CKPT" ]]; then
  if [[ -f "$STAGE2/model_best.slam" ]]; then
    SLAM_CKPT="$STAGE2/model_best.slam"
    echo "警告: stage3 无 model_best.slam，回退 stage2: $SLAM_CKPT"
  elif [[ -f "$PRETRAINED/model_best.slam" ]]; then
    SLAM_CKPT="$PRETRAINED/model_best.slam"
    echo "警告: 回退 pretrained: $SLAM_CKPT"
  else
    echo "错误: 找不到 model_best.slam" >&2
    exit 1
  fi
fi

LOCAL_CKPT="$WEIGHT_SRC/model_best.local"
if [[ ! -f "$LOCAL_CKPT" && -f "$STAGE2/model_best.local" ]]; then
  LOCAL_CKPT="$STAGE2/model_best.local"
  echo "警告: stage3 无 model_best.local，回退 stage2: $LOCAL_CKPT"
fi

LOAD_ARGS=(
  --load_slam "$SLAM_CKPT"
  --load_global "$GLOBAL_CKPT"
  --load_local "$LOCAL_CKPT"
)
if [[ -n "${REACH:-}" && -f "$REACH" ]]; then
  RPN_CH="$(bash "$SCRIPT_DIR/infer_rpn_channels.sh" "$REACH")"
  LOAD_ARGS+=(--goal_reachability_model_path "$REACH" --rpn_in_channels "$RPN_CH")
  echo "  reach: $REACH (${RPN_CH}ch, 自动推断)"
fi

PRIOR_STEPS=0
if [[ -f "$RUN_ROOT/models/stage4_ssc_loop/train.log" ]]; then
  PRIOR_STEPS="$(grep -oP 'num timesteps \K[0-9]+' "$RUN_ROOT/models/stage4_ssc_loop/train.log" 2>/dev/null | sort -n | tail -1 || echo 0)"
fi
echo "  历史步数(日志): ${PRIOR_STEPS}（本次从 checkpoint 热启动，步数计数器归零）"

exec bash "$SCRIPT_DIR/run_nso.sh" \
  --habitat_version 2 \
  --task_config pointnav_gibson.yaml \
  --split "${NSO_GIBSON_SPLIT:-train}" \
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
