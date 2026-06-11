#!/usr/bin/env bash
# 评估 paper_fast 训练权重（论文模式，val split）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
export MPLBACKEND=Agg

RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
PRETRAINED="${PRETRAINED_DIR:-$PROJECT_DIR/pretrained_models}"
MODEL_DIR="${PAPER_FAST_DIR:-$RUN_ROOT/models/paper_fast}"
EVAL_EPISODES="${1:-${EVAL_EPISODES:-20}}"
SPLIT="${EVAL_SPLIT:-val}"
EXP="paper_fast_eval"

GLOBAL_CKPT="$MODEL_DIR/model_best.global"
REACH_CKPT="$MODEL_DIR/model_best.reach"
SLAM_CKPT="$PRETRAINED/model_best.slam"
LOCAL_CKPT="$PRETRAINED/model_best.local"

for f in "$GLOBAL_CKPT" "$SLAM_CKPT" "$LOCAL_CKPT"; do
  [[ -f "$f" ]] || { echo "缺少权重: $f" >&2; exit 1; }
done

EXTRA=()
if [[ -f "$REACH_CKPT" ]]; then
  RPN_CH="$(bash "$SCRIPT_DIR/infer_rpn_channels.sh" "$REACH_CKPT")"
  EXTRA+=(--goal_reachability_model_path "$REACH_CKPT" --rpn_in_channels "$RPN_CH")
  echo "  reach: $REACH_CKPT (${RPN_CH}ch)"
fi

echo "========== paper_fast 评估 | $EVAL_EPISODES ep | split=$SPLIT =========="
echo "  global: $GLOBAL_CKPT"
echo "  slam:   $SLAM_CKPT"
echo "  local:  $LOCAL_CKPT"

bash "$SCRIPT_DIR/run_nso.sh" \
  --habitat_version 2 \
  --task_config pointnav_gibson.yaml \
  --split "$SPLIT" \
  --eval 1 \
  --paper_mode \
  --semantic_use_all_classes \
  --semantic_conf_thresh 0.15 \
  --semantic_interval 5 \
  --train_slam 0 --train_local 0 --train_global 0 \
  --use_loop_detection 0 \
  --rpn_mc_samples 5 \
  --num_processes 1 \
  --num_episodes "$EVAL_EPISODES" \
  --max_episode_length 500 \
  --log_interval 10 \
  -na 0 -no 0 -v 0 --print_images 0 \
  -d "$RUN_ROOT/" \
  --exp_name "$EXP" \
  --load_slam "$SLAM_CKPT" \
  --load_global "$GLOBAL_CKPT" \
  --load_local "$LOCAL_CKPT" \
  "${EXTRA[@]}"
