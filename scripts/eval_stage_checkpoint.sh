#!/usr/bin/env bash
# 通用 val 评估：按阶段加载 checkpoint 并输出覆盖率
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
export MPLBACKEND=Agg
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
PRETRAINED="${PRETRAINED_DIR:-$SCRIPT_DIR/../pretrained_models}"
STAGE3_DIR="${STAGE3_LOAD_DIR:-$RUN_ROOT/models/stage3_rpn}"
STAGE2_DIR="${STAGE2_LOAD_DIR:-$RUN_ROOT/models/stage2_paper_global}"

STAGE="${1:?用法: eval_stage_checkpoint.sh <stage1|stage2|stage3|stage4> [episodes]}"
EVAL_EPISODES="${2:-${EVAL_EPISODES:-20}}"
SPLIT="${EVAL_SPLIT:-val}"

case "$STAGE" in
  stage1|1) STAGE_DIR="${STAGE_LOAD_DIR:-$RUN_ROOT/models/stage1_slam_local}"; EXP="stage1_eval" ;;
  stage2|2) STAGE_DIR="${STAGE_LOAD_DIR:-$RUN_ROOT/models/stage2_paper_global}"; EXP="stage2_eval" ;;
  stage3|3) STAGE_DIR="${STAGE_LOAD_DIR:-$RUN_ROOT/models/stage3_rpn}"; EXP="stage3_eval" ;;
  stage4|4) STAGE_DIR="${STAGE_LOAD_DIR:-$RUN_ROOT/models/stage4_ssc_loop}"; EXP="stage4_eval" ;;
  *) echo "未知阶段: $STAGE" >&2; exit 1 ;;
esac

bash "$SCRIPT_DIR/restore_global_ckpt.sh" "$(basename "$STAGE_DIR")" >/dev/null

resolve_ckpt() {
  local name="$1"
  local primary="$STAGE_DIR/model_best.$name"
  if [[ -f "$primary" ]]; then
    echo "$primary"
    return 0
  fi
  if [[ "$STAGE" == stage4|4 ]]; then
    for fb in "$STAGE3_DIR/model_best.$name" "$STAGE2_DIR/model_best.$name" "$PRETRAINED/model_best.$name"; do
      if [[ -f "$fb" ]]; then
        echo "警告: $STAGE_DIR 无 model_best.$name，回退 $fb" >&2
        echo "$fb"
        return 0
      fi
    done
  fi
  return 1
}

SLAM_CKPT="$(resolve_ckpt slam)" || { echo "缺少 model_best.slam（已查 stage 目录与回退）" >&2; exit 1; }
LOCAL_CKPT="$(resolve_ckpt local)" || { echo "缺少 model_best.local" >&2; exit 1; }

GLOBAL_CKPT="$STAGE_DIR/model_best.global"
[[ -f "$GLOBAL_CKPT" ]] || GLOBAL_CKPT="$PRETRAINED/model_best.global"

EXTRA=()
case "$STAGE" in
  stage2|2|stage3|3|stage4|4)
    EXTRA+=(--paper_rewards 1 --use_structural_reward 1 --use_intrinsic_goal_penalty 1 --use_semantic)
    ;;
esac
case "$STAGE" in
  stage3|3|stage4|4)
    EXTRA+=(--use_goal_reachability)
    REACH_CKPT="$STAGE_DIR/model_best.reach"
    if [[ -f "$REACH_CKPT" ]]; then
      EXTRA+=(--goal_reachability_model_path "$REACH_CKPT")
      RPN_CH="$(bash "$SCRIPT_DIR/infer_rpn_channels.sh" "$REACH_CKPT")"
      EXTRA+=(--rpn_in_channels "$RPN_CH")
      echo "  reach: $REACH_CKPT (${RPN_CH}ch)"
    fi
    ;;
esac
case "$STAGE" in
  stage4|4)
    EXTRA+=(--use_ssc_completion --use_loop_detection --loop_pose_correction 1)
    ;;
esac

echo "========== 评估 $STAGE | $EVAL_EPISODES ep | split=$SPLIT =========="
echo "  dir: $STAGE_DIR"
echo "  global: $GLOBAL_CKPT"
echo "  slam: $SLAM_CKPT"
echo "  local: $LOCAL_CKPT"

bash "$SCRIPT_DIR/run_nso.sh" \
  --habitat_version 2 \
  --task_config pointnav_gibson.yaml \
  --split "$SPLIT" \
  --eval 1 \
  --train_slam 0 --train_local 0 --train_global 0 \
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

LOG="$RUN_ROOT/models/$EXP/train.log"
if [[ -f "$LOG" ]]; then
  python3 - "$LOG" "$STAGE" "$EVAL_EPISODES" <<'PY'
import re, sys
log, stage, n_ep = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(log, encoding="utf-8", errors="ignore").read()
ratios = []
for m in re.finditer(r"Final Exp Ratio:\s*\n([0-9., \n]+)", text):
    vals = [float(x.strip()) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    if vals:
        ratios.append(max(vals))
if ratios:
    import statistics as st
    print(f"[结果] {stage}: episodes={len(ratios)} max_ratio={max(ratios):.4f} "
          f"mean_max={st.mean(ratios):.4f} median={st.median(ratios):.4f}")
else:
    print(f"[结果] {stage}: 未在日志中找到 Final Exp Ratio，见 {log}")
PY
fi
