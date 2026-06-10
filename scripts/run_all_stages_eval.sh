#!/usr/bin/env bash
# 四阶段统一 val 评估并汇总覆盖率
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/eval_summary_$(date +%Y%m%d_%H%M%S).txt"
EPISODES="${EVAL_EPISODES:-20}"
SPLIT="${EVAL_SPLIT:-val}"

export NSO_RUN_ROOT="$RUN_ROOT"
export MPLBACKEND=Agg
export EVAL_EPISODES="$EPISODES"
export EVAL_SPLIT="$SPLIT"

{
  echo "===== NSO 四阶段 val 评估 ====="
  echo "时间: $(date '+%F %T')"
  echo "split=$SPLIT episodes=$EPISODES"
  echo

  bash "$SCRIPT_DIR/restore_global_ckpt.sh"

  for stage in stage1 stage2 stage3 stage4; do
    echo "--- $stage ---"
  done

  for stage in 1 2 3 4; do
    echo
    bash "$SCRIPT_DIR/eval_stage_checkpoint.sh" "$stage" "$EPISODES" 2>&1 || \
      echo "[警告] stage$stage 评估失败"
  done

  echo
  echo "===== 汇总 ====="
  for f in stage1_eval stage2_eval stage3_eval stage4_eval; do
    log="$RUN_ROOT/models/$f/train.log"
    if [[ -f "$log" ]]; then
      ratio="$(grep -A1 'Final Exp Ratio' "$log" | tail -1 | tr ',' '\n' | grep -oP '[0-9.]+' | sort -g | tail -1 || true)"
      echo "$f: max_exp_ratio=${ratio:-N/A}"
    else
      echo "$f: 无日志"
    fi
  done
} 2>&1 | tee "$SUMMARY"

echo "汇总已写入: $SUMMARY"
