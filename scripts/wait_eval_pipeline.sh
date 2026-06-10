#!/usr/bin/env bash
# 等待阶段 4：10k 步做中期评估，25k 步做四阶段完整对比评估
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/eval_pipeline.log"

export NSO_RUN_ROOT="$RUN_ROOT"
export MPLBACKEND=Agg
export ENV_NAME="${ENV_NAME:-nso_h2}"

STAGE4_LOG="$RUN_ROOT/models/stage4_ssc_loop/train.log"
MID_TARGET="${STAGE4_MID_STEPS:-10000}"
FINAL_TARGET="${STAGE4_FINAL_STEPS:-25000}"
POLL_SEC="${POLL_SEC:-120}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PIPELINE_LOG"; }

current_steps() {
  [[ -f "$STAGE4_LOG" ]] || { echo 0; return; }
  grep -oP 'num timesteps \K[0-9]+' "$STAGE4_LOG" 2>/dev/null | sort -n | tail -1 || echo 0
}

wait_steps() {
  local target="$1" label="$2"
  log "等待阶段4 ${label} (${target} 步)..."
  while true; do
    local s; s="$(current_steps)"
    if [[ "$s" -ge "$target" ]]; then
      log "阶段4 已达 ${s} 步 (>= ${target})"
      return 0
    fi
    if ! pgrep -f "main.py.*stage4_ssc_loop" >/dev/null 2>&1; then
      log "阶段4 进程已结束于 ${s} 步"
      return 0
    fi
    log "阶段4 进度: ${s}/${target}"
    sleep "$POLL_SEC"
  done
}

log "========== 评估流水线启动 =========="
log "中期目标: ${MID_TARGET} | 最终: ${FINAL_TARGET} | eval_episodes=${EVAL_EPISODES}"

# 先恢复已有阶段的 global
bash "$SCRIPT_DIR/restore_global_ckpt.sh" 2>&1 | tee -a "$PIPELINE_LOG"

# 若阶段 4 未达 10k，等待
if [[ "$(current_steps)" -lt "$MID_TARGET" ]]; then
  wait_steps "$MID_TARGET" "中期(SSC/回环)"
fi

log "===== 中期评估 (stage4 @ >=${MID_TARGET}步) ====="
bash "$SCRIPT_DIR/restore_global_ckpt.sh" stage4_ssc_loop 2>&1 | tee -a "$PIPELINE_LOG"
export EVAL_EPISODES="$EVAL_EPISODES"
bash "$SCRIPT_DIR/eval_stage_checkpoint.sh" stage4 "$EVAL_EPISODES" 2>&1 | tee -a "$LOG_DIR/eval_stage4_mid.txt" || true

# 对比 stage3 vs stage4（SSC/回环收益）
log "===== 中期对比: stage3 vs stage4 ====="
bash "$SCRIPT_DIR/eval_stage_checkpoint.sh" stage3 "$EVAL_EPISODES" 2>&1 | tee -a "$LOG_DIR/eval_stage3_mid.txt" || true

# 等待阶段 4 完成
if [[ "$(current_steps)" -lt "$FINAL_TARGET" ]]; then
  wait_steps "$FINAL_TARGET" "最终"
fi

log "===== 最终四阶段对比评估 ====="
bash "$SCRIPT_DIR/restore_global_ckpt.sh" 2>&1 | tee -a "$PIPELINE_LOG"
bash "$SCRIPT_DIR/run_all_stages_eval.sh" 2>&1 | tee -a "$LOG_DIR/eval_all_stages_final.txt"

log "========== 评估流水线完成 =========="
