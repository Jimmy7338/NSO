#!/usr/bin/env bash
# 等待阶段 4：10k 步做中期评估，25k 步做四阶段完整对比评估
# 进程提前退出 != 达到目标步数；可自动重启 stage4 训练
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/eval_pipeline.log"

export NSO_RUN_ROOT="$RUN_ROOT"
export MPLBACKEND=Agg
export ENV_NAME="${ENV_NAME:-nso_h2}"
export NSO_GIBSON_SPLIT="${NSO_GIBSON_SPLIT:-train}"

STAGE4_LOG="$RUN_ROOT/models/stage4_ssc_loop/train.log"
STAGE4_CONSOLE="$LOG_DIR/stage4_ssc_loop.log"
MID_TARGET="${STAGE4_MID_STEPS:-10000}"
FINAL_TARGET="${STAGE4_FINAL_STEPS:-25000}"
POLL_SEC="${POLL_SEC:-120}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
STAGE4_AUTO_RESTART="${STAGE4_AUTO_RESTART:-1}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PIPELINE_LOG"; }

current_steps() {
  [[ -f "$STAGE4_LOG" ]] || { echo 0; return; }
  grep -oP 'num timesteps \K[0-9]+' "$STAGE4_LOG" 2>/dev/null | sort -n | tail -1 || echo 0
}

is_stage4_alive() {
  pgrep -f "main.py.*stage4_ssc_loop" >/dev/null 2>&1
}

start_stage4() {
  log "启动/重启阶段4 训练 (STAGE4_RESUME=1)..."
  export STAGE4_RESUME=1
  bash "$SCRIPT_DIR/run_stage_in_tmux.sh" nso_stage4 \
    "$SCRIPT_DIR/train_stage4_ssc_loop.sh" "$STAGE4_CONSOLE"
  sleep 60
}

wait_steps() {
  local target="$1" label="$2"
  local restarts=0
  log "等待阶段4 ${label} (${target} 步，当前 $(current_steps))..."
  while true; do
    local s; s="$(current_steps)"
    if [[ "$s" -ge "$target" ]]; then
      log "阶段4 已达 ${s} 步 (>= ${target})"
      return 0
    fi
    if ! is_stage4_alive; then
      log "阶段4 进程已退出于 ${s}/${target} 步，未达目标"
      if [[ "$STAGE4_AUTO_RESTART" == "1" && "$restarts" -lt "$MAX_RESTARTS" ]]; then
        restarts=$((restarts + 1))
        log "自动重启阶段4 (${restarts}/${MAX_RESTARTS})..."
        start_stage4 || true
        continue
      fi
      log "错误: 阶段4 无法继续至 ${target} 步，流水线中止"
      return 1
    fi
    log "阶段4 进度: ${s}/${target}"
    sleep "$POLL_SEC"
  done
}

log "========== 评估流水线启动 =========="
log "中期目标: ${MID_TARGET} | 最终: ${FINAL_TARGET} | eval_episodes=${EVAL_EPISODES}"
log "自动重启: ${STAGE4_AUTO_RESTART} | 最大重启: ${MAX_RESTARTS}"

bash "$SCRIPT_DIR/restore_global_ckpt.sh" 2>&1 | tee -a "$PIPELINE_LOG"

# 若未在跑且未达中期目标，先拉起训练
if [[ "$(current_steps)" -lt "$MID_TARGET" ]] && ! is_stage4_alive; then
  start_stage4
fi

if [[ "$(current_steps)" -lt "$MID_TARGET" ]]; then
  wait_steps "$MID_TARGET" "中期(SSC/回环)" || exit 1
fi

log "===== 中期评估 (stage4 @ >=${MID_TARGET}步) ====="
bash "$SCRIPT_DIR/restore_global_ckpt.sh" stage4_ssc_loop 2>&1 | tee -a "$PIPELINE_LOG"
export EVAL_EPISODES="$EVAL_EPISODES"
bash "$SCRIPT_DIR/eval_stage_checkpoint.sh" stage4 "$EVAL_EPISODES" 2>&1 | tee -a "$LOG_DIR/eval_stage4_mid.txt" || true

log "===== 中期对比: stage3 vs stage4 ====="
bash "$SCRIPT_DIR/eval_stage_checkpoint.sh" stage3 "$EVAL_EPISODES" 2>&1 | tee -a "$LOG_DIR/eval_stage3_mid.txt" || true

if [[ "$(current_steps)" -lt "$FINAL_TARGET" ]]; then
  if ! is_stage4_alive; then
    start_stage4
  fi
  wait_steps "$FINAL_TARGET" "最终" || exit 1
fi

log "===== 最终四阶段对比评估 ====="
bash "$SCRIPT_DIR/restore_global_ckpt.sh" 2>&1 | tee -a "$PIPELINE_LOG"
bash "$SCRIPT_DIR/run_all_stages_eval.sh" 2>&1 | tee -a "$LOG_DIR/eval_all_stages_final.txt"

log "========== 评估流水线完成 =========="
