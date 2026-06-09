#!/usr/bin/env bash
# 监控阶段 2 步数，达到目标后自动停止并依次跑阶段 3 → 4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

export NSO_RUN_ROOT="$RUN_ROOT"
export NSO_GIBSON_SPLIT="${NSO_GIBSON_SPLIT:-train}"
export MPLBACKEND=Agg
export ENV_NAME="${ENV_NAME:-nso_h2}"

STAGE2_SESSION="${STAGE2_SESSION:-nso_stage2}"
STAGE2_TARGET="${STAGE2_TARGET_STEPS:-30000}"
STAGE3_TARGET="${STAGE3_TARGET_STEPS:-25000}"
STAGE4_TARGET="${STAGE4_TARGET_STEPS:-25000}"
POLL_SEC="${POLL_SEC:-60}"

STAGE2_LOG="$RUN_ROOT/models/stage2_paper_global/train.log"
STAGE3_LOG="$RUN_ROOT/models/stage3_rpn/train.log"
STAGE4_LOG="$RUN_ROOT/models/stage4_ssc_loop/train.log"
PIPELINE_LOG="$LOG_DIR/pipeline_34.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PIPELINE_LOG"; }

# 从 train.log 取最新 num timesteps
current_steps() {
  local logfile="$1"
  [[ -f "$logfile" ]] || { echo 0; return; }
  grep -oP 'num timesteps \K[0-9]+' "$logfile" 2>/dev/null | tail -1 || echo 0
}

# 等待指定日志步数达到目标
wait_for_steps() {
  local logfile="$1" target="$2" label="$3"
  log "等待 $label 达到 ${target} 步 (当前: $(current_steps "$logfile"))..."
  while true; do
    local steps
    steps="$(current_steps "$logfile")"
    if [[ "$steps" -ge "$target" ]]; then
      log "$label 已达 ${steps} 步 (目标 ${target})"
      return 0
    fi
    log "$label 进度: ${steps}/${target} 步"
    sleep "$POLL_SEC"
  done
}

# 停止 tmux 训练会话
stop_session() {
  local session="$1"
  if tmux has-session -t "$session" 2>/dev/null; then
    log "停止 tmux 会话: $session"
    tmux send-keys -t "$session" C-c
    sleep 10
    local i
    for i in $(seq 1 12); do
      tmux has-session -t "$session" 2>/dev/null || break
      sleep 5
    done
    if tmux has-session -t "$session" 2>/dev/null; then
      log "强制结束: $session"
      tmux kill-session -t "$session"
    fi
  fi
}

# 在 tmux 中启动训练脚本
start_stage() {
  local session="$1" script="$2" logfile="$3"
  stop_session "$session"
  log "启动 $session -> $script"
  tmux new-session -d -s "$session" \
    "export NSO_RUN_ROOT=$RUN_ROOT NSO_GIBSON_SPLIT=$NSO_GIBSON_SPLIT MPLBACKEND=Agg ENV_NAME=$ENV_NAME && \
     bash $script 2>&1 | tee -a $logfile"
  sleep 30
  if ! tmux has-session -t "$session" 2>/dev/null; then
    log "错误: $session 启动失败，查看 $logfile"
    exit 1
  fi
  log "$session 已启动"
}

log "========== 流水线启动 =========="
log "阶段2目标: ${STAGE2_TARGET} 步 | 阶段3: ${STAGE3_TARGET} | 阶段4: ${STAGE4_TARGET}"

# --- 阶段 2：等待 30k 步 ---
if tmux has-session -t "$STAGE2_SESSION" 2>/dev/null; then
  wait_for_steps "$STAGE2_LOG" "$STAGE2_TARGET" "阶段2"
  stop_session "$STAGE2_SESSION"
else
  steps="$(current_steps "$STAGE2_LOG")"
  if [[ "$steps" -lt "$STAGE2_TARGET" ]]; then
    log "阶段2 未运行且仅 ${steps} 步，先启动阶段2"
    start_stage "$STAGE2_SESSION" "$SCRIPT_DIR/train_stage2_paper_global.sh" "$RUN_ROOT/logs/stage2_paper_global.log"
    wait_for_steps "$STAGE2_LOG" "$STAGE2_TARGET" "阶段2"
    stop_session "$STAGE2_SESSION"
  else
    log "阶段2 已有 ${steps} 步，跳过等待"
  fi
fi

# 确认 checkpoint
for f in slam local; do
  if [[ ! -f "$RUN_ROOT/models/stage2_paper_global/model_best.$f" ]]; then
    log "警告: 缺少 stage2 model_best.$f，回退 stage1"
    break
  fi
done

# --- 阶段 3 ---
start_stage "nso_stage3" "$SCRIPT_DIR/train_stage3_rpn.sh" "$LOG_DIR/stage3_rpn.log"
wait_for_steps "$STAGE3_LOG" "$STAGE3_TARGET" "阶段3"
stop_session "nso_stage3"

# --- 阶段 4 ---
start_stage "nso_stage4" "$SCRIPT_DIR/train_stage4_ssc_loop.sh" "$LOG_DIR/stage4_ssc_loop.log"
wait_for_steps "$STAGE4_LOG" "$STAGE4_TARGET" "阶段4"
stop_session "nso_stage4"

log "========== 阶段 3/4 流水线完成 =========="
log "Checkpoint:"
ls -lh "$RUN_ROOT/models/stage3_rpn"/model_best.* 2>/dev/null | tee -a "$PIPELINE_LOG" || true
ls -lh "$RUN_ROOT/models/stage4_ssc_loop"/model_best.* 2>/dev/null | tee -a "$PIPELINE_LOG" || true
