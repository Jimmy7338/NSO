#!/usr/bin/env bash
# 监控阶段 3 步数，达标后自动停止并启动阶段 4
# 若阶段 3 已在运行，仅监控不重启
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

export NSO_RUN_ROOT="$RUN_ROOT"
export NSO_GIBSON_SPLIT="${NSO_GIBSON_SPLIT:-train}"
export MPLBACKEND=Agg
export ENV_NAME="${ENV_NAME:-nso_h2}"

STAGE3_TARGET="${STAGE3_TARGET_STEPS:-25000}"
STAGE4_TARGET="${STAGE4_TARGET_STEPS:-25000}"
POLL_SEC="${POLL_SEC:-60}"
STALL_SEC="${STALL_SEC:-600}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"

STAGE3_LOG="$RUN_ROOT/models/stage3_rpn/train.log"
STAGE4_LOG="$RUN_ROOT/models/stage4_ssc_loop/train.log"
PIPELINE_LOG="$LOG_DIR/pipeline_stage4.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PIPELINE_LOG"; }

current_steps() {
  local logfile="$1"
  [[ -f "$logfile" ]] || { echo 0; return; }
  grep -oP 'num timesteps \K[0-9]+' "$logfile" 2>/dev/null | sort -n | tail -1 || echo 0
}

log_mtime() {
  local logfile="$1"
  [[ -f "$logfile" ]] || { echo 0; return; }
  stat -c %Y "$logfile" 2>/dev/null || echo 0
}

is_training_alive() {
  local pattern="${1:-stage3_rpn}"
  pgrep -f "main.py.*${pattern}" >/dev/null 2>&1 || \
    pgrep -f "main.py.*exp_name.*${pattern}" >/dev/null 2>&1
}

stop_session() {
  local session="$1"
  local alive_pattern="${2:-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    log "停止 tmux 会话: $session"
    tmux send-keys -t "$session" C-c
    sleep 10
    local i
    for i in $(seq 1 12); do
      if [[ -z "$alive_pattern" ]] || ! is_training_alive "$alive_pattern"; then
        break
      fi
      sleep 5
    done
    if tmux has-session -t "$session" 2>/dev/null; then
      log "强制结束: $session"
      tmux kill-session -t "$session"
    fi
  fi
  if [[ -n "$alive_pattern" ]]; then
    pkill -f "main.py.*${alive_pattern}" 2>/dev/null || true
    sleep 3
  fi
}

start_stage() {
  local session="$1" script="$2" console_log="$3" alive_pattern="${4:-}"
  stop_session "$session" "$alive_pattern"
  log "启动 $session -> $script"
  bash "$SCRIPT_DIR/run_stage_in_tmux.sh" "$session" "$script" "$console_log"
  sleep 45
  if ! tmux has-session -t "$session" 2>/dev/null; then
    log "错误: tmux 会话 $session 已退出，查看 $console_log"
    return 1
  fi
  if [[ -n "$alive_pattern" ]] && ! is_training_alive "$alive_pattern"; then
    log "警告: $session 已创建但 main.py 未检测到，查看 $console_log"
    tail -5 "$console_log" 2>/dev/null | tee -a "$PIPELINE_LOG" || true
    return 1
  fi
  log "$session 已启动"
  return 0
}

wait_for_steps_with_watchdog() {
  local logfile="$1" target="$2" label="$3" session="$4" script="$5" console_log="$6" alive_pattern="$7"
  local restarts=0 last_steps=-1 last_mtime=0 stall_since=0

  log "等待 $label 达到 ${target} 步 (当前: $(current_steps "$logfile"))..."
  while true; do
    local steps now mtime
    steps="$(current_steps "$logfile")"
    now="$(date +%s)"
    mtime="$(log_mtime "$logfile")"
    local alive=0
    if is_training_alive "$alive_pattern"; then alive=1; fi

    if [[ "$steps" -ge "$target" ]]; then
      log "$label 已达 ${steps} 步 (目标 ${target})"
      return 0
    fi

    if [[ "$steps" != "$last_steps" ]] || [[ "$mtime" != "$last_mtime" ]]; then
      stall_since="$now"
      last_steps="$steps"
      last_mtime="$mtime"
    fi

    local stalled=$(( now - stall_since ))
    local need_restart=0
    if [[ "$alive" -eq 0 ]]; then
      need_restart=1
      log "$label 进程未运行 (步数 ${steps}/${target})"
    elif [[ "$stalled" -ge "$STALL_SEC" ]]; then
      need_restart=1
      log "$label 日志 ${stalled}s 无更新 (步数 ${steps}/${target})"
    fi

    if [[ "$need_restart" -eq 1 ]]; then
      if [[ "$restarts" -ge "$MAX_RESTARTS" ]]; then
        log "错误: $label 重启次数达上限 ${MAX_RESTARTS}，流水线中止"
        exit 1
      fi
      restarts=$((restarts + 1))
      log "重启 $label (第 ${restarts}/${MAX_RESTARTS} 次)..."
      if ! start_stage "$session" "$script" "$console_log" "$alive_pattern"; then
        log "重启失败，${POLL_SEC}s 后重试..."
      fi
      stall_since="$(date +%s)"
      sleep "$POLL_SEC"
      continue
    fi

    log "$label 进度: ${steps}/${target} 步 (alive=${alive})"
    sleep "$POLL_SEC"
  done
}

STAGE3_CONSOLE="$LOG_DIR/stage3_rpn.log"
STAGE4_CONSOLE="$LOG_DIR/stage4_ssc_loop.log"

log "========== 阶段3→4 流水线启动 =========="
log "阶段3目标: ${STAGE3_TARGET} | 阶段4: ${STAGE4_TARGET}"

# --- 阶段 3 ---
steps="$(current_steps "$STAGE3_LOG")"
if [[ "$steps" -lt "$STAGE3_TARGET" ]]; then
  if ! is_training_alive "stage3_rpn"; then
    start_stage "nso_stage3" "$SCRIPT_DIR/train_stage3_rpn.sh" \
      "$STAGE3_CONSOLE" "stage3_rpn" || true
  else
    log "阶段3 已在运行 (步数 ${steps})，仅监控"
  fi
  wait_for_steps_with_watchdog "$STAGE3_LOG" "$STAGE3_TARGET" "阶段3" \
    "nso_stage3" "$SCRIPT_DIR/train_stage3_rpn.sh" \
    "$STAGE3_CONSOLE" "stage3_rpn"
  stop_session "nso_stage3" "stage3_rpn"
else
  log "阶段3 已有 ${steps} 步，跳过"
fi

# --- 阶段 4 ---
start_stage "nso_stage4" "$SCRIPT_DIR/train_stage4_ssc_loop.sh" \
  "$STAGE4_CONSOLE" "stage4_ssc_loop"
wait_for_steps_with_watchdog "$STAGE4_LOG" "$STAGE4_TARGET" "阶段4" \
  "nso_stage4" "$SCRIPT_DIR/train_stage4_ssc_loop.sh" \
  "$STAGE4_CONSOLE" "stage4_ssc_loop"
stop_session "nso_stage4" "stage4_ssc_loop"

log "========== 阶段 4 流水线完成 =========="
ls -lh "$RUN_ROOT/models/stage3_rpn"/model_best.* 2>/dev/null | tee -a "$PIPELINE_LOG" || true
ls -lh "$RUN_ROOT/models/stage4_ssc_loop"/model_best.* 2>/dev/null | tee -a "$PIPELINE_LOG" || true
