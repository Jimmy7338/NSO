#!/usr/bin/env bash
# 等待阶段 1 tmux 结束后，依次执行 eval → 阶段 2 → 3 → 4
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

WAIT_SESSION="${WAIT_SESSION:-nso_stage1}"
export NSO_RUN_ROOT="$RUN_ROOT"
export MPLBACKEND=Agg

log() { echo "[$(date '+%F %T')] $*"; }

log "等待 tmux 会话 $WAIT_SESSION 结束..."
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep 60
done
log "阶段 1 已结束，开始 eval"

bash "$SCRIPT_DIR/train_stage1_eval.sh" \
  --num_episodes "${EVAL_EPISODES:-50}" \
  2>&1 | tee "$LOG_DIR/stage1_eval.log"

for stage in 2 3 4; do
  log "启动阶段 $stage"
  bash "$SCRIPT_DIR/train_stage${stage}_"*.sh \
    2>&1 | tee "$LOG_DIR/stage${stage}.log"
  log "阶段 $stage 完成"
done

log "全部阶段完成"
