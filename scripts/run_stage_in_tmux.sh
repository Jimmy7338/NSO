#!/usr/bin/env bash
# 在 tmux 中稳健启动训练（避免 pipefail+tee 管道误杀进程）
set -euo pipefail

SESSION="${1:?用法: run_stage_in_tmux.sh <tmux_session> <train_script.sh> <console_log>}"
SCRIPT="${2:?}"
CONSOLE_LOG="${3:?}"

RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
export NSO_RUN_ROOT="$RUN_ROOT"
export NSO_GIBSON_SPLIT="${NSO_GIBSON_SPLIT:-train}"
export MPLBACKEND=Agg
export ENV_NAME="${ENV_NAME:-nso_h2}"

mkdir -p "$(dirname "$CONSOLE_LOG")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

# 不用 2>&1 | tee：tee/管道断开会在 pipefail 下连带杀掉训练进程
tmux new-session -d -s "$SESSION" bash -lc "
  set +o pipefail
  export NSO_RUN_ROOT='$RUN_ROOT' NSO_GIBSON_SPLIT='$NSO_GIBSON_SPLIT' MPLBACKEND=Agg ENV_NAME='$ENV_NAME'
  echo \"[\$(date '+%F %T')] 启动: $SCRIPT\" >> '$CONSOLE_LOG'
  bash '$SCRIPT' >> '$CONSOLE_LOG' 2>&1
  ec=\$?
  echo \"[\$(date '+%F %T')] 训练退出 exit=\$ec\" >> '$CONSOLE_LOG'
  exit \$ec
"
