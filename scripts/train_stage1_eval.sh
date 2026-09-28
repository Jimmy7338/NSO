#!/usr/bin/env bash
# 阶段 1 评估：加载 stage1 checkpoint，在 val 上测覆盖率
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
export MPLBACKEND=Agg
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
STAGE1_DIR="${STAGE1_LOAD_DIR:-$RUN_ROOT/models/stage1_slam_local}"
PRETRAINED="${PRETRAINED_DIR:-$SCRIPT_DIR/../pretrained_models}"

for f in slam local; do
  if [[ ! -f "$STAGE1_DIR/model_best.$f" ]]; then
    echo "缺少 $STAGE1_DIR/model_best.$f" >&2
    exit 1
  fi
done

bash "$SCRIPT_DIR/restore_global_ckpt.sh" stage1_slam_local >/dev/null
GLOBAL_CKPT="$STAGE1_DIR/model_best.global"

echo "阶段 1 评估 | slam/local: $STAGE1_DIR | global: $GLOBAL_CKPT"

exec bash "$SCRIPT_DIR/run_nso.sh" \
  --habitat_version 2 \
  --task_config pointnav_gibson.yaml \
  --split val \
  --eval 1 \
  --train_slam 0 --train_local 0 --train_global 0 \
  --num_processes 1 \
  --num_episodes "${EVAL_EPISODES:-50}" \
  --max_episode_length 500 \
  --log_interval 10 \
  -na 0 -no 0 -v 0 --print_images 0 \
  -d "$RUN_ROOT/" \
  --exp_name stage1_eval \
  --load_slam "$STAGE1_DIR/model_best.slam" \
  --load_global "$GLOBAL_CKPT" \
  --load_local "$STAGE1_DIR/model_best.local" \
  "$@"
