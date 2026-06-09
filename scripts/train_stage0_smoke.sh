#!/usr/bin/env bash
# 阶段 0：环境 smoke（~50 步，不训练，验证 Gibson H2 + checkpoint 路径）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
export MPLBACKEND=Agg
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
mkdir -p "$RUN_ROOT"

exec bash "$SCRIPT_DIR/run_nso.sh" \
  --habitat_version 2 \
  --task_config pointnav_gibson.yaml \
  --split "${NSO_GIBSON_SPLIT:-val}" \
  --eval 0 \
  --train_slam 0 --train_local 0 --train_global 0 \
  --num_processes 1 \
  --num_episodes 1 \
  --max_episode_length "${STAGE0_STEPS:-50}" \
  --log_interval 10 \
  -na 0 -no 0 -v 0 --print_images 0 \
  -d "$RUN_ROOT/" \
  --exp_name stage0_smoke \
  --load_slam pretrained_models/model_best.slam \
  --load_global pretrained_models/model_best.global \
  --load_local pretrained_models/model_best.local \
  "$@"
