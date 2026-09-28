#!/usr/bin/env bash
# 使用 Habitat 2 环境运行 NSO（conda 环境 nso_h2）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENV_NAME="${ENV_NAME:-nso_h2}"
export NSO_HABITAT_VERSION=2

# 默认使用 habitat-test-scenes；命令行参数可覆盖（写在后面生效）
exec bash "$SCRIPT_DIR/run_nso.sh" \
  --habitat_version 2 \
  --task_config pointnav_habitat_test.yaml \
  --split val \
  --num_episodes 1 \
  --max_episode_length 100 \
  --load_global "${SCRIPT_DIR}/../pretrained_models/model_best.global" \
  --load_local "${SCRIPT_DIR}/../pretrained_models/model_best.local" \
  --load_slam "${SCRIPT_DIR}/../pretrained_models/model_best.slam" \
  --train_global 0 --train_local 0 --train_slam 0 \
  -n 1 --auto_gpu_config 0 -na 0 -no 0 \
  -v 0 --print_images 1 \
  -d "${SCRIPT_DIR}/../tmp/" --exp_name h2_vis \
  "$@"
