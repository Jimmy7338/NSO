#!/usr/bin/env bash
# Habitat 2 + Gibson PointNav — 训练（保存 checkpoint，无可视化）
#
# 数据前提:
#   - 场景: data/scene_datasets/gibson/*.glb
#   - PointNav: data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz
#   - train 为空时需重新解压 pointnav_gibson_v1.zip（见下方说明）
#
# 用法:
#   bash scripts/run_nso_h2_gibson_train.sh
#   bash scripts/run_nso_h2_gibson_train.sh --split val --max_episode_length 500
#   bash scripts/run_nso_h2_gibson_train.sh --use_semantic --exp_name gibson_sem
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$PROJECT_DIR/data/scene_datasets/gibson/Cantwell.glb" ]]; then
  echo "请先运行: bash scripts/setup_gibson_habitat.sh" >&2
  exit 1
fi

# 默认保存到云盘，避免占满系统盘
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
mkdir -p "$RUN_ROOT"

export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"

SPLIT="${NSO_GIBSON_SPLIT:-val}"
TRAIN_JSON="$PROJECT_DIR/data/datasets/pointnav/gibson/v1/train/train.json.gz"
if [[ -f "$TRAIN_JSON" ]]; then
  EP_COUNT=$(python3 -c "import gzip,json; print(len(json.load(gzip.open('$TRAIN_JSON'))['episodes']))")
  if [[ "$EP_COUNT" -gt 0 ]]; then
    SPLIT="${NSO_GIBSON_SPLIT:-train}"
  fi
fi

PRETRAINED="$PROJECT_DIR/pretrained_models"
LOAD_ARGS=()
if [[ "${NSO_FINETUNE:-1}" == "1" ]]; then
  if [[ -f "$PRETRAINED/model_best.global" ]]; then
    LOAD_ARGS=(
      --load_global "$PRETRAINED/model_best.global"
      --load_local "$PRETRAINED/model_best.local"
      --load_slam "$PRETRAINED/model_best.slam"
    )
    echo "[训练] 从预训练权重微调: $PRETRAINED"
  else
    echo "[训练] 未找到 pretrained_models/，从零开始（可先 bash scripts/download_pretrained.sh）"
  fi
fi

echo "Gibson PointNav 训练 | split=$SPLIT | 输出: $RUN_ROOT/models/<exp_name>/"
echo "  model_best.{slam,global,local}  与  dump/<exp_name>/periodic_*"

exec bash "$SCRIPT_DIR/run_nso.sh" \
  --habitat_version 2 \
  --task_config pointnav_gibson.yaml \
  --split "$SPLIT" \
  --eval 0 \
  --train_global 1 \
  --train_local 1 \
  --train_slam 1 \
  -na 0 \
  -no 0 \
  -v 0 \
  --print_images 0 \
  --auto_gpu_config 1 \
  --num_episodes 1000000 \
  --max_episode_length 500 \
  -d "$RUN_ROOT/" \
  --exp_name gibson_h2_train \
  --save_interval 1 \
  --save_periodic 50000 \
  --log_interval 10 \
  "${LOAD_ARGS[@]}" \
  "$@"
