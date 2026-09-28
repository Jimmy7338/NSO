#!/usr/bin/env bash
# 优化后的论文模式训练：多进程 + 冻结 SLAM/Local + 降频语义 + 关回环
# 比 run_nso_paper_train.sh 快 5-15 倍
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
export MPLBACKEND=Agg
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
mkdir -p "$RUN_ROOT"

PRETRAINED="$PROJECT_DIR/pretrained_models"
EXP_DIR="$RUN_ROOT/models/paper_fast"
LOAD_ARGS=()
if [[ -f "$EXP_DIR/model_best.global" ]]; then
  LOAD_ARGS=(
    --load_global "$EXP_DIR/model_best.global"
    --load_local "$PRETRAINED/model_best.local"
    --load_slam "$PRETRAINED/model_best.slam"
  )
elif [[ -f "$PRETRAINED/model_best.global" ]]; then
  LOAD_ARGS=(
    --load_global "$PRETRAINED/model_best.global"
    --load_local "$PRETRAINED/model_best.local"
    --load_slam "$PRETRAINED/model_best.slam"
  )
fi
if [[ -f "$EXP_DIR/model_best.reach" ]]; then
  LOAD_ARGS+=(--goal_reachability_model_path "$EXP_DIR/model_best.reach")
fi

echo "=============================================="
echo "快速论文模式训练"
echo "优化项："
echo "  - 多进程并行（auto_gpu_config 默认 4 进程）"
echo "  - 只训练全局策略（冻结 SLAM/Local）"
echo "  - 语义检测降频（每 5 步）"
echo "  - RPN MC 采样降至 5"
echo "  - 关闭回环检测（评估时再开）"
echo "  - 回合加长至 1000 步"
echo "  - GT 地图磁盘缓存"
echo "输出: $RUN_ROOT/models/paper_fast/"
echo "=============================================="

exec bash "$SCRIPT_DIR/run_nso.sh" \
  --habitat_version 2 \
  --task_config pointnav_gibson.yaml \
  --split "${NSO_GIBSON_SPLIT:-val}" \
  --eval 0 \
  --paper_mode \
  --semantic_use_all_classes \
  --semantic_conf_thresh 0.15 \
  --semantic_interval 5 \
  --train_global 1 --train_local 0 --train_slam 0 \
  --use_loop_detection 0 \
  --rpn_mc_samples 5 \
  --auto_gpu_config 1 \
  --num_episodes 1000000 \
  --max_episode_length 1000 \
  --save_interval 100 \
  --save_periodic 100000 \
  --log_interval 10 \
  -na 0 -no 0 -v 0 --print_images 0 \
  -d "$RUN_ROOT/" \
  --exp_name paper_fast \
  "${LOAD_ARGS[@]}" \
  "$@"
