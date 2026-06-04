#!/usr/bin/env bash
# Habitat 1 + 项目原始可视化（matplotlib TkAgg，经 Xming/X11 转发）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "错误: DISPLAY 未设置。请先开 Xming 并启用 SSH X11 转发。" >&2
  exit 1
fi

export ENV_NAME="${ENV_NAME:-nso}"
export NSO_VIS_NATIVE=1
export MPLBACKEND=TkAgg
unset NSO_USE_XVFB_GPU NSO_X11_DISPLAY NSO_LIVE_VIS NSO_LIVE_FAST
unset NSO_LIVE_VIS_DIR NSO_VIEWER_PID NSO_VIEWER_EXTERNAL
unset __GLX_VENDOR_LIBRARY_NAME

PRETRAINED="$PROJECT_DIR/pretrained_models"
LOAD_ARGS=()
if [[ -f "$PRETRAINED/model_best.global" ]]; then
  LOAD_ARGS=(
    --load_global "$PRETRAINED/model_best.global"
    --load_local "$PRETRAINED/model_best.local"
    --load_slam "$PRETRAINED/model_best.slam"
  )
fi

echo "模式: 原始实时窗口 (matplotlib TkAgg → DISPLAY=$DISPLAY)"

DEFAULT_ARGS=(
  -n 1
  --auto_gpu_config 0
  -na 0
  -no 0
  --task_config tasks/pointnav_habitat_test.yaml
  --split val
  --num_episodes 1
  --max_episode_length 100
  "${LOAD_ARGS[@]}"
  --train_global 0
  --train_local 0
  --train_slam 0
  -v 1
  --vis_type 1
  --print_images 0
  -d "$PROJECT_DIR/tmp/"
  --exp_name vis
)

bash "$SCRIPT_DIR/run_nso.sh" "${DEFAULT_ARGS[@]}" "$@"
