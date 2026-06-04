#!/usr/bin/env bash
# Habitat 2 + 项目原始可视化（matplotlib TkAgg，经 Xming/X11 转发）
# 前置: Windows 开 Xming，SSH 勾选 X11 forwarding，echo $DISPLAY 非空
# 用法: bash scripts/run_nso_h2_vis.sh [额外 main.py 参数...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "错误: DISPLAY 未设置。" >&2
  echo "  1) Windows 启动 Xming" >&2
  echo "  2) SSH 客户端开启 X11 转发（MobaXterm / PuTTY+Xming）" >&2
  echo "  3) 验证: echo \$DISPLAY && xeyes" >&2
  exit 1
fi

export NSO_HABITAT_VERSION=2
export ENV_NAME="${ENV_NAME:-nso_h2}"
export NSO_VIS_NATIVE=1
export NSO_VIS_FULL_MAP=0
export NSO_VIS_MAP_ZOOM=1
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
echo "与 README 一致: python main.py -v 1 （不经 NSO Live / 帧导出）"
echo "步数更长: bash scripts/run_nso_h2_vis.sh --max_episode_length 300"

DEFAULT_ARGS=(
  --habitat_version 2
  --task_config pointnav_habitat_test.yaml
  --split val
  -n 1
  --auto_gpu_config 0
  -na 0
  -no 0
  --num_episodes 1
  --max_episode_length 200
  "${LOAD_ARGS[@]}"
  --train_global 0
  --train_local 0
  --train_slam 0
  -v 1
  --vis_type 1
  --print_images 0
  -d "$PROJECT_DIR/tmp/"
  --exp_name h2_vis
)

bash "$SCRIPT_DIR/run_nso.sh" "${DEFAULT_ARGS[@]}" "$@"
