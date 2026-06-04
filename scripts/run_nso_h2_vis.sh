#!/usr/bin/env bash
# Habitat 2 + 实时可视化（MobaXterm X11 + 服务器 GPU 渲染）
# 前置: MobaXterm X server 已开，SSH 已勾选 X11-Forwarding，echo $DISPLAY 非空
# 用法: bash scripts/run_nso_h2_vis.sh [额外 main.py 参数...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-nso_h2}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "错误: DISPLAY 未设置，无法弹出实时窗口。" >&2
  echo "请用 MobaXterm SSH 连接，并确认 X server 已启动、X11-Forwarding 已勾选。" >&2
  echo "验证: echo \$DISPLAY && xeyes" >&2
  exit 1
fi

export NSO_HABITAT_VERSION=2
export NSO_X11_DISPLAY="${DISPLAY}"
export NSO_USE_XVFB_GPU=1
export NSO_LIVE_VIS=1
export NSO_LIVE_VIS_DIR="/tmp/nso_vis_live_h2"
export MPLBACKEND=Agg
export NSO_LIVE_FAST=1
export NSO_VIS_HIGH_QUALITY=1
export NSO_VIS_FULL_MAP=1
export NSO_VIS_JPEG=1
export NSO_VIS_JPEG_QUALITY=95
export NSO_VIS_MAP_ZOOM=1
export NSO_LIVE_OBS_W=720
export NSO_LIVE_OBS_H=540
export NSO_LIVE_MAP_SIZE=540
export NSO_VIEWER_GEOM=1280x720
export NSO_VIEWER_MAX_W=1280
export NSO_VIEWER_MAX_H=720
export NSO_VIS_EVERY=1
export NSO_VIEWER_POLL_MS=40

PRETRAINED="$PROJECT_DIR/pretrained_models"
LOAD_ARGS=()
if [[ -f "$PRETRAINED/model_best.global" ]]; then
  LOAD_ARGS=(
    --load_global "$PRETRAINED/model_best.global"
    --load_local "$PRETRAINED/model_best.local"
    --load_slam "$PRETRAINED/model_best.slam"
  )
fi

echo "模式: Habitat 2 实时可视化（测试场景 pointnav_habitat_test）"
echo "窗口: NSO Live（MobaXterm）  帧目录: $NSO_LIVE_VIS_DIR"
echo "步数更长可看完整建图: bash scripts/run_nso_h2_vis.sh --max_episode_length 300"
echo "高清模式已默认开启 (NSO_VIS_HIGH_QUALITY=1, 全局地图 NSO_VIS_FULL_MAP=1)"
echo "同时保存每步 PNG: bash scripts/run_nso_h2_vis.sh --print_images 1"

if [[ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]]; then
  echo "未找到 conda: $CONDA_DIR" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "$NSO_LIVE_VIS_DIR"
VIEWER_LOG="$NSO_LIVE_VIS_DIR/viewer.log"
: > "$VIEWER_LOG"

NSO_VIEWER_KEEP="${NSO_VIEWER_KEEP:-1}"

start_viewer() {
  unset MPLBACKEND
  nohup python "$SCRIPT_DIR/nso_live_viewer.py" "$NSO_LIVE_VIS_DIR" >>"$VIEWER_LOG" 2>&1 &
  NSO_VIEWER_PID=$!
  disown "$NSO_VIEWER_PID" 2>/dev/null || true
  export NSO_VIEWER_PID NSO_VIEWER_EXTERNAL=1
  sleep 1.2
  if ! kill -0 "$NSO_VIEWER_PID" 2>/dev/null; then
    echo "[可视化] 查看器启动失败，详见 $VIEWER_LOG:" >&2
    cat "$VIEWER_LOG" >&2
    return 1
  fi
  echo "[可视化] NSO Live 已启动 pid=$NSO_VIEWER_PID"
  return 0
}

pkill -f "nso_live_viewer.py.*nso_vis_live_h2" 2>/dev/null || true
sleep 0.3
start_viewer || exit 1

cleanup_viewer() {
  [[ "$NSO_VIEWER_KEEP" == 1 ]] && return
  kill "$NSO_VIEWER_PID" 2>/dev/null || true
}
trap cleanup_viewer EXIT INT TERM

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
  --env_frame_width 512
  --env_frame_height 512
  "${LOAD_ARGS[@]}"
  --train_global 0
  --train_local 0
  --train_slam 0
  -v 1
  --vis_type 1
  --print_images 0
  -d "$PROJECT_DIR/tmp/"
  --exp_name h2_vis_live
)

bash "$SCRIPT_DIR/run_nso.sh" "${DEFAULT_ARGS[@]}" "$@"
RUN_EXIT=$?

if [[ -f "$NSO_LIVE_VIS_DIR/frame.jpg" ]] || [[ -f "$NSO_LIVE_VIS_DIR/frame.png" ]]; then
  echo "[可视化] 仿真结束。NSO Live 窗口应保留最后一帧。"
else
  echo "[可视化] 未找到 frame.jpg，请确认 -v 1 且日志无报错"
fi

if [[ "$NSO_VIEWER_KEEP" == 1 ]]; then
  echo "[可视化] 关闭窗口: pkill -f nso_live_viewer"
fi

exit "$RUN_EXIT"
