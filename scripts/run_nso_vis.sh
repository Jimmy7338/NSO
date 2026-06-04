#!/usr/bin/env bash
# 实时可视化（MobaXterm X11 + 服务器 GPU 渲染）
# 前置: MobaXterm X server 已开，SSH 已勾选 X11-Forwarding，echo $DISPLAY 非空
# 用法: bash scripts/run_nso_vis.sh [额外 main.py 参数...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-nso}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "错误: DISPLAY 未设置，无法弹出实时窗口。" >&2
  echo "请用 MobaXterm SSH 连接，并确认 X server 已启动、X11-Forwarding 已勾选。" >&2
  echo "验证: echo \$DISPLAY && xeyes" >&2
  exit 1
fi

# 保存 MobaXterm 的 DISPLAY；xvfb 子进程会改掉 DISPLAY
export NSO_X11_DISPLAY="${DISPLAY}"
export NSO_USE_XVFB_GPU=1
export NSO_LIVE_VIS=1
export NSO_LIVE_VIS_DIR="/tmp/nso_vis_live"
export MPLBACKEND=Agg
# 实时帧：OpenCV JPEG 快速路径（比每步 matplotlib savefig 快很多）
export NSO_LIVE_FAST=1
export NSO_VIS_HIGH_QUALITY=1
export NSO_VIS_FULL_MAP=1
export NSO_VIS_JPEG=1
export NSO_VIS_JPEG_QUALITY=95
export NSO_LIVE_OBS_W=640
export NSO_LIVE_OBS_H=480
export NSO_LIVE_MAP_SIZE=480
export NSO_VIEWER_GEOM=1280x720
export NSO_VIEWER_MAX_W=1280
export NSO_VIEWER_MAX_H=720
export NSO_VIS_EVERY=1
export NSO_VIEWER_POLL_MS=40

PRETRAINED="$PROJECT_DIR/pretrained_models"
if [[ ! -f "$PRETRAINED/model_best.local" ]] && [[ -f "$PRETRAINED/model_best.download" ]]; then
  echo "将 model_best.download 重命名为 model_best.local"
  mv "$PRETRAINED/model_best.download" "$PRETRAINED/model_best.local"
fi

LOAD_ARGS=()
if [[ -f "$PRETRAINED/model_best.global" ]]; then
  LOAD_ARGS=(
    --load_global "$PRETRAINED/model_best.global"
    --load_local "$PRETRAINED/model_best.local"
    --load_slam "$PRETRAINED/model_best.slam"
  )
  echo "已检测到预训练权重，将加载 pretrained_models/model_best.*"
else
  echo "提示: 未找到预训练权重，轨迹为随机策略。"
fi

echo "模式: 实时可视化（OpenCV 快速帧 + MobaXterm 窗口，场景在 GPU 渲染）"
echo "仍卡顿时可: NSO_VIS_EVERY=2 bash scripts/run_nso_vis.sh  （每 2 步刷新一帧）"
echo "高清存档: bash scripts/run_nso_vis.sh --print_images 1"

# 在 xvfb 之前于当前 SSH 会话启动查看器（继承 MobaXterm X11 认证）
# 若在 xvfb-run 内的 Python 子进程启动会报 MoTTY Unsupported authorisation protocol
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

# 默认保留窗口（仿真很快结束时不至于一闪就关）
NSO_VIEWER_KEEP="${NSO_VIEWER_KEEP:-1}"

start_viewer() {
  unset MPLBACKEND
  nohup python "$SCRIPT_DIR/nso_live_viewer.py" "$NSO_LIVE_VIS_DIR" >>"$VIEWER_LOG" 2>&1 &
  NSO_VIEWER_PID=$!
  disown "$NSO_VIEWER_PID" 2>/dev/null || true
  export NSO_VIEWER_PID
  export NSO_VIEWER_EXTERNAL=1
  sleep 1.2
  if ! kill -0 "$NSO_VIEWER_PID" 2>/dev/null; then
    echo "[可视化] 查看器启动失败，详见 $VIEWER_LOG:" >&2
    cat "$VIEWER_LOG" >&2
    return 1
  fi
  echo "[可视化] NSO Live 窗口已启动 pid=$NSO_VIEWER_PID DISPLAY=$DISPLAY"
  echo "[可视化] 请查看 Windows 任务栏「NSO Live」（可能被其他窗口挡住）"
  return 0
}

if [[ "${NSO_VIEWER_REUSE:-0}" == 1 ]] && pgrep -f "scripts/nso_live_viewer.py" >/dev/null 2>&1; then
  NSO_VIEWER_PID="$(pgrep -f "scripts/nso_live_viewer.py" | head -1)"
  export NSO_VIEWER_PID NSO_VIEWER_EXTERNAL=1
  echo "[可视化] 复用已有查看器 pid=$NSO_VIEWER_PID"
else
  pkill -f "scripts/nso_live_viewer.py" 2>/dev/null || true
  sleep 0.3
  start_viewer || exit 1
fi
echo "[可视化] 查看器日志: $VIEWER_LOG"

cleanup_viewer() {
  if [[ "$NSO_VIEWER_KEEP" == 1 ]]; then
    return
  fi
  kill "$NSO_VIEWER_PID" 2>/dev/null || true
}
trap cleanup_viewer EXIT INT TERM

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
  -d ./tmp/
  --exp_name vis_live
)

bash "$SCRIPT_DIR/run_nso.sh" "${DEFAULT_ARGS[@]}" "$@"
RUN_EXIT=$?

# 仿真结束后：确保查看器仍在且已加载最后一帧
if ! kill -0 "$NSO_VIEWER_PID" 2>/dev/null; then
  echo "[可视化] 查看器已退出，正在重新打开以显示最后一帧…"
  start_viewer || true
  sleep 0.8
fi

if [[ -f "$NSO_LIVE_VIS_DIR/frame.jpg" ]] || [[ -f "$NSO_LIVE_VIS_DIR/frame.png" ]]; then
  if grep -q "首帧已显示" "$VIEWER_LOG" 2>/dev/null; then
    echo "[可视化] 仿真结束。NSO Live 窗口应显示最后一帧。"
  else
    echo "[可视化] 等待查看器加载帧…（约 1 秒）"
    sleep 1.2
  fi
else
  echo "[可视化] 警告: 未找到 frame.jpg，请确认 -v 1 且 visualize 已开启"
fi

if [[ "$NSO_VIEWER_KEEP" == 1 ]]; then
  echo "[可视化] 窗口将保持打开。关闭: pkill -f nso_live_viewer  或点窗口 X"
fi

exit "$RUN_EXIT"
