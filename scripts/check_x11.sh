#!/usr/bin/env bash
# 检查 X11 转发是否就绪（在 MobaXterm SSH 里运行）
set -euo pipefail

echo "=== X11 环境检查 ==="
echo "DISPLAY=${DISPLAY:-<未设置>}"
echo "SSH_CLIENT=${SSH_CLIENT:-<无>}"
echo "SSH_CONNECTION=${SSH_CONNECTION:-<无>}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo ""
  echo "[失败] DISPLAY 为空，当前会话没有 X11 转发。"
  echo ""
  echo "常见原因："
  echo "  1) 用了 Cursor/VS Code 远程终端、普通 ssh，而不是 MobaXterm 的 SSH 会话"
  echo "  2) MobaXterm 未开启 X server（右上角绿色 X）"
  echo "  3) 该 SSH 会话未勾选 X11-Forwarding"
  echo "  4) 服务器 sshd 日志有 'Failed to allocate internet-domain X11 display socket'"
  echo "     → 需 /etc/ssh/sshd_config.d/99-x11-ipv4.conf (AddressFamily inet) 并重连 SSH"
  echo ""
  echo "请按下列步骤重连："
  echo "  • MobaXterm 右上角确认 X server 已启动"
  echo "  • 新建/编辑 Session → SSH → Advanced SSH settings → 勾选 X11-Forwarding"
  echo "  • 关闭当前终端，用该 Session 重新 SSH 登录"
  echo "  • 登录后再运行: bash scripts/check_x11.sh"
  exit 1
fi

if ! command -v xauth >/dev/null 2>&1; then
  echo "[警告] 未安装 xauth: sudo apt install -y xauth"
fi

if command -v xeyes >/dev/null 2>&1; then
  echo "[测试] 正在启动 xeyes（应在你 Windows 上弹出窗口）..."
  timeout 3 xeyes 2>/dev/null || xeyes &
  sleep 1
  echo "若看到双眼窗口，可运行: bash scripts/run_nso_vis.sh"
  echo "（会弹出标题为「NSO Live」的实时窗口）"
else
  echo "[警告] 未安装 xeyes: sudo apt install -y x11-apps"
fi

exit 0
