#!/usr/bin/env bash
# 解压本地上传的 habitat-lab v0.2.4 源码（服务器无法访问 GitHub 时使用）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$PROJECT_DIR/third_party"
ARCHIVE="${1:-$TP/habitat-lab-v0.2.4.tar.gz}"
TARGET="$TP/habitat-lab"

if [[ -d "$TARGET/habitat-lab" ]] || [[ -f "$TARGET/habitat/__init__.py" ]]; then
  echo "已存在: $TARGET"
  exit 0
fi

if [[ ! -f "$ARCHIVE" ]]; then
  echo "未找到压缩包: $ARCHIVE" >&2
  echo "" >&2
  echo "请在本机浏览器下载后 SFTP 上传:" >&2
  echo "  https://github.com/facebookresearch/habitat-lab/archive/refs/tags/v0.2.4.tar.gz" >&2
  echo "  -> $TP/habitat-lab-v0.2.4.tar.gz" >&2
  echo "" >&2
  echo "然后重新运行: bash scripts/fetch_habitat2_source.sh" >&2
  exit 1
fi

mkdir -p "$TP"
rm -rf "$TP/habitat-lab-0.2.4" "$TARGET"
tar -xzf "$ARCHIVE" -C "$TP"
mv "$TP/habitat-lab-0.2.4" "$TARGET"
echo "已解压到 $TARGET"
ls "$TARGET" | head -10
