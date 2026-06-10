#!/usr/bin/env bash
# 下载 Active Neural SLAM 预训练权重（多镜像，国内服务器 Google Drive 常超时）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$PROJECT_DIR/pretrained_models"
mkdir -p "$OUT"

CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-15}"
MAX_TIME="${MAX_TIME:-120}"
# Google Drive 在国内常超时，默认跳过（设 TRY_GDRIVE=1 可强制尝试）
TRY_GDRIVE="${TRY_GDRIVE:-0}"

download_one() {
  local name="$1"
  local out="$2"
  shift 2
  local urls=("$@")

  if [[ -f "$out" ]] && [[ -s "$out" ]]; then
    if file "$out" | grep -qE 'HTML|text'; then
      echo "删除无效文件(HTML): $out"
      rm -f "$out"
    else
      echo "已存在: $out ($(du -h "$out" | cut -f1))"
      return 0
    fi
  fi

  echo ""
  echo "======== 下载 ${name} ========"
  for url in "${urls[@]}"; do
    echo "尝试: $url"
    if curl -fL --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
        -C - -o "$out" "$url"; then
      if [[ -s "$out" ]] && ! file "$out" | grep -qE 'HTML|text'; then
        echo "成功: $out ($(du -h "$out" | cut -f1))"
        return 0
      fi
      echo "无效内容(可能为网页)，删除并重试..."
      rm -f "$out"
    else
      echo "失败，换下一个源..."
      rm -f "$out"
    fi
  done

  echo "错误: 无法下载 ${name}"
  return 1
}

try_gdown() {
  local id="$1" out="$2"
  if [[ "${TRY_GDOWN:-0}" != "1" ]]; then
    return 1
  fi
  if ! command -v gdown >/dev/null 2>&1; then
    return 1
  fi
  echo "尝试 gdown: $id（最多 60s）"
  timeout 60 gdown "$id" -O "$out" 2>/dev/null || return 1
  [[ -s "$out" ]] && ! file "$out" | grep -qE 'HTML|text'
}

# 各文件 Google Drive ID（与 README 一致）
GLOBAL_ID="1UK2hT0GWzoTaVR5lAI6i8o27tqEmYeyY"
LOCAL_ID="1A1s_HNnbpvdYBUAiw2y1JmmELRLfAJb8"
SLAM_ID="1o5OG7DIUKZyvi5stozSqRpAEae1F2BmX"

CMU_BASE="http://www.cs.cmu.edu/~dchaplot/projects/active_neural_slam"

FAILED=0

GDRIVE_GLOBAL=(
  "https://drive.google.com/uc?export=download&id=${GLOBAL_ID}"
)
GDRIVE_LOCAL=(
  "https://drive.google.com/uc?export=download&id=${LOCAL_ID}"
)
GDRIVE_SLAM=(
  "https://drive.google.com/uc?export=download&id=${SLAM_ID}"
)
if [[ "$TRY_GDRIVE" == "1" ]]; then
  echo "TRY_GDRIVE=1：将尝试 Google Drive（可能较慢或失败）"
else
  GDRIVE_GLOBAL=()
  GDRIVE_LOCAL=()
  GDRIVE_SLAM=()
fi

# model_best.global
if ! download_one "model_best.global" "$OUT/model_best.global" \
  "${CMU_BASE}/model_best.global" \
  "${GDRIVE_GLOBAL[@]}"; then
  try_gdown "$GLOBAL_ID" "$OUT/model_best.global" || FAILED=1
fi

# model_best.local
if ! download_one "model_best.local" "$OUT/model_best.local" \
  "${CMU_BASE}/model_best.local" \
  "${GDRIVE_LOCAL[@]}"; then
  try_gdown "$LOCAL_ID" "$OUT/model_best.local" || FAILED=1
fi

# model_best.slam
if ! download_one "model_best.slam" "$OUT/model_best.slam" \
  "${CMU_BASE}/model_best.slam" \
  "${GDRIVE_SLAM[@]}"; then
  try_gdown "$SLAM_ID" "$OUT/model_best.slam" || FAILED=1
fi

echo ""
ls -lh "$OUT" 2>/dev/null || true

if [[ "$FAILED" -ne 0 ]] || [[ ! -s "$OUT/model_best.global" ]] || [[ ! -s "$OUT/model_best.local" ]] || [[ ! -s "$OUT/model_best.slam" ]]; then
  cat <<'EOF'

【自动下载未全部成功】服务器无法访问 Google Drive 时，请在本机 Windows 下载后上传：

1) 浏览器打开（需可访问 Google）：
   https://drive.google.com/uc?export=download&id=1UK2hT0GWzoTaVR5lAI6i8o27tqEmYeyY  → 另存为 model_best.global
   https://drive.google.com/uc?export=download&id=1A1s_HNnbpvdYBUAiw2y1JmmELRLfAJb8  → 另存为 model_best.local
   https://drive.google.com/uc?export=download&id=1o5OG7DIUKZyvi5stozSqRpAEae1F2BmX  → 另存为 model_best.slam

2) MobaXterm 左侧 SFTP 拖到服务器目录：
   /home/ubuntu/NSO/pretrained_models/

3) 验证后运行：
   ls -lh /home/ubuntu/NSO/pretrained_models/
   bash /home/ubuntu/NSO/scripts/run_nso_vis.sh

EOF
  exit 1
fi

echo "全部预训练权重已就绪。"
