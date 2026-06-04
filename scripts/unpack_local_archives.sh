#!/usr/bin/env bash
# 解压放在 NSO 项目根目录下的本地上传压缩包
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- habitat-lab 源码 ---
HL_TGZ="$PROJECT_DIR/habitat-lab-0.2.4.tar.gz"
if [[ -f "$HL_TGZ" ]]; then
  mkdir -p "$PROJECT_DIR/third_party"
  cp -f "$HL_TGZ" "$PROJECT_DIR/third_party/habitat-lab-v0.2.4.tar.gz"
  bash "$PROJECT_DIR/scripts/fetch_habitat2_source.sh"
fi

mkdir -p "$DATA_DIR/scene_datasets" "$DATA_DIR/datasets/pointnav"

# --- Gibson PointNav 配置 ---
GIBSON_ZIP="$PROJECT_DIR/pointnav_gibson_v1.zip"
if [[ -f "$GIBSON_ZIP" ]] && [[ ! -f "$DATA_DIR/datasets/pointnav/gibson/v1/val/val.json.gz" ]]; then
  log "解压 Gibson PointNav 配置..."
  mkdir -p "$DATA_DIR/datasets/pointnav/gibson"
  unzip -q -o "$GIBSON_ZIP" -d "$DATA_DIR/datasets/pointnav/gibson/"
  mkdir -p "$DATA_DIR/datasets/pointnav/gibson/v1/val" \
           "$DATA_DIR/datasets/pointnav/gibson/v1/train"
  [[ -f "$DATA_DIR/datasets/pointnav/gibson/val/val.json.gz" ]] && \
    cp -n "$DATA_DIR/datasets/pointnav/gibson/val/"*.json.gz \
      "$DATA_DIR/datasets/pointnav/gibson/v1/val/" 2>/dev/null || true
  [[ -f "$DATA_DIR/datasets/pointnav/gibson/train/train.json.gz" ]] && \
    cp -n "$DATA_DIR/datasets/pointnav/gibson/train/train.json.gz" \
      "$DATA_DIR/datasets/pointnav/gibson/v1/train/" 2>/dev/null || true
  [[ -d "$DATA_DIR/datasets/pointnav/gibson/train/content" ]] && \
    cp -rn "$DATA_DIR/datasets/pointnav/gibson/train/content" \
      "$DATA_DIR/datasets/pointnav/gibson/v1/train/" 2>/dev/null || true
  log "Gibson PointNav -> data/datasets/pointnav/gibson/v1/"
fi

# --- MP3D PointNav 配置 ---
MP3D_PN_ZIP="$PROJECT_DIR/pointnav_mp3d_v1.zip"
if [[ -f "$MP3D_PN_ZIP" ]] && [[ ! -f "$DATA_DIR/datasets/pointnav/mp3d/v1/val/val.json.gz" ]]; then
  log "解压 MP3D PointNav 配置..."
  mkdir -p "$DATA_DIR/datasets/pointnav/mp3d"
  unzip -q -o "$MP3D_PN_ZIP" -d "$DATA_DIR/datasets/pointnav/mp3d/"
  mkdir -p "$DATA_DIR/datasets/pointnav/mp3d/v1/val" \
           "$DATA_DIR/datasets/pointnav/mp3d/v1/train"
  [[ -f "$DATA_DIR/datasets/pointnav/mp3d/val/val.json.gz" ]] && \
    cp -n "$DATA_DIR/datasets/pointnav/mp3d/val/"*.json.gz \
      "$DATA_DIR/datasets/pointnav/mp3d/v1/val/" 2>/dev/null || true
  [[ -f "$DATA_DIR/datasets/pointnav/mp3d/train/train.json.gz" ]] && \
    cp -n "$DATA_DIR/datasets/pointnav/mp3d/train/train.json.gz" \
      "$DATA_DIR/datasets/pointnav/mp3d/v1/train/" 2>/dev/null || true
  [[ -d "$DATA_DIR/datasets/pointnav/mp3d/train/content" ]] && \
    cp -rn "$DATA_DIR/datasets/pointnav/mp3d/train/content" \
      "$DATA_DIR/datasets/pointnav/mp3d/v1/train/" 2>/dev/null || true
  log "MP3D PointNav -> data/datasets/pointnav/mp3d/v1/"
fi

# --- MP3D 示例场景 ---
MP3D_EX_ZIP="$PROJECT_DIR/mp3d_example_v1.1.zip"
if [[ -f "$MP3D_EX_ZIP" ]] && [[ ! -f "$DATA_DIR/scene_datasets/17DRP5sb8fy/17DRP5sb8fy.glb" ]]; then
  log "解压 MP3D 示例场景..."
  unzip -q -o "$MP3D_EX_ZIP" -d "$DATA_DIR/scene_datasets/"
  log "MP3D example -> data/scene_datasets/"
fi

log "========== 数据检查 =========="
log "Gibson 场景: bash scripts/setup_gibson_habitat.sh"
for f in \
  "$DATA_DIR/datasets/pointnav/gibson/v1/val/val.json.gz" \
  "$DATA_DIR/datasets/pointnav/mp3d/v1/val/val.json.gz" \
  "$DATA_DIR/scene_datasets/gibson/Cantwell.glb" \
  "$DATA_DIR/scene_datasets/17DRP5sb8fy/17DRP5sb8fy.glb" \
  "$DATA_DIR/datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz"; do
  if [[ -f "$f" ]]; then echo "  OK $f"; else echo "  -- 缺失 $f"; fi
done

log "done"
