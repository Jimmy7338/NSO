#!/usr/bin/env bash
# NSO 数据集下载脚本（Gibson PointNav 配置 / Habitat 测试场景 / MP3D 示例）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
mkdir -p "$DATA_DIR/scene_datasets" "$DATA_DIR/datasets/pointnav"

log() { echo "[$(date +%H:%M:%S)] $*"; }

download() {
  local url="$1" out="$2"
  local min_bytes="${3:-0}"
  if [[ -f "$out" ]]; then
    local size
    size=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if unzip -t "$out" >/dev/null 2>&1; then
      log "已存在且校验通过，跳过: $out"
      return 0
    fi
    if [[ "$min_bytes" -gt 0 ]] && [[ "$size" -ge "$min_bytes" ]]; then
      log "已存在（${size} 字节），跳过: $out"
      return 0
    fi
    if [[ "$size" -gt 0 ]]; then
      log "续传不完整文件 (${size} 字节): $out"
    fi
  fi
  log "下载: $url -> $out"
  curl -fL --retry 10 --retry-delay 5 --retry-all-errors -C - -o "$out" "$url"
}

# --- 1. Gibson PointNav 配置（约 385MB，官方 CDN）---
GIBSON_PN_ZIP="$DATA_DIR/datasets/pointnav/gibson/pointnav_gibson_v1.zip"
if [[ ! -f "$DATA_DIR/datasets/pointnav/gibson/v1/val/val.json.gz" ]]; then
  download "https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip" \
    "$GIBSON_PN_ZIP" 380000000
  unzip -q -o "$GIBSON_PN_ZIP" -d "$DATA_DIR/datasets/pointnav/gibson/"
  mkdir -p "$DATA_DIR/datasets/pointnav/gibson/v1/val" \
           "$DATA_DIR/datasets/pointnav/gibson/v1/train"
  [[ -f "$DATA_DIR/datasets/pointnav/gibson/val/val.json.gz" ]] && \
    cp -n "$DATA_DIR/datasets/pointnav/gibson/val/"*.json.gz "$DATA_DIR/datasets/pointnav/gibson/v1/val/" 2>/dev/null || true
  [[ -f "$DATA_DIR/datasets/pointnav/gibson/train/train.json.gz" ]] && \
    cp -n "$DATA_DIR/datasets/pointnav/gibson/train/train.json.gz" "$DATA_DIR/datasets/pointnav/gibson/v1/train/"
  [[ -d "$DATA_DIR/datasets/pointnav/gibson/train/content" ]] && \
    cp -rn "$DATA_DIR/datasets/pointnav/gibson/train/content" "$DATA_DIR/datasets/pointnav/gibson/v1/train/" 2>/dev/null || true
  log "Gibson PointNav 配置已解压到 data/datasets/pointnav/gibson/v1/"
else
  log "Gibson PointNav 配置已就绪"
fi

# --- 2. Habitat 测试场景 + PointNav（快速验证，约 90MB + 1MB）---
HT_PN_ZIP="/tmp/habitat_test_pointnav.zip"
HT_SC_ZIP="/tmp/habitat_test_scenes.zip"
if [[ ! -f "$DATA_DIR/scene_datasets/habitat-test-scenes/apartment_1.glb" ]]; then
  download "http://dl.fbaipublicfiles.com/habitat/habitat-test-pointnav-dataset_v1.0.zip" "$HT_PN_ZIP" 800000
  unzip -q -o "$HT_PN_ZIP" -d "$DATA_DIR/datasets/pointnav/habitat-test-scenes"
  download "http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes_v1.0.zip" "$HT_SC_ZIP" 85000000
  mkdir -p "$DATA_DIR/scene_datasets/habitat-test-scenes"
  unzip -q -o "$HT_SC_ZIP" -d "$DATA_DIR/scene_datasets/habitat-test-scenes/"
  log "Habitat 测试数据已安装（可用 tasks/pointnav 的 habitat_test 配置）"
fi

# --- 3. MP3D PointNav 配置（约数百 MB）---
MP3D_PN_ZIP="/tmp/pointnav_mp3d_v1.zip"
if [[ ! -f "$DATA_DIR/datasets/pointnav/mp3d/v1/val/val.json.gz" ]]; then
  mkdir -p "$DATA_DIR/datasets/pointnav/mp3d/v1/"{train,val,test}
  download "https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip" \
    "$MP3D_PN_ZIP" 390000000
  unzip -q -o "$MP3D_PN_ZIP" -d "$DATA_DIR/datasets/pointnav/mp3d/"
  log "MP3D PointNav 配置已解压"
fi

# --- 4. MP3D 示例场景（单场景，约 65MB，无需完整 Matterport 授权）---
MP3D_EX_ZIP="/tmp/mp3d_example.zip"
if [[ ! -f "$DATA_DIR/scene_datasets/17DRP5sb8fy/17DRP5sb8fy.glb" ]]; then
  download "http://dl.fbaipublicfiles.com/habitat/mp3d/mp3d_example_v1.1.zip" "$MP3D_EX_ZIP" 60000000
  unzip -q -o "$MP3D_EX_ZIP" -d "$DATA_DIR/scene_datasets/"
  log "MP3D 示例场景已安装"
fi

log ""
log "========== 状态检查 =========="
source "${CONDA_DIR:-$HOME/miniconda3}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate nso 2>/dev/null || true
python3 "$PROJECT_DIR/verify_mp3d_setup.py" 2>/dev/null || true

log ""
log "========== 仍需手动完成 =========="
log "Gibson 全量场景（.glb + .navmesh）："
log "  1. 在 https://github.com/StanfordVL/GibsonEnv 同意协议并获取下载方式"
log "  2. 将场景放入: $DATA_DIR/scene_datasets/gibson/"
log ""
log "Matterport3D 全量场景："
log "  1. 在 https://niessner.github.io/Matterport/ 申请"
log "  2. 将场景放入: $DATA_DIR/scene_datasets/mp3d/"
log ""
log "habitat-sim / habitat-api：请运行 bash scripts/install_server.sh（conda 环境 nso）"
