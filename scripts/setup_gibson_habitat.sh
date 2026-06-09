#!/usr/bin/env bash
# 解压 Gibson Habitat 场景（.glb）并校验 PointNav 配置
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CLOUD_ROOT="${NSO_CLOUD_DATA:-/mnt/nso_data}"
DATA_DIR="$PROJECT_DIR/data"
GIBSON_DIR="$DATA_DIR/scene_datasets/gibson"
# 大 zip 优先放云盘，避免占满 40G 系统盘
ARCHIVE_DIR="$CLOUD_ROOT/archives"
if [[ -d "$CLOUD_ROOT" ]]; then
  mkdir -p "$ARCHIVE_DIR"
fi
CONFIG_URL="https://dl.fbaipublicfiles.com/habitat/gibson/config_v1/gibson_semantic.scene_dataset_config.json"
EXPECTED_TRAINVAL_SIZE=10833075327

log() { echo "[$(date +%H:%M:%S)] $*"; }

check_zip_ok() {
  python3 - <<PY >/dev/null 2>&1 || return 1
import sys, zipfile
with zipfile.ZipFile("$1") as f:
    bad = f.testzip()
    if bad:
        sys.exit(1)
PY
}

pick_zip() {
  local trainval="" challenge=""
  for base in "$ARCHIVE_DIR" "$PROJECT_DIR"; do
    [[ -f "$base/gibson_habitat_trainval.zip" ]] && trainval="$base/gibson_habitat_trainval.zip"
    [[ -f "$base/gibson_habitat.zip" ]] && challenge="$base/gibson_habitat.zip"
  done
  trainval="${trainval:-$PROJECT_DIR/gibson_habitat_trainval.zip}"
  challenge="${challenge:-$PROJECT_DIR/gibson_habitat.zip}"
  if [[ -f "$trainval" ]]; then
    local sz
    sz=$(stat -c%s "$trainval")
    if [[ "$sz" -ge "$((EXPECTED_TRAINVAL_SIZE - 50000000))" ]] && check_zip_ok "$trainval"; then
      echo "$trainval"
      return 0
    fi
    log "警告: gibson_habitat_trainval.zip 不完整 ($(($sz / 1024 / 1024))MB / 约 10100MB)"
    log "续传（建议下到云盘）: wget -c -O $ARCHIVE_DIR/gibson_habitat_trainval.zip \\"
    log "  https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat_trainval.zip"
  fi
  if [[ -f "$challenge" ]] && check_zip_ok "$challenge"; then
    echo "$challenge"
    return 0
  fi
  return 1
}

finalize_gibson_extract() {
  local tmp="$1"
  if [[ -d "$tmp/gibson" ]]; then
    find "$tmp/gibson" -maxdepth 1 \( -name '*.glb' -o -name '*.navmesh' \) \
      -exec mv -n {} "$GIBSON_DIR/" \;
  fi
  find "$tmp" \( -name '*.glb' -o -name '*.navmesh' \) -exec mv -n {} "$GIBSON_DIR/" \;
  rm -rf "$tmp"
}

extract_zip() {
  local zip="$1"
  mkdir -p "$GIBSON_DIR"
  local tmp="$GIBSON_DIR/.unpack_tmp"
  rm -rf "$tmp"
  mkdir -p "$tmp"
  log "解压 $zip ..."
  unzip -q -o "$zip" -d "$tmp"
  finalize_gibson_extract "$tmp"
  if [[ "${DELETE_ZIP_AFTER:-0}" == 1 ]]; then
    log "删除压缩包: $zip"
    rm -f "$zip"
  fi
}

install_pointnav_gibson_config() {
  local z="$PROJECT_DIR/pointnav_gibson_v1.zip"
  local val="$DATA_DIR/datasets/pointnav/gibson/v1/val/val.json.gz"
  [[ -f "$val" ]] && return 0
  [[ -f "$z" ]] || { log "缺少 pointnav_gibson_v1.zip"; return 1; }
  log "解压 Gibson PointNav 配置..."
  mkdir -p "$DATA_DIR/datasets/pointnav/gibson"
  unzip -q -o "$z" -d "$DATA_DIR/datasets/pointnav/gibson/"
  mkdir -p "$DATA_DIR/datasets/pointnav/gibson/v1/val" \
           "$DATA_DIR/datasets/pointnav/gibson/v1/train"
  [[ -f "$DATA_DIR/datasets/pointnav/gibson/val/val.json.gz" ]] && \
    cp -n "$DATA_DIR/datasets/pointnav/gibson/val/"*.json.gz \
      "$DATA_DIR/datasets/pointnav/gibson/v1/val/" 2>/dev/null || true
  [[ -f "$DATA_DIR/datasets/pointnav/gibson/train/train.json.gz" ]] && \
    cp -n "$DATA_DIR/datasets/pointnav/gibson/train/train.json.gz" \
      "$DATA_DIR/datasets/pointnav/gibson/v1/train/" 2>/dev/null || true
}

main() {
  command -v unzip >/dev/null || { echo "需要 unzip"; exit 1; }
  install_pointnav_gibson_config || true

  local zip
  if ! zip=$(pick_zip); then
    log "未找到可用 zip。可下载子集（约 1.4GB）："
    log "  wget -O $PROJECT_DIR/gibson_habitat.zip \\"
    log "    https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat.zip"
    exit 1
  fi

  local glb_count
  glb_count=$(find "$GIBSON_DIR" -maxdepth 1 -name '*.glb' 2>/dev/null | wc -l)
  if [[ "${FORCE_GIBSON_REEXTRACT:-0}" == 1 ]]; then
    log "强制重新解压（清空旧场景）..."
    rm -f "$GIBSON_DIR"/*.glb "$GIBSON_DIR"/*.navmesh 2>/dev/null || true
    extract_zip "$zip"
  elif [[ "$glb_count" -lt 5 ]]; then
    extract_zip "$zip"
  elif [[ "$zip" == *trainval* ]] && [[ "$glb_count" -lt 200 ]]; then
    log "检测到 trainval 包但场景数偏少 ($glb_count)，重新解压..."
    rm -f "$GIBSON_DIR"/*.glb "$GIBSON_DIR"/*.navmesh 2>/dev/null || true
    extract_zip "$zip"
  else
    log "Gibson 场景已存在 ($glb_count 个)，跳过解压"
  fi
  if [[ ! -f "$GIBSON_DIR/gibson_semantic.scene_dataset_config.json" ]]; then
    log "下载 scene_dataset 配置..."
    wget -q -O "$GIBSON_DIR/gibson_semantic.scene_dataset_config.json" "$CONFIG_URL"
  fi

  local cnt=0
  cnt=$(find "$GIBSON_DIR" -maxdepth 1 -name '*.glb' | wc -l)
  log "Gibson .glb 数量: $cnt"
  if [[ -f "$GIBSON_DIR/Cantwell.glb" ]]; then
    log "OK Cantwell.glb"
  else
    log "警告: 缺少 Cantwell.glb，val 部分 episode 可能无法加载"
  fi
  log "运行: bash scripts/run_nso_h2_gibson_vis.sh"
}

main "$@"
