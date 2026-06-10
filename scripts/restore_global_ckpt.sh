#!/usr/bin/env bash
# 为指定阶段目录补齐 model_best.global（优先已有 best，其次 periodic_*.global，最后 pretrained）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${NSO_RUN_ROOT:-/mnt/nso_data/nso_runs}"
PRETRAINED="${PRETRAINED_DIR:-$SCRIPT_DIR/../pretrained_models}"

stage_dir() {
  local name="$1"
  if [[ -d "$name" ]]; then echo "$name"
  elif [[ -d "$RUN_ROOT/models/$name" ]]; then echo "$RUN_ROOT/models/$name"
  else echo "$RUN_ROOT/models/$name"; fi
}

restore_one() {
  local stage_name="$1"
  local dir
  dir="$(stage_dir "$stage_name")"
  local exp="${stage_name##*/}"
  local dump="$RUN_ROOT/dump/$exp"
  local out="$dir/model_best.global"

  if [[ -s "$out" ]]; then
    echo "[OK] $out 已存在 ($(du -h "$out" | awk '{print $1}'))"
    return 0
  fi

  mkdir -p "$dir"
  local periodic
  periodic="$(find "$dump" -name 'periodic_*.global' 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "$periodic" && -s "$periodic" ]]; then
    cp -f "$periodic" "$out"
    echo "[恢复] $out <- $periodic"
    return 0
  fi

  if [[ -s "$PRETRAINED/model_best.global" ]]; then
    cp -f "$PRETRAINED/model_best.global" "$out"
    echo "[回退] $out <- pretrained"
    return 0
  fi

  echo "[失败] 无法为 $stage_name 找到 global 权重" >&2
  return 1
}

main() {
  local stages=("$@")
  if [[ ${#stages[@]} -eq 0 ]]; then
    stages=(stage1_slam_local stage2_paper_global stage3_rpn stage4_ssc_loop)
  fi
  for s in "${stages[@]}"; do
    restore_one "$s"
  done
}

main "$@"
