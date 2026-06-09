#!/usr/bin/env bash
# 挂载 120G 云盘到 /mnt/nso_data（仅需执行一次；已写入 /etc/fstab）
set -euo pipefail

CLOUD_DEV="${NSO_CLOUD_DEV:-/dev/vdb}"
MOUNT="${NSO_CLOUD_DATA:-/mnt/nso_data}"

if mountpoint -q "$MOUNT" 2>/dev/null; then
  echo "已挂载: $MOUNT ($(df -h "$MOUNT" | tail -1))"
  exit 0
fi

if ! sudo blkid "$CLOUD_DEV" | grep -q ext4; then
  echo "格式化 $CLOUD_DEV -> ext4 ..."
  sudo mkfs.ext4 -L nso_cloud -F "$CLOUD_DEV"
fi

sudo mkdir -p "$MOUNT"
sudo mount "$CLOUD_DEV" "$MOUNT"
sudo chown -R "${USER:-ubuntu}:${USER:-ubuntu}" "$MOUNT"
echo "OK: $MOUNT  $(df -h "$MOUNT" | tail -1)"
