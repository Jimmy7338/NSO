#!/usr/bin/env bash
set -eu
# One passive diagnostic window. Password input belongs only to local sudo.
exec env ROS_MASTER_URI=http://127.0.0.1:11382 \
  PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages \
  /usr/bin/python3 \
  /home/ouc/nso_robot/analyses/joint_coverage_pair_20261107/raw_scan_probe/capture.py \
  --sudo --output /home/ouc/nso_robot/analyses/joint_coverage_pair_20261107/raw_scan_probe/capture_01
