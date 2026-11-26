#!/usr/bin/env bash
set -euo pipefail
set +u
source /opt/ros/noetic/setup.bash
set -u
export ROS_MASTER_URI=http://127.0.0.1:11382
exec taskset -c 0,1,2,4,5,6,7 nice -n 5 /usr/bin/python3 -s -B "$(dirname -- "${BASH_SOURCE[0]}")/optimized_preview.py" "$@"
