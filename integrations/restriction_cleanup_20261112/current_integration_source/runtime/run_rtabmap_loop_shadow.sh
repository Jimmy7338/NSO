#!/usr/bin/env bash
set -euo pipefail
work=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# These overlays affect this child only. No shell startup files are changed.
set +u
source /opt/ros/noetic/setup.bash
source /home/ouc/hjd/catkin_ws/devel/setup.bash
set -u
export ROS_MASTER_URI=http://127.0.0.1:11382
export ROS_PACKAGE_PATH="$work/loop_module:${ROS_PACKAGE_PATH:-}"
if [[ ${1:-} == --check-only ]]; then
  rospack find rtabmap_slam
  rospack find rtabmap_sync
  rospack find nso_loop_integration
  /usr/bin/python3 -c 'import rospy; from rtabmap_msgs.msg import Info, MapGraph; print("RTAB-Map messages available")'
  roslaunch --nodes "$work/config/rtabmap_loop_shadow.launch" database_path:=/tmp/nso_loop_check.db output_dir:=/tmp/nso_loop_check
  exit 0
fi
output=$(/usr/bin/python3 "$work/runtime/rtabmap_loop_session.py" "$@")
mode=$(/usr/bin/python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["mode"])' "$output/startup.json")
exec roslaunch "$work/config/rtabmap_loop_shadow.launch" database_path:="$output/rtabmap.db" output_dir:="$output" mode:="$mode"
