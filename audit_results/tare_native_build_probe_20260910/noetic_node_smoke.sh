#!/usr/bin/env bash
# Prepared only. Native ROS node liveness with no sensor input; NOT a demo.
set -euo pipefail
source /opt/tare_ws/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
roslaunch tare_planner explore_garage.launch rviz:=false rosbag_record:=false &
tare_smoke_pid=$!
trap 'kill -INT "$tare_smoke_pid" 2>/dev/null || true; wait "$tare_smoke_pid" 2>/dev/null || true' EXIT
tare_node_ready=0
for tare_attempt in $(seq 1 20); do
  if rosnode ping -c 1 /sensor_coverage_planner/tare_planner_node >/dev/null 2>&1; then
    tare_node_ready=1
    break
  fi
  sleep 0.5
done
test "$tare_node_ready" -eq 1
kill -0 "$tare_smoke_pid"
rosnode info /sensor_coverage_planner/tare_planner_node
printf '%s\n' '{"scope":"native planner node liveness only","sensor_inputs":false,"navigation_demo":false}'
