#!/usr/bin/env bash
set -euo pipefail
source /home/ouc/scout_nav/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11382
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=127.0.0.1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
task_work=/home/ouc/nso_robot/workspaces/nso_hybrid_20261029
exec /usr/bin/python3 -s -B "$task_work/runtime/mission.py" "$@"
