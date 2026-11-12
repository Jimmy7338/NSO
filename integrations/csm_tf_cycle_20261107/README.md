# Isolated CSM TF-cycle replay

Input: the 60 serialized LaserScans in `inputs/scans.npz`. These are a later stationary capture, not the historical moving failure. Original files were read without alteration; hashes are in `metadata.json`.

Only ROS master `http://127.0.0.1:11398` was used. The real installed matcher and the existing bridge ran in isolation, with the captured CSM private parameters and static base-to-laser transform. Default group omitted `use_tf` (binary default true); false group explicitly disabled it. No actuator, planner, CAN publisher, live scan subscription, or live-master connection was used. Source stamps were shifted by a constant to each group's current clock; per-source index, original sequence, source stamp, replay stamp, and publication time are retained in each JSON. The paced run doubled wall-clock spacing while retaining original relative source stamps.

| Replay | Outputs / inputs | TF warnings | Median publish-to-pose latency |
|---|---:|---:|---:|
| Original ~10 Hz, default | 57 / 60 | 57 | 150.10 ms |
| Original ~10 Hz, false | 60 / 60 | 0 | 5.92 ms |
| 2x slower pacing, default | 60 / 60 | 60 | 107.81 ms |
| 2x slower pacing, false | 60 / 60 | 0 | 6.62 ms |

Default 10 Hz omitted source indices 15, 32, 51. Across 57 common source stamps, maximum XY difference was 0.0394 mm and maximum yaw difference 0.00002695 rad. In the paced comparison all 60 source stamps were present and poses were numerically identical (maximum XY/yaw differences exactly zero).

All four groups had XY coordinate spans 1.399 / 1.408 mm and maximum adjacent XY displacement 1.214 mm. Full 60-scan groups had XY standard deviations 0.340 / 0.324 mm and yaw span 0.002050 rad. This stationary sample did not reproduce the historical 43.282 mm moving-frame anomaly.

Conclusion: explicitly disabling TF prediction removes the demonstrated circular wait, reduces output latency, and avoids three lost inputs in this one 10 Hz replay. Under equal complete inputs it does not change estimated poses or stationary jitter. This establishes a timing/configuration repair, not a demonstrated localization-accuracy improvement or repair of the historical moving jump. The research policy and jump guard remain untouched.

Reproduce after ensuring port 11398 is free:

```bash
source /opt/ros/noetic/setup.bash
python3 integrations/csm_tf_cycle_20261107/replay.py
```

The script refuses an occupied port, starts its own isolated master, runs the four bounded groups, stores logs/source mappings/poses, regenerates `summary.json`, and stops its child processes. Reported elapsed times in JSON include the final 1.5-second drain interval; per-pose latencies above measure publication to receipt.

Dependencies: ROS 1 Noetic, Python 3 NumPy, rospy, tf2_ros, geometry_msgs, sensor_msgs, nav_msgs, roslib, and laser_scan_matcher 0.4.0. The bridge is resolved relative to this repository; matcher executable is resolved through the ROS package index. The original host used Debian package ros-noetic-laser-scan-matcher 0.4.0-1focal.20250521.032611. No binary is bundled and no clean-machine environment is provisioned. Run only on an isolated host with port 11398 free. `recorded_summary.json` preserves the original measured results; reruns generate `summary.json` and group logs/JSON. Original local absolute paths are not required.
