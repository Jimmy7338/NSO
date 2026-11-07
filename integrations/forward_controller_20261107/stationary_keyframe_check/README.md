# Isolated stationary keyframe negative control

The 60 original scans and transform/parameter receipt are already stored at `integrations/csm_tf_cycle_20261107/inputs/`; this directory does not duplicate them. `default.json` and `every_frame.json` retain the source-time mappings and output poses, with original parameter receipts and `summary.json`. The six-second stationary-context replay used independent new matcher processes. Default thresholds produced 0.268 mm start-to-end XY displacement; zero thresholds produced 2.136 mm and larger spread. This is a single input without independent ground truth, not long-run drift or moving accuracy evidence. The candidate was not deployed in this record.

From this directory in the existing compatible ROS Noetic installation:

```bash
export STATIONARY_OUTPUT_DIR="$(mktemp -d)"
/usr/bin/python3 -B run.py
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -B summarize.py
```

The publication copies read the existing package-relative inputs, use isolated localhost master 11402 and write only to the supplied output directory. A rerun on the originating machine produced 60/60 observations for each profile and identical geometric summary values; wall-clock latency naturally changes. No live ROS graph or robot command is accessed. Reproduction on a clean installation is not established.
