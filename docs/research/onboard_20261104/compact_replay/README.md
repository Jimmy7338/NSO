# Derived measured-sensor replay: S40

This is a compact derived-input record of 41 accepted observations from one physical episode, with canonical 320×180 color, depth, intrinsic and cached semantic labels; measured camera/laser transforms, scan, base pose, timestamps and the observed navigation grid are retained. `manifest.json` records original packet and converted-packet hashes, frame IDs, and each retained array's dtype, shape and SHA-256. The original labels are cached automatic YOLOE results with an explicit category prior, not ground truth. The files cannot independently verify the original detection, registration, sensor calibration, or model inference: full-resolution RGB-D, ROS serialized messages, detector masks and weights are omitted. One episode is not an independent repeat or accuracy evidence.

`reference/live0.npz` is a separate later actual observation in the same externally checked continuous odometry epoch. It bootstraps offline replay of all 41 historical frames using the bundled bridge and configuration; it is not the 42nd accepted frame of the original episode. Its semantic labels are cached. Prefix decoding intentionally reconstructs maps from historical sensors rather than importing their navigation grids; the original grids remain in each frame for other offline decision replays. `reference/decisions.jsonl` records historical actions, not proof that this fresh-initialization next decision was physically executed. `execution_allowed` is forced false.

With Python 3.12, NumPy 1.26.4 and SciPy 1.11.4 in the tested science312 environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B replay.py
```

This validates 41 per-array hashes and compares the four mapper arrays, resulting state and next action with `reference/replay_expected.json`. To additionally compare the unabridged converted source arrays and rerun them through the same map reconstruction:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/ouc/nso_robot/envs/science312/bin/python -B replay.py --compare-original /home/ouc/nso_robot/analyses/continuation_20261104
```

The original converted input directory is local and excluded from this repository. Matching replays establish faithful conversion under this setup; they do not independently validate the sensor observations or detector. On another numerical environment exact results may differ.

`code/joint_vendor` is the selected preserved research source, with original source manifest and license. `code/nso_bridge` is the local physical adaptation from `/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge/nso_bridge`; it uses real camera/laser inputs and discretizes measured base poses. No changes to policy were made for this package. `reference/configuration.json` is the existing offline configuration; absolute paths in descriptive calibration metadata are provenance, not input dependencies. Code snapshots, archive and source metadata are local artifacts; no publication is implied.
