# Native actor

The actor uses the archived whole-episode driver, measured TSDF and weighted same-goal chooser. Each subprocess owns its simulator and mapper. The global policy is neutral G throughout. The weighted local G chooser also runs before and after the one experimental selection. A cache rebuild collects unreachable Python/Open3D objects before calling the same native cache function. This memory lifecycle repair is active in every arm.

The first candidate bank is eligible through the inclusive clock of 64 if an acquired frame contains valid depth from the registered shelf and there are at least two distinct admitted outbound traces ending at the same position and heading. Each placement contains a single shelf of visible class 2. The actor counts actual valid class-2 depth support in the acquired prefix. The experimental D, S and D+ scores are evaluated only after the native candidate gate accepts this shared bank.

A triggered actor stores the full candidate bank, rankings and selected path with its exact return certificate. The shared fork receipt binds the raw observations, fused belief and mesh fields, ordered candidates, matched policy history, current budget, target, sensor seed/state and continuation version. A non-G actor checks the shared and selected-route hashes against the G bank before motion. The whole trajectory is independently recorded up to its actual native stop. The supervisor is responsible for aliasing arms only when their verified fork, route and continuation hashes agree.

Run G first, then any distinct non-G branch with the verified G bank.

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
"$NSO_PY" runtime/run.py shelf_dev_open G --freeze "$NSO_PHASE_FREEZE" --out "$NSO_G_RUN"
"$NSO_PY" runtime/run.py shelf_dev_open D --freeze "$NSO_PHASE_FREEZE" --out "$NSO_D_RUN" --fork-bank "$NSO_G_RUN/fork_bank.json"
```

The supervisor can pass `--allow-unbound-fallback` after a technical failure prevents it from verifying the G bank. In that case each remaining assignment runs independently. An untriggered episode reports a null alternative bank, retains the examined sensor prefix and continues with the shared neutral method.

The native whole-episode output records the untouched start, each actually observed checkpoint, the measured terminal map and the shared fork if eligible. Each charged action records successful native authorization, whether the robot was returning, route progress, pose, wall time and raw RGB-D and laser digests. The frozen available checkpoints have raw measured meshes. The compact `common_prefix.npz` includes measured RGB, semantic label, depth, calibration, pose and planar range arrays for the initial-to-fork sequence. The manifest hashes every archived actor output.
