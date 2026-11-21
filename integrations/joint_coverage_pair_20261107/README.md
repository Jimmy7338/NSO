# Ordered physical C/S exploration pair

One scene and one ordered C-then-S pair, with fresh independently initialized mapping for each arm. C completed 40 paid actions, 40 arrivals and 41 observations and returned to the anchor; S stopped after 32 paid actions, 32 arrivals and 33 observations with `current_footprint_not_known_safe`, without completing return. Both stopped with CAN zero confirmed. This is not an independent replication or a matched efficacy result; starting poses, initial maps, and observed data differ. Initial 3D display TSDF and localization were retained. See `comparison/` and both summary files for actual data, assumptions and limitations.

The aligned, derived canonical arrays preserve the recorded depth, color, geometry, map and cached semantic labels; they are not original full RGB-D packets and cannot independently validate detection. The actual S termination is retained without invented observations. The prior public research code is reused from commit `763c107794f3fdcc3a0d281b2eff88928226836d` under `integrations/exploration_resume_20261107/code`; `code/` here contains the changed physical adapters and their tests, not a second vendor copy. Both arms use the declared measured extent 3m geometry adaptation; do not present this as unchanged original 2.2m applicability.

For the same offline cached-label replay, with science312 dependencies installed, run from any directory using the repository paths:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python integrations/joint_coverage_pair_20261107/replay_portable.py --arm C --vendor-code integrations/exploration_resume_20261107/code --output /tmp/nso_pair_C_replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python integrations/joint_coverage_pair_20261107/replay_portable.py --arm S --vendor-code integrations/exploration_resume_20261107/code --output /tmp/nso_pair_S_replay
```

The wrapper stages byte-identical executed runners and points them to existing vendor code. Recorded replay outputs compare actual engine responses and selection scores, with declared tolerances and frontend-specific exceptions; no inference or robot is used. The original protocol README is retained as a historical pre-execution document, and its preparation-time status is not the final outcome. Scripts that capture or start epochs operate physical ROS infrastructure and are retained as provenance, not required for offline replay.

The mapper-only source audit can also be reproduced without a planner or detector:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python integrations/joint_coverage_pair_20261107/footprint_failure/mapper_portable.py --vendor-code integrations/exploration_resume_20261107/code --output /tmp/nso_pair_mapper
```

The wrapper redirects only paths in a temporary copy of the original audit. Its independent validation matches the original frame 31/32 belief hashes and insertion-stage hashes; see `footprint_failure/portable_validation.json`. The observed laser return's physical identity remains unverified.
