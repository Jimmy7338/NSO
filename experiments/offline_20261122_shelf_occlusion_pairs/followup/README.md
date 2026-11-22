# Separate diagnostic experiments

The formal shelf experiment uses a fixed global planner and mapper. The investigations below have separate inputs and outputs.

- [Global region attribution and outbound rewards](https://github.com/Jimmy7338/NSO/blob/2f6c97352aeb7fed719bb82c422418c7412eaa8b/experiments/offline_20261122_region_global_revision/next/global_attribution/results/RESULTS.md) traces each change to the archived candidate pools and documents the independent floor comparison.
- [Full-floor residual audit](global_residual/RESULTS.md) locates later surface differences by room, wall boundary and orientation using the original fused meshes.
- [Passive camera replay](global_residual/SENSOR_RESULTS.md) measures which missed boundary surfaces passed through the camera frustum, had an unobstructed first hit and matched measured depth.
- [Local scoring diagnostics](local_scoring_diagnostic/) relates the D+ increment to the score margin within the formal banks and checks actual S–G divergence.
- [Local route witnesses](local_witness/README.md) counts regret against the executed subset of an independently sampled route bank in two development placements.
- [West projection shadow](west_projection_shadow/README.md) tests direction-sensitive endpoint ownership with fixed raw measurements and an independent continuous geometry check.
- [West cold-start simulation](west_projection_native/RESULTS.md) checks the versioned endpoint change in one newly executed neutral G episode.
