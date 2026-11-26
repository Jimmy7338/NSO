# Frozen recursive reference / prediction comparisons

This is an offline sensor-estimator adaptation study. No live source, matcher, bridge, coordinates, TSDF, motion command, protection threshold or NSO policy was changed. It supports a bounded diagnostic candidate, not a reconstruction-accuracy, navigation-efficiency or NSO-benefit claim.

Three configurations share native installed scan conversion, ICP parameters and original ICP output:

- A: default 0.1 m / 10-degree reference thresholds, zero relative-motion prediction.
- B: only existing `kf_dist_linear=0` and `kf_dist_angular=0`. Reference replacement still uses the unmodified `newKeyframeNeeded`; exactly zero translation and rotation may retain a reference. There is no forced update, negative threshold, output clipping or new admission filter.
- C: default reference thresholds; only the previous-to-current motion prediction is supplied from source-time-aligned original CAN records, using piecewise-constant body velocity integration. This is a sampled motion model, not independent motion truth. The original `use_vel` subscriber was not enabled because it does not retain the required source-time semantics.

The helper links the original installed `liblaser_scan_matcher.so`, permits private-state access only in this finite offline executable, and mirrors the original reference/pose state update. Every row records original scan/reference/previous timestamps, previous/keyframe poses, first guess, native valid/x/error/nvalid/iterations, ICP elapsed time, result, reference switch and unchanged guard-bound audit. All results are unfiltered. Offline replay continues after a would-stop violation to expose the subsequent reverse jump; that continuation is not a completed physical trajectory.

Each recursive run uses the actual pose only once for a shared initial anchor. Thereafter each variant's previous pose and reference pose come exclusively from that variant's own preceding ICP result. `recursion_previous_equals_own_previous_result=true` was verified for every variant. B's `reference_replaced` equals the original switch decision on every row.

## Probe06: recovered failure state, eight recursive steps

The initial reference and pose are actual scan096, already identified by exact live-fault reproduction. Targets097–104 then run recursively. A reproduces the original sequence exactly, including both jump099 and reverse jump100. B avoids both violations in this finite segment.

| Mode | Valid / steps | Original guard violations | Maximum XY step | Cumulative XY path |
|---|---:|---|---:|---:|
| A | 8 / 8 | 099, 100 | 50.106 mm | 175.362 mm |
| B, native zero thresholds | 8 / 8 | none | 14.557 mm | 64.655 mm |

The earlier `../frozen_abc06/results.json` paired experiment reset the previous pose to the recorded pose on each step. That result is retained and labeled separately. The current `06_results.json` is genuinely recursive; A's recursive outputs match those frozen-pair outputs exactly. Shorter path is not proof of better accuracy.

C was not presented as a valid probe06 fault test. The retained original-kernel samples around099 have a 200.025 ms gap, exceeding the original 150 ms source TTL. No offset-reconstructed timestamps, future-sample backfill, fake zero or stale timestamp renewal was used to close this gap. The valid complete CAN test is probe07 instead; it cannot retroactively establish C's effect on fault099.

## Probe07: complete CAN data, 149 recursive steps

All modes begin at a common scan000 reference and its recorded actual pose. The unknown historical pre-ring keyframe is not claimed to be restored. `../../can_source_audit07/intervals.json` supplies 149 intervals with full original kernel coverage, 5–6 records including the preceding boundary per interval, no future velocity, and maximum held source age20.347 ms, below150 ms. The two stored ring files preserve the original scans and complete bounded CAN records. Probe07 physically succeeded under the original configuration at a different starting pose; it is not a matched efficacy comparison with failed probes05/06.

| Mode | Valid / steps | Guard violations | Maximum XY step | Reference switches | Cumulative XY path |
|---|---:|---:|---:|---:|---:|
| A | 149 / 149 | 0 | 15.668 mm | 9 | 551.285 mm |
| B | 149 / 149 | 0 | 13.085 mm | 149 | 443.392 mm |
| C | 149 / 149 | 0 | 14.529 mm | 9 | 549.563 mm |

B and C end8.48 mm and5.23 mm from A respectively, not from ground truth. They do not demonstrate accuracy gains. Timings are recorded for diagnostics; this one interleaved run is not a performance benchmark. C did not introduce a guard violation in this healthy frozen sequence, but its ability to fix the original failed pair remains untested with complete data.

## Negative control and interpretation

The separate installed-matcher stationary replay (`../../stationary_keyframe_check/summary.json`) used the same60 real scans: default start-to-end XY drift0.268 mm versus zero-threshold2.136 mm; maximum step1.214 versus1.546 mm; yaw span0.00205 versus0.00418 rad. Both consumed every scan and produced no ICP/TF warning. This is one six-second stationary input, not independent replication or external ground truth. The extra stationary jitter must remain beside the moving-segment improvement.

The independent correspondence audit (`../../correspondence_audit`) found the ref096 failure's retained point-to-line residual concentrated at ordinary2–6 m returns, particularly laser beam angles+90 to+180 degrees; it is not evidence that rare long-range tails alone caused the jump. Reference choice, objective correspondences and initialization remain relevant. The present findings justify treating B's existing zero-threshold setting as a minimal **diagnostic** candidate. They do not justify range clipping, a new residual rejection threshold, output smoothing or a general claim that frame-to-frame matching is more accurate.

## Reproduction and boundaries

In the original study layout and existing ROS Noetic / CSM environment:

```bash
source /opt/ros/noetic/setup.bash
python3 build.py
python3 run_recursive.py 07
python3 run_recursive.py 06
```

Each invocation refuses an occupied11399, starts its own isolated master, loads only its fixture parameters, executes a finite sequence and stops that master. It starts no navigation, bridge, CAN receiver or motion publisher. Serialized scan copies are temporary and removed after execution. Original NPZ/JSON files remain unchanged. No packages or clean-machine environment are provisioned. Input paths and hashes are in `freeze_manifest.json`; full scalar native outputs and source-prior metadata remain in `06_results.json` / `07_results.json`.

`candidate_B_parameters.json` contains the only proposed matcher-parameter difference. The installed matcher reads these in its constructor; this file is not a live hot-update mechanism. Deployment, coordinate alignment and any physical diagnostic remain root-owned and outside this artifact.
