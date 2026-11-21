# Actual-state fixture reproduces probe06's live jump

No live restart was needed. The 150 recorded scans and their actual serialized PoseStamped observations were sufficient to derive six possible final reference frames and reproduce fault099 exactly with reference096.

Before fault099, a retained reference must keep every later measured pose through098 within the unchanged 0.1 m / 10-degree keyframe bounds. Checking those recorded poses leaves only indices093–098. Ring000 to098 changed yaw by0.700096 rad, exceeding twice the10-degree bound, so a single reference older than the ring cannot remain valid throughout. The bounds select reference candidates; no ICP parameter was swept. `state_fixture_candidate_audit.json` retains each excluded candidate's first observed threshold crossing.

The fixture links the actual installed `/opt/ros/noetic/lib/liblaser_scan_matcher.so` and calls its original scan conversion, zero-predictor `getPrediction`, and exported `sm_icp`. The offline C++ helper uses `-fno-access-control` to assign the original class's previous/keyframe transforms and laser input pointers. It neither patches that library nor runs its scan callbacks. Reference and target scans are converted from their unchanged serialized data; actual previous and keyframe poses are restored from serialized PoseStamped. Base-to-laser TF remains the captured translation(-0.011,0.0035,0.451), identity rotation. The original formula `laser_from_base * inverse(keyframe_pose) * previous_pose * base_from_laser` supplies the first guess. No filtering, guess tuning, guard change or new prediction is involved.

| Reference index | Resulting jump, mm | XY error against live099, mm |
|---|---:|---:|
| 093 | 41.300701 | 15.942031 |
| 094 | 41.773373 | 15.211154 |
| 095 | 42.470630 | 14.839153 |
| **096** | **50.105966472451156** | **0** |
| 097 | 15.705681 | 34.890694 |
| 098 | 10.173318 | 39.984652 |

Reference096 stamp:1794053131381242990. Previous098 stamp:1794053131581063032. Fault099 stamp:1794053131681172132. For096, first guess was[0.010185340688086832,-0.004402547859409271,0.054821639273336456]. ICP reported valid=true, error=64.7565731504332, nvalid=1652, iterations=11. It did not request a keyframe switch. Its result matched live099 XY exactly and yaw within4.44e-16 rad. The original guard's bound was40.021820 mm; this fixture reproduces the50.105966 mm violation without moving the robot.

This identifies an actual-state/reference-dependent registration jump, rather than merely observing noise under a new initialization. The adjacent098/099 fresh-reference result is10.173318 mm and does not represent the historical096/099 registration. The fixture does not establish which physical surfaces or correspondences caused the failure, whether point times were distorted, or which estimator repair will preserve accuracy. ICP's `valid` flag alone admitted the reproduced outlier; solver error/nvalid are diagnostics, not independent reconstruction ground truth. No automatic rejection threshold is proposed here.

The publication copy reads the already-published `../probes/control_can_offset_06/failure_scan_ring.npz` and JSON; no duplicate sensor payload is included. `state_fixture_results.json` contains archived matcher parameters and results. Paths were changed only in the publication copy to use these inputs and a separate writable output directory. ROS Noetic, the original installed matcher and CSM libraries, Python ROS messages and NumPy are required. On the originating environment:

```bash
export FIXTURE_OUTPUT_DIR="$(mktemp -d)"
/usr/bin/python3 -B instrumented_csm_candidate/build_state_fixture.py
/usr/bin/python3 -B instrumented_csm_candidate/state_fixture.py
```

Run from `integrations/forward_controller_20261107`. The copied runner refuses an occupied port 11402, starts its own isolated master, and closes it. It does not connect to live11382. The publication paths and isolated port differ from the original evidence scripts; the ICP logic, observations and parameters are unchanged. Fresh reconstructed inputs are written to `FIXTURE_OUTPUT_DIR`; archived result files are not overwritten. This exact portable copy reproduced candidate096 on the original machine, not a clean-room installation.
