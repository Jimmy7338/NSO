# Forward-controller diagnosis, 2026-11-07

Three physical standard-navigation diagnostic probes, not NSO actions or a matched method-efficiency comparison. Their actual starting poses and scene observations differ; `forward_point_distance` also differs for the last probe. The source checkpoint retains all three outcomes:

| Probe | Outcome | Minimum logged position error | CAN distance / absolute yaw |
|---|---|---:|---:|
| short_forward_02 | navigation setup goal reached | 0.07397 m | 0.1223 m / 0.0932 rad |
| offset_forward_03 | interrupted: odometry jump | 0.09269 m | 0.4218 m / 3.1430 rad |
| offset_fp0_04 | interrupted: setup navigation duration elapsed | 0.09146 m | 0.3513 m / 2.9544 rad |

All were reported stopped. These observations do not establish controller efficacy, comparative success rates, or moving localization accuracy. The cached diagnosis records sign changes, planner failure warnings and possible dynamic-window/guard mismatch; per-cycle critic rejection, oscillation flags, and exact DWA input timing were not recorded. Temporal relationships do not prove causation. The 10 Hz telemetry contains latest received measurements, not synchronized controller cycles.

`probes/<name>/` preserves original telemetry, goal, configuration and post-probe parameter restoration. `checkpoint_summary.json` and `temporal_turn_diagnosis.json` are unchanged historical analyses; absolute local report paths therein are provenance, not portable links. From this directory, run `python3 -B verify_telemetry.py` to recompute minimum position errors, sample counts within 8 cm, and the 6 and 17 command-turn sign flips in the two interrupted probes. It checks that those flips agree with the archived diagnosis. The original diagnosis also used external ROS navigation and event logs not included here; this script does not independently regenerate failure-warning associations, planner-state claims or the full diagnosis.

## Isolated DWA mechanism reproduction

`dynamic_window_repro/` retains the unmodified observed results, representative measured-twist sample, C++ harness, candidate generator diff and companion `checkTrajectory` diff. The candidate was tested only in a separate generated class; it was not installed or deployed. The companion diff has not been integrated and tested. The finite synthetic cases support a dynamic-window mechanism, not an explanation of every physical reversal or complete closed-loop behavior. Rejecting an empty intersection could increase no-path/replan events; it is not a demonstrated convergence fix. CAN substitution and later experiments remain separate, pending evidence.

The reproduction requires an existing compatible ROS Noetic development environment with the original `base_local_planner` source and matching compiled `libbase_local_planner.so`, Eigen, C++ compiler, and transitive ROS libraries. The observed source SHA-256 values are `584e776487561956f6fb678c034d9c4f9dc5799452943e1bcd79bf1b8d080860` for `src/simple_trajectory_generator.cpp` and `73fa3282f4f913316bb86717c2b820b31dd4279fda6ffe85654888b5a73346be` for its header. The linked local library's recorded hash is `176b9d8dc15840b02d68fdf728ba131478003255058c4c6f5b8fde6b9c7e4227`; matching source and library provenance matters. None of these external dependencies is bundled here.

Set `PLANNER_SRC` to the `base_local_planner` source directory, `PLANNER_LIB` to the directory containing its existing library, and `ROS_INCLUDE` to the ROS include directory. In this package directory:

```sh
work=$(mktemp -d)
python3 dynamic_window_repro/prepare.py --source "$PLANNER_SRC" --output "$work"
cmp "$work/proposed_source.diff" dynamic_window_repro/proposed_source.diff
g++ -std=c++14 -O2 -I"$work" -I"$PLANNER_SRC/include" -I"$ROS_INCLUDE" -I/usr/include/eigen3 dynamic_window_repro/repro.cpp "$work/checked_generator.cpp" -L"$PLANNER_LIB" -Wl,-rpath,"$PLANNER_LIB" -lbase_local_planner -o "$work/repro"
"$work/repro" > "$work/results.jsonl"
cmp "$work/results.jsonl" dynamic_window_repro/results.jsonl
```

This command generates files in a separate temporary directory, links the pre-existing library, and does not need a ROS master or hardware. It was checked on the original machine; exact reproduction on a different ROS build is not established. Do not apply the candidate diff to production on the basis of this isolated test.

## Additional CAN control-twist probe

`probes/control_can_offset_05/` and `probes_addendum.json` preserve a later single-variable CAN control-velocity trial with default DWA and no candidate DWA patch. It was interrupted by the odometry-jump guard after 8.214 s and 0.178526 m, with zero verified arrivals. CAN feedback confirmed zero before and after closure; the result explicitly records no independent physical stop observation. This was one trial at its own starting pose, not a matched efficacy comparison or NSO action. Further fault investigation is pending.

A second CAN control-twist probe, `control_can_offset_06`, was interrupted by the odometry-jump guard after 9.668820 s and 0.184944 m, with zero verified arrivals. CAN zero feedback was confirmed before and after closure; independent physical stop observation was not recorded. Its `failure_scan_ring.npz` and companion JSON preserve a bounded history of 150 laser scans and 150 CSM poses. This is not the complete matcher state; a separate scan-replay investigation remains pending. Neither failed CAN trial establishes a final diagnosis.

## CSM scan replay and source audit

`failure_scan_replay06/` retains the three original fresh-initialization output JSON files (all 150 scans, last 100, and the 98–99 pair), their parameter audits, TF comparison, source audit and concise results. `probes/control_can_offset_05/analysis.json` and `probes/control_can_offset_06/analysis.json` preserve the two subsequent analyses. The three fresh matcher initializations did not reproduce the observed 0.050106 m live step: their maximum replay steps were approximately 0.022191 m, 0.022191 m, and 0.010173 m. This does not exonerate the original matcher: its actual internal reference keyframe, previous pose and first guess were not captured. Keyframe switch indices in the setup audit are inferred from replay and release defaults, not observed live state. TF and startup parameter checks do not independently recreate that missing state. Package-version-matched public source was audited, but exact equivalence to the installed matcher binary was not proven.

With ROS Noetic and the installed `laser_scan_matcher` executable, run from this directory on an otherwise isolated host:

```sh
work=$(mktemp -d)
/usr/bin/python3 -B failure_scan_replay06/repeat_clean_setup.py --output "$work"
```

Each case starts a fresh localhost ROS master on **11398**, refuses an already occupied port, forces its child processes to that master, and reads the bundled 06 capture and `failure_scan_replay06/inputs/tf_receipt.json`. It does not use the robot master on 11382. It requires ROS Python messages, NumPy, tf2 and the matcher binary; these dependencies are not bundled. Outputs are written outside the package. With the original installed matcher, the adapted package script was run on the isolated master and produced 150/150, 100/100 and 2/2 input/output counts with the same respective maximum steps above. Timing and absolute replay timestamps are not expected to be identical.
