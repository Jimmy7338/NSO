Staged candidate only; no deployed file or running process has been changed.

This bridge can authenticate `/nso_csm_zero_keyframe/pose_stamped` from caller `/nso_csm_zero_keyframe/matcher` in frame `odom_zero_keyframe_20261107`. Its output and TF remain `odom -> base_link`. Only one bridge may own those outputs/TF during root's controlled cutover. Keep the original matcher available as an independent shadow reference; both matchers must have `publish_tf=false`.

Use a fresh pair of original old-odom and candidate poses with exactly the same original scan nanosecond stamp. Compute constant `T_odom_from_source = T_old_odom_from_base * inverse(T_candidate_from_base)`. Write:

```json
{
  "schema": "csm_fixed_se2_alignment/1",
  "source_topic": "/nso_csm_zero_keyframe/pose_stamped",
  "expected_caller": "/nso_csm_zero_keyframe/matcher",
  "source_frame": "odom_zero_keyframe_20261107",
  "target_frame": "odom",
  "odom_from_source": ["tx (actual number)", "ty (actual number)", "yaw (actual number)"],
  "anchor": {
    "source_stamp_ns": "actual positive integer",
    "odom_stamp_ns": "same actual integer",
    "source_pose": ["candidate x", "candidate y", "candidate yaw"],
    "odom_pose": ["old odom x", "old odom y", "old odom yaw"]
  }
}
```

The strings above are schema placeholders and intentionally do not pass validation. Root supplies real finite numbers and real same-stamp poses. Loading validates source bindings and anchor equivalence within 1e-7 m/rad, then freezes the transform/receipt/SHA. Do not modify an alignment file to re-anchor a running bridge; it does not reload. Candidate restarts or coordinate-epoch changes require another explicitly recorded alignment/cutover, never an automatic fallback.

Add these options to root's existing bridge command, preserving its existing CAN selection/interface arguments:

```
--pose-topic /nso_csm_zero_keyframe/pose_stamped
--expected-caller /nso_csm_zero_keyframe/matcher
--source-frame odom_zero_keyframe_20261107
--alignment-json /ABSOLUTE/REAL_ALIGNMENT.json
```

No flags retain the original source and original pose object/value behavior. An alignment applies a fixed world SE2 transform, keeps measured Z/tilt, preserves the exact incoming scan stamp object, and preserves body-frame twist under world-coordinate rotation. It never smooths, clamps registration, holds a pose under a new stamp, fabricates a caller, replaces failed ICP with CAN pose, or changes CAN reader/TTL/guards. The existing first-sample withholding and nonincreasing-stamp behavior are unchanged.

Both modes overwrite `~pose_source`: default declares `original_csm_pose_source_v1` and empty alignment, clearing stale candidate metadata. Aligned mode declares `fixed_se2_pose_source_diagnostic_v1` and bindings/full receipt/SHA. ROS parameter receipt `*_ns` fields are decimal strings with an explicit encoding label because XMLRPC cannot marshal int64; the alignment JSON file and topic provenance retain original integer nanoseconds. Aligned runs also publish `/nso_csm/pose_source_provenance` and annotate existing CAN provenance. `raw_pose_stamp_tf_unchanged` is false in aligned CAN provenance; `fixed_coordinate_transform_applied`, `raw_registration_preserved`, and `source_pose_stamp_unchanged` describe the actual change. Original source PoseStamped and transformed odometry should be captured independently.

This is an explicitly labeled sensor/localization diagnostic adaptation. Fixed coordinate alignment preserves an existing coordinate epoch; it does not establish calibration or physical accuracy. More frequent keyframes increased millimeter-scale stationary drift in the existing six-second regression. One different-start successful probe is not a paired NSO method or accuracy result.
