# Visibility-consistent template exposure: B2 offline evidence

This package records a disclosed correction to the full-template joint-exploration extension in [the preceding offline release](../offline_20261119/). The repair skips a hypothetical self-occluding ray intersection only when valid acquired depth strictly contradicts it, with a measured-surface veto. It does not change the score weights, candidate generator, template dimensions, real-TSDF occlusion, action budget, return rule, furniture geometry, or initial conditions.

The exact repair source is `57c5cd1bb58a2e816d81184014cbfbad2a3d5ddeb872c43c7b5d9895378bb2fb`. All 12 declared slots are retained: C, J, G, S, S_no_fit, and S_no_history in two furniture permutations of **one authored floorplan**. These are deterministic paired conditions, not twelve independent environments. The input is ideal visible category and partial instance support for the two checked structures, not recognition by YOLOE. The method is a new combination of the original V10 joint-coverage candidate policy with the V3 complete two-template estimator; it is not an unchanged reproduction of either original planner or a physical-robot experiment.

G and S remain trajectory- and mesh-identical within both layouts. Relative to the preceding B version, both improve left-layout final F1 at 5 cm by 0.238652 percentage points, with no change in first full-coverage action. Their right-layout outputs remain unchanged. C and J are unchanged negative controls. The other ablation outcomes, including deterioration, are preserved. S_no_history removes historical reward-sample filtering but retains the visibility repair; the original global sample cap is still active in some of its planning calls, so that comparison is not a pure isolated history effect. No semantic-specific efficiency or reconstruction advantage is established.

The two saved-route sample-matching intervals in `replay/attempt_003/prediction_matching.json` are post-hoc diagnostics, not new policy trajectories. Predicted sample support is measured against source triangles at 5 cm. Weights are hypothetical template area, can include alternative hypotheses and duplicate nearby support, and are not independent real-surface area. The repair restores some source-supported samples but leaves geometry mismatch. The earlier replay retains its original source-provenance limitation: attempt_001 loaded a pre-final repair while its original trajectory, sensor hashes and old score checks remained valid. Attempts 002 and 003 preserve exact frozen source snapshots and receipts; no receipt is silently replaced.

## Contents

- `run.py`, `protocol.json`, and `method/`: frozen B2 runtime and the single repair.
- `layouts/*/prepared/`, all terminal `result.json` and `attempt.json`: both pre-run freezes and all declared outcomes.
- `audit/`: independent surface evaluation, mechanism diagnostics, and final 12-slot summary.
- `model_audit/`, `candidate_audit/`: meaningful tests, the preserved initial boundary-test failure, final passing tests, and independent reviews of successive identified source hashes.
- `replay/`: three fixed-route replay receipts, source snapshots, summary, and same-domain matching. The two small attempt_003 prediction NPZ files contain the algorithm's own template samples and weights. Evaluator-distance NPZ files include coordinates derived from external source geometry and are excluded; their hashes and aggregate matching results remain available.
- `archives/`: lossless exact choices, steps, template, owner/frame and replay logs, with every archive member's byte count and SHA-256 in `ARCHIVE_INDEX.json`.
- `paper_update/`: one measured B-to-B2 comparison figure with editable SVG, PDF, PNG, CSV and plotting source; claim-evidence audit; the compact saved-state matching table. The current manuscript is added separately by the parent release workflow.

## Dependencies and limits of reproduction

This release is **not self-contained or one-command runnable**. The runner retains the original absolute paths rooted at `/home/ouc/nso_robot/analyses/`. It imports the preceding release's frozen scientific source, original B method and controlled-world adapter, the shared instance-support modules, and the fixed evaluator. Retrieve the source archives and dependency receipts from [offline_20261119](../offline_20261119/). `DEPENDENCIES.json` records their hashes and parent freezes. The separately licensed ReplicaCAD furniture assets, source meshes, fixed-reference geometry, raw sensors and native TSDF meshes are deliberately excluded. Re-executing the actor or full surface evaluator requires those exact external assets and path/dependency setup; source and result bytes can be verified without them.

The exact local Python/package metadata is in `ENVIRONMENT.json`; it is not a portable dependency lockfile. No robot, ROS or remote-control actions were performed while preparing this package. Verification uses the standard library only:

```
python3 verify_package.py
```

`RELEASE_MANIFEST.json` hashes every packaged file except itself. `content_manifest.json` is maintained outside this payload for the parent upload workflow. No upload is performed by the builder or verifier. Excluded data include PROJECT_PRIORITIES, credentials, raw sensor frames/video, native mesh checkpoints, reference/source assets and their closest-point arrays, caches and manuscript compilation logs.
