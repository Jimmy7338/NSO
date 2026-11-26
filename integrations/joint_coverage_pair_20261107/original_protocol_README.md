# One ordered physical C/S pair — prepared, not executed

Predeclared order **C then S**, each40 paid actions and600s. Use `exploration_joint_v1`/`v10_3_1` and the existing score modes. C is same-candidate coverage scoring, not native frontier: it removes quality and all object-prior terms but keeps the original candidate generator, circle, posterior coverage weights and complete-heading return. Keep the same explicit automatic semantic frontend, sensor/adapter/controller configuration in both arms.

Root supplies a **new independent gmapping map and actual fresh initialization for each arm**. CSM and the retained display TSDF are not reset by these scripts. The final map-source topic/interface is deliberately not hard-coded: each fresh capture's actual settings and optional source receipt are retained. The script does not capture, replay, launch ROS or claim an epoch was verified.

Prepare one arm only after its actual capture, then repeat with S's new capture:

```bash
python prepare_arm.py --pair-id PAIR_ID --arm C --fresh-settings /absolute/fresh_C.json --map-source-receipt /absolute/fresh_C_map_receipt.json
python prepare_arm.py --pair-id PAIR_ID --arm S --fresh-settings /absolute/fresh_S.json --map-source-receipt /absolute/fresh_S_map_receipt.json
```

Each invocation creates a unique configuration/receipt and reserved unique report path. It copies settings, changing only `configuration.score_mode`, fixed `total_budget=40`, identity `episode_id`, plus calibration-only comparison provenance. It preserves the actual map/TF source fields, scene ID, original circle/anchor/candidates, all semantic settings and execution permission. No motion script or RUN_NEXT is generated. Use root's existing execution command with the returned `--joint-configuration`, `--duration 600` and `--report` arguments. Preparation itself is not evidence of an executed episode.

After both actual attempts (including failures):

```bash
python compare_pair.py --c-report /absolute/C_report --s-report /absolute/S_report --output /absolute/comparison
```

`comparison.json` and `.csv` retain each actual action/observation count, elapsed/CAN cost, native coverage-proxy gain, old-quality-proxy delta, grid return, failure and shutdown information. Initial packets, actual poses, transforms, map hashes and configuration differences are exposed. Known-cell masks are projected via each `planning_from_odom` into C's0.2m grid; domain losses are reported. This cell-center rasterization is lossy when transforms differ and assumes the retained odom epoch is the same, which actual source receipts must establish. It is neither ground-truth coverage nor mapping accuracy. Zero-observation failures are recorded with unavailable metrics, never zero-filled or dropped.

The two fresh captures/maps are **not claimed byte-identical**. Different initial knowledge and pose remain confounds. Old S40 is not a paired arm. One scene, one ordered pair and two episodes do not supply independent environmental replication; frames/actions are repeated measurements. Per-arm fixed old-surface support may differ, so quality-proxy deltas are not an accuracy ranking. No budget/weight selection after outcomes.
