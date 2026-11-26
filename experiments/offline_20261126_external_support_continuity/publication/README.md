# External-scan support continuity study

Run the local staging command after the sealed episodes, evaluator, and collision audit finish.

```sh
python3 publication/stage_publication.py
```

The command writes `publication/payload/` and `publication/upload.json`. Each manifest entry records the intended GitHub path, Git blob ID, SHA-256, and size. The destination is `experiments/offline_20261126_external_support_continuity/`.

The package contains the frozen four-start, three-policy, 192-action protocol, validation and run receipts, the two sets of 21 geometry controls, fixed-clock and terminal metrics, original-mask sensitivity, and the scripts and JSON for the HM3D plateau and topology analyses. The native evidence ZIP contains the action, decision, checkpoint, and worker records for all retained attempts. The earlier support-repair study has its own destination and record.

Licensed scene assets, downloaded navigation meshes, sensor frames, occupancy truth, reference or reconstructed meshes, and rendered topology plots remain local. Obtain the original scene data from its source to reproduce the geometry diagnostics.
