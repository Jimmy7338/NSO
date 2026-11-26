# Sensor-only swept-footprint experiment

Run the local staging command after the sealed episodes and evaluator finish.

```sh
python3 publication/stage_publication.py
```

The command writes `publication/payload/` and `publication/upload.json`. Each manifest entry records the intended GitHub path, Git blob ID, SHA-256, and size. The destination is `experiments/offline_20261126_external_online_sweep/`.

The package contains observation-side motion admission code, blinded control results, fixed replay of archived sensor packets, four paired starts with three strategies and a 192-action budget, and final outcomes on the unchanged V2 evaluation reference. The V1-mask comparison records the earlier HM3D plateau of 510 observed free cells out of 1,640. The native evidence ZIP holds decision, action, checkpoint, and worker records for the retained attempts.

Licensed scene assets, sensor frames, occupancy truth, downloaded navigation meshes, and reference or reconstructed meshes remain local. Obtain the original data from its respective source for full reproduction. The V1 and V2 studies have separate publication directories.
