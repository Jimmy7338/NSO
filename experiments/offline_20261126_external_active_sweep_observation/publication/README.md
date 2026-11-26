# Active observation experiment evidence

Use the gate bundle after the one from-zero HM3D diagnostic and its independent audit finish, even if the trial misses its prerequisite.

```sh
python3 publication/stage_publication.py --phase gate
```

Use the formal bundle only after the gate qualifies and all twelve fresh runs and their evaluations are complete.

```sh
python3 publication/stage_publication.py --phase formal
```

Each command writes a separate local payload and uploader manifest under `publication/gate/` or `publication/formal/`. The intended GitHub namespace is `experiments/offline_20261126_external_active_sweep_observation/`. Formal individual and paired metrics exclude the gate run. The native archive retains compact gate provenance in either bundle and separate formal action and decision receipts when applicable.

Scene assets, sensor frames, occupancy arrays, navigation meshes, and reconstructed geometry remain local.
