# Region and structure in first-entry floor exploration

This independent study separates an observed regional global switch from a local category-conditioned structure switch. The strict nearest-reachable-frontier arm retains its own selected commands and uses the same renderer, observed guard, return anchor and measured map. The four held-out floors are fixed before their first strategy run. The known Cycle 18 floor and older results are development and regression evidence.

## Verified environment and source

- Jetson host `ouc-desktop`, AArch64, eight available CPU cores, 29 GiB RAM, Python 3.12.15 at `/home/ouc/nso_robot/envs/science312/bin/python`, and 264 GiB disk available at initial check. Native workers cap BLAS and OpenMP at one thread.
- `ExplorationJointCPUFourModules.select_target` is loaded from the `reproduction_20261104` source snapshot. Its SHA-256 and exact switch-off parity are in [the binding](regression/active_binding.json) and [the replay receipt](regression/verified.json). The latter used 252.3 s of task time and reproduced all 192 charged commands, 28 macro selections, occupancy, fused mesh and map metrics.
- [Native cold-start reference](https://github.com/Jimmy7338/NSO/tree/4c388a5c06d51e53251a9c901bbefd511d5b2a95/experiments/offline_20261121_same_goal_cold_holdout) and [geometry calibration](https://github.com/Jimmy7338/NSO/tree/8071a50fc7d04980d2eb390f633be7cae950706e/experiments/offline_20261122_weighted_geometric_control) were independently checked against repository blobs. The active simulator, trajectory ledger and estimator audit are described in the [methods](docs/method.md) and [evidence index](docs/evidence_index.md).

## Reproduce

Use the absolute hash-bound paths with their original files. The commands below generate a fresh evidence directory structure. Executing the same commands in a directory with an already sealed run will preserve its output and stop. Existing records can be remeasured with the evaluation command and its separate output option.

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$NSO_PY" regression.py run
"$NSO_PY" regression.py verify
"$NSO_PY" study.py seal development
"$NSO_PY" study.py run development --workers 1
"$NSO_PY" evaluate.py development
"$NSO_PY" formal_study.py seal
"$NSO_PY" formal_study.py run --workers 5
"$NSO_PY" evaluate.py formal
```

The [scene procedure](scenes/SCENES.md) constructs cold starts and keeps evaluator geometry separate. [Regional unit checks](policy/test_observed_regions.py) exercise observed portal, frontier qualification and traversal/blocked memory. `local/` contains the separate measured-coverage direct-inspection experiment, including the prospective placement seal and raw replay commands. The papers use separate output paths from [this evidence index](docs/evidence_index.md).
