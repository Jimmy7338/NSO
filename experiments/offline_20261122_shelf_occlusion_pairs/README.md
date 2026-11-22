# Frontal occlusion and structure-guided shelf inspection

This simulated experiment isolates frontal occlusion within six room–shelf units. The two conditions of each unit share the shelf, reference samples, start, seed and swept-footprint graph. Four rules nominate a route once at the same observed, eligible native goal. The common neutral planner and measured RGB-D fusion then run to the actual terminal state.

- [Experiment protocol](PROTOCOL.json) fixes the outcomes, weights, first eligible trigger, stage order and representative-case rule.
- [Paired scene construction](scenes/SCENES.md) gives the asset reuse, reference surface sampling, exact collision checks and cold-start scene preparation.
- [Source and prior evidence binding](source_receipt.json) links both earlier experiments to the pinned NSO commit. [Historical area receipt](measurement/historical_receipt.json) is the independent remeasurement of the two previous comparisons.
- [Formal outcomes](results/formal/) contain the 48 scheduled assignments, parent-level contrasts, all failure rows and raw mask indexes. [Paper figures and English result text](paper/) use those results.
- [Separate diagnostic work](followup/) checks the earlier full-floor and local route studies using their own frozen inputs.

## Reproduction on the original NSO workspace

The frozen JSON paths assume the existing NSO analyses tree on the Jetson and its Python 3.12 science environment. Follow the scene instructions to generate the cold observations and independently sampled references. Use a fresh study directory or preserve the original execution bundle before re-running. Published compact bundles contain the source bindings, measured ledger and required hashes. The original evaluation PLYs remain in the indexed native run directories.

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$NSO_PY" measurement/historical.py
"$NSO_PY" study.py seal development
"$NSO_PY" study.py run development --workers 4
"$NSO_PY" measurement/evaluate.py --phase development --repeatability --out results/development
"$NSO_PY" study.py seal formal
"$NSO_PY" study.py run formal_first --workers 6
"$NSO_PY" measurement/evaluate.py --phase formal --first 2 --out results/formal_first
"$NSO_PY" release.py
"$NSO_PY" study.py run formal_rest --workers 6
"$NSO_PY" measurement/evaluate.py --phase formal --out results/formal
"$NSO_PY" figures.py
"$NSO_PY" write_paper.py
"$NSO_PY" audit.py
```

Every placement begins with an actual G execution. Within the same observed fork and continuation, the same outbound and return identity allows a policy assignment to reuse that measured trajectory. A distinct selected route replays the common prefix and repeats the complete decision internally. Guard or early terminal records remain in the matrix. The action-96 meter uses actual paid actions. Reported return retention requires both policies to have returned.
