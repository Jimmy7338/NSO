# Independent region-planning floor comparison

Four prospectively registered simulated floors compare the published revised regional scorer, the original regional scorer, and strict nearest-reachable frontier. Both regional arms use the same local S policy. The measured outcome is the first actual paid action with 95% correct reachable free-space coverage. Whole-scene quality at action 96 and at actual return tests the corresponding reconstruction trade-off.

- [Formal protocol](PROTOCOL.json) · [Pinned published source](SOURCE_RECEIPT.json)
- [Geometry, admissions, excluded anchor and scene reproduction](scenes/SCENES.md)
- [Previously known two floors](known/README.md)
- [New four-floor briefing](paper/BRIEF.md) · [All checkpoint metrics](paper/TABLES.md) · [Paper figure](paper/coverage_joint_quality.pdf)
- [Paired metrics](results/formal/pairs.csv) · [Actual trajectory](results/formal/trajectory.csv) · [Measured checkpoints](results/formal/checkpoints.csv) · [Every executed coverage clock](results/formal/coverage_curve.csv)

## Reproduction

Use the pinned NSO Jetson workspace and its Python 3.12 scientific environment. The new directory is separate from all prior studies. The attached scene ledgers distinguish acquired actor input, simulator-internal geometry and independent offline references. Arms share the initial sensor packet and use the same exact-pose simulation interface as their trajectories advance. The budget includes turns and return. A rejected or blocked navigation action follows the existing replan, stop and return rules in the retained controller. The frontier arm uses its original nearest-reachable choice without an NSO regional invocation.

### Original execution rules

The regional planner checks the current and anchor footprints in the observed map and admits an oriented outbound option only when its native return fits the remaining budget. The local controller checks the next action and can try a fresh target once after rejection. If the second attempt is unavailable or rejected, the controller tries an observed-map return. A missing or unsafe return, an exhausted budget, or a native episode terminal ends the run. At the anchor, a run can end when no positive outbound option remains.

Strict frontier selects actual measured-grid frontier cells reachable on the observed safe grid. It retains the published representative cap and shortest-distance ordering, with gain and cell ordering for ties. A rejected outbound action can trigger one fresh frontier choice and then an observed-map return. A frontier stop requests that same observed return. All three arms preserve the native guards and recorded stops.

After preparing and freezing the four scene inputs as described in the scene documentation, run the following commands from the study directory in a clean reproduction or preserve the original outputs first.

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$NSO_PY" known/recompute.py --verify
"$NSO_PY" scenes/seal_scenes.py --verify
"$NSO_PY" study.py seal
"$NSO_PY" study.py run --workers 6
"$NSO_PY" evaluation/evaluate_recovery.py
"$NSO_PY" evaluation/raw_qa.py . results/formal | cmp - evaluation/raw_qa_receipt.json
"$NSO_PY" paper.py
"$NSO_PY" audit.py
```

Each episode records actual process time and decision durations. Reported wall times describe this concurrent simulation environment. The execution ledger retains technical attempts, native output and file hashes. Compact bundles carry raw trajectories, grid histories and checkpoint receipts. Original fused PLY snapshots remain at the indexed native measurement paths on the Jetson.

All twelve actor runs completed on their first attempts. The first evaluation stopped on a format difference in five regional guard records. The evaluation recovery adds a derived label to those exported records and preserves every original field, actor run, frozen evaluator and map measurement rule. The first traceback and separate recovery hashes are part of the evidence.
