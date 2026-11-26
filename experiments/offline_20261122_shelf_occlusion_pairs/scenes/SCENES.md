# Paired shelf scenes

Each parent has a single ordered pair of placements. The `open` placement has a clear front view. The `screen` placement adds an opaque, unlabeled overhead panel in front of the same shelf. The panel is separated from the robot’s swept collision band by 0.06 m. Its height allows it to hide upper portions of the shelf at the original camera height while leaving the planar scan plane clear. Room geometry, shelf geometry, start pose, native calibration, action budget and simulator seed are shared within a pair. The camera and mapper still acquire the panel through simulation in the screened condition.

The seven rooms use a new arrangement of partitions and assets in the existing synthetic four-room simulator. The system uses the native RGB-D renderer, scan, owner IDs, motion and mapper. The shelf inserts are procedural measured geometries that are visible in the renderer. Each has distinct shelf deck heights and interior divider locations. The six test parents use three shelf assets twice each. Shelf asset use across parents is recorded explicitly in `test_geometry.json`. The distinct development parent is used for pipeline testing before the test pipeline is frozen.

| Parent | Role | Shelf asset | Paired seed |
|---|---|---|---:|
| `shelf_dev` | Development | `dev_split` | 202611228801 |
| `shelf_01` | Held out | `A_divided` | 202611228821 |
| `shelf_02` | Held out | `B_tall_bay` | 202611228827 |
| `shelf_03` | Held out | `C_staggered` | 202611228829 |
| `shelf_04` | Held out | `B_tall_bay` | 202611228847 |
| `shelf_05` | Held out | `A_divided` | 202611228857 |
| `shelf_06` | Held out | `C_staggered` | 202611228869 |

Independent source meshes and reference coordinates are stored under `evaluator_references/`. Shelf evaluation uses the same point array and area weight for both conditions. The reference includes area samples visible in each condition from the fixed shared reachable camera lattice, the start and the independently nominated side pose. The full-scene F1 evaluator uses each condition’s actual source scene, including the panel where present. Front-view fractions use each parent’s shared eligible shelf reference as denominator. The visible side subset counts shared shelf points seen from the open front pose, hidden from the screened front pose and visible from the screened side pose. All of this preflight uses source geometry with zero policy selections.

The pair audit verifies equal non-panel meshes and transforms, equal robot footprint grids and directed sweep edges, equal planar scans, and a reachable side pose that restores first-hit visibility to screened points. Actual policy feasibility and new area are assessed from the subsequent run. The geometric side pose remains in the evaluator ledger.

| Test parent | Front visible, open / screened | Front-hidden area seen from side (m²) | Side round trip (actions) | Minimum swept centreline clearance (m) |
|---|---:|---:|---:|---:|
| `shelf_01` | 57.01% / 34.54% | 0.942 | 36 | 0.265 |
| `shelf_02` | 54.84% / 34.89% | 1.311 | 36 | 0.330 |
| `shelf_03` | 56.07% / 36.36% | 0.378 | 42 | 0.205 |
| `shelf_04` | 68.91% / 39.98% | 1.485 | 32 | 0.340 |
| `shelf_05` | 57.86% / 35.08% | 0.719 | 48 | 0.330 |
| `shelf_06` | 56.88% / 35.47% | 0.978 | 40 | 0.270 |

Round trips count every turn and end at the original heading. The swept radius is 0.20 m. The centreline clearance is the smallest distance to an obstacle or floor boundary along the independently certified side witness.

## Reproduce the scene inputs

Use the previous study’s directory structure and the same scientific Python environment. Keep existing frozen scene files and copy them into a new version before regenerating. Formal scene inputs are sealed before running any formal candidate probes. Exports place independent evaluator truth and simulator-only collision caches in ledgers separate from actor inputs.

```sh
PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" prepare.py shelf_dev
"$PY" seal.py development
"$PY" seal.py development --verify

for parent in shelf_01 shelf_02 shelf_03 shelf_04 shelf_05 shelf_06
do
  "$PY" prepare.py "$parent"
done
"$PY" seal.py formal
"$PY" seal.py formal --verify
```

The per-parent records are in `pair_audits/`. Frozen `DEVELOPMENT_SCENE_FREEZE.json` and `FORMAL_SCENE_FREEZE.json` list ordered placement names and SHA-256 for cold inputs, simulator internals and evaluator data.
