# NSO regional scoring factorial study, Phase A

Four simulated floors contribute independent paired comparisons across four scoring combinations.
The original regional planner D0G0 and full regional planner D1G1 are archive reuses.
The two single-change arms D1G0 and D0G1 are registered as new closed-loop runs.
The study records 8 new and 8 reused complete runs,
with 0 failed terminal dispositions. All arms use the same local S policy,
initial observations, sensors, TSDF and 192-action budget including the return.

- [Protocol](factor/PROTOCOL.json) · [Actor freeze](factor/FORMAL_FREEZE.json) · [Reuse gate](factor/REUSE_GATE.json) · [Source audit](factor/SOURCE_AUDIT.json)
- [Archived score and target replay](factor_audit/reuse_receipt.json)
- [A_FACTOR_RESULTS](factor_audit/A_FACTOR_RESULTS.md)
- [Episodes](factor/results/formal/episodes.csv) · [Checkpoints](factor/results/formal/checkpoints.csv) · [Factor effects](factor/results/formal/factor_effects.csv)
- [A96 figure PDF](paper/checkpoints_a96.pdf) · [PNG](paper/checkpoints_a96.png) · [Editable SVG](paper/checkpoints_a96.svg) · [Caption and source](paper/CAPTION.md)
- [Factor interactions PDF](factor_audit/factor_interactions.pdf) · [PNG](factor_audit/factor_interactions.png) · [Editable SVG](factor_audit/factor_interactions.svg)
- [Native trace and figure evidence index](EVIDENCE_INDEX.json) · [Reused scene identity](SCENE_REUSE.json)

The archive reference commit is `74ecf883503d5a117d1e566417aff0213f76472a`.
The published actor source baseline is `f482722f6cbfc3135358080b08f27274f54cb535`.
The [original four-floor archive](../offline_20261123_region_independent_floors/README.md) contains the scene material
and the eight reused trajectories. This package includes the raw outcomes and logs from
all new attempts, with PLY snapshots identified by their authoritative local paths and SHA-256.
Large derived trace tables are stored in the evidence bundle when indicated by the index.

## Reproduction

Use the original Jetson workspace with the frozen four-floor and inherited
source directories available as siblings of this study. Preserve existing
outputs before running into a fresh study directory.

```sh
nso_python=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$nso_python" factor_audit/replay_rankings.py | cmp - factor_audit/reuse_receipt.json
"$nso_python" factor/factor_study.py seal
"$nso_python" factor/factor_study.py run --workers 6
"$nso_python" factor/evaluate_factor.py
"$nso_python" factor_audit/summarize_factor.py
"$nso_python" paper/render_checkpoints.py
```
