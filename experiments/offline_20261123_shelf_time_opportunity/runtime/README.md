# Native closure of a registered outbound branch

Run from the study directory with a sealed candidate bank and an unused output
path. The original workspace contains the pinned sensor fixtures and native
runtime.

```bash
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
"$NSO_PY" runtime/run.py --bank bank/BANK_FREEZE.json \
  --placement increment_open --route via_25_13_3 \
  --out execution/increment_open/via_25_13_3/native
```

The runner rebuilds the published observed fork and verifies the source actor
inputs, target position and heading, candidate population, observed occupancy,
mesh digests, sensor packet and return certificate. The chosen outbound is
bound through the archived same-goal sidecar. The original driver applies its
per-action collision and return-budget guard, including replan at cancellation,
then uses the frozen neutral global G and weighted local G for further choices.
Every charged action contributes to the 192-action episode budget.

The run keeps each executed checkpoint at task actions 16, 24, 48, 96, 144 and
192, plus the untouched initial, fork and actual terminal states. The manifest
records guard cancellation separately from native episode termination. Earlier
stops mark unexecuted clocks as missing. A branch with a matching short archive
verifies its charged motion and observation digests through the archived
outbound event after the new native episode has finished. Prefix raw arrays,
per-frame digests, acquired occupancy and measured checkpoint meshes are
available to the independent evaluator. Evaluation reference files are outside
the runner's inputs.

These branches implement the stage-I opportunity test. The bank fixes the one
outbound intervention and source observations before any new episode is run.
