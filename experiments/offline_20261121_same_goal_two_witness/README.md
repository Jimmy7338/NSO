# Actual continuation of two same-goal choices

The two development witnesses are the gallery-left state after 56 paid actions and the warehouse-right state after 142. Each pair starts from the exact archived action and observation prefix. One native target's local choice uses the full frozen candidate pool and either neutral G or category S scoring. Subsequent decisions use the source neutral global G with the original four-alternate local G. Existing guard checks, new sensor frames and mapping, route cancellation, replanning, return and terminal accounting remain active through the actual stop.

- [Frozen protocol](PROTOCOL.json), [source inputs](SOURCE_FREEZE.json), [execution freeze](runner/freeze.json), and [actor receipt](runner/supervisor_receipt.json).
- [Paired metrics at equal-budget and terminal clocks](results/paired_results.csv), [actual command and observation comparison](measurement/results/pair_causal_audit.json), and [terminal engineering gate](results/decision.json).
- [Measured joint quality versus paid actions (PDF)](results/J_by_paid_action.pdf), [editable SVG](results/J_by_paid_action.svg), and [Chinese paper paragraph](RESULTS_ZH.md).
- [Completion audit](completion_audit.json) and [raw evidence index](EVIDENCE_INDEX.json). The four full step logs include original archived controls and actual new commands, guard events, poses, sensor hashes, and native return records. Native mesh and belief snapshots remain indexed on the experimental host. GitHub preserves each arm's compact raw telemetry under `evidence/`.

## Recompute on the original host

Use Python 3.12 with the archived NSO environment, its deterministic simulator and independent surface evaluator. Read-only metric and report reproduction can run into fresh output directories.

```sh
python measurement/evaluate.py --out /tmp/nso-two-witness-measurements
python report.py --out /tmp/nso-two-witness-report
```

`runner/supervise.py` is the four-launch entry point and checks the predeclared schedule and input freeze. Only rerun it in a fresh copy of the study if authorized to reproduce actual continuations. A fresh plot can use the published measurement JSON tables without the simulator. The four independent holdout buildings remain outside this experiment.
