# Independent floor measurement

Run from the study root after all scheduled native processes have an exit record.

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
"$NSO_PY" evaluation/evaluate_recovery.py
```

The default output is `results/formal`. Use `--out` with a fresh directory to verify again. The run freeze identifies the original code, actor inputs, independent references and complete four-floor, three-arm schedule. The evaluator checks the sealed launch, exit, native file hashes, exact starting packet and fusion state across the three arms. It validates the original motion and return reservation and independently translates every charged motion to a pose update. The two regional arms use the same fixed local S scoring. The repaired global scoring sidecar is checked only where a region ranking was available.

The first evaluation stopped because the native regional adapter emits guard rejections with `paid`, `option_id`, `selection_call_id`, `reason` and `motion`, without a separate event name. The recovery runner reuses the frozen evaluator and supplies the derived table label `regional_guard_rejection` only for that exact schema in `regional_guards.json`. The raw event dictionary is stored verbatim as the parsed `event_data`. Other untagged event shapes remain errors. Already tagged audit and frontier records retain the original parsing and selection. The original freeze, evaluator, actor outputs, native navigation fallback and stops remain intact. See `results/evaluation_attempt_001/supervisor.log` and `results/formal/evaluation_recovery.json` for the two measurement attempts.

| File | Contents |
| --- | --- |
| `episodes.csv` | Actual paid forward and turn counts, native terminal and return, pose, path length, collision and guard counts, first A95 and actual wall timings |
| `checkpoints.csv` | Start, steps 48/96/144/192 and actual terminal. Native mesh, belief and reference paths, exact observation receipts, coverage, precision, recall, F1 and J |
| `coverage_curve.csv` | Correct reachable free cells and their denominator at every paid action, including action zero |
| `trajectory.csv` | Every charged action and observed pose, with packet hashes and guard results from the native trace |
| `guard_events.csv` | Native decision and rejection events, with the terminal record and source row index |
| `decisions.csv`, `room_crossings.csv`, `first_divergence.csv` | Existing regional or frontier evidence bound to actually executed motion |
| `pairs.csv`, `paired_checkpoints.csv` | Revised minus strict and revised minus original region. Signed percentage-point differences are left minus right. Positive saved actions mean revised first attained the threshold sooner |
| `summary.json`, `measurement_audit.json` | Floor-level denominators and attributes, separate returns and failures, result hashes and raw evidence bindings |

A95 uses the first exact charged clock satisfying the integer equivalent of 95% coverage. The evaluator independently indexes the true reachable cells in every stored observation. A native run that never reached the threshold retains its measured terminal coverage. An unmatched A95 pair keeps the two explicit statuses with an empty action difference. Surface metrics use the published distance tolerance of 5 cm and independently checks the precision–recall harmonic mean and the coverage–F1 product.

Each native terminal has its actual charged clock and original pose. Return-to-return differences require both policies to arrive at their anchor. Unequal actual endpoint clocks are marked in the paired table. Checkpoints unavailable after an early stop and partial captures from a failed process are explicit.

The read-only validation mode checks the wrapper against the two existing published floors and can write to a new temporary directory.

```sh
NSO_EVAL_SMOKE=$(mktemp -d /tmp/nso-floor-smoke.XXXXXX)
"$NSO_PY" evaluation/evaluate.py --known-smoke --out "$NSO_EVAL_SMOKE/results"
```
