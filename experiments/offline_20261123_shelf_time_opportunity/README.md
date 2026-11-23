# Timed shelf observation and route opportunity

The study asks whether a legal same-goal outbound choice can deliver more measured shelf surface than direct inspection D by paid task action 96. The first stage measures the opportunity in three already seen placements. Each has the archived choices of D and S and two distinct geometry-only alternatives at the exact same outbound costs. Every route shares the archived initial observation, fork, oriented target, return reservation and subsequent neutral G/G_perf feedback.

- [Protocol and stopping criteria](PROTOCOL.json) specify which measured results justify developing a causal-time score or investing in six new pairs.
- [Published source binding](SOURCE_RECEIPT.json) identifies the exact shelf-study Git blobs from the pinned NSO base.
- [Historical independent remeasurement](history/README.md) audits the six original screened comparisons and the 13 prior outbound-only runs.
- [Candidate bank and sampling rules](bank/README.md) give all legal candidate counts, fixed selection and archived reuse.
- [Development route opportunities](paper/OPPORTUNITY_TABLE.md) and [submission briefing](paper/BRIEF.md) report results with the actual clock and their applicability.

## Reproduce the development study

Use the indexed NSO Jetson source directories and the same Python 3.12 scientific runtime. Original inputs and outputs carry their native SHA-256. Run in a fresh copy of the study directory or preserve the existing output paths.

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$NSO_PY" history/recompute.py
"$NSO_PY" bank/seal.py --check
"$NSO_PY" study.py seal
"$NSO_PY" study.py run --workers 6
"$NSO_PY" recovery.py seal
"$NSO_PY" recovery.py run --workers 6
"$NSO_PY" recovery.py measure
"$NSO_PY" report.py
```

The first execution version remains for provenance. It stopped before any charged action when archived direct-observation category keys were compared to their in-memory integer-key representation. Its meter also encountered a scalar-count interface when given an area-weight vector. Recovery keeps the bank, native motion, prediction, recorder and stops bound while comparing canonical JSON bytes and summing the actual reference-area weights by mask. Two no-motion probes independently matched the encoded category content and all other fork descriptors. The archived full D and S runs in the screened confirmation placement have no captured action-16 or action-24 mesh. They remain unavailable at those clocks.

The compact evidence bundles include native actions, branch receipts, logs and checkpoint hashes. The full raw RGB-D fusion PLYs remain in their original indexed execution folders on the Jetson. An archived full trajectory is reused only for the identical observed fork and route. An outbound-only archive serves as a prefix check for a newly executed closed loop.
