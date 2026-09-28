# Article comparison snapshot

Results are separated by phase; all supplied slots remain listed. No minimum-positive-gain selection is applied.

## development

{"reviewed_qualified": 5, "running_reserved": 2, "unstarted": 5}

Qualified completed pairs: 6.

## main

{}

Qualified completed pairs: 0.

## Reviewed outcome rows

| Run | Qualified | C_nav | P | R | F1 | J_nav |
| --- | --- | --- | --- | --- | --- | --- |
| dev_AISLE_B_b160_n92801 | True | 0.999289 | 0.921930 | 0.819272 | 0.857363 | 0.856753 |
| dev_AISLE_G_b160_n92801 | True | 0.999289 | 0.921930 | 0.819272 | 0.857363 | 0.856753 |
| dev_AISLE_NBV_b160_n92801 | True | 0.999289 | 0.921930 | 0.819272 | 0.857363 | 0.856753 |
| dev_AISLE_S_b160_n92801 | True | 0.999289 | 0.921930 | 0.819272 | 0.857363 | 0.856753 |
| dev_CELL_B_b160_n92801 | True | 0.936450 | 0.962245 | 0.756044 | 0.818539 | 0.766521 |

## Interpretation boundaries

- Development and main are separate; development results are engineering observations, not independent confirmation.
- Only reviewed, SHA-bound sealed records provide metrics. Missing/unstarted/running slots have blank metrics, never zero scores.
- Unqualified reviewed episodes remain visible but are excluded from qualified performance deltas.
- Mechanism comparisons use a common actual observation/action prefix; score differences alone do not establish improved reconstruction.
- Zero forecast diagnoses concern stored supplied candidates, not all possible views.
- Cross-run observation equality uses all saved sensor arrays except run-namespaced frame_id; original receipt hashes remain recorded.
- Elapsed/planning times are recorded wall times and can be affected by simultaneous jobs; no isolated runtime benchmark is claimed.
- All metric values are copied from completed independent reviews; no new mesh evaluation occurs.
- Positive gains are not a reporting or qualification filter. Every supplied frozen slot and failure is retained.
- Ledgers are captured once; later completions belong in a new snapshot.
