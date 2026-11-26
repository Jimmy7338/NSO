# Prior shelf evidence audit

Run `~/nso_robot/envs/science312/bin/python -B recompute.py --out /path/to/audit` from this directory in the archived NSO analyses tree. The program reads the pinned `3a0f74ff` study and its native output snapshots. It reruns the frozen 5 cm mesh evaluator and checks each recovered shelf mask against the fixed reference points, archived trajectory, and released table.

| File | Contents |
| --- | --- |
| [formal_screened_pairs.csv](formal_screened_pairs.csv) | Six screened S/D pairs, actual fork and route end clocks, native terminal outcomes, signed S−D at fixed clocks and return, and the gain from retrospectively selecting the better of the two runs at action 96. |
| [short_witness_routes.csv](short_witness_routes.csv) | Thirteen earlier outbound branches, expanded by checkpoint. Each row keeps the absolute paid clock, time since fork, and exact measurement or guard status. |
| [short_witness_clocks.csv](short_witness_clocks.csv) | Candidate counts, registered denominators, evaluated equal-clock maxima and availability. The outbound event records each route's realized clock. |
| [historical_receipt.json](historical_receipt.json) | Source, manifest, trace, sample archive, captured mesh, and output hashes. |

The two-record hindsight gain describes the archived S and D executions only. Each short witness ends at outbound arrival or guard cancellation. Its archived return reserve is a certificate, so every witness has a missing return measurement and a missing task-action-96 measurement. The new continuation experiment is needed to measure those endpoints.
