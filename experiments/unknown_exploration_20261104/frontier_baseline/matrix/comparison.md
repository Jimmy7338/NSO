# Fixed Q0 frontier comparison

Q0 two related arrangements, one deterministic trajectory per condition; not independent scenes or physical evidence. Original G/S and joint G/S/C are previously run results, not rerun. C is same NSO candidate generator with coverage scoring, not frontier. Frontier uses the local observed FrontierPolicy plus common guard and permanent return on the first rejected action or policy stop; it is not ROS explore_lite and does not have the same policy as NSO. Surface and 5 cm F1 are simulator-reference evaluation only.

| Layout | Budget | Method | Coverage gain m² | Path m | New visible surface m² | F1 5 cm | Used / limit | Returned | Failed |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| shelf_west | 48 | frontier+guard | 15.00 | 8.4 | 84.12 | 0.8158 | 48/48 | True | False |
| shelf_west | 48 | original-G | 12.52 | 7.6 | 95.52 | 0.8857 | 48/48 | True | False |
| shelf_west | 48 | original-S | 12.52 | 7.6 | 95.52 | 0.8857 | 48/48 | True | False |
| shelf_west | 48 | joint-G | 14.16 | 8.4 | 96.58 | 0.8909 | 48/48 | True | False |
| shelf_west | 48 | joint-S | 14.16 | 8.4 | 96.58 | 0.8909 | 48/48 | True | False |
| shelf_west | 48 | joint-C | 14.16 | 8.4 | 96.58 | 0.8909 | 48/48 | True | False |
| shelf_west | 96 | frontier+guard | 16.88 | 17.6 | 107.23 | 0.9359 | 96/96 | True | False |
| shelf_west | 96 | original-G | 16.96 | 16.8 | 114.11 | 0.9689 | 96/96 | True | False |
| shelf_west | 96 | original-S | 16.92 | 16.0 | 111.44 | 0.9483 | 96/96 | True | False |
| shelf_west | 96 | joint-G | 16.92 | 17.2 | 109.51 | 0.9456 | 96/96 | True | False |
| shelf_west | 96 | joint-S | 16.96 | 16.4 | 118.52 | 0.9829 | 96/96 | True | False |
| shelf_west | 96 | joint-C | 16.92 | 17.2 | 108.56 | 0.9426 | 96/96 | True | False |
| shelf_east | 48 | frontier+guard | 15.00 | 8.4 | 84.02 | 0.8130 | 48/48 | True | False |
| shelf_east | 48 | original-G | 12.52 | 7.6 | 95.39 | 0.8830 | 48/48 | True | False |
| shelf_east | 48 | original-S | 12.52 | 7.6 | 95.39 | 0.8830 | 48/48 | True | False |
| shelf_east | 48 | joint-G | 14.16 | 8.4 | 96.46 | 0.8883 | 48/48 | True | False |
| shelf_east | 48 | joint-S | 14.16 | 8.4 | 96.46 | 0.8883 | 48/48 | True | False |
| shelf_east | 48 | joint-C | 14.16 | 8.4 | 96.46 | 0.8883 | 48/48 | True | False |
| shelf_east | 96 | frontier+guard | 16.76 | 16.8 | 96.56 | 0.8751 | 96/96 | True | False |
| shelf_east | 96 | original-G | 16.96 | 16.8 | 118.71 | 0.9857 | 96/96 | True | False |
| shelf_east | 96 | original-S | 16.96 | 16.8 | 118.71 | 0.9857 | 96/96 | True | False |
| shelf_east | 96 | joint-G | 16.92 | 16.8 | 114.00 | 0.9638 | 96/96 | True | False |
| shelf_east | 96 | joint-S | 16.92 | 16.8 | 114.00 | 0.9638 | 96/96 | True | False |
| shelf_east | 96 | joint-C | 16.92 | 17.2 | 112.11 | 0.9552 | 94/96 | True | False |

All frontier runs returned with no collisions and ended on budget exhaustion. Guard intervention and return trigger (step count denotes completed actions):

- shelf_west 48: {"action": "forward", "event": "action_rejected_return", "phase": "explore", "reason": "known_return_exceeds_remaining_budget", "step": 23}
- shelf_west 96: {"action": "forward", "event": "action_rejected_return", "phase": "explore", "reason": "known_return_exceeds_remaining_budget", "step": 76}
- shelf_east 48: {"action": "forward", "event": "action_rejected_return", "phase": "explore", "reason": "known_return_exceeds_remaining_budget", "step": 23}
- shelf_east 96: {"action": "forward", "event": "action_rejected_return", "phase": "explore", "reason": "known_return_exceeds_remaining_budget", "step": 53}

No policy-stop-to-return occurred in these four frontier runs. West/48 steps and before/after metrics exactly match the retained smoke. Initial packet, anchor and full before metrics were asserted against the matching original, joint and C results. Source and input result hashes were checked after all runs.
