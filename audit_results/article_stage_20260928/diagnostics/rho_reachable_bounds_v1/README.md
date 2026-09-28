# One-peer rho interval bounds on saved direct candidates

Fixed own evidence, recorded costs and area vectors. Endpoint sums are conservative envelopes; not-excluded is not an observed or executable benefit.

| Run | Direct pools | Variable current-peer pools | Current peer: changes not excluded | Hypothetical peer: changes not excluded |
|---|---:|---:|---:|---:|
| dev_AISLE_B_b160_n92801 | 10 | 0 | 0 | 0 |
| dev_AISLE_G_b160_n92801 | 10 | 0 | 0 | 0 |
| dev_AISLE_NBV_b160_n92801 | 10 | 0 | 0 | 0 |
| dev_AISLE_S_b160_n92801 | 10 | 0 | 0 | 0 |
| dev_CELL_B_b160_n92801 | 34 | 24 | 0 | 0 |
| dev_CELL_NBV_b160_n92801 | 7 | 0 | 0 | 0 |
| dev_CELL_S_b160_n92801 | 34 | 24 | 0 | 0 |
| dev_LOOP_B_b160_n92801 | 18 | 0 | 0 | 0 |
| dev_LOOP_G_b160_n92801 | 18 | 0 | 0 | 0 |
| dev_LOOP_NBV_b160_n92801 | 18 | 0 | 0 | 0 |
| dev_LOOP_S_b160_n92801 | 18 | 0 | 0 | 0 |
| ground_AISLE_B_b160_n92801 | 14 | 0 | 0 | 0 |
| ground_AISLE_G_b160_n92801 | 14 | 0 | 0 | 0 |

G/NBV rows preserve their deliberately absent semantic class. No private class/GT information is inserted.
Initialization is separated. At diagnostic-selected steps, only the direct subproblem is bounded. Missing or inconsistent records are unavailable.
Single-instance score contrasts are linear fractional in rho with positive denominator; their derivative sign is constant, so extrema lie at endpoints. Summing extrema over instances is conservative and can combine incompatible endpoints.
Full per-candidate contrast bounds, endpoint posteriors and exact current selection kinds are retained in bounds.json. No frozen execution source is modified.
