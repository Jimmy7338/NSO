# Frozen actual-observation budget comparison, 2026-11-09

Only this analysis directory was written. No production source, ROS, navigation, controller, robot motion, map input or semantic label was modified. Original circular radius 0.7212185868375829 m, rear offset >0.15 m, entry projection <=0.35 m, entry-before-deep, maximum 12 candidates, region filtering and all scoring weights remain unchanged.

`replay.py C` and `replay.py S` run in `/home/ouc/nso_robot/envs/science312/bin/python -B`. Historical canonical semantic caches are loaded with their original source, mask-file and canonical-label SHA256 checks. They replace inference only. The actual saved NPZ observations pass through the original JointSession mapper and runtime. All 21 historical next-action calls through paid20 match action and complete selection (candidate paths, candidate audit, all scores and score audit) at 1e-8 tolerance. Both histories are retained, not independent draws from the same frozen state. All other summary/receipt keys were not claimed to match.

At paid0 and paid20, the original backend select_target is called with ledger.total_budget set to 40, 80 or 96. Paid actions remain unchanged: paid20 has 20/60/76 remaining. Selection bookkeeping and calls are restored before continuing the baseline. A shared physical history is an offline counterfactual, not evidence that an 80/96 physical run would acquire the same observations or improve outcomes. `fresh.py` runs the newly recorded post-reboot initial packet with one natural frontend inference; its own canonical masks and receipts are retained under fresh/semantics. Fresh coordinates and source packet remain separate from historical C/S epochs.

| History / paid | Feasible states at total 40 / 80 / 96 | Candidates 40 / 80 / 96 | G versus S at total96 |
|---|---|---|---|
| C / 0 | 1207 / 1603 / 1603 | 5 / 5 / 5 | exact tie |
| C / 20 | 91 / 1589 / 1807 | 3 / 5 / 5 | scores differ; same top1 |
| S / 0 | 1207 / 1675 / 1675 | 5 / 5 / 5 | exact tie |
| S / 20 | 21 / 1695 / 2003 | 3 / 5 / 5 | exact tie |
| Fresh / 0 | 1239 / 1695 / 1711 | 5 / 5 / 5 | exact tie |

No entry or deep candidate is admitted in any of these 15 comparisons. Candidates retain four coverage roles plus old-surface rotation, except historical paid20 budget40 with two coverage roles plus rotation. Every G/S top1 is identical within the same frozen-state/budget comparison. More budget can change coverage top1 and predicted support, but does not establish a semantic action advantage here.

C paid20 total96 is the only compared state with unequal G/S score arrays. Its leading coverage candidate [217,227,0], complete planned cost76, acquires incremental aperture0.17895587958042738 for asset2. G prior1.5783573132029372 versus S0.7291087188330381 gives G score22.628771121290157 versus S21.77952252692026. Both still choose this same candidate. The entry pose is rejected with projection error1.9060889400679164 m >0.35 m, and deep requires an unavailable entry. Thus coverage-route aperture and explicit entry admission are distinct. S paid20 total96 also has nonzero aperture, but its G/S arrays remain equal; natural semantic availability/association matters separately from geometry.

Fresh initial total40 chooses [213,196,3], cost40, G=S22.790125. Total80 chooses [218,216,1], cost76, G=S54.52687588624854. Total96 chooses [220,219,1], cost92, G=S63.96687588624854. All fresh aperture increments remain zero. Its only two assets have no admitted entry: one has no unused feasible aperture pose, the other has nearest projection error2.151795659778532 m at total80/96, far above0.35 m. Larger budget alone does not open fresh initial semantic scoring.

A single separately frozen longer run can be justified to test coverage progression and whether new observations change this applicability barrier, while retaining all null outcomes. Total96 has more diagnostic range than80 in the old paid20 state and current fresh initial state. Neither budget is supported as a reliable fix for semantic benefit; no present comparison predicts a G/S action difference. Do not launch a repeated long-run matrix expecting semantic success. A long physical episode must preserve original policy and protections and report actual exploration, surface observations, semantic scoring, return, and failures separately.

Machine-readable results: C/result.json, S/result.json and fresh/result.json. Full original selections for every state/budget are saved next to them. The full selections are authoritative for aperture; summarize.py corrected initial convenience fields that used a nonexistent role/aperture key, without changing any selection or replay. Original executed-script hashes remain in historical result provenance, with the corrected rerunnable script hash recorded separately.
