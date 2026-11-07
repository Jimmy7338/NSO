# Evidence and section map for the focused submission candidate

This is an index to retained evidence, not a claim that all underlying raw recordings are publicly distributed. The full review manuscript is `../submission_20261104_two_room/main.tex` and `main.pdf`; the original is unchanged. The focused paper preserves all four primary paired outcomes, eight changed-background outcomes, the 128-attempt extension including the 32 low-budget failures and the budget-54 reversal, and the ordered physical C/S outcomes and failed S return. These observations are not independent scene replications.

| Full manuscript | Focused paper | Additional retained detail |
|---|---|---|
| Abstract, §§1–4, Fig. 1 and Table 1 | Abstract, §§1–4, Fig. 1 and Table 1 | Public two-template and supplied-graph assumptions remain explicit. |
| §5.1–5.3, Table 2 | §5.1–5.3 and Table 2 | Full Fig. 2–6 paths, stage meshes and temporal curves remain in the full review manuscript and `../submission_20261104_two_room/ESM_1.pdf`–`ESM_3.pdf`. Checkpoints reuse trajectories. |
| §5.4, Table 3 | Baseline scope summarized in related work and discussion | Full common-CPU adapted mechanism comparison in full manuscript §5.4; adapted methods are not full original systems. |
| §5.5–5.6, Tables 4–5 | Sensitivity and background tables retained as Tables 3–4 | Full Fig. 7–8 and all failed/tied/negative conditions remain in full manuscript. |
| §5.7, Table 6 | Attribution limits in discussion | Full component evidence and bundled correction/diagnosis limits in full manuscript §5.7. |
| §6.1–6.2 and Tables 7, physical pair | Physical deployment case and fresh-map C/S table | Full robot history, including unsuccessful earlier sessions, dependent 4 November continuation, 7 November S40 and C/S pilot, remains in full manuscript and `integrations/joint_coverage_pair_20261107/` in the project repository. |
| §6.3–6.5, Tables 8–9 | Not used as independent replication or primary evidence | Q0-derived 24 trajectories, frontier adaptation, two-room diagnostics and geometric-weight control remain indexed in the full manuscript; shared arrangements and post-hoc analyses do not add independent scenes. |
| §7, multi-instance paragraph | Condensed limits in discussion and conclusion | Neutral/adverse cross-instance findings, nine ties, 35/36 original pipeline completions and the separately corrected evaluation remain in the full manuscript and `../submission_20261104_two_room/ESM_4.pdf`–`ESM_5.pdf`. |
| §8–9 and declarations | Condensed conclusion, reproducibility and draft declarations | Existing code/evidence archive and the exact source records remain authoritative. |

For the C/S pilot, `integrations/joint_coverage_pair_20261107/comparison/comparison.json` holds complete outcomes and initial-condition differences. `integrations/joint_coverage_pair_20261107/footprint_failure/mapper_source_result.json` and `integrations/joint_coverage_pair_20261107/footprint_failure/mapper_portable.py` retain the S stopping diagnosis: a recorded 0.504 m laser hit was inserted into the fused belief, while the input navigation grid alone passed the circle test. The object's identity and whether the return was noise remain unverified. `integrations/joint_coverage_pair_20261107/C/` and `integrations/joint_coverage_pair_20261107/S/` contain actual derived canonical frames, semantic receipts, recorded decisions and cached-label replay results. These replays evaluate the downstream policy and do not independently rerun detection. The original full-resolution packets are retained locally and are not part of that public compact package.

For the completed 7 November S40, `integrations/exploration_resume_20261107/` contains the 41-frame canonical packet manifest, decisions, replay and semantic audit. Its initial ROS map retained prior observations. G/S equality in its recorded states is a same-state score comparison, not a separately executed physical G trajectory. The physical pilot uses the separately declared 3 m measured-extent adaptation and fixed SE2 alignment; it does not establish equivalence to the controlled supplied-graph experiment.

## Post-hoc common-voxel diagnostic

The 145 shared 15 cm bootstrap voxel keys were selected from the two actual arms. This protocol was written before calculating this metric but after the original outcomes were known; it is not a preregistered comparison. Same-key points differ in position by a median 0.079986 m and maximum 0.198045 m, so the keys do not establish exact physical surface correspondence.

| Arm / action | Common-key proxy, bootstrap → action 32 | Information contribution | Residual contribution | Additional terminal value |
|---|---:|---:|---:|---:|
| C | 0.279212 → 0.253034 | +0.010359 | −0.036538 | 0.215996 at action 40 |
| S | 0.265727 → 0.335260 | +0.165932 | −0.096400 | stopped at action 32 |

These components explain the recorded original metric, not added algorithm modules. The S trajectory acquired stronger information on these coarse keys while also incurring a residual penalty. One ordered pair, unequal starts and the failed S return do not support attribution to semantics or policy, nor do these proxies measure reconstruction accuracy. `integrations/joint_coverage_pair_20261107/common_support/result/{protocol.json,result.json,curves.csv,common_keys.json}` and `integrations/joint_coverage_pair_20261107/common_support/analyze.py` retain the analysis. Mapper-only reproduction matched each arm's own-support logged quality to 1e-12.
