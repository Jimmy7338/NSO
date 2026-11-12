# Evidence index for the 8 November focused submission candidate

This is an index to retained evidence, not a claim that all underlying raw recordings are publicly distributed. The 7 November 16-page focused source and full evidence manuscript are retained unchanged. The new `main.pdf` is accompanied by `supplement.pdf`, which includes all 24 current-source joint-exploration matrix trajectories and the related attribution controls. The full review manuscript is `../submission_20261104_two_room/main.tex` and `main.pdf`; the original is unchanged. The focused paper preserves all four primary paired outcomes, eight changed-background outcomes, the 128-attempt extension including the 32 low-budget failures and the budget-54 reversal, and the ordered physical C/S outcomes and failed S return. These observations are not independent scene replications.

The correspondence table records the 7 November extraction. In the 8 November revision, the new joint diagnostic is Section 6, physical evidence is Section 7 and discussion follows in Section 8.

| Full manuscript | Historical focused extraction | Additional retained detail |
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

At the actual C action-21 state, an unchanged-method post-hoc resource counterfactual evaluated 57 remaining actions (78 total including 21 already paid); the recorded remaining-19 control reproduced the native candidates and scores. All five remaining-57 candidates had zero incremental aperture, with exactly equal G/S/X scores. G/S/X chose the same 19-outbound plus 38-return path; C chose a 35-plus-22 path under its different objective, which is not a semantic effect. The marked entry's nearest budget-eligible projection error was 1.437 m against the original 0.35 m limit; the earlier 0.889 m value used the entire connected graph without that budget restriction. This was not a physical S78 trial; native aperture is not an occlusion certificate. See `integrations/joint_coverage_pair_20261107/C_prefix21_original_remaining57/` for the original audit, results and verification.

The C21 original-method audit and the raw-scan synthetic snapshot analysis have portable offline command-line entry points and recorded relocated-path checks in their respective repository README files. The earlier semantic-chain and C21 availability scripts retain machine-specific source paths; hardware capture also remains machine-specific and is not an offline replay.


## 8 November revision inputs

The new main-paper Section 6 retains unknown-exploration evidence separately from the supplied-graph category mechanism; `supplement.pdf` reports every original/joint/C/frontier matrix outcome. The shared scene context and deterministic conditions do not increase independent scene counts. Expanded two-room and post-hoc interpretation is in `joint_diagnostic_detail.tex`. Original matrix source records are under `/home/ouc/nso_robot/analyses/frontier_semantic_publication_stage/experiments/unknown_exploration_20261104/`; the source file identities are recorded in `revision_review.json`.

`physical_update_20261108.tex` is the single insertion point for new physical evidence. Its calibrated S24 values are traced to `/home/ouc/nso_robot/analyses/calibrated_trial_20261108/terminal_summary.json` and the corresponding terminal report. Actual totals are 24 paid actions, 24 verified arrivals, 25 observations, 3.590760972 m CAN-estimated travel and 368.846735337 s. The operator's subsequent mapping-drift observation is retained as a limitation. Six new planning states have equal G/S scores and zero incremental aperture contribution. This is integration evidence, with no natural-semantic benefit, comparative efficiency or independent reconstruction-accuracy claim.

No uncompleted run or planned result is reported. Author identity, affiliation, correspondence and venue remain intentionally blank at user direction; no submission was made.


## Additional Q1 comparison

The prospectively fixed additional Q1 synthetic parent is separate from the 24-trajectory Q0 matrix. All G/S/X endpoints, common-state ranking checks and generic-weight ambiguity are retained in supplementary Table S3 and its adjacent analysis. Canonical new comparison records are in `/home/ouc/nso_robot/analyses/rapid_evidence_20261108/semantic/`, including `execution/protocol.json`, `comparison.json`, `conclusion.json` and `Q1_curves.csv`. All three arms complete 96 actions with zero simulated collisions and full-anchor return. S gains 11.7190812 m² reference-visible surface and 0.048558302 F1 relative to G/X, but all recorded actual S choices admit a common nonnegative geometric-prior weight; no weighted rollout or claim of category-specific advantage is made.


## Final calibrated ordered physical pair

Terminal records are `/home/ouc/nso_robot/analyses/rapid_evidence_20261108/physical/C_summary.json`, `S_summary.json` and `comparison/comparison.json`. The original protocol was corrected to a common 600 s external limit before first motion after a 900 s startup attempt was rejected by the unchanged input contract without motion. Each arm retained a 40-action allowance. C stopped normally at the anchor after 24 actions/25 observations/24 verified arrivals (`no_positive_option_at_anchor`), 3.737542498 m and 342.344998064 s. S stopped with the original `anchor_footprint_not_known_safe` failure after 13 actions/14 observations/13 verified arrivals, 1.856685546 m and 184.862006604 s, without return. Both logged zero feedback after close. Neither endpoint was an external timeout or interruption.

Initial odometry differed by 0.066946348 m and −0.062592959 rad; common-frame initial known-mask XOR was 430 cells (17.20 m²). Own-support surface-proxy supports differ (479 versus 437), so their changes cannot establish matched accuracy or policy superiority. All five candidate-bearing planning states had equal G/S scores; the terminal S state had no candidates. A shorter failed S route is not an efficiency benefit. Both episodes belong to one ordered pair in one physical scene.

The calibrated S24 drift report was made after the run and referred to preceding manual relocation to the test area. Its onset was not established; its initial map may already have been affected.
