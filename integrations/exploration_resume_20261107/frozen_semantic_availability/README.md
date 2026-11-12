# Optional final-state semantic eligibility audit

This separate read-only diagnostic runs after the 41-frame S replay has fully matched. It does not change the session, choose a candidate, modify the original zero remaining budget, or claim an original policy outcome. Hypothetical additional B40/B80 budgets are conditional on the already-paid 40-action prefix and its retained map; they are not free exploration, matched trials, physical reachability or efficacy evidence.

The final quality-evidence sample contains 4,228 points, of which 859 carry class-3 labels. Applying the existing 0.2 m grid, 3×3 closing, connected grouping and 5–95% geometry tests yields three large groups holding all 859 marked points (133, 518 and 208). Their maximum horizontal extents are 4.911, 6.588 and 5.023 m, so each fails the active 3 m applicability cap. The two accepted groups have no marked points; the audit therefore cannot report a marked-asset B*. It does not establish actual object identities, ground-truth dimensions, or whether closing merged physically distinct objects. The map contains 449 known-safe cells and 979 reachable nonzero-outbound orientation states under the original circular safety graph. The earlier 40 paid actions and 7.3935 m CAN distance remain part of the context.

From the parent evidence directory, after installing the replay dependencies:

```sh
work=$(mktemp -d)
nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -B replay.py --output "$work" --post-audit-module frozen_semantic_availability/audit_module.py
PYTHONPATH=code/joint_vendor:code/nso_bridge OPENBLAS_NUM_THREADS=1 python -B frozen_semantic_availability/test_costs.py
```

The replay result and `frozen_semantic_availability.json` are written separately to the output directory. Five isolated tests cover paid turns and return heading, connectivity and original grouping thresholds. The audit checks that passed group IDs, supports and marked counts equal native final assets, and checks its read-only state fingerprint. Timings and machine-local source paths in the archived JSON are provenance, not expected portable outputs. No ROS or detector is used.

## Recorded action-20 prefix

`prefix20_semantic_availability.json` is a separate frozen audit invoked immediately after observation action20 and before its next planning call. The JSON phrase “final native packet” means the latest packet at the prefix hook, not the final packet of the full episode. Its complete surrounding replay continued to the end with zero differences (`prefix20_replay_result.json`); the audit did not change the recorded decisions. The state was `(200,218,0)` with anchor `(200,200,0)`, 20 paid actions and 20 remaining. CAN accumulation was 3.736396 m. The reported 260.739 s is the difference between the recorded arrival-pose source timestamp and `running.started_unix_s`, not exact monotonic runtime or observation completion time.

For marked group 3 (78 class-3 points), one known-safe aperture orientation exists outside the component connected to the current state and anchor; no finite B* is found in this graph. For group 5 (27 class-3 points), the cheapest **arbitrary-heading aperture-support** state costs 3 outbound plus 19 to return including heading, so it requires 22 remaining actions versus the actual 20. Its heading differs from the native requested heading; restricting to the requested heading requires 26. Hypothetical remaining budgets 40 and 80 correspond to total budgets 60 and 100 after the already-paid prefix, and yield 40 and 46 aperture states for group 5. This is a graph/aperture availability audit: native entry projection, entry/deep sequencing, candidate quotas and eventual selection were not evaluated. It does not establish physical reachability, correct object identity, improved policy outcomes or free access to the prefix.

To reproduce the separate prefix audit within one otherwise unchanged full replay:

```sh
work=$(mktemp -d)
nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -B replay_prefix20.py --output "$work" --prefix-audit-module frozen_semantic_availability/prefix20_hook.py
```

The copied hook resolves the byte-identical original `result.json` and `running.json` from `source/`; the source receipt records their SHA-256. This changes only the path of offline inputs. The full trace is checked both before and after the hook; this package has not run a second full replay solely to validate the relocated hook, and does not silently replace the source receipt.

## One conditional remaining-40 candidate/score audit

At the same real prefix20, a separate read-only call to original candidate generation and `_scores_v10_1` used a hypothetical 40 remaining actions (total 60, including the 20 already paid). This is a resource counterfactual, not an executed S60 or a matched physical trial. The original region filter and scorer were retained; neither the live ledger nor the original continuation was changed. The audit produced five candidates. G, S and X all selected candidate 2, target `[196,221,2]`, first action `forward`, with respective scores 11.614301, 5.547137 and 17.681465. For candidates 2 and 3, S−G was −6.067164: the category-dependent component affected scores but not the leading choice in this one conditional state. The marked asset 2 entry was rejected because its nearest candidate was 0.873847 m from the desired entry, exceeding the native 0.35 m projection limit; its deep role requires an available entry. These are candidate/model constraints, not proof of physical unreachability or correct object association.

`prefix20_budget40_candidate_audit.json` retains candidates, scores and eligibility details; `prefix20_budget40_replay_result.json` records that the full original trace continued with zero differences after the read-only audit. To reproduce in the tested environment:

```sh
work=$(mktemp -d)
nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -B replay_prefix20_budget40.py --output "$work" --prefix-audit-module frozen_semantic_availability/prefix20_candidates_hook.py
```

The publication hook only relocates `result.json` and `running.json` to byte-identical `source/` inputs. The original single offline run used the same bytes from their local report path; the relocated wrapper was not separately full-replayed.
