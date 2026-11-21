# S96 terminal semantic-chain and safe-graph diagnosis

The actual episode ended after87 paid actions and88 observations, during return, with `current_footprint_not_known_safe`; it did not reach the1200s deadline. Root's reduction records87 verified arrivals,15.9443m and964.59s. This audit does not diagnose the terminal footprint/grid failure; another independent audit handles that. It diagnoses why the original semantic entry/scoring mechanism remained inactive.

All15 recorded global plans have exact G/S score-array equality. Fourteen have nonempty candidate sets; final plan15 has zero candidates, so its equality is an empty-array equality. Every retained candidate in the14 nonempty plans has zero incremental aperture and zero prior potentials. There is no recorded entry/deep candidate and no semantic G/S action divergence. These are actual-policy null results, not evidence that semantic recognition failed entirely.

Natural predictions across the episode include floor, closed equipment cabinet (first paid1), display counter (paid3), cabinet and wall (paid11), and chair (paid25). No shelves/bookcase detection is recorded, and measured quality class2 remains0. Final cumulative quality labels are class3=1255, class2=0, while the accepted geometry group at the final observation has zero marked points. At paid76, three of four accepted assets have measured class-vote1 support (20,58 and89 marked points); retained candidate aperture still remains zero. Thus missing category predictions alone cannot explain the whole episode's zero prior. Predicted classes and nominal aperture remain separate from independently verified object truth or measured hidden surfaces.

The larger-budget episode reached a new far-side accepted geometry at plan2/paid36, center[46.294,39.158,0.947],373 support points. Geometry group IDs are not persistent object IDs; group renumbering alone is not counted as discovering a new object. This is within-episode mapping progression, not a matched causal comparison with old40-action trials.

## Two original-history reconstructed checkpoints

One offline process replayed every actual observation/action from the initial state through paid36, with original canonical semantic masks, mapper, feedback ledger, totalbudget96 and original policy. All37 next-action returns and complete selection objects (routes, candidate audits, scores and score audits) matched the recorded baseline at1e-8 tolerance. No thresholds/weights/labels or production code were modified. Configuration and original remaining paid budget are preserved. Each checkpoint uses one original safe orientation graph and directed Dijkstra distances from current state/to the fixed return anchor.

| Checkpoint / asset | Full graph finding | Earliest established blocker |
|---|---|---|
| paid0 / group1 | No known-safe state passes rear>0.15. Maximum connected rear=-1.3161m. | Observed-safe geometry |
| paid0 / group3 |8 rear states and2 nonzero-aperture states, connected and budget-feasible. Both fail the original wanted heading2; no wanted-heading aperture pose exists. | Original heading admission |
| paid36 / group2 |36 raw-safe rear states, but zero connected rear states; connected max=-2.1161m. | Current/anchor connected safe geometry |
| paid36 / group4 |44 raw-safe rear states and3 raw-safe aperture states, but zero connected rear states; connected max=-0.3021m. | Current/anchor connected safe geometry |
| paid36 / new group5 |228 connected nonzero-aperture states, including90 wanted-heading3 states; none fits remaining60 actions. | Reserved outward+return budget |

The initial group3 all-heading aperture witness is[188,192,1], outward23+return23=46 actions, rear0.2972m. It does not pass original wanted heading2. Its entry-point projection error is2.1303m, so this witness would not satisfy the0.35m entry test either; it is not a proposed replacement action.

For paid36's new far-side asset, the cheapest all-heading aperture witness is[218,234,0], outward22+return56=78 actions, exceeding remaining60 by18. The cheapest original-wanted-heading aperture witness is[218,239,3], outward28+return60=88, exceeding by28. These particular witnesses have entry-projection errors2.8739m/2.9426m. Total budgets114/124 would merely admit these respective aperture witnesses at that same hypothetical historical state; they do not establish entry feasibility, semantic selection change, physical accessibility, or benefits of a new run. We have not swept larger budgets or changed the method to seek success.

The actual retained-route rear offsets are all negative across the14 nonempty plans. That finding is consistent with the full graph containing a few discarded aperture poses: retained routes are not the entire safe graph. The two-checkpoint audit distinguishes geometric/connectedness absence, heading restriction and reserved-budget restriction. Candidate cap12 and region filtering are not the earliest blockers at these checkpoints; no entry survived the earlier stages to make them causal. Generic-weight score equivalence is also not needed to explain the actual retained-candidate null, because their aperture increments are identically zero. Later global states are retained in JSON but were not individually reconstructed as new graph experiments.

Further repeated long physical matrices are not justified by these observations alone. The current evidence establishes precise applicability restrictions and a failed physical return, while leaving unknown-space coverage efficiency, semantic benefit and reconstruction accuracy as separate unproven claims. Any proposed change to heading, projection, safe radius, budget/stopping or scoring requires explicit disclosure and authorization; no change is proposed or deployed by this audit.

Files: `final_semantic_chain.json` (all15 plans), `paid0_graph.json`, `paid36_graph.json`, `replay_result.json` (37 matched calls), `replay_graph.py`, `replay.log`, and `provenance.json`. The original live snapshot was not overwritten.
