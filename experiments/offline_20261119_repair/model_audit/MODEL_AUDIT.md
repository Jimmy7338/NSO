# B → B2: measured-free template occlusion consistency

Scope: offline fixed-observation diagnosis and one declared visibility repair. B and the original scientific source remain unchanged. No candidate generation, weights, template sizes, execution/return rule, map update or scene is modified here. A repaired expected gain is a counterfactual prediction, not an observed improvement in coverage or reconstruction.

## Confirmed inconsistency and minimal repair

The original `ObjectCompletionModel` removes reward samples with projected depth at or in front of an acquired depth measurement (`z <= measured + 0.10 m`). Its later self-occlusion query still raycasts against the complete original box/shelf meshes. A template surface explicitly contradicted by observed free space can therefore keep blocking a remaining, unobserved reward sample. Removing every historically explained occluder would be incorrect: an actually measured solid surface must remain opaque.

B2 distinguishes these cases at each ray intersection using acquired full-scene keyframes only. It may skip an intersection only if a valid, in-image depth sample establishes `z_hit < measured_depth - 0.10 m`, with no keyframe placing that same intersection in the closed measured-surface tolerance band. A measured surface, a point behind measured depth, an out-of-image projection, invalid depth and an unobserved point remain opaque. It does not remove an entire triangle because one intersection was contradicted. All preceding intersections are examined, so a free first blocker does not make a later unknown or measured blocker transparent. Real TSDF occlusion is unchanged and runs first.

This changes the template visibility rule and is explicitly a new version. It does not change geometry-fit likelihoods or make the box/shelf pseudo-posterior a free-space-calibrated shape distribution. It also does not carve or fuse template geometry into the measured map.

## Executable evidence

`test_occlusion_repair.py` contains ten passing tests. A calibrated oblique history view sees free space through a synthetic front board while the later target remains outside that old view. Original gain incorrectly remains zero because the board is still in the template ray mesh; B2 recovers the target. Separate tests retain a measured blocker, a subsequent unknown blocker, conflicting measured-surface support, real-TSDF occlusion and independent alternative hypotheses. Multiple strictly contradicted blockers can be skipped. Zero, NaN, infinite and out-of-image depth do not establish free space.

The disabled repair matches original `ObjectCompletionModel.gain` elementwise on original source-model fixtures, and installation does not edit B. A first test run exposed floating-point ambiguity in `abs(z-d) <= 0.10`; the final implementation uses the same explicit `d-0.10` lower bound for both the strict-free and closed surface intervals. The initial failure log is retained. A new process is required to change the repair on/off after evaluating gain, preventing stale original route caches from mixing visibility versions.

The first fixed S-trajectory replay reproduces 256 actions, 257 packets, all original scores and the native TSDF exactly. Its on-repair hook was loaded before that final floating-point band correction, so its reported recovered gains are explicitly pre-boundary counterfactuals, not outputs attributed to the final source hash. The trace shows the intended defect occurs in actual saved states, including L54 and L109, and can suppress all old selected-route template gain at L167; it does not remove all zero-gain cases (L208 remains zero). Final-hash fixed-state validation is recorded separately by the replay agent.

## Remaining limitations, not changed by this repair

| Item | Audit finding and boundary |
|---|---|
| Visibility | The source uses axial depth, consistent with this renderer. Per-alternative ray scenes correctly prevent a box and its mutually exclusive shelf hypothesis from occluding each other. The repaired history check still uses nearest-pixel projection and the original tolerance, not a calibrated uncertain-depth visibility model. |
| Reward area | `_box` assigns face-area quadrature weights; the route takes a pointwise maximum over outbound views, avoiding repeated-view addition. Coefficient 1 replaces the old analytic term. These are area proxies, not independently measured surface quality. No evidence here warrants changing the coefficient. |
| Point cap | Global deterministic thinning above 3,500 points preserves only total weighted area. It need not preserve each object's, alternative's or view's weighted mass. B's S_no_history activates this approximation while its other arms do not. B2 retains the cap and records activations; it is not silently combined with a sampling repair. |
| Partial geometry | The source constructs axis-aligned templates from measured quantiles and normals, minimum XY sizes 0.9×0.65 m and grounded height clipped to 0.9–1.5 m. This is a restrictive shape/pose assumption, not full object-pose estimation. Actual source furniture lies outside that height prior; no hidden asset bounds are used to repair it. |
| Probability | Fit uses mean clipped distance and starts from the original prior on each rebuild. It is not sequential Bayes or calibrated confidence; adding historical free-space evidence to its likelihood would be a separate method change. |
| Candidate/return limits | Late single-candidate states can be forced by the remaining return budget. Zero template gain alone does not prove a missing semantic observation or a visibility bug. The separate candidate audit supplies that distinction. |

## Runner API and ablation meaning

Import `method/occlusion_repair.py` under a distinct module name, then call `install(old_B_adapter.FullTemplateModel, enabled=True)` once in the new process before any scores. `diagnostics()` provides JSON-compatible global and per-variant counters; `diagnostics(model)` provides per-model counters. These counters sum evaluations, not unique surfaces or independent observations. Default storage contains no point arrays. `record_calls=True` optionally retains compact per-view receipts.

The repair is common to both G and S. In B2 `S_no_history` still disables the two historical **reward** filters, while the measured-free visibility repair remains enabled. This prevents the ablation from reintroducing a known contradictory occluder, but differs from B's old visibility behavior and must be disclosed. The old B adapter's static `reused_original_gain_function=True` field must be overridden in B2 output: disabled-gain parity is a negative control, not evidence that the enabled gain is unchanged.
