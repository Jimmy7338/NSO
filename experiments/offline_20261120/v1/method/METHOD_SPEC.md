# B3 staged measured-only template revision

This directory implements a proposed revision, separate from B and B2. It changes the internal hypothetical surface model, not candidates, coverage objective, score coefficients, motion, map fusion, budget or stop rules. No source furniture bounds, meshes, labels outside actually observed pixels or evaluator ground truth enter the model.

## API and staged settings

`FullTemplateModel(mapper, semantic, fine_categories=True, fit_update=True, history_rejection=True, geometry_fit=None, free_space_likelihood=None, measured_surface_consumption=True)`.

`configure(geometry_fit=False, free_space_likelihood=False)` returns a fresh bound subclass. `None` inherits those class flags; an explicit constructor boolean overrides it. Stages are sampling `(False,False)`, geometry `(True,False)` and geometry_free `(True,True)`. `gain` invokes the unmodified final-hash B2 visibility implementation, with positive-weight samples evaluated then scattered back to the complete point basis. `diagnostics(model)` exposes B2 gain counters. Do not call B2 `install()` on this new class: that historical installer intentionally validates only B's source path.

`fit_update=False` disables only the inherited measured-surface-distance pseudo-likelihood. It does not disable the independent free-space pseudo-likelihood; rollout ablations should use the independent geometry/free/history controls. The legacy `history_rejection=False` still bypasses all reward history. For the new history-consumption ablation pass `history_rejection=True, measured_surface_consumption=False`: measured-free contradictions are still removed and B2 visibility remains enabled.

## Sampling and evidence basis

All template samples and their hypothesis IDs are retained, including zero-reward samples. There is no 3,500 point cap or global rescaling. History changes only the weights on this common basis. G and S have identical points and hypotheses for a common geometry stage; posterior weights differ. Every stage rebuilds from the original prior and current aggregate measured evidence, without recursively reusing an old posterior.

B2 historical depth classification is used for zero-weight masks: only finite positive valid-image depth is evidence; strict free means `z < depth - 0.10`; the closed surface band `[depth-0.10,depth+0.10]` vetoes conflicting free evidence. Unknown samples do not become free. With consumption enabled, both free and measured samples get zero future reward. This differs from B constructor behavior for invalid positive-infinite depths; its valid finite-depth, noncapped positive-weight integral is matched by tests. Source renderer frames use finite or zero depth.

Original measured-point grouping is retained, including `.12 < z < 1.8`, maximum 10,000 measured quality points, 0.2 m component grouping, and original minimum group/span criteria. These constraints mean the object envelope is fitted to partial evidence, not the full object.

## Geometric estimator (not a full shape optimization)

For near-vertical measured faces (`|normal_z| < .35`, horizontal normal norm > .7), a fourth angular moment estimates a Manhattan orientation modulo 90 degrees. At least eight normals and moment concentration >= .25 are needed; otherwise use the inherited world axes. This is a robust finite estimator, not learned shape completion or nonlinear optimization.

Transform acquired points and normals into these axes. Retain 5%/95% observed quantiles. For each horizontal axis, normals with component > .7 and < -.7 supply low/high inward-facing plane candidates. Eight samples per side and ordered median separation > .08 m identify opposite-face support. In that case median planes, expanded to contain the observed quantile interval, determine the extent. Thus measured extents can be smaller than original .9/.65 m minimums. A single supported face anchors the prior .9/.65 m extent behind that face; observed span is a lower bound. Unresolved endpoints are recorded with an unbounded upper extent, not presented as measured dimensions or a calibrated uncertainty distribution. The implementation still uses a point estimate for scoring.

The grounded z=0 assumption remains. If at least eight inward horizontal top-facing normals (`normal_z < -.7`) occupy the upper candidate band (within .04 m of their 95% height and within .08 m of the overall measured 95% height), the candidate plane anchors height without the old .9 m minimum or extra .08 m offset. Otherwise keep the inherited prior height and extend it to at least the measured upper quantile plus .08 m; there is no 1.5 m clipping of higher observations. A candidate could be an internal shelf level, so vertical completeness remains unresolved in the audit even when this estimate is used. This limitation must be tested, not described as complete object-pose estimation.

Box and shelf meshes retain the original parts and sampling; both use the same estimated pose and dimensions. Their predicted samples and ray meshes are rotated consistently once. Backboard direction is transformed into the same local axes. S affects no geometry parameter.

## Free-space pseudo-likelihood

For each alternative, classify each quadrature point once across acquired keyframes. Define weighted contradiction `f = strictly_free_area / (strictly_free_area + measured_surface_area)`, with zero penalty when there is no resolved evidence. Behind-surface, invalid and out-of-view points provide neither positive nor negative evidence. Repeated frames do not multiply a sample's evidence. Surface evidence vetoes contradictory free evidence.

Add `(f_box - f_shelf) * .15^2 / (2*.06^2)` to the original shelf log odds, independently of the original mean-distance term. The .15 m clipping and .06 m discrepancy scales are inherited from the original likelihood; there is no independently tuned coefficient. Both G and S use the same acquired evidence and expression. The result remains a heuristic Gibbs pseudo-posterior, not calibrated Bayesian confidence. Class priors remain .5 for G and `.1 + .8*visible_category_vote` for S. Log-odds clipping remains [-6,6].

## Validation and limitations

Twelve tests cover unchanged uncapped positive-weight integrals at three viewpoints, G/S common geometry in all switch combinations, fixed history-ablation basis, retained free-space rejection, independent free-likelihood evidence, duplicates, invalid/unknown depth, exact strict-free boundaries, measured-surface veto, 30/90 degree geometry, opposite observed extents below legacy minimums, partial-face uncertainty, measured height beyond 1.5 m and upper plane below .9 m.

Tests establish code behavior. They do not establish improvement. Replay must separately evaluate surface support, free-space contradiction, actual incremental reconstruction and candidate ranking. Oriented rectangular priors, internal shelf faces, finite measured support and unobserved extents remain substantive limitations. No all-scene advantage or guarantee of G/S separation follows from these changes.
