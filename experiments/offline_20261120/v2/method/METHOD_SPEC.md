# Instance pooling and absolute free-consistency mass

This explicit revision preserves the frozen geometry_20261120 method in its own directory. It introduces two independent model switches. Geometry, its unresolved extent annotations, original likelihood constants, shared full sample basis, historical consumption controls and B2 visibility remain available unchanged. Candidates, scoring coefficients, map fusion, actions, coverage and stopping rules are unchanged. No model code receives source assets or evaluator truth.

## API

`configure(geometry_fit=False, free_space_likelihood=False, instance_pooling=False, applicability=False)` returns a bound model class. The same optional keywords can be passed to `FullTemplateModel`. Other constructor keywords remain `fit_update`, `history_rejection` and `measured_surface_consumption`. Applicability requires `free_space_likelihood=True`; it is not meaningful to condition one distribution while discounting by a different undeclared evidence term.

The parent runner may use geometry on/off crossed with applicability on/off while pooling is enabled. An applicability-on variant also turns on its relative-free conditional likelihood, with the absolute mass applied as described below. Separate pooling-only/relative-free-only comparisons remain possible. `binding()` records all switches and source hashes.

## Observed instance pooling

First obtain the inherited measured components and their actually acquired support-track associations. Each `track>0` is represented by the unique sorted union of its measured indices, regardless of connectivity of those observed fragments. Unknown track0 components are copied unchanged. There is no truth-based merge or deletion of unknown groups.

Only one box/shelf pair is then constructed for that acquired instance, rather than a complete pair for each disconnected fragment. This does not assume that every estimated track is correct. Merged source-component IDs and counts are recorded. The inherited minimum points and size eligibility remain; a pooled track skipped by the inherited 2.2m horizontal span or .25m vertical span criterion is explicitly recorded in `skipped_pooled_groups`. Declared-state evaluation must report such skips, not attribute them silently to improved applicability.

## Absolute evidence attenuation

Let q be the original geometry-only conditional probability of the shelf hypothesis after its measured-surface discrepancy update. Let f_box and f_shelf be the v1 area-weighted fractions of resolved template samples contradicted by acquired free space. Valid finite depth, the 0.10m band, measured-surface veto, unknown-space treatment and union over frames are unchanged.

Use K = .15² / (2*.06²) = 3.125 from the existing discrepancy scales. Set L_h = exp(-K f_h), and compute the evidence marginal:

`Z = (1-q)*L_box + q*L_shelf`.

Keep exactly the v1 conditional shape posterior p, including its [-6,6] log-odds clip. The template probability masses used in surface quadrature are:

`m_box = Z*(1-p)` and `m_shelf = Z*p`.

The remaining `1-Z` is retained as unmodeled residual accounting. It is not reassigned to either template. Without conditional clipping, these masses equal the geometry-only hypothesis masses times each absolute free likelihood. With clipping, the inherited clipped conditional distribution is preserved and only its total mass changes. There is no second multiplication by the relative free likelihood.

G and S use identical acquired free evidence, scales, geometry and this formula. Their Z values may legitimately differ because their conditional geometry distributions have different semantic priors. The original object-probability factor is still applied once after these masses. When neither hypothesis has resolved contradictory evidence, both L are 1 and Z is 1. When both are completely contradicted, Z=exp(-3.125), so roughly 95.6% of total hypothesis mass is not redistributed back into unsupported templates.

This is a heuristic free-consistency attenuation, not a calibrated probability that the object is one of three true classes. Unknown geometry receives no penalty, so it does not prove template applicability. Relative ranking improvements, reduction in prediction magnitude and actual physical surface recovery must be evaluated separately. No gain improvement is guaranteed by reducing unsupported mass.

## Validation

Nineteen targeted tests pass. New tests check unique-data instance pooling, unchanged unknown components, one model for two tracked fragments, G/S common point basis, all-bad mass reduction, no-evidence unchanged mass, no squared free likelihood, preservation of the conditional clip and deterministic nonrecursive posterior rebuilding. The twelve v1 tests are retained, including observed-plane geometry, valid-depth/strict-free boundaries, measured veto, unchanged uncapped integral, shared history basis and independent evidence switches.
