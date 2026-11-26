# B8 independent distribution calibration and acceptance specification

This specification is written before new inference or calibration outcomes. It does not alter the actor, generate objects, launch policies or change the old B5/O8/B6/B7 evidence.

## Units and isolation

The generator freezes 24 procedural objects, split by whole object into 12 training, six validation and six held-out objects, with four observation conditions each. Repeated frames and condition prefixes remain measurements of the same object. The expected 96 cases / 288 input frames are not 96 independent geometries. All outcomes, missing masks/tracks and out-of-support cases remain recorded. Ideal visible instance masks are not a natural detector.

The inference process receives sanitized observed frames, masks and public constants only. Neither split membership nor true vertices, dimensions, pose, part labels or reference geometry enters inference. The evaluator loads a private truth file only for cases in the requested split. Training alone selects the scale. Validation and held-out outcomes are evaluated after a selected-scale receipt is written and hashed; no validation retuning or selection by G/S outcome is allowed. The old six prefixes/66 branches remain inspected development diagnostics.

## Shared geometry and likelihood

The method exports a common particle bank k, its prior masses q0(k), acquired surface/free discrepancy arrays E_surface(k,h), E_free(k,h), and the inferred world-space depth-axis unit vector. Shape h has two states, box/shelf. Calibration uses pi(h)=0.5. For sigma in metres:

`w(k,h) = normalize(q0(k) * pi(h) * exp(-(E_surface(k,h)+E_free(k,h))/(2*sigma**2)))`.

There is no second multiplication by the old B6 fitting likelihood. Duplicate depths caused by location or shape hypotheses are summed into one depth PMF for size evaluation. Particle-bank support, observed-span conditioning and any out-of-prior fallback are recorded explicitly. The prior-only baseline uses the same pre-likelihood bank q0(k); it therefore retains the same observed-span/anchor constraints. The fixed-0.65 baseline is a point distribution used as a dimension benchmark, with no invented confidence interval. Out-of-support truth is not clipped.

G/S subsequently share observations, support, particles and geometric likelihood; only the class prior changes. Their marginal depth distributions can differ through the class mixture. Calling their marginal depth distributions identical would be incorrect.

## Evaluation target and scores

The private evaluator projects all true object vertices onto the actor's inferred world depth axis and takes max minus min. This is the extent corresponding to the inferred coordinate, not automatically the object's intended depth. In parallel report intended local depth, error to the declared true depth axis (sign ignored), best matching horizontal true axis, swapped-axis flag, and non-horizontal axis error. These diagnostics prevent a rotated or swapped axis from being hidden by the projected target. Do not match/reassign axes to minimize prediction error.

For PMF values x_i and masses p_i, CRPS in metres is exactly:

`sum_i p_i * abs(x_i-y) - 0.5 * sum_i,j p_i*p_j*abs(x_i-x_j)`.

The posterior median is the smallest x with cumulative probability at least 0.5. Its absolute error is reported. Central 50/80/95 percent intervals use this same left-inverse discrete CDF at tails (0.25,0.75), (0.10,0.90), (0.025,0.975). Report each interval's inclusive truth-in-interval count, denominator and mean width; finite grid intervals are not continuous confidence bounds. For fixed-0.65, report point MAE/CRPS and degenerate interval width/count only as a baseline, never as calibrated uncertainty.

Aggregate first equally across four conditions of each object, then equally across objects in the split. Also report each condition equally across objects and all individual object/condition rows. Missing, multiple-track or unusable inference cases must not disappear from denominators. They receive an evaluator-side, sigma-independent unconditional uniform-density prior on [0.2,1.4] m using the public 0.1 m nodes and Voronoi masses. Their evaluation axis is the fixed public world-y direction, explicitly marked as fallback; no hidden geometry selects a track or axis. They retain failure reasons, and full-denominator results are accompanied by valid-only summaries and failure counts. A scalar selected with fallback cases is provisional and does not establish calibration of a complete tracking-plus-fitting pipeline. Hash violations in an existing frozen input remain integrity failures, not statistical fallback cases.

## Frozen scale choice

Candidate sigma values are `[0.01,0.015,0.025,0.04,0.06,0.09,0.14,0.22]` metres. Minimize training equal-object CRPS only. Numerical ties within 1e-12 metres choose the larger sigma. Save every scale's training rows and score, not just the winner. Validation/held-out are not loaded during scale selection. The selected scalar is a generalized-posterior temperature, not an estimate of real sensor noise.

## Predeclared acceptance and limitations

1. Input isolation, bank normalization, finite scores, all-case accounting and immutable train/validation/held-out split checks must pass.
2. An identical-observation counterfactual with a different hidden depth must produce an identical bank, likelihood and posterior. Separately, a synthetic strictly front-only, depth-independent likelihood must leave the depth prior unchanged. A nominal front view of an open shelf is not necessarily depth-uninformative.
3. For front-side and around-object validation/held-out cases, report CRPS and median MAE against both prior-only and fixed-0.65 baselines, with interval coverage counts and widths. Any improvement must be accompanied by these complete paired results; broad intervals alone are not success, and a lower point error alone is not uncertainty calibration.
4. Geometry readiness uses full-denominator informative-view improvement in equal-object CRPS relative to both prior-only and fixed-0.65 baselines, assessed separately on untouched validation and held-out subsets. Failed cases remain in these scores through the declared fallback, with their rates and interpretation visible; any failure does not add a separate automatic no-experiment gate. Informative views mean front-side plus around-object, averaged first within object then equally across objects. Median MAE is an important reported diagnostic, not an additional hard acceptance criterion. All per-condition tradeoffs remain visible. If the small suite cannot establish this, record the failure, perform only the old-six-prefix diagnostic to investigate it and do not launch a new closed-loop policy or tune the scale on those subsets to force acceptance. This is a conservative experimental progression rule, not a navigation-candidate filter or statistical claim of general superiority.
5. Report whether informative-view intervals become narrower than front-only intervals while retaining truth; do not impose a nominal population coverage claim from six held-out objects. No universal confidence/calibration claim follows from these descriptive counts.
6. Passing the geometry checks permits a separately frozen prediction/policy evaluation, not a claim that S outperforms G. Reused old six-state evaluation still checks absolute supported and realized weight, rank, and actual selected object/global gain and cost, not support precision alone.

## Subsequent policy matrix, separately authorized and frozen

Recommended five arms on each fixed layout: G_fixed, S_fixed, G_uncertainty, S_uncertainty, S_mean-collapse. All use B8's same observed support, B6AB non-depth geometry, calibrated likelihood, candidates, fusion and execution budget. Fixed mode preserves directly observed opposing-face extent; unresolved depth is max(0.65 m, acquired lower span). It retains the same location-allocation a nodes, prior and acquired-data updating as the full model, isolating the extent distribution rather than removing position uncertainty too. Old O8 remains a historical control, not the sole isolation of dimension uncertainty.

S_mean-collapse first computes the full S joint posterior, preserves each structural marginal and then collapses that structure's geometry distribution to its conditional mean extent/location. It does not recompute likelihood or class weights after collapse. This is an approximate representative-geometry ablation; it isolates removal of spread while preserving posterior structural belief. A full G/S by fixed/uncertain factorial identifies their interaction, while mean-collapse distinguishes spread from simply replacing a bad fixed size with a better fitted location/size.
