# B8: uncertainty in unobserved extent, before calibration and policy outcomes

The user authorized this revision on 20 November 2026. This is an offline method
revision, not a robot deployment. All earlier B5/O8, B6 and B7 files and negative
results remain frozen. Unknown-space coverage remains the primary planning task.

## Model change

Retain B6's observed-instance input, qualified outer orientation, width,
box/leg height and absolute-layer construction as conditional geometry. Model
local-axis-1 extent and, where no unique outer face is observed, the allocation
of unobserved extent on either side of acquired support as latent variables.
This does not claim a posterior over all shape, orientation or object pose.

Use a common class-neutral extent prior on 0.20--1.40 m, deterministic 0.10 m
quadrature and acquired-support boundary handling recorded by the method. The
prior is conditioned on actual acquired support, not hidden dimensions. A
single credible observed endpoint anchors position. Without one, relative
unseen-extent allocation uses nodes 0, 0.5, 1 with masses 0.25, 0.5, 0.25.
Record out-of-support and inconsistent-anchor cases rather than silently
excluding objects. No outcome-dependent pruning of particles is allowed.

For structure h and geometric particle k, use the joint normalized weight

    w_m(h,k) proportional to pi_m(h) q0(k) exp[-E(h,k) / (2 sigma^2)].

The acquired-point and valid-free-ray errors, bounded discrepancy and
cross-history measured-surface veto follow the separately tested B7 diagnostic.
G and S share acquired instances, particle meshes, ray basis and geometric
errors. Only their class prior differs; consequently their marginal extent
weights may legitimately differ. Do not claim identical marginal dimensions.
Do not apply the old surface-distance likelihood a second time.

Each particle retains the existing B6 parts and area quadrature. Compute an
expectation across mutually exclusive hypotheses, not the area of their union.
Retain per-sample maximum visibility across a route and the B2 history mask.
Particle duplication with split probability and repeated route views must not
change gain. No candidate rules, reward coefficients, navigation, mapping,
action budget or stopping rules are changed by this revision.

## Independent calibration

Create 24 procedural objects independently of the old scene assets and the
inference mesh implementation. Split entire geometries into 12 training,
6 validation and 6 held-out objects, with all views of an object in one split.
Four fixed observation conditions per object give 96 dependent cases, not 96
independent objects. The generator and evaluator alone access the true geometry.
Inference receives sanitized RGB-D and visible instance support only.

Choose one shared sigma from [0.01, 0.015, 0.025, 0.04, 0.06, 0.09, 0.14,
0.22] m by training CRPS, averaging conditions within each object and then
equally across objects. Exact ties choose the larger sigma. Validation and
held-out results do not trigger retuning. Record prior-only and fixed-0.65 m
baselines, per-object errors, central interval coverage counts and widths,
failed cases and local-axis/orientation limitations. All interval calibration
claims are limited to these small synthetic splits.

The old six trajectory prefixes are repeatedly inspected development data,
not calibration or held-out data. Their 66 candidate outcomes may be reused
only with explicit dependence and zero new acquisition counts. Freeze the
policy matrix before new candidate scores or closed-loop outcomes; report all
arms, including ties and negative effects. A semantic win is not a calibration
criterion.

## Statistical rationale and limits

Loss-based exponential belief updates are a general framework, not a guarantee
that this particular mean-loss model is a correctly specified sensor posterior:
Bissiri, Holmes and Walker, *A General Framework for Updating Belief
Distributions*, https://arxiv.org/abs/1306.6430 (verified 2026-11-20).

Posterior spread does not automatically have nominal frequentist coverage:
Syring and Martin, *Calibrating general posterior credible regions*,
https://arxiv.org/abs/1509.00922 (verified 2026-11-20). Our proposed synthetic
CRPS selection is not their coverage-calibration algorithm, and must not be
described as inheriting its guarantees.

Fixing incorrect width, yaw, shelf parts or instance support can still bias an
extent distribution. These conditional-model limitations must remain visible
even when the depth marginal improves. A broad interval alone is not success;
nor does better geometry automatically establish better autonomous exploration.
