# B9 object-group diagnostic residual intervals

This revision corrects **reported intervals**, not the probability mass function,
template gains or policy. It leaves every B8 method file frozen. The original
G-neutral structure prior and sigma = 0.01 m remain fixed. The returned raw PMF,
its CRPS and point estimates are unchanged. No new planning benefit is implied.

## Observed failure mechanism

The frozen B8 training study has 12 objects and four dependent conditions per
object. Eight `two_observed_depth_faces` cases set depth to a single estimate.
Their 95% intervals have zero width and cover 0/8 targets, although their mean
absolute extent error is 0.0262055705 m. Their mean best horizontal-axis error
is only 0.513501 degrees. Thus axis error alone does not explain the zero-width
failure: finite surface sampling and conditional shape errors are treated as
zero uncertainty. Two additional multi-node cases also have zero-width 95%
intervals because a discrete CDF concentrates on one value. Temperature tuning
cannot widen a one-node bank. Training-only per-case results and source hashes
are preserved in `training_undercoverage_diagnostic.json`.

The target remains the true object extent projected onto the predicted depth
axis, as in B8. This correction does not estimate uncertain orientation or
resolve an incorrect axis. Width, layers and instance failures remain visible.

## Three separate data roles

1. **B8 training only:** fit a fixed positive residual scale from its original
   12 objects/48 condition cases. Old validation and held-out results do not
   choose this scale, a threshold or a band width.
2. **Fresh B9 calibration:** 24 independently sampled objects from a frozen
   procedural mixture. Their four conditions form one statistical unit per
   object. They determine one order statistic q after inference is complete.
3. **Fresh B9 test:** 24 more objects from the same independently sampled
   mixture, opened for evaluation only after q is frozen. No redraws or tuning
   based on outcomes. The test suite has 24 geometry units, not 96 independent
   samples. All clean-view conditions, failures and fallback cases remain.

The separate generator/inference/evaluator records define the exact independent
sampling and truth-access boundary. Interval application consumes predicted
depths and masses plus frozen calibration, and never receives hidden geometry.

## Fixed adaptive residual score

For B8 training object o and condition c, let m_oc be its raw PMF median and
y_oc its projected true extent. Fix

    s0 = median_o median_c |y_oc - m_oc|.

All four conditions, including the declared evaluation fallback, are retained.
The resulting fixed training value is 0.04755561642039245 m. A 1e-12 m numerical
floor applies only if this median is zero; it was not activated on these data.
It is not a fitted physical sensor noise level.

The base interval [L_oc,U_oc] uses B8's unchanged discrete 2.5%/97.5% quantiles.
For each fresh calibration object, define

    s_oc = max((U_oc - L_oc)/2, s0)
    R_o = max_c max(L_oc - y_oc, y_oc - U_oc, 0) / s_oc.

Set k = ceil((n+1)*0.95), and q = the k-th smallest of the n object scores.
For n=24 this is the maximum score, which is sensitive to an extreme
calibration object; this finite-sample cost must remain visible. If k>n, use
an unbounded interval rather than substituting the observed maximum. In
particular, fewer than 19 objects cannot give this procedure a finite 95%
interval. Decimal arithmetic prevents binary rounding from changing k.

For a new prediction use

    [max(0, L - q*s), U + q*s],  s = max((U-L)/2, s0).

The lower bound may use the known nonnegative extent domain; the upper bound
is **not clipped to 1.4 m**. Out-of-prior/fallback predictions remain eligible
for correction and retain their original failure labels in evaluator records.
Unbounded upper endpoints and widths are encoded as JSON null, with `finite`
false. The original distribution is still returned verbatim.

## Statistical interpretation

This is split conformal prediction using one score per object. Taking the
maximum over the four condition residuals gives a simultaneous event for those
four condition-specific extents of a new object. If calibration and new
objects are exchangeable and the fitted score is fixed independently of
calibration targets, the usual rank argument gives at least 95% marginal
coverage over calibration/new-object sampling. It is not a guarantee
conditional on each object, class, observation condition, real environment,
or the particular realized calibration set. IID procedural generation is an
assumption of this experiment; deterministic balanced quotas or a different
test distribution cannot simply be called IID. Coverage on 24 test objects
remains a noisy empirical estimate, not proof of real-world calibration.

The basic rank construction follows Angelopoulos and Bates, *A Gentle
Introduction to Conformal Prediction and Distribution-Free Uncertainty
Quantification*, Sections 1.1, 2.2–2.4 and Appendix D:
https://arxiv.org/abs/2107.07511. The four-condition maximum and retained B8
conditional geometry are the explicitly declared adaptation here.

## API and evaluation boundary

- `fit_training_scale(rows)` validates complete unique object/condition units,
  retains fallback rows, and returns the training scale and per-object audit.
- `calibrate_object_intervals(rows, training_fit, alpha=.05)` requires disjoint
  training/calibration object IDs and returns every score, q and finite status.
  This version deliberately permits only 95% base/target intervals.
- `apply_to_distribution(depth_values_m, probability_mass, calibration)` takes
  no true extent. It returns the raw PMF, original 95% interval, new interval,
  correction scale and receipt hash. It never outputs a new posterior or CRPS.

Evaluation must report original-PMF CRPS/median error separately from interval
coverage, width and interval score, including object-level all-four coverage
and every individual condition. Compare corrected and uncorrected intervals;
very broad intervals alone are not good estimation. Preserve nonfinite-width
counts and original inference failure/out-of-prior flags. No source meshes,
new candidate construction, scoring weights, stops, fitting, or policy runs
are introduced by this module.

Tests cover grouping/duplicate rejection, sample-size rank boundary, the
point-mass failure, raw-PMF invariance, adaptive width, fallback/out-of-prior
behavior, and all 20 possible held-out choices in a rank example, where the
19-object calibration procedure covers exactly 19/20 choices.
