"""Object-group split-conformal intervals; never modifies the B8 posterior.

Training fixes an adaptive residual scale, fresh calibration objects fix one
order statistic, and application accepts predicted distributions only. Truth
belongs exclusively to the separate training/calibration/evaluation stages.
"""
import hashlib
import json
import math
from decimal import Decimal, ROUND_CEILING

import numpy as np

CONDITIONS = ('front', 'front_side', 'around', 'occluded')
NUMERICAL_SCALE_FLOOR_M = 1e-12
FAMILY = 'B8_neutral_structure_prior_sigma_0.01'


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _groups(rows):
    if not rows:
        raise ValueError('At least one complete object is required')
    groups = {}
    for row in rows:
        oid = str(row['object_id'])
        condition = row['condition']
        if not oid or condition not in CONDITIONS:
            raise ValueError('Valid object ID and declared condition required')
        values = [float(row[k]) for k in
                  ('target_extent_m', 'median_m', 'lower_m', 'upper_m')]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('All target and base interval values must be finite')
        target, median, lower, upper = values
        if target < 0 or lower < 0 or not lower <= median <= upper:
            raise ValueError('Nonnegative extent and ordered base interval required')
        if condition in groups.setdefault(oid, {}):
            raise ValueError('Duplicate condition within object')
        groups[oid][condition] = dict(object_id=oid, condition=condition,
                                     target_extent_m=target, median_m=median,
                                     lower_m=lower, upper_m=upper,
                                     fallback=bool(row.get('fallback', False)))
    if any(set(group) != set(CONDITIONS) for group in groups.values()):
        raise ValueError('Every object must retain all four conditions')
    return groups


def fit_training_scale(rows):
    """Freeze median-object/median-condition absolute residual from training."""
    groups = _groups(rows)
    per_object = []
    for oid, group in sorted(groups.items()):
        errors = [abs(group[c]['target_extent_m'] - group[c]['median_m'])
                  for c in CONDITIONS]
        per_object.append(dict(object_id=oid,
                               condition_absolute_residuals_m=errors,
                               median_absolute_residual_m=float(np.median(errors)),
                               fallback_conditions=sum(group[c]['fallback'] for c in CONDITIONS)))
    raw = float(np.median([row['median_absolute_residual_m'] for row in per_object]))
    ordered = [groups[oid][c] for oid in sorted(groups) for c in CONDITIONS]
    return dict(schema='B9_training_residual_scale_v1',
                status='fitted_on_B8_training_objects_only',
                prediction_family=FAMILY, sigma_m=.01,
                s0_m=max(raw, NUMERICAL_SCALE_FLOOR_M),
                raw_median_scale_m=raw,
                numerical_floor_m=NUMERICAL_SCALE_FLOOR_M,
                numerical_floor_applied=raw < NUMERICAL_SCALE_FLOOR_M,
                definition='median_object(median_four_conditions(abs(target - base_median)))',
                object_ids=sorted(groups), objects=len(groups),
                conditions=list(CONDITIONS), cases=len(ordered),
                fallback_cases=sum(row['fallback'] for row in ordered),
                per_object=per_object, training_rows_sha256=_digest(ordered),
                posterior_or_policy_modified=False)


def finite_sample_rank(n, alpha=.05):
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError('Positive integer calibration object count required')
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    # Decimal avoids an upward round caused only by binary 0.05 arithmetic.
    return int((Decimal(n + 1) * (Decimal(1) - Decimal(str(alpha))))
               .to_integral_value(rounding=ROUND_CEILING))


def calibrate_object_intervals(rows, training_fit, alpha=.05):
    """Use fresh objects, with max score over all four dependent conditions."""
    if alpha != .05:
        raise ValueError('This frozen revision uses 95% base and target intervals only')
    if training_fit.get('schema') != 'B9_training_residual_scale_v1':
        raise ValueError('Frozen training scale receipt required')
    if training_fit.get('prediction_family') != FAMILY:
        raise ValueError('Only the declared neutral B8 distribution is calibrated')
    s0 = float(training_fit['s0_m'])
    if not math.isfinite(s0) or s0 <= 0:
        raise ValueError('Positive finite frozen scale required')
    groups = _groups(rows)
    if set(groups) & set(training_fit['object_ids']):
        raise ValueError('Training and calibration object IDs overlap')
    k = finite_sample_rank(len(groups), alpha)
    per_object = []
    for oid, group in sorted(groups.items()):
        cases = []
        for condition in CONDITIONS:
            row = group[condition]
            lower, upper, target = (row[k] for k in
                                     ('lower_m', 'upper_m', 'target_extent_m'))
            scale = max((upper - lower) / 2, s0)
            residual = max(lower - target, target - upper, 0.)
            score = residual / scale
            if not math.isfinite(score):
                raise ValueError('Non-finite normalized residual')
            cases.append(dict(condition=condition, base_lower_m=lower,
                              base_upper_m=upper, target_extent_m=target,
                              scale_m=scale, residual_m=residual, score=score,
                              fallback=row['fallback']))
        per_object.append(dict(object_id=oid, score=max(c['score'] for c in cases),
                               cases=cases))
    scores = sorted(row['score'] for row in per_object)
    finite = k <= len(scores)
    q = scores[k - 1] if finite else None
    ordered = [groups[oid][c] for oid in sorted(groups) for c in CONDITIONS]
    return dict(schema='B9_group_conformal_interval_v1',
                status='finite_group_calibration' if finite else 'insufficient_objects_for_finite_interval',
                prediction_family=FAMILY, alpha=float(alpha), coverage_level=1-float(alpha),
                s0_m=s0, q=q, finite=finite, order_statistic_rank=k,
                calibration_objects=len(groups), calibration_cases=len(ordered),
                train_object_ids=list(training_fit['object_ids']),
                calibration_object_ids=sorted(groups), conditions=list(CONDITIONS),
                fallback_cases=sum(row['fallback'] for row in ordered),
                per_object=per_object, sorted_object_scores=scores,
                training_fit_sha256=_digest(training_fit),
                calibration_rows_sha256=_digest(ordered),
                simultaneous_target='All four declared condition-specific projected extents of one new object',
                coverage_scope='Marginal over exchangeable calibration/new objects from the same frozen distribution; not conditional coverage for each geometry or class',
                insufficient_sample_rule='Use [0,+infinity), encoded upper_m=null; never replace the unavailable order statistic with the maximum observed score',
                posterior_or_policy_modified=False)


def _pmf(depths, masses):
    x = np.asarray(depths, dtype=np.float64)
    p = np.asarray(masses, dtype=np.float64)
    if (x.ndim != 1 or x.shape != p.shape or not len(x) or
            not np.isfinite(x).all() or not np.isfinite(p).all() or
            np.any(x < 0) or np.any(p < 0) or not math.isfinite(float(p.sum())) or p.sum() <= 0):
        raise ValueError('Aligned finite nonnegative depth values and masses required')
    values, inverse = np.unique(x, return_inverse=True)
    probabilities = np.bincount(inverse, weights=p / p.sum(), minlength=len(values))
    return x, p, values, probabilities / probabilities.sum()


def apply_to_distribution(depth_values_m, probability_mass, calibration):
    """Sensor-prediction-only application; no true extent or object class input."""
    if calibration.get('schema') != 'B9_group_conformal_interval_v1':
        raise ValueError('Calibration receipt required')
    if calibration.get('prediction_family') != FAMILY:
        raise ValueError('Only neutral B8 probabilities are covered by this receipt')
    alpha = float(calibration['alpha'])
    if alpha != .05:
        raise ValueError('This frozen revision uses 95% base and target intervals only')
    rank = finite_sample_rank(int(calibration['calibration_objects']), alpha)
    if bool(calibration['finite']) != (rank <= calibration['calibration_objects']):
        raise ValueError('Finite calibration flag is inconsistent with object count')
    s0 = float(calibration['s0_m'])
    if not math.isfinite(s0) or s0 <= 0:
        raise ValueError('Positive finite scale required')
    raw_x, raw_p, values, probabilities = _pmf(depth_values_m, probability_mass)
    cdf = np.cumsum(probabilities)
    def quantile(level):
        return float(values[min(int(np.searchsorted(cdf, level, side='left')), len(values)-1)])
    lower, upper = quantile(alpha / 2), quantile(1-alpha / 2)
    scale = max((upper-lower) / 2, s0)
    finite = bool(calibration['finite'])
    if finite:
        q = float(calibration['q'])
        if not math.isfinite(q) or q < 0:
            raise ValueError('Finite nonnegative calibrated order statistic required')
        radius = q * scale
        if not math.isfinite(radius):
            raise ValueError('Non-finite interval radius')
        corrected = dict(lower_m=max(0., lower-radius), upper_m=upper+radius,
                         width_m=upper+radius-max(0., lower-radius), finite=True)
    else:
        q, radius = None, None
        corrected = dict(lower_m=0., upper_m=None, width_m=None, finite=False)
    return dict(schema='B9_diagnostic_interval_output_v1', prediction_family=FAMILY,
                raw_distribution=dict(depth_values_m=raw_x.tolist(), probability_mass=raw_p.tolist()),
                raw_distribution_sha256=_digest(dict(depth_values_m=raw_x.tolist(), probability_mass=raw_p.tolist())),
                base_interval=dict(level=1-alpha, lower_m=lower, upper_m=upper,
                                   width_m=upper-lower, median_m=quantile(.5)),
                calibrated_interval=dict(level=1-alpha, **corrected),
                correction=dict(scale_m=scale, q=q, radius_m=radius),
                posterior_distribution_changed=False, actor_or_gain_changed=False,
                CRPS_scope='Only the unchanged raw distribution has a CRPS; the calibrated set is not a probability distribution',
                calibration_receipt_sha256=_digest(calibration))
