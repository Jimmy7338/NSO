"""Scale-safe candidate/asset features for the V12 CPU planning prototype.

V11 is frozen evidence and intentionally remains limited to two assets.  This
module adds a new versioned representation for an arbitrary non-empty set of
measured assets.  It consumes only decision-time audits.  Ground truth and
post-action outcomes are not accepted by this interface.
"""
from dataclasses import dataclass

import numpy as np

from nso.semantic_gain_v11 import _asset_confidence


COMMON_FEATURE_NAMES = (
    "common_proxy_per_action",
    "generic_opportunity_per_action",
    "cost_fraction",
    "inverse_cost",
    "log_asset_count",
    "aperture_mean",
    "aperture_max",
    "target_aperture",
    "non_target_aperture_mean",
    "target_generic_opportunity_per_action",
    "support_opportunity_per_action",
    "target_support_opportunity_per_action",
)

SEMANTIC_SET_FEATURE_NAMES = (
    "signed_target_aperture",
    "signed_target_generic_opportunity_per_action",
    "signed_set_generic_opportunity_per_action",
    "signed_non_target_generic_opportunity_per_action",
    "signed_set_support_opportunity_per_action",
    "signed_aperture_mean",
    "signed_aperture_second_moment",
    "signed_remaining_debt_opportunity_per_action",
)

GEOMETRY_CAPACITY_FEATURE_NAMES = (
    "target_aperture_square",
    "target_generic_opportunity_square",
    "set_generic_opportunity_square",
    "non_target_generic_opportunity_square",
    "set_support_opportunity_square",
    "aperture_mean_square",
    "aperture_second_moment",
    "remaining_debt_opportunity_square",
)


@dataclass(frozen=True)
class SetFeatureConfig:
    """Declared physical scaling; no statistic is fitted on confirmation data."""

    action_budget_scale: float = 192.0
    maximum_assets: int = 24

    def __post_init__(self):
        if not np.isfinite(self.action_budget_scale) or self.action_budget_scale <= 0:
            raise ValueError("positive finite action budget scale required")
        if not isinstance(self.maximum_assets, int) or self.maximum_assets < 1:
            raise ValueError("maximum_assets must be a positive integer")


def required_candidate_capacity(asset_count, *, coverage_slots=4, include_rotation=True):
    """Cap needed to offer coverage plus entry/deep roles for every asset."""
    if (isinstance(asset_count, (bool, np.bool_)) or not isinstance(asset_count, (int, np.integer))
            or int(asset_count) < 1):
        raise ValueError("asset_count must be a positive integer")
    if (isinstance(coverage_slots, (bool, np.bool_))
            or not isinstance(coverage_slots, (int, np.integer)) or int(coverage_slots) < 1):
        raise ValueError("coverage_slots must be a positive integer")
    return int(coverage_slots) + int(bool(include_rotation)) + 2 * int(asset_count)


def _finite_vector(values, name, count):
    array = np.asarray(values, dtype=float)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain one finite value per measured asset")
    return array


def _stable_sum(values):
    """Order-independent floating reduction for hash-stable replay."""
    values = np.asarray(values, dtype=float).reshape(-1)
    return float(np.sum(np.sort(values), dtype=np.float64))


def _stable_mean(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return _stable_sum(values) / len(values) if len(values) else 0.0


def candidate_set_features(prediction_audit, candidate, measured_assets, *,
                           confidence_scale=1.0, config=SetFeatureConfig()):
    """Return matched 20-D semantic and label-invariant geometry features.

    The representation is candidate-centric: the selected target asset is kept
    explicit and every other asset is reduced with symmetric sums/means.  A
    simultaneous permutation of assets, audit arrays and ``asset_index`` cannot
    change the result.  Coverage candidates with ``asset_index=None`` use a
    symmetric set mean and therefore have no privileged object.
    """
    count = len(measured_assets)
    if count < 1 or count > config.maximum_assets:
        raise ValueError("V12 requires 1..maximum_assets measured assets")
    cost = float(candidate["cost"])
    if not np.isfinite(cost) or cost <= 0:
        raise ValueError("candidate cost must be positive and finite")
    confidence_scale = float(confidence_scale)
    if not np.isfinite(confidence_scale) or not 0.0 <= confidence_scale <= 1.0:
        raise ValueError("semantic confidence scale must be in [0,1]")

    aperture = _finite_vector(prediction_audit["aperture_factors"],
                              "aperture_factors", count)
    generic = _finite_vector(prediction_audit["generic_area_proxies"],
                             "generic_area_proxies", count)
    prefix = _finite_vector(prediction_audit.get("prefix_aperture_support", [0.] * count),
                            "prefix_aperture_support", count)
    if np.any(aperture < 0) or np.any(generic < 0) or np.any(prefix < 0):
        raise ValueError("aperture, generic area and prefix support must be nonnegative")

    votes = _finite_vector([asset["class_vote"] for asset in measured_assets],
                           "class_vote", count)
    confidence = _finite_vector([_asset_confidence(asset) for asset in measured_assets],
                                "asset_confidence", count) * confidence_scale
    support = _finite_vector([asset["support_m2_proxy"] for asset in measured_assets],
                             "support_m2_proxy", count)
    if np.any(support < 0):
        raise ValueError("support_m2_proxy must be nonnegative")
    signed = votes * confidence

    target_index = candidate.get("asset_index")
    if target_index is not None:
        if isinstance(target_index, (bool, np.bool_)) or not isinstance(target_index, (int, np.integer)):
            raise ValueError("asset_index must be an integer or None")
        target_index = int(target_index)
        if not 0 <= target_index < count:
            raise ValueError("asset_index outside measured asset set")
        target = np.zeros(count, dtype=bool)
        target[target_index] = True
        other = ~target
        target_reduce = lambda values: float(values[target_index])
        other_reduce = lambda values: _stable_mean(values[other])
    else:
        target = np.ones(count, dtype=bool)
        other = np.ones(count, dtype=bool)
        target_reduce = lambda values: _stable_mean(values)
        other_reduce = lambda values: _stable_mean(values)

    generic_opportunity = aperture * generic / cost
    support_opportunity = aperture * support / cost
    remaining_debt = aperture * np.clip(1.0 - prefix, 0.0, 1.0) * generic / cost
    total_generic = _stable_sum(generic_opportunity)
    total_support = _stable_sum(support_opportunity)
    non_target_generic = (_stable_sum(generic_opportunity[other])
                          if target_index is not None else total_generic)
    aperture_second_moment = _stable_mean(aperture ** 2)
    debt_total = _stable_sum(remaining_debt)

    common = np.asarray((
        float(prediction_audit["common_proxy_per_action"]),
        total_generic,
        cost / config.action_budget_scale,
        1.0 / cost,
        np.log1p(count),
        _stable_mean(aperture),
        float(aperture.max()),
        target_reduce(aperture),
        other_reduce(aperture),
        target_reduce(generic_opportunity),
        total_support,
        target_reduce(support_opportunity),
    ), dtype=float)

    semantic_tail = np.asarray((
        target_reduce(signed * aperture),
        target_reduce(signed * generic_opportunity),
        _stable_sum(signed * generic_opportunity),
        _stable_sum((signed * generic_opportunity)[other]),
        _stable_sum(signed * support_opportunity),
        _stable_mean(signed * aperture),
        _stable_mean(signed * aperture ** 2),
        _stable_sum(signed * remaining_debt),
    ), dtype=float)

    geometry_tail = np.asarray((
        target_reduce(aperture) ** 2,
        target_reduce(generic_opportunity) ** 2,
        total_generic ** 2,
        non_target_generic ** 2,
        total_support ** 2,
        _stable_mean(aperture) ** 2,
        aperture_second_moment,
        debt_total ** 2,
    ), dtype=float)

    semantic_x = np.r_[common, semantic_tail]
    geometry_x = np.r_[common, geometry_tail]
    if semantic_x.shape != (20,) or geometry_x.shape != (20,):
        raise RuntimeError("V12 feature schema drift")
    if not np.isfinite(semantic_x).all() or not np.isfinite(geometry_x).all():
        raise ValueError("nonfinite V12 feature")
    return semantic_x, geometry_x, {
        "schema_version": "semantic_gain_v12_candidate_set_features/1",
        "asset_count": count,
        "target_asset_index": target_index,
        "permutation_invariant_non_target_pool": True,
        "truth_or_future_outcome_used": False,
        "confidence_scale": confidence_scale,
        "asset_confidence": confidence.tolist(),
        "semantic_feature_names": list(COMMON_FEATURE_NAMES + SEMANTIC_SET_FEATURE_NAMES),
        "geometry_feature_names": list(COMMON_FEATURE_NAMES + GEOMETRY_CAPACITY_FEATURE_NAMES),
    }
