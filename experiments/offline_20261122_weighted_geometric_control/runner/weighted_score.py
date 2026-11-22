"""Re-rank the frozen full same-goal candidate pool without changing admission."""

from copy import copy, deepcopy
from dataclasses import asdict, dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class Weights:
    unknown_weight: float = 0.0
    action_cost: float = 0.0

    def __post_init__(self):
        if any(not math.isfinite(w) or w < 0 for w in (self.unknown_weight, self.action_cost)):
            raise ValueError("Extra geometric weights must be finite and nonnegative")


def rank_admitted(rows, native, prior, weights, cell_area, tolerance):
    ranked = []
    for row in rows:
        if not row["admitted"]:
            continue
        # In particular the neutral branch indexes only G, even if a table
        # also carries independent S measurements from a semantic replay.
        area = float(row["scores"][prior])
        unknown = (int(row["unknown_union_cells"]) - int(native["unknown_union_cells"])) * cell_area
        extra = int(row["outbound_cost"]) - int(native["outbound_cost"])
        if extra < 0:
            raise ValueError("Admitted route cheaper than the native shortest outbound trace")
        gain = area + weights.unknown_weight * unknown - weights.action_cost * extra
        if not math.isfinite(gain):
            raise ValueError("Non-finite weighted route utility")
        ranked.append({"route_id": row["route_id"], "area_m2": area, "delta_unknown_union_m2": unknown,
                       "extra_outbound_actions": extra, "score_m2": gain})
    best = max(row["score_m2"] for row in ranked)
    candidates = {row["route_id"]: row for row in rows}
    eligible = [row for row in ranked if best - row["score_m2"] <= tolerance]
    winning_id = min(
        (row["route_id"] for row in eligible),
        key=lambda identifier: (candidates[identifier]["outbound_cost"], identifier != native["route_id"],
                                candidates[identifier]["outbound_sha256"]),
    )
    return candidates[winning_id], ranked


def exchanged_model(model, reweight):
    """Reuse the previously archived two-shape prior exchange, keeping the point bank fixed."""
    assert float(model.audit["structural_temperature"]) == 16.0
    swapped = copy(model)
    swapped.audit = dict(model.audit)
    swapped.objects = deepcopy(model.objects)
    exchanged = []
    for source, copied in zip(model.objects, swapped.objects, strict=True):
        previous = list(source["shape_prior"])
        assert len(previous) == 2 and all(math.isfinite(value) for value in previous)
        copied["shape_prior"] = previous[::-1]
        exchanged.append({"track_id": source["support_track"], "S": previous, "X": copied["shape_prior"]})
    reweight(swapped, 16.0, model.audit["calibration"])
    assert swapped.bank is model.bank
    for field in ("points", "hypothesis_ids", "history_keep"):
        np.testing.assert_array_equal(getattr(swapped, field), getattr(model, field))
    return swapped, exchanged


def choose_route(legacy, frozen_policy, state, native, prior, weights, route_union_area, config, reweight=None):
    """Delegate exact route generation to legacy code, then re-rank its admitted rows."""
    if prior not in ("G", "S", "X"):
        raise ValueError(f"Unknown shape prior: {prior}")
    if native is None:
        return None, None
    cache = state["mapper"]._dual_template_state["cache"]
    assert cache["predictor"] == "visibility"
    neutral = cache["models"]["G"]
    provider_calls = {"G": 0, "S": 0, "X": 0}
    audit_neutral_wall_s = 0.0

    def score_area(model, path, ray, gain_cache):
        # The frozen selector enumerates both legacy priors. Returning a
        # discarded placeholder for its unused S branch leaves the generator
        # untouched while the neutral/X arm does no local S scoring.
        if model is not neutral and prior != "S":
            return 0.0, 0
        name = "G" if model is neutral else "S"
        provider_calls[name] += 1
        nonlocal audit_neutral_wall_s
        started = time.monotonic()
        area = route_union_area(model, path, ray, gain_cache)
        if name == "G" and prior == "S":
            audit_neutral_wall_s += time.monotonic() - started
        return area

    legacy_mode = "S" if prior == "S" else "G"
    unweighted_option, audit = frozen_policy(state, native, legacy_mode, score_area, config)
    if prior != "S":
        for row in audit["candidates"]:
            row["scores"].pop("S", None)
        audit["hypothetical_selected_route_id"].pop("S")
    exchanged = []
    if prior == "X":
        swapped, exchanged = exchanged_model(cache["models"]["S"], reweight)
        gains = {}
        for row in audit["candidates"]:
            if row["admitted"]:
                provider_calls["X"] += 1
                value, _ = route_union_area(swapped, row["outbound_states"][1:], cache["ray"], gains)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid exchanged-prior predicted area")
                row["scores"]["X"] = float(value)
    rows = audit["candidates"]
    native_row = next(row for row in rows if row["route_id"] == legacy.NATIVE_ID)
    cell_area = float(state["mapper"].config.resolution_m) ** 2
    tolerance = float(config["score_equality_tolerance"])
    winner, ranked = rank_admitted(rows, native_row, prior, weights, cell_area, tolerance)
    if prior == "X":
        baseline, _ = rank_admitted(rows, native_row, prior, Weights(), cell_area, tolerance)
        audit["hypothetical_selected_route_id"]["X"] = baseline["route_id"]
    selected = legacy._same_goal_option(native, winner)
    if weights == Weights() and prior in ("G", "S"):
        assert winner["route_id"] == audit["selected_route_id"]
        assert selected == unweighted_option
    audit.update(selected_mode=prior, selected_route_id=winner["route_id"], chosen_outbound_sha256=winner["outbound_sha256"],
                 score_table_sha256=legacy._json_sha([row["scores"] for row in rows]))
    audit["weighted_ranking"] = dict(
        prior=prior, **asdict(weights), grid_cell_area_m2=cell_area,
        native_unknown_union_cells=native_row["unknown_union_cells"], native_outbound_cost=native_row["outbound_cost"],
        score_equality_tolerance=tolerance, ranking_scores_sha256=legacy._json_sha(ranked),
        unweighted_same_prior_route_id=audit["hypothetical_selected_route_id"][prior],
        local_area_provider_calls=provider_calls, supplemental_neutral_audit_wall_s=audit_neutral_wall_s,
        choice_prior_uses_category=prior != "G",
        scores=ranked, observed_instance_shape_prior_exchange=exchanged)
    return selected, audit
