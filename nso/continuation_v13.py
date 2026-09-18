"""V13 observed candidate descriptors and explicit development interventions.

No simulator or evaluator is accepted. Descriptors are measured proxies, not
calibrated uncertainty or evidence of semantic efficacy.
"""
from copy import deepcopy

import numpy as np

from nso.competition_prior_v8 import score_routes
from nso.cpu_sensor_contract_v10 import digest, json_value
from nso.semantic_gain_v11 import _asset_confidence


def describe_candidates(runtime, routes):
    state = runtime.components._cpu_backend.scenes[0]
    mapper, ledger = state["mapper"], state["ledger"]
    assets = state["assets"]
    history = ledger.planning_camera_poses()
    predicted = score_routes(mapper, routes, history)
    attempted = state["gain"].attempted_camera_mask(mapper, history)
    rows = []
    for route, audit in zip(routes, predicted["audit"]):
        radar, camera = state["gain"].route_masks(mapper, route["states"][1:], attempted)
        rows.append(dict(candidate_id=route["candidate_id"], group=route["group"],
            remaining_budget=ledger.remaining_budget,
            total_budget=ledger.total_budget, roundtrip_cost=route["cost"],
            outbound_cost=route["outbound_cost"], return_cost=route["return_cost"],
            unknown_allocated_grid_fraction=float(np.mean(mapper.belief == -1)),
            expected_new_camera_cells=int(camera.sum()), expected_new_radar_cells=int(radar.sum()),
            camera_yield_posterior=state["gain"].posterior_mean("camera"),
            radar_yield_posterior=state["gain"].posterior_mean("radar"),
            observed_assets=[dict(asset_index=i, class_vote=a["class_vote"],
                semantic_confidence=_asset_confidence(a),
                measured_support_points=int(a["support_points"]),
                inverse_sqrt_support_uncertainty_proxy=1. / np.sqrt(max(1, a["support_points"])),
                direction_unobserved_fractions=[1. - float(np.mean(
                    (np.asarray(a["bits"], dtype=np.int64) & (1 << sector)) != 0)) for sector in range(8)])
                for i, a in enumerate(assets)],
            fixed_prediction_audit=audit,
            feature_boundary="observations only; marker confidence and inverse support are uncalibrated proxies"))
    return json_value(rows)


def force_full_candidate(runtime, candidate_id, expected_pool_hash):
    """Intervene on one STGHP choice; keep RPN checks and IGCR updates live.

    The first option commits its full current->target->home route. After that,
    the unmodified geometry policy spends the same remaining task budget.
    This is an offline development intervention, never a learned-policy result.
    """
    state = runtime.states[0]
    if state["closed"] or state["pending"] is not None or state["active_actions"]:
        raise ValueError("candidate intervention requires an idle decision boundary")
    backend = runtime.components._cpu_backend
    backend.select_target(0)
    bs = backend.scenes[0]
    candidates = bs["last_selection"]["candidates"]
    if digest(candidates) != expected_pool_hash:
        raise ValueError("candidate pool changed before intervention")
    matches = [row for row in candidates if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("candidate is not uniquely identified in the frozen pool")
    option = deepcopy(matches[0])
    option.update(option_id=f"v13-forced-{bs['plans']}-{candidate_id}",
        selection_call_id=len(backend.calls) + 1, selected_map_version=bs["mapper"].frames,
        selected_feedback_version=bs["feedback"]["feedback_version"],
        parent_region=bs["graph"]["current_region"])
    bs["selected"] = option
    state["option"] = deepcopy(option)
    state["active_actions"] = list(option["actions"])
    state["phase"] = "outbound"
    backend._record(0, "STGHP", "development_forced_full_candidate",
        dict(candidate_pool_sha256=expected_pool_hash, candidate_id=candidate_id), option)
    runtime.audit.append(dict(event="development_forced_full_candidate", option=deepcopy(option)))
    return option
