"""V13.2 experimental first-outbound intervention with the live return guard.

This changes the intervention's commitment horizon, not semantic scoring or
the shared candidate pool. It has no claim of improved reconstruction utility.
"""
from copy import deepcopy

from nso.cpu_sensor_contract_v10 import digest


def force_outbound_candidate(runtime, candidate_id, expected_pool_hash):
    state = runtime.states[0]
    if state["closed"] or state["pending"] is not None or state["active_actions"]:
        raise ValueError("candidate intervention requires an idle decision boundary")
    backend = runtime.components._cpu_backend
    backend.select_target(0)
    backend_state = backend.scenes[0]
    candidates = backend_state["last_selection"]["candidates"]
    if digest(candidates) != expected_pool_hash:
        raise ValueError("candidate pool changed before intervention")
    matches = [row for row in candidates if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("candidate is not uniquely identified in the frozen pool")
    option = deepcopy(matches[0])
    outbound = list(option["outbound_actions"])
    if not outbound or len(outbound) != option["outbound_cost"]:
        raise ValueError("positive complete outbound option required")
    if option["cost"] > backend_state["ledger"].remaining_budget:
        raise ValueError("full return reservation exceeds the remaining budget")
    option.update(option_id=f"v13.2-forced-outbound-{backend_state['plans']}-{candidate_id}",
        selection_call_id=len(backend.calls) + 1,
        selected_map_version=backend_state["mapper"].frames,
        selected_feedback_version=backend_state["feedback"]["feedback_version"],
        parent_region=backend_state["graph"]["current_region"],
        intervention_contract="first_outbound_then_geometry_replan_with_live_return_guard")
    # Keep the complete return route and its cost in both module records.
    backend_state["selected"] = option
    state["option"] = deepcopy(option)
    state["active_actions"] = outbound
    state["phase"] = "outbound"
    backend._record(0, "STGHP", "development_forced_outbound_candidate",
        dict(candidate_pool_sha256=expected_pool_hash, candidate_id=candidate_id), option)
    runtime.audit.append(dict(event="development_forced_outbound_candidate", option=deepcopy(option)))
    return option
