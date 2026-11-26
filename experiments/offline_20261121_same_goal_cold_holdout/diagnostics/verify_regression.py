"""Independently verify both development-regression receipts. Reads no test-building files."""

import json
import hashlib
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RUNNER = STUDY / "runner"
PRIOR = STUDY.parent / "same_goal_two_witness_20261121"
ARRAY_FIELDS = ("depth_values_sha256", "rgb_values_sha256", "scan_ranges_sha256")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_arm(root, witness, contract, mode, seal, inputs):
    layout, paid, call = witness.split(":")
    paid, call = int(paid), int(call)
    path = root / f"{layout}_{mode}.json"
    result = read(path)
    require(result == seal["witnesses"][witness]["modes"][mode], "Witness differs from sealed aggregate")
    archived = PRIOR / "runner/runs" / layout / mode
    manifest = read(archived / "manifest.json")
    require(manifest["paid_actions"] == 256 and manifest["returned"] and manifest["collisions"] == 0,
            "Source is not the complete prior development witness")
    require(result["compared_archive_manifest_sha256"] == sha(archived / "manifest.json"), "Wrong replay archive")
    for filename in ("steps.json", "intervention.json"):
        filepath = archived / filename
        require(sha(filepath) == manifest["files_sha256"][filename], "Source archive corrupt")
        inputs[str(filepath)] = sha(filepath)
    inputs[str(archived / "manifest.json")] = sha(archived / "manifest.json")
    source = read(archived / "intervention.json")
    stop = contract["paid_first_divergence"]
    require(
        result["matched_before_intervention"] == paid and result["compared_steps"] == stop
        and result["local_call_id"] == call and result["local_route_id"] == contract["winners"][mode]
        and result["last_action"] == contract["actions"][mode]
        and not result["holdout_opened"] and not result["complete_exploration_trial"] and result["no_collision"],
        "Development regression condition changed",
    )
    full = result["full_pool"]
    require(full["enabled"] and full["original_small_pool_cap"] == 4 and full["unassessed_after_cap"] == 0
            and full["exhaustive_cap_upper_bound"] > full["original_distinct_alternates"]
            and full["full_candidate_pool_sha256"] == source["expanded_candidate_pool_sha256"]
            and full["full_score_sha256"] == source["expanded_score_table_sha256"],
            "Adapter failed to reconstruct original full-pool score table")
    raw = result["measured_action_pose_and_sensor_receipts"]
    steps = read(archived / "steps.json")[:stop]
    require(len(raw) == len(steps) == stop, "Wrong charge count at development witness")
    for current, old in zip(raw, steps):
        require(all(current[key] == old[key] for key in ("step", "action", "position", "heading", "packet_sha256", *ARRAY_FIELDS)),
                f"Archived action, pose or sensor differences detected at {layout}/{mode}/{old['step']}")
    if layout == "warehouse_right":
        refuse, select, authorize = result["guard_sequence"]
        legacy_refuse = source["pre_call_native_guard"][-1]
        legacy_authorize = source["post_call_native_guard"][-1]
        require(
            [refuse["call_id"], select["call_id"], authorize["call_id"]]
            == [contract["original_guard_rejection_call"], contract["native_reselection_call"], contract["authorization_call"]]
            and refuse["method"] == authorize["method"] == "assess_local_action"
            and select["method"] == "select_topo_target"
            and refuse["outputs"] == legacy_refuse["outputs"] and authorize["outputs"] == legacy_authorize["outputs"]
            and not refuse["outputs"]["allowed"] and refuse["outputs"]["action"] == "forward"
            and refuse["outputs"]["reason"] == "next_footprint_not_known_safe"
            and select["target"] == source["native_target"]
            and authorize["outputs"]["allowed"] and authorize["outputs"]["action"] == contract["actions"][mode],
            "Native warehouse rejection/selection/authorization chronology changed",
        )
    return raw


def verify():
    protocol_path = STUDY / "PROTOCOL.json"
    protocol = read(protocol_path)
    data = RUNNER / "regression"
    receipt = data / "receipt.json"
    seal = read(receipt)
    require(seal["status"] == "passed" and seal["source_commit"] == protocol["source_commit"]
            and seal["actor_code_sha256"] == sha(RUNNER / "cold_run.py")
            and seal["regression_code_sha256"] == sha(RUNNER / "regression.py")
            and seal["protocol_sha256"] == sha(protocol_path),
            "The sealed development validation is bound to different method or protocol bytes")
    inputs = {str(p): sha(p) for p in (receipt, Path(__file__), protocol_path, RUNNER / "cold_run.py", RUNNER / "regression.py")}
    for path, checksum in seal["files_sha256"].items():
        require(Path(path).parent == data and sha(Path(path)) == checksum,
                "Regression appendix outside sealed path or hash mismatch")
        inputs[path] = checksum
    rows = []
    for witness in seal["witnesses"]:
        contract = protocol["engineering_regression_before_holdout"][witness]
        g, s = (verify_arm(data, witness, contract, mode, seal, inputs) for mode in ("G", "S"))
        require(g[:-1] == s[:-1] and all(g[-1][field] != s[-1][field] for field in ("action", *ARRAY_FIELDS)),
                "Expected identical raw acquired history then actual charged action and data-channel divergence")
        if witness.startswith("gallery_left"):
            exact_after_choice = len(g[:-1]) - int(witness.split(":")[1])
            require(exact_after_choice == contract["equal_prior_actions_and_observations_after_injection"] == 9,
                    "Gallery common observations after local choice are incomplete")
        rows.append({
            "witness": witness, "compared_first_divergence_paid": g[-1]["step"],
            "G_action": g[-1]["action"], "S_action": s[-1]["action"],
            "full_raw_prefix_equality": True, "all_three_sensor_content_hashes_diverged": True,
            "actual_closed_loop_through_first_divergence": True,
            "verified_native_rejection_then_replan": witness.startswith("warehouse_right"),
        })
    result = {
        "status": "passed_independent_readonly_regression_review",
        "heldout_files_opened": False, "new_paid_actions_by_this_script": 0,
        "source_commit": protocol["source_commit"], "rows": rows, "inputs_sha256": dict(sorted(inputs.items())),
    }
    output = Path(__file__).with_name("regression_verification.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["rows"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    verify()
