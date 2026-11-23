#!/usr/bin/env python3
"""Freeze and verify the development bank using archived forks and route geometry."""

import argparse
import csv
from fractions import Fraction
from hashlib import file_digest, sha256
import json
from pathlib import Path
import time


BANK = Path(__file__).resolve().parent
STUDY = BANK.parent
ANALYSES = STUDY.parent
SHELF = ANALYSES / "shelf_occlusion_pairs_20261122"
INCREMENT = ANALYSES / "region_global_revision_20261122/next/local_increment"
WITNESS = SHELF / "followup/local_witness"
EXPECTED_ORDER = ("increment_open", "increment_screen", "shelf_01_screen")
FORMS = ("D", "S")
GEOMETRY_MAX_COST_GAP = 2
GEOMETRY_PREFIX_ACTIONS = 16


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    with Path(path).open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def referenced_file(path):
    return str(path), file_sha(path)


def verify_run(run, expected_manifest):
    assert file_sha(run/"manifest.json") == expected_manifest
    manifest = read(run/"manifest.json")
    for name, expected in manifest["files_sha256"].items():
        assert file_sha(run/name) == expected, run/name
    return manifest


def first_poses(route):
    return {tuple(pose) for pose in route["outbound_states"][1:GEOMETRY_PREFIX_ACTIONS+1]}


def route_distance(left, right):
    return Fraction(len(left ^ right), max(1, len(left | right)))


def select_geometry(eligible, nominations, filter_ids):
    """Match each incumbent's route cost and maximize minimum early diversity."""
    by_id = {row["route_id"]:row for row in eligible}
    visited = {name: first_poses(row) for name, row in by_id.items()}
    recorded = dict(nominations)
    decisions = {}
    for form in FORMS:
        focal = by_id[nominations[form]]
        excluded = set(recorded.values())
        options = [row for row in eligible if
                   row["route_id"] not in excluded and row["route_id"] in filter_ids and
                   abs(row["outbound_cost"] - focal["outbound_cost"]) <= GEOMETRY_MAX_COST_GAP]
        assert options, f"No eligible geometry candidate near {focal['route_id']}"

        def key(row):
            diversity = min(route_distance(visited[row["route_id"]], visited[old]) for old in excluded)
            return abs(row["outbound_cost"] - focal["outbound_cost"]), -diversity, row["outbound_sha256"]

        winner = min(options, key=key)
        difference, negative_diversity, _ = key(winner)
        role = f"geometry_{form}"
        recorded[role] = winner["route_id"]
        decisions[role] = {
            "matched_nominee_route_id": focal["route_id"],
            "candidates_within_cost_bound_after_exclusions": len(options),
            "outbound_action_gap": difference,
            "minimum_first_16_jaccard_distance": {
                "numerator": (-negative_diversity).numerator, "denominator": negative_diversity.denominator},
        }
    assert len({recorded["geometry_D"], recorded["geometry_S"]}) == 2
    assert not set(nominations.values()) & {recorded["geometry_D"], recorded["geometry_S"]}
    return recorded, decisions


def original_inputs(layout, shelf_formal_path):
    if layout.startswith("increment_"):
        route_path = INCREMENT/"ROUTE_FREEZE.json"
        old_inputs = INCREMENT/"INPUT_FREEZE.json"
        archive = read(route_path)
        assert archive["inputs_sha256"] == file_sha(old_inputs)
        fork = archive["layouts"][layout]
        assert fork["status"] == "eligible_fork"
        sealed_witness = WITNESS/"WITNESS_FREEZE.json"
        prior = read(sealed_witness)["placements"][layout]
        assert prior["fork_candidate_pool_sha256"] == fork["candidate_pool_sha256"]
        assert prior["fork_packet_sha256"] == fork["packet_sha256"]
        assert all(prior["policies"][arm] == fork["selected"][arm] for arm in FORMS)
        return {
            "source":"increment", "source_bank":route_path, "source_candidates":route_path,
            "source_input_freeze":old_inputs, "candidate_pool":fork["assessed_candidates"],
            "eligible":fork["eligible_full_pool"], "nominations":{arm:fork["selected"][arm] for arm in FORMS},
            "sampling_set":{row["route_id"] for row in prior["routes"] if row["geometric_witness"]},
            "sampling_source":sealed_witness, "paid_before":fork["paid_before"], "target_pose":fork["target_pose"],
            "fork_pose":fork["fork_receipt"]["pose"], "fork_packet_sha256":fork["packet_sha256"],
            "cold_packet_sha256":fork["initial_raw"]["packet_sha256"], "fork_belief_sha256":fork["fork_belief_sha256"],
            "candidate_pool_sha256":fork["candidate_pool_sha256"], "return_actions":fork["return_actions"],
            "return_states":fork["return_states"], "rankings":fork["rankings"],
            "source_scope":"original local increment development fork under the historical visibility/weighted planner",
        }

    receipt_path = SHELF/"execution/formal"/layout/"placement_receipt.json"
    receipt = read(receipt_path)
    assert receipt["phase_freeze_sha256"] == file_sha(shelf_formal_path)
    registration = receipt["arms"]["G"]
    first = Path(registration["run_directory"])
    bank_path, candidates_path = first/"fork_bank.json", first/"fork_candidates.json"
    manifest = verify_run(first, registration["manifest_sha256"])
    assert file_sha(bank_path) == registration["fork_bank_sha256"]
    assert manifest["continuation_policy"] == "G_perf"
    fork = read(bank_path)
    assert fork["status"] == "triggered" and fork["freeze_sha256"] == file_sha(shelf_formal_path)
    candidates = read(candidates_path)
    rows = candidates["candidates"]
    return {
        "source":"shelf", "source_bank":bank_path, "source_candidates":candidates_path,
        "source_input_freeze":shelf_formal_path, "candidate_pool":rows, "eligible":[r for r in rows if r["admitted"]],
        "nominations":{arm:fork["arms"][arm]["route_id"] for arm in FORMS},
        "sampling_set":{r["route_id"] for r in rows if r["admitted"]}, "sampling_source":candidates_path,
        "paid_before":fork["paid_before"], "target_pose":fork["target_pose"], "fork_pose":fork["pose"],
        "fork_packet_sha256":fork["fork_packet_sha256"], "cold_packet_sha256":fork["initial_packet_sha256"],
        "fork_belief_sha256":fork["belief_sha256"], "candidate_pool_sha256":fork["candidate_pool_sha256"],
        "return_actions":fork["arms"]["D"]["return_actions"], "return_states":fork["arms"]["D"]["return_states"],
        "rankings":candidates["rankings"], "source_scope":"already-observed shelf 01 screened diagnosis; never held out",
    }


def bind_closed_loop(layout, data, route, roles):
    receipt_path = SHELF/"execution/formal"/layout/"placement_receipt.json"
    receipt = read(receipt_path)
    for arm in FORMS:
        if arm not in roles:
            continue
        proof = receipt["arms"][arm]
        assert proof["status"] == "verified_native_execution" and proof["technical_exit_code"] == 0
        directory = Path(proof["run_directory"])
        manifest = verify_run(directory, proof["manifest_sha256"])
        fork_path = directory/"fork_bank.json"
        assert file_sha(fork_path) == proof["fork_bank_sha256"]
        fork = read(fork_path)
        original = read(data["source_bank"])
        handover = fork["arms"][arm]
        assert fork["shared_identity_sha256"] == original["shared_identity_sha256"]
        assert fork["continuation_id"] == original["continuation_id"]
        assert all(handover[k] == route[k] for k in ("outbound_actions", "outbound_states", "return_actions", "return_states",
                                                    "route_id", "route_identity_sha256", "outbound_sha256"))
        assert manifest["continuation_policy"] == "G_perf"
        assert manifest["initial_packet_sha256"] == data["cold_packet_sha256"]
        return {
            "run_directory":str(directory), "manifest_sha256":proof["manifest_sha256"],
            "fork_bank_sha256":proof["fork_bank_sha256"], "receipt_path":str(receipt_path),
            "receipt_sha256":file_sha(receipt_path), "execution_arm":proof["execution_arm"],
            "continuation_id":fork["continuation_id"], "route_identity_sha256":proof["route_identity_sha256"],
            "shared_identity_sha256":fork["shared_identity_sha256"],
            "archive_captures":{name: row["status"] for name, row in manifest["checkpoints"].items()},
        }
    return None


def bind_short(layout, data, route):
    seal = WITNESS/"WITNESS_FREEZE.json"
    gathered = WITNESS/"execution_receipt.json"
    receipt = read(gathered)
    assert receipt["freeze_sha256"] == file_sha(seal)
    evidence = next(r for r in receipt["outcomes"] if r["layout"] == layout and r["route_id"] == route["route_id"])
    assert evidence["status"] == "verified" and evidence["exit_code"] == 0
    directory = Path(evidence["directory"])
    witness_path = directory/"witness_receipt.json"
    assert file_sha(witness_path) == evidence["witness_receipt_sha256"]
    witness = read(witness_path)
    old = verify_run(directory, witness["native_manifest_sha256"])
    assert old["same_observed_prefix_verified"] and not old["evaluator_inputs_read_by_actor"]
    assert old["candidate_pool_sha256"] == data["candidate_pool_sha256"]
    assert old["fork_packet_sha256"] == data["fork_packet_sha256"]
    assert old["target_pose"] == data["target_pose"]
    assert old["route_id"] == route["route_id"] and old["first_packet_sha256"] == data["cold_packet_sha256"]
    return {
        "run_directory":str(directory), "manifest_sha256":witness["native_manifest_sha256"],
        "witness_receipt_path":str(witness_path), "witness_receipt_sha256":evidence["witness_receipt_sha256"],
        "execution_receipt_path":str(gathered), "execution_receipt_sha256":file_sha(gathered),
        "archived_trace_sha256":file_sha(directory/"steps.json"), "verified_in_prior_short_study":True,
        "clock_dispositions":witness["paid_checkpoints"], "observed_outbound_event":old["route_end"],
    }


def assemble():
    protocol_path = STUDY/"PROTOCOL.json"
    protocol = read(protocol_path)
    receipt = SHELF/"publication/receipt.json"
    assert read(receipt)["head"] == protocol["base_commit"] == "3a0f74ff09d2d38aaa6b641392256edd2ea9f4db"
    assert read(receipt)["branch"] == protocol["branch"] == "codex/robot-evidence-20261104"
    assert tuple(protocol["development_order"]) == EXPECTED_ORDER
    formal_path = SHELF/"FORMAL_FREEZE.json"
    formal = read(formal_path)
    numeric_path = SHELF/"measurement/development_repeatability.json"
    tolerance = read(numeric_path)["tie_tolerance_m2"]
    assert tolerance == formal["tie_tolerance_m2"] == protocol["tie_tolerance_m2"]
    assert formal["native_budget"] == protocol["native_budget"] == 192
    placements, csv_rows = {}, []
    evidence_files = dict(referenced_file(path) for path in (protocol_path, receipt, formal_path, numeric_path, Path(__file__)))
    for layout in protocol["development_order"]:
        data = original_inputs(layout, formal_path)
        eligible = data["eligible"]
        overall = data["candidate_pool"]
        assert digest([{k:v for k,v in r.items() if k != "scores"} for r in overall]) == data["candidate_pool_sha256"]
        assert len(eligible) == sum(bool(row["admitted"]) for row in overall)
        by_id = {row["route_id"]:row for row in eligible}
        assert len(by_id) == len(eligible) == len({r["outbound_sha256"] for r in eligible})
        assert len(data["return_states"]) == len(data["return_actions"])+1
        assert data["return_states"][0] == data["target_pose"]
        for row in eligible:
            assert row["admitted"] and len(row["outbound_states"]) == row["outbound_cost"]+1
            assert row["outbound_cost"] == len(row["outbound_actions"])
            assert len(data["return_actions"]) == row["reserved_return_cost"]
            assert data["paid_before"]+row["outbound_cost"]+len(data["return_actions"]) <= 192
            assert row["outbound_states"][0] == data["fork_pose"]
            assert row["outbound_states"][-1] == row["goal_pose"] == data["target_pose"]

        selected, decisions = select_geometry(eligible, data["nominations"], data["sampling_set"])
        selected_rows = []
        ranking = {arm:{x["route_id"]:x for x in data["rankings"][arm]} for arm in FORMS}
        for route_id in dict.fromkeys(selected.values()):
            route = by_id[route_id]
            roles = [role for role, name in selected.items() if name == route_id]
            identity = {"outbound_states":route["outbound_states"], "outbound_actions":route["outbound_actions"],
                        "return_states":data["return_states"], "return_actions":data["return_actions"],
                        "oriented_target":data["target_pose"]}
            materialized = identity | {
                "route_id":route_id, "outbound_sha256":route["outbound_sha256"], "outbound_cost":route["outbound_cost"],
                "reserved_return_cost":route["reserved_return_cost"], "route_identity_sha256":digest(identity), "roles":roles,
                "archived_D_weighted_score_m2":ranking["D"][route_id]["score_m2"],
                "archived_S_weighted_score_m2":ranking["S"][route_id]["score_m2"],
            }
            materialized["archived_short"] = bind_short(layout, data, materialized) if data["source"] == "increment" else None
            materialized["closure_reuse"] = bind_closed_loop(layout, data, materialized, roles) if data["source"] == "shelf" else None
            selected_rows.append(materialized)

        frozen = {key:data[key] for key in ("source", "source_scope", "paid_before", "target_pose", "fork_pose",
                  "fork_packet_sha256", "cold_packet_sha256", "fork_belief_sha256", "candidate_pool_sha256")}
        frozen["initial_packet_sha256"] = data["cold_packet_sha256"]
        for name in ("source_bank", "source_candidates", "source_input_freeze", "sampling_source"):
            location, signature = referenced_file(data[name])
            frozen[name+"_path"], frozen[name+"_sha256"] = location, signature
            evidence_files[location] = signature
        frozen.update(n_assessed_native=len(overall), n_legal_native=len(eligible),
                      n_in_geometry_sampling_set=len(data["sampling_set"]), selected=selected, routes=selected_rows,
                      geometry_decisions=decisions)
        placements[layout] = frozen
        for row in overall:
            route_id = row["route_id"]
            csv_rows.append({
                "placement":layout, "route_id":route_id, "outbound_sha256":row["outbound_sha256"],
                "native_admitted":row["admitted"], "exclusion_reason":row["exclusion_reason"],
                "in_geometry_sampling_set":route_id in data["sampling_set"],
                "assigned_roles":"|".join(role for role, name in selected.items() if name == route_id),
                "outbound_actions":row["outbound_cost"], "reserved_return_actions":row["reserved_return_cost"],
                "abs_outbound_gap_to_D":abs(row["outbound_cost"]-by_id[selected["D"]]["outbound_cost"]),
                "abs_outbound_gap_to_S":abs(row["outbound_cost"]-by_id[selected["S"]]["outbound_cost"]),
                "predicted_D_area_m2":ranking["D"].get(route_id,{}).get("area_m2"),
                "predicted_S_area_m2":ranking["S"].get(route_id,{}).get("area_m2"),
                "weighted_D_score_m2_equivalent":ranking["D"].get(route_id,{}).get("score_m2"),
                "weighted_S_score_m2_equivalent":ranking["S"].get(route_id,{}).get("score_m2"),
            })
    return {
        "schema":"nso_shelf_opportunity_development_bank_v1", "phase":"development",
        "base_commit":protocol["base_commit"], "branch":protocol["branch"], "native_budget":192,
        "primary_task_paid":protocol["primary_task_paid"], "tolerance_m2":tolerance,
        "fixed_task_clocks":protocol["fixed_task_clocks"], "development_order":protocol["development_order"],
        "geometry_rule":{
            "sources":"Use previously frozen score-blind witness geometry IDs in the original increment setups. Use all admitted native candidates in known shelf_01_screen.",
            "slots":"Match D first, then original S. Exclude incumbent routes and any previous geometry selection.",
            "selection":"Lexicographically minimize absolute cost gap, then maximize the minimum Jaccard distance from any already selected route in the first 16 oriented poses, then lowest outbound SHA-256.",
            "max_cost_gap_actions":GEOMETRY_MAX_COST_GAP, "prefix_actions":GEOMETRY_PREFIX_ACTIONS,
            "uses_measured_candidate_outcome":False,
        },
        "placements":placements, "evidence_files_sha256":evidence_files,
    }, csv_rows


def audit_text(rows):
    import io
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Rebuild without changing the seal")
    args = parser.parse_args()
    target = BANK/"BANK_FREEZE.json"
    audit = BANK/"candidate_bank_rows.csv"
    record, rows = assemble()
    content = audit_text(rows)
    record["candidate_bank_rows_sha256"] = sha256(content.encode()).hexdigest()
    if args.check:
        reference = read(target)
        assert reference.pop("frozen_unix_s") > 0
        assert reference == record
        assert audit.read_text() == content
    else:
        assert not (STUDY/"execution").exists(), "The candidate registry must precede execution"
        assert not audit.exists()
        with target.open("x") as destination:
            json.dump(record | {"frozen_unix_s":time.time()}, destination, indent=2, allow_nan=False)
            destination.write("\n")
        with audit.open("x") as destination:
            destination.write(content)
    print(file_sha(target))
    for name, case in record["placements"].items():
        print(name, case["n_legal_native"], [
            (row["roles"], row["route_id"], row["outbound_cost"], bool(row["closure_reuse"]))
            for row in case["routes"]])


if __name__ == "__main__":
    main()
