"""Bind the paired published same-goal development data without running its actor."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
SOURCE = STUDY.parent / "same_goal_route_choice_20261121"
COMMIT = "02e9c5dcc0da7df61f23d8683fec2dd92c145c1e"
ARMS = ("newG", "newS")
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def read(path):
    return json.loads(Path(path).read_text())


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def require(passed, reason):
    if not passed:
        raise ValueError(reason)


def verify(path, expected):
    observed = sha_file(path)
    require(observed == expected, f"Source checksum changed: {path}")
    return observed


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temporary.replace(path)


def next_pose(before, action):
    r, c, heading = before
    require(0 <= heading < len(DIRS), f"Invalid packed heading {before!r}")
    if action == "forward":
        dr, dc = DIRS[heading]
        return [r + dr, c + dc, heading]
    if action == "left":
        return [r, c, (heading - 1) % len(DIRS)]
    if action == "right":
        return [r, c, (heading + 1) % len(DIRS)]
    raise ValueError(f"Unknown historical action {action!r}")


def verify_path(actions, states, start, goal, label):
    require(len(states) == len(actions) + 1 and states[0] == start and states[-1] == goal,
            f"Candidate route endpoints or action counts disagree: {label}")
    for index, action in enumerate(actions):
        require(next_pose(states[index], action) == states[index + 1],
                f"Packed heading/transition disagrees with {action} at {label}:{index}")


def baseline_candidate_pool(core, route, cfg):
    candidates = core["candidates"]
    native = next(item for item in candidates if item["route_id"] == "native")
    require(core["original_route_id"] == "native", "Original native route was displaced")
    require(core["assessed_candidate_count"] == len(candidates), "Candidate counts disagree")
    require(len({row["route_id"] for row in candidates}) == len(candidates), "Duplicate candidate IDs")
    admitted = [row for row in candidates if row["admitted"]]
    require(len(admitted) == core["admitted_candidate_count"], "Admitted route count is wrong")
    require(len(admitted) <= cfg["max_alternates"] + 1 and native["admitted"], "Original cap/native route changed")
    require(core["candidate_pool_sha256"] == digest([
        {key: value for key, value in row.items() if key != "scores"} for row in candidates
    ]), "Original candidate pool checksum disagrees")
    require(core["score_table_sha256"] == digest([row["scores"] for row in candidates]),
            "Original scores checksum disagrees")
    require(len({tuple(row["outbound_actions"]) for row in candidates}) == len(candidates),
            "Candidate pool contains duplicated physical command sequences")
    require(native["outbound_actions"] == route["native_outbound_actions"]
            and native["outbound_states"] == route["native_outbound_states"], "Native trace differs from original plan")
    require(core["native_outbound_sha256"] == native["outbound_sha256"], "Native route checksum differs")
    require(route["native_return_actions"] == route["rebound_return_actions"]
            and route["native_return_states"] == route["rebound_return_states"],
            "Archived alternative changed the native reserve")
    verify_path(route["native_return_actions"], route["native_return_states"], route["same_global_target"],
                core["return_anchor"], route["option_id"] + "/return")
    for row in candidates:
        actions, states = row["outbound_actions"], row["outbound_states"]
        verify_path(actions, states, route["pose"], route["same_global_target"], row["route_id"])
        require(digest({"outbound_actions": actions, "outbound_states": states}) == row["outbound_sha256"],
                f"Route checksum is wrong: {row['route_id']}")
        require(len(actions) == row["outbound_cost"] and row["reserved_return_cost"] == len(route["native_return_actions"])
                and row["goal_pose"] == route["same_global_target"] and row["common_score"] == 0.0,
                f"Route cost, target pose, or scoring fee changed: {row['route_id']}")
        if row["route_id"] != "native":
            require(row["novel_view_tags"] and row["waypoint_pose"] in states[1:-1]
                    and route["same_global_target"] not in states[:-1]
                    and len(actions) + row["reserved_return_cost"] <= route["remaining_budget"]
                    and len(actions) <= native["outbound_cost"] + cfg["max_extra_outbound_actions"],
                    f"Alternative action trace fails one-waypoint/gain/budget conditions: {row['route_id']}")
            enough = row["unknown_union_cells"] + 1e-12 >= cfg["min_unknown_union_ratio"] * native["unknown_union_cells"]
            require(enough == row["admitted"], f"Original coverage admission changed: {row['route_id']}")
        require(set(row["scores"]) == ({"G", "S"} if row["admitted"] else set()),
                f"Scored/excluded candidate identity differs: {row['route_id']}")
    for mode in ("G", "S"):
        require(all(math.isfinite(row["scores"][mode]) and row["scores"][mode] >= 0 for row in admitted),
                f"Invalid original score on mode {mode}")
        best = max(row["scores"][mode] for row in admitted)
        near = [row for row in admitted if best - row["scores"][mode] <= cfg["score_equality_tolerance"]]
        expected = min(near, key=lambda row: (row["outbound_cost"], row["route_id"] != "native", row["outbound_sha256"]))
        require(core["hypothetical_selected_route_id"][mode] == expected["route_id"], "Original mode winner changed")
    selected = next(row for row in admitted if row["route_id"] == core["selected_route_id"])
    require(route["active_local_route_id"] == selected["route_id"]
            and core["chosen_outbound_sha256"] == route["active_outbound_sha256"] == selected["outbound_sha256"]
            and route["selected_outbound_actions"] == selected["outbound_actions"]
            and route["selected_outbound_states"] == selected["outbound_states"], "Published actuator bind changed")
    return native, admitted


def verify_trace(data, run, cap):
    trace = read(run / "steps.json")
    require(len(trace) == data["paid_actions"] == cap and data["returned"] and data["collisions"] == 0,
            f"Published episode lacks native stop/return: {run}")
    packet = data["initial_packet_sha256"]
    chain = hashlib.sha256(bytes(32) + bytes.fromhex(packet)).hexdigest()
    pose = data["initial_pose"]
    for index, step in enumerate(trace, start=1):
        packet = step["packet_sha256"]
        chain = hashlib.sha256(bytes.fromhex(chain) + bytes.fromhex(packet)).hexdigest()
        destination = [*step["position"], step["heading"]]
        require(step["step"] == index and step["packet_chain_sha256"] == chain
                and step["guard_allowed"] and not step["collision"] and next_pose(pose, step["action"]) == destination,
                f"Historical command/pose/packet path failed at {run}:{index}")
        pose = destination
    require(data["final_packet_chain_sha256"] == chain and pose == data["final_pose"] == data["initial_pose"],
            f"Terminal pose or packet differs from archived return: {run}")
    return trace


def verify_arms(layout, cap, index):
    records, bindings = {}, {}
    for arm in ARMS:
        name = f"runner/runs/development/{layout}/{arm}/manifest.json"
        path = SOURCE / name
        hashes = index[name]
        verified_manifest = verify(path, hashes["manifest_sha256"])
        manifest = read(path)
        require(manifest["files_sha256"] == hashes["files_sha256"], f"Published source manifest list changed: {name}")
        for filename, checksum in hashes["files_sha256"].items():
            verify(path.parent / filename, checksum)
        require(manifest["status"] == "completed_execution" and manifest["arm"] == arm and manifest["layout"] == layout
                and manifest["global_mode"] == "G" and manifest["local_mode"] == arm[-1]
                and manifest["actual_closed_loop"] and not manifest["forced_route_or_goal"]
                and not manifest["evaluator_inputs_read_by_actor"], f"Archived actor identity differs: {name}")
        trace = verify_trace(manifest, path.parent, cap)
        with np.load(path.parent / "observed_belief_trace.npz", allow_pickle=False) as raw:
            require(list(raw["paid"]) == list(range(cap + 1)), f"Belief ledger differs from paid actions: {name}")
            belief = raw["belief"]
        records[arm] = {"manifest": manifest, "trace": trace, "belief": belief,
                        "local": read(path.parent / "local_choices.json"), "global": read(path.parent / "global_choices.json")}
        bindings[arm] = {"manifest_path": str(path), "manifest_sha256": verified_manifest,
                         "files_sha256": hashes["files_sha256"]}
    g, s = (records[arm] for arm in ARMS)
    require(g["global"] == s["global"] and np.array_equal(g["belief"], s["belief"])
            and all({k: v for k, v in x.items() if k != "decision_wall_s"}
                    == {k: v for k, v in y.items() if k != "decision_wall_s"}
                    for x, y in zip(g["trace"], s["trace"], strict=True))
            and g["manifest"]["initial_pose"] == s["manifest"]["initial_pose"]
            and g["manifest"]["initial_packet_sha256"] == s["manifest"]["initial_packet_sha256"]
            and g["manifest"]["files_sha256"]["terminal/native_mesh.ply"]
            == s["manifest"]["files_sha256"]["terminal/native_mesh.ply"],
            f"Historical G/S trajectories already diverged in {layout}")
    return records, bindings


def normalize_local(row):
    return {k: v for k, v in row.items() if k not in ("local_mode", "choice_audit")} | {
        "choice_audit": {k: v for k, v in row["choice_audit"].items() if k != "selected_mode"}
    }


def bind_layout(layout, prior_config, index):
    cap = prior_config["development"]["paid_action_budget"]
    records, bindings = verify_arms(layout, cap, index)
    g, s = (records[arm] for arm in ARMS)
    pairs = list(zip(g["local"], s["local"], strict=True))
    expected_global = {event["call_id"]: event for event in g["global"]}
    require(len(expected_global) == len(g["global"]), f"Global selection IDs are ambiguous for {layout}")
    states, matched_keys = [], set()
    for ordinal, (neutral, semantic) in enumerate(pairs):
        core = neutral["choice_audit"]
        paid = neutral["paid_before"]
        option_id = neutral["option_id"]
        selection_id = neutral["selection_call_id"]
        frame = g["trace"][paid - 1] if paid else None
        initial = g["manifest"]
        packet = frame["packet_sha256"] if frame else initial["initial_packet_sha256"]
        paid_pose = [*frame["position"], frame["heading"]] if frame else initial["initial_pose"]
        require(0 <= paid < cap and normalize_local(neutral) == normalize_local(semantic)
                and neutral["local_mode"] == core["selected_mode"] == "G"
                and semantic["local_mode"] == semantic["choice_audit"]["selected_mode"] == "S"
                and core["paid_before"] == paid and core["packet_sha256"] == packet == neutral["packet_sha256"]
                and core["belief_sha256"] == neutral["observed_belief_sha256"] == array_digest(g["belief"][paid])
                and neutral["pose"] == paid_pose and neutral["remaining_budget"] == cap - paid,
                f"Different observations or local audits in archived pair {layout}:{ordinal}")
        native_call = expected_global[selection_id]
        native = native_call["outputs"]["selected"]
        require(native_call["action_id"] == paid and native_call["outputs"]["mode"] == "G"
                and native["option_id"] == option_id == core["native_option_id"]
                and native["pose"] == neutral["same_global_target"] == core["native_goal_pose"]
                and native["outbound_actions"] == neutral["native_outbound_actions"]
                and native["outbound_states"] == neutral["native_outbound_states"]
                and native["return_actions"] == neutral["native_return_actions"]
                and native["return_states"] == neutral["native_return_states"],
                f"Archived local event falsely reports unchanged global target: {layout}:{ordinal}")
        state_key = (paid, option_id, selection_id, packet)
        require(state_key not in matched_keys, f"Duplicate archived decision key {layout}:{ordinal}")
        matched_keys.add(state_key)
        native_candidate, admitted = baseline_candidate_pool(core, neutral, prior_config["local_route"])
        if len(admitted) <= 1:
            continue
        states.append({
            "layout": layout, "ordinal": ordinal, "paid_before": paid, "global_call_id": selection_id,
            "native_option_id": option_id, "packet_sha256": packet, "belief_sha256": core["belief_sha256"],
            "observed_pose": paid_pose, "goal_pose": core["native_goal_pose"], "return_anchor": core["return_anchor"],
            "original_row_G_sha256": digest(neutral), "original_row_S_sha256": digest(semantic),
            "original_core_G_sha256": digest(core), "original_core_S_sha256": digest(semantic["choice_audit"]),
            "original_pool_sha256": core["candidate_pool_sha256"], "original_scores_sha256": core["score_table_sha256"],
            "native_outbound_sha256": native_candidate["outbound_sha256"],
            "native_outbound_cost": native_candidate["outbound_cost"],
            "native_return_cost": native_candidate["reserved_return_cost"], "original_admitted_candidates": len(admitted),
            "original_winners": core["hypothetical_selected_route_id"], "source_manifest_G_sha256": bindings["newG"]["manifest_sha256"],
            "source_manifest_S_sha256": bindings["newS"]["manifest_sha256"],
        })
    return states, bindings, len(pairs)


def main(output):
    source = {name: sha_file(SOURCE / name) for name in (
        "publication/receipt.json", "EVIDENCE_INDEX.json", "PROTOCOL.json", "runner/development_freeze.json",
        "measurement/development_decision.json", "completion_audit.json", "route_choice.py",
    )}
    require(read(SOURCE / "publication/receipt.json")["head"] == COMMIT, "Unexpected published evidence commit")
    prior_config = read(SOURCE / "PROTOCOL.json")
    frozen = read(SOURCE / "runner/development_freeze.json")
    require(frozen["status"] == "frozen_before_any_new_paid_episode", "Original actor configuration was not frozen")
    verify(SOURCE / "route_choice.py", frozen["files_sha256"][str(SOURCE / "route_choice.py")])
    verify(SOURCE / "PROTOCOL.json", frozen["files_sha256"][str(SOURCE / "PROTOCOL.json")])
    index = {row["manifest"]: row for row in read(SOURCE / "EVIDENCE_INDEX.json")["run_files_rooted_at_each_manifest"]}
    rows, bindings = [], {}
    total_pairs = 0
    for layout in prior_config["development"]["layouts"]:
        states, per_layout, count = bind_layout(layout, prior_config, index)
        rows.extend(states)
        bindings[layout] = per_layout
        total_pairs += count
    require(total_pairs == 116 and len(rows) == 25, "Independent original-pair enumeration differs from published evidence")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "archived_states.json", {"status": "verified_original_pairs", "source_commit": COMMIT,
              "paired_source_decisions": total_pairs, "eligible_original_states": len(rows),
              "source_study_files_sha256": source, "run_file_bindings": bindings, "rows": rows})
    write_csv(output / "archived_states.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    main(parser.parse_args().out)
