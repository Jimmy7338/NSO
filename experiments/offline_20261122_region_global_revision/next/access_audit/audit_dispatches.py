#!/usr/bin/env python3
"""Replay two independent scoring changes on frozen region_S dispatch pools."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path


GRID_M = 0.20
BUDGET = 192
LAYOUTS = ("floor26_ordinary", "floor26_branch", "floor26_corridor", "floor26_loop")
MODES = (
    "old", "target_region_only", "outbound_measured_only", "both",
    "pure_native_N", "pure_native_G",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode("utf-8")).hexdigest()


def dump_csv(rows):
    stream = io.StringIO()
    csv_writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
    csv_writer.writeheader()
    csv_writer.writerows(rows)
    return stream.getvalue()


def topology(room):
    return (
        1.25 * (room is not None and not room["visited"]),
        0.25 * min(3.0, GRID_M * room["remaining_frontier_cells"]) if room else 0.0,
    )


def audit_pool(uid, choice, native):
    view = choice["ranking"]
    rooms = {room["region_id"]: room for room in view["regions"]}
    doors = {door["entrance_id"]: door for door in view["entrances"]}
    cert = {tuple(row["pose"]): row for row in choice["certification"]}
    source_candidates = native["outputs"]["candidates"]
    scores = native["outputs"]["scores"]
    audit = native["outputs"]["score_audit"]
    assert scores.keys() >= {"C", "N", "G"}
    plans = {tuple(x["pose"]): (x, audit[i], i) for i, x in enumerate(source_candidates)}
    assert len(plans) == len(source_candidates)

    shadow = min(
        (i for i, c in enumerate(source_candidates) if not c.get("regional_proposal")),
        key=lambda i: (-scores["G"][i], source_candidates[i]["cost"], source_candidates[i]["candidate_id"]),
    )
    assert source_candidates[shadow]["pose"] == choice["native_shadow_pose"]

    rows = []
    for rank, record in enumerate(view["ranking"], 1):
        pose = tuple(record["pose"])
        candidate, raw, score_index = plans[pose]
        assert candidate["candidate_id"] == raw["candidate_id"]
        actual = cert.get(pose, {})
        proposal = actual.get("proposal") or candidate.get("regional_proposal") or {}
        standing = record["region_id"]
        hint = proposal.get("target_region_id")
        assert hint == record["target_region_hint"], (uid, pose, proposal, record)
        entry = doors.get(record["entrance_id"])
        assert record["entrance_id"] == proposal.get("entrance_id")
        explicit = proposal.get("proposal_kind") == "doorway_view" and hint in rooms
        use_region = hint if explicit else standing
        if explicit:
            assert entry and hint in entry["regions"]
        unvisited, frontier = topology(rooms.get(standing))
        opposite_unvisited, opposite_frontier = topology(rooms.get(use_region))
        assert bool(unvisited) == record["unvisited_region"]
        visible = record["visible_unknown_cells"]
        clipped = min(visible * GRID_M**2, 2.0)
        entrance = 0.35 * (entry is not None and not entry["crossed_paid_steps"])
        cost = 0.0125 * candidate["cost"]
        retry = 0.35 * record["previous_dispatches"]
        assert candidate["cost"] == record["outbound_actions"] + record["return_actions"]
        assert 0 < candidate["outbound_cost"] <= candidate["cost"] <= BUDGET - choice["paid_before"]
        assert candidate["outbound_cost"] == len(candidate["outbound_actions"])
        assert candidate["pose"] == candidate["outbound_states"][-1]
        coverage, measured = scores["C"][score_index], raw["quality_total_proxy"]
        native_n, native_g = scores["N"][score_index], scores["G"][score_index]
        assert abs(coverage + measured - native_n) < 1e-10
        assert native_n == raw["common_total_proxy"]
        assert abs(native_n + raw["template_expected_visible_area_m2"]["G"] - native_g) < 1e-10
        old = unvisited + frontier + clipped + entrance - cost - retry
        assert abs(old - record["score_m2"]) < 1e-12
        rows.append({
            "state_uid": uid, "old_rank": rank, "candidate_id": candidate["candidate_id"],
            "pose": ",".join(map(str, pose)), "origin": record["origin"],
            "proposal_kind": proposal.get("proposal_kind"),
            "standing_region": standing, "explicit_target_hint": hint, "effective_target_region": use_region,
            "entrance_id": record["entrance_id"], "standing_visited": not bool(unvisited) if standing in rooms else None,
            "target_visited": not bool(opposite_unvisited) if use_region in rooms else None,
            "standing_frontier_cells": rooms.get(standing, {}).get("remaining_frontier_cells", 0),
            "target_frontier_cells": rooms.get(use_region, {}).get("remaining_frontier_cells", 0),
            "visible_unknown_endpoint_cells": visible, "clipped_endpoint_area_m2": clipped,
            "outbound_actions": candidate["outbound_cost"], "return_actions": candidate["return_cost"],
            "roundtrip_actions": candidate["cost"], "native_C_full_route": coverage, "quality_total_proxy": measured,
            "native_N_measured": native_n, "native_G_with_template": native_g,
            "roundtrip_penalty": cost, "repeat_goal_penalty": retry, "uncrossed_entrance_bonus": entrance,
            "old": record["score_m2"],
            "target_region_only": opposite_unvisited + opposite_frontier + clipped + entrance - cost - retry,
            "outbound_measured_only": unvisited + frontier + native_n + entrance - cost - retry,
            "both": opposite_unvisited + opposite_frontier + native_n + entrance - cost - retry,
            "pure_native_N": native_n, "pure_native_G": native_g,
            "outbound_orientation_route_sha256": stable_hash(candidate["outbound_states"]),
            "outbound_action_sequence_sha256": stable_hash(candidate["outbound_actions"]),
        })
    return rows


def order(pool, mode):
    if mode in {"pure_native_N", "pure_native_G"}:
        return sorted(pool, key=lambda row: (-row[mode], row["roundtrip_actions"], row["candidate_id"]))
    return sorted(pool, key=lambda row: (-row[mode], row["roundtrip_actions"], tuple(map(int, row["pose"].split(",")))))


def make(base):
    decisions_path = base / "results/formal/decisions.csv"
    summary_path = base / "results/formal/summary.json"
    freeze_path = base / "FORMAL_FREEZE.json"
    products = [decisions_path, summary_path, freeze_path, base / "policy/observed_regions.py",
                base / "regional_adapter.py"]
    decisions = [r for r in read_csv(decisions_path) if r["arm"] == "region_S" and r["first_dispatched_paid"]]
    summary = read(summary_path)
    freeze = read(freeze_path)
    assert summary["run_freeze_sha256"] == sha(freeze_path)
    assert summary["complete_endpoints"] == 20
    for item in products[3:]:
        assert sha(item) == freeze["actor_files"][str(item)]

    states, candidates, selected = [], [], []
    for layout in LAYOUTS:
        run = base / "execution/formal" / layout / "region_S/attempt_001/run"
        manifest_path = run / "manifest.json"
        manifest = read(manifest_path)
        episode = next(e for e in summary["episodes"] if e["layout"] == layout and e["arm"] == "region_S")
        assert sha(manifest_path) == episode["source_manifest_sha256"]
        products.append(manifest_path)
        for name in ("regional_choices.json", "global_choices.json", "steps.json"):
            path = run / name
            assert sha(path) == manifest["files_sha256"][name]
            products.append(path)
        regional = {r["selection_call_id"]: r for r in read(run / "regional_choices.json")}
        native = {r["call_id"]: r for r in read(run / "global_choices.json")}
        steps = {r["step"]: r for r in read(run / "steps.json")}
        assert len(regional) == len(native) == episode["global_decisions"] == episode["dispatched_macros"]
        for r in (row for row in decisions if row["layout"] == layout):
            paid, call = int(r["selected_paid_before"]), int(r["selection_call_id"])
            uid = f"{layout}/region_S/paid-{paid:03d}/call-{call}"
            command = steps[int(r["first_dispatched_paid"])]
            choice, source = regional[call], native[call]
            assert source["action_id"] == choice["paid_before"] == paid
            assert command["selection_call_id"] == call and command["active_option_id"] == r["option_id"]
            assert command["step"] == paid + 1
            assert choice["actual_target_pose"] == command["selected_target_pose"] == json.loads(r["target_pose"])
            assert r["selection_origin"] == choice["decision_origin"]
            fallback = choice["decision_origin"] == "native_fallback"
            pool = audit_pool(uid, choice, source)
            assert fallback == (not pool)
            go, back = int(r["planned_go"]), int(r["planned_return"])
            slack = BUDGET - paid - go - back
            assert slack >= 0
            old_pose = ",".join(map(str, choice["actual_target_pose"]))
            shadow_pose = ",".join(map(str, choice["native_shadow_pose"]))
            assert (old_pose != shadow_pose) == (r["changed_from_own_native_shadow"] == "True")
            assert not pool or order(pool, "old") == pool
            if pool:
                assert pool[0]["pose"] == old_pose
                for mode in MODES:
                    for nrank, candidate in enumerate(order(pool, mode), 1):
                        candidate[f"rank_{mode}"] = nrank
                candidates.extend(pool)
            row = {
                "state_uid": uid, "layout": layout, "paid_before": paid, "selection_call_id": call,
                "option_id": r["option_id"], "first_dispatched_paid": command["step"],
                "first_action": command["action"], "phase_before_action": command["phase_before_action"],
                "decision_origin": r["selection_origin"], "decision_reason": r["regional_reason"],
                "overrode_native_shadow": old_pose != shadow_pose, "recorded_pose": old_pose,
                "native_shadow_pose": shadow_pose, "available_paid": BUDGET - paid, "outbound_actions": go,
                "return_actions": back, "remaining_after_go_and_return": slack,
                "exact_budget_one_step_fallback": fallback and go == 1 and slack == 0,
                "all_logged_candidates": len(source["outputs"]["candidates"]),
                "historically_qualified_pool": len(pool), "proposal_count": choice["raw_proposal_count"],
                "certification_statuses": json.dumps(dict(sorted(Counter(
                    item["status"] for item in choice["certification"]).items())), separators=(",", ":")),
                "runtime_dispatch_packet_sha256": command["packet_sha256"],
                "archived_global_outputs_sha256": source["output_sha256"],
                "steps_archive": (run / "steps.json").relative_to(base).as_posix(),
                "regional_choice_archive": (run / "regional_choices.json").relative_to(base).as_posix(),
                "native_score_archive": (run / "global_choices.json").relative_to(base).as_posix(),
            }
            for mode in MODES:
                winner = next((c for c in pool if c[f"rank_{mode}"] == 1), None)
                row[f"{mode}_pose"] = winner["pose"] if winner else old_pose
                row[f"{mode}_route_differs"] = bool(winner and (
                    winner["outbound_orientation_route_sha256"] != pool[0]["outbound_orientation_route_sha256"]))
                if winner:
                    selected.append({
                        "state_uid": uid, "mode": mode, "pose": winner["pose"],
                        "candidate_id": winner["candidate_id"],
                        "score": winner[mode], "outbound_actions": winner["outbound_actions"],
                        "return_actions": winner["return_actions"], "origin": winner["origin"],
                        "changed_pose": winner["pose"] != old_pose,
                    })
            states.append(row)
    counts = {
        "all_dispatches": len(states),
        "qualified": sum(bool(r["historically_qualified_pool"]) for r in states),
        "overrides": sum(r["overrode_native_shadow"] for r in states),
        "fallbacks": sum(not r["historically_qualified_pool"] for r in states),
        "exact_budget_one_step_fallbacks": sum(r["exact_budget_one_step_fallback"] for r in states),
    }
    assert tuple(counts.values()) == (133, 17, 15, 116, 93)
    measured_counts = {
        mode: sum(r[f"{mode}_route_differs"] for r in states)
        for mode in MODES
    }
    compare_first = next(r for r in states if r["layout"] == "floor26_corridor" and r["paid_before"] == 0)
    compare_loop = next(r for r in states if r["layout"] == "floor26_loop" and r["paid_before"] == 59)
    earliest_target = min(
        (r for r in states if r["target_region_only_route_differs"]),
        key=lambda r: (r["paid_before"], LAYOUTS.index(r["layout"])),
    )
    fixed_branch_candidates = {}
    for decision in (compare_first, compare_loop, earliest_target):
        uid = decision["state_uid"]
        branches = [r for r in selected if r["state_uid"] == uid and r["mode"] in MODES[:4]]
        unique = {}
        for branch in branches:
            cell = unique.setdefault(branch["pose"], {"pose": branch["pose"],
                                     "candidate_id": branch["candidate_id"], "score_modes": []})
            cell["score_modes"].append(branch["mode"])
        fixed_branch_candidates[uid] = list(unique.values())
    report = {
        "scope": "frozen region_S factual decision states, candidate scores only, no new sensor/trajectory outcomes",
        "counts": counts, "changed_route_counts_on_17_eligible_states": measured_counts,
        "fallback_counts_by_floor": {
            layout: dict(Counter(
                ("exact_budget_one_step" if r["exact_budget_one_step_fallback"] else "other_fallback")
                for r in states if r["layout"] == layout and not r["historically_qualified_pool"]
            )) for layout in LAYOUTS
        },
        "fixed_branch_candidates": fixed_branch_candidates,
        "recorded_stage": {"selection": "old ranking eligibility", "execution": "actual post-dispatch steps"},
        "formulas": {
            "old": "standing topology + capped endpoint area + entrance - costs",
            "target_region_only": "explicit valid doorway-destination topology replaces standing topology",
            "outbound_measured_only": "old formula with capped endpoint area replaced by native N = C + quality_total_proxy",
            "both": "combine the target-region and outbound-measured changes",
            "pure_native_N": "native full-route C + measured-surface quality; control with native ties",
            "pure_native_G": "native measured subtotal with original neutral template term; control with native ties",
        },
        "fallback": "no qualified archived regional-ranked candidate. Existing native choice used for all versions",
        "source_sha256": {str(path.relative_to(base)): sha(path) for path in products},
    }
    return {
        "dispatches.csv": dump_csv(states),
        "candidate_scores.csv": dump_csv(candidates),
        "selected_routes.csv": dump_csv(selected),
        "receipt.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--archive", type=Path, default=here.parents[2] / "region_structure_floor_20261122")
    parser.add_argument("--output", type=Path, default=here)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = make(args.archive)
    for name, content in files.items():
        path = args.output / name
        if args.check:
            assert path.read_text(encoding="utf-8") == content, path
        else:
            with path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(content)
    print(f"{'Verified' if args.check else 'Wrote'} {len(files)} files")


if __name__ == "__main__":
    main()
