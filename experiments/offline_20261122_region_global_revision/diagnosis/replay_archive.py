#!/usr/bin/env python3
"""Verify archived evidence and replay the coverage-cap change on fixed states."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np


CASES = {
    "floor26_corridor": {
        "compare": "original_G",
        "decisions": (0, 23, 30, 51, 96),
        "clocks": (0, 19, 20, 23, 30, 33, 42, 48, 51, 60, 73, 96, 118, 131, 144, 192),
    },
    "floor26_loop": {
        "compare": "F_strict",
        "decisions": (0, 28, 59),
        "clocks": (0, 18, 24, 48, 52, 57, 59, 60, 66, 83, 92, 96, 104, 114, 122, 124, 125,
                   131, 141, 144, 192),
    },
}
REGIONAL = "region_S"
CAP_M2 = 2.0
GRID_M = 0.20
FIELDS = ("layout", "arm", "actual_paid", "correct_free_cells", "reachable_cells", "C_free",
          "F1_scene_05cm", "J")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_text(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def measure_candidate(layout, paid, row, doors, native, belief, occupied, reachable, visible_mask):
    pose = row["pose"]
    visible = visible_mask(belief == 1, pose[:2], pose[2], 40, 90) & (belief == -1)
    count = int(visible.sum())
    assert count == row["visible_unknown_cells"], (layout, paid, pose, count, row)

    # The optimistic mask gives unknown cells no occluding behavior. Restore
    # source obstacle occlusion only for this evaluator-side geometry check.
    true_los = visible_mask((belief == 1) | occupied, pose[:2], pose[2], 40, 90)
    true_count = int((true_los & reachable & visible).sum())
    entry = doors.get(row["entrance_id"])
    virgin = 1.25 * row["unvisited_region"]
    frontier = 0.25 * min(row["remaining_frontier_cells"] * GRID_M, 3.0)
    visible_area = count * GRID_M**2
    door = 0.35 * (entry is not None and not entry["crossed_paid_steps"])
    outbound, returning = row["outbound_actions"], row["return_actions"]
    travel = 0.0125 * (outbound + returning)
    revisit = 0.35 * row["previous_dispatches"]
    capped_area = min(CAP_M2, visible_area)
    old_score = virgin + frontier + capped_area + door - travel - revisit
    uncapped_score = virgin + frontier + visible_area + door - travel - revisit
    assert abs(old_score - row["score_m2"]) < 1e-12, (layout, paid, pose, old_score, row)
    native_candidate = native[tuple(pose)]
    return {
        "layout": layout,
        "paid_before": paid,
        "pose": ",".join(map(str, pose)),
        "candidate_id": native_candidate["candidate_id"],
        "origin": row["origin"],
        "region_id": row["region_id"],
        "entrance_id": row["entrance_id"],
        "visible_unknown_cells": count,
        "offline_geometry_los_reachable_unknown_cells": true_count,
        "remaining_frontier_cells": row["remaining_frontier_cells"],
        "outbound_actions": outbound,
        "return_actions": returning,
        "unvisited_region_bonus_m2": virgin,
        "frontier_bonus_m2": frontier,
        "capped_unknown_area_m2": capped_area,
        "uncapped_unknown_area_m2": visible_area,
        "uncrossed_entrance_bonus_m2": door,
        "roundtrip_cost_m2": travel,
        "repeat_goal_cost_m2": revisit,
        "recorded_score_m2": row["score_m2"],
        "uncapped_score_m2": uncapped_score,
        "native_full_route_coverage_proxy_C": native_candidate["coverage_proxy"],
        "native_joint_proxy_G": native_candidate["joint_proxy"],
        "native_predicted_laser_cells": native_candidate["predicted_radar_cells"],
        "native_predicted_camera_cells": native_candidate["predicted_camera_cells"],
    }


def decision_rows(layout, spec, base, run, visible_mask):
    region = {decision["paid_before"]: decision for decision in read(run / "regional_choices.json")}
    global_scores = {x["action_id"]: x["outputs"] for x in read(run / "global_choices.json")}
    with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as archive:
        beliefs = archive["belief"].copy()
    truth_path = base / "scenes/evaluator_references" / layout / "evaluator_truth.npz"
    with np.load(truth_path, allow_pickle=False) as truth:
        occupied = truth["occupancy"] == 1
        reachable = truth["reachable"].astype(bool)
    rows, overview = [], []
    for paid in spec["decisions"]:
        decision = region[paid]
        ranking = decision["ranking"]
        doors = {door["entrance_id"]: door for door in ranking["entrances"]}
        output = global_scores[paid]
        native = {
            tuple(candidate["pose"]): {
                "candidate_id": candidate["candidate_id"],
                "coverage_proxy": output["scores"]["C"][i],
                "joint_proxy": output["scores"]["G"][i],
                **{key: output["score_audit"][i][key]
                   for key in ("predicted_radar_cells", "predicted_camera_cells")},
            }
            for i, candidate in enumerate(output["candidates"])
        }
        ranked = [measure_candidate(layout, paid, r, doors, native, beliefs[paid],
                                    occupied, reachable, visible_mask) for r in ranking["ranking"]]
        assert ranked[0]["pose"] == ",".join(map(str, decision["actual_target_pose"]))
        revised = sorted(ranked, key=lambda r: (-r["uncapped_score_m2"],
                         r["outbound_actions"] + r["return_actions"], tuple(map(int, r["pose"].split(",")))))
        new_ranks = {r["pose"]: i for i, r in enumerate(revised, 1)}
        for old_rank, candidate in enumerate(ranked, 1):
            rows.append({**candidate, "recorded_rank": old_rank, "uncapped_rank": new_ranks[candidate["pose"]]})
        overview.append({
            "paid_before": paid,
            "robot_pose": decision["start_pose"],
            "old_ranked_count": len(ranked),
            "originally_clamped_count": sum(r["uncapped_unknown_area_m2"] > CAP_M2 for r in ranked),
            "native_shadow_pose": decision["native_shadow_pose"],
            "selected_old": ranked[0],
            "selected_uncapped_fixed_state": revised[0],
        })
    return rows, overview, truth_path


def interval(steps, initial_pose, correct, lo, hi):
    prior_cell = tuple(initial_pose[:2])
    edges = set()
    translations = repeats = 0
    for row in steps[:hi]:
        cell = tuple(row["position"])
        if cell == prior_cell:
            continue
        edge = tuple(sorted((cell, prior_cell)))
        if row["step"] >= lo:
            translations += 1
            repeats += edge in edges
        edges.add(edge)
        prior_cell = cell
    return {
        "paid_interval_inclusive": [lo, hi],
        "translation_steps": translations,
        "repeated_translating_edges": repeats,
        "net_new_correct_free_cells": correct[hi] - correct[lo - 1],
    }


def make(base, geometry_source):
    sys.path.insert(0, str(geometry_source))
    from utils.grid_geometry import visible_mask

    evaluated = base / "results/formal"
    summary_path = evaluated / "summary.json"
    summary = read(summary_path)
    freeze = base / "FORMAL_FREEZE.json"
    assert summary["run_freeze_sha256"] == digest(freeze)
    assert summary["complete_endpoints"] == 20
    geometry_hash = digest(geometry_source / "utils/grid_geometry.py")
    assert any(path.endswith("/utils/grid_geometry.py") and sha == geometry_hash
               for path, sha in read(freeze)["actor_files"].items())
    source_paths = [summary_path, freeze, evaluated / "checkpoints.csv", evaluated / "coverage_curve.csv",
                    evaluated / "room_crossings.csv"]
    checkpoints = read_csv(evaluated / "checkpoints.csv")
    coverage = read_csv(evaluated / "coverage_curve.csv")
    crossings = read_csv(evaluated / "room_crossings.csv")
    pair_reports = {}
    for layout, spec in CASES.items():
        arms = (spec["compare"], REGIONAL)
        pair_reports[layout] = {
            "arms": {
                arm: next(x for x in summary["episodes"] if (x["layout"], x["arm"]) == (layout, arm))
                for arm in arms
            },
            "first_pair_divergence": next(x for x in summary["pairs"]
                                          if x["layout"] == layout and {x["arm_a"], x["arm_b"]} == set(arms)),
            "first_G_S_divergence_while_regional": next(x for x in summary["pairs"]
                                          if (x["layout"], x["arm_a"], x["arm_b"]) ==
                                          (layout, "region_G", REGIONAL)),
        }
    ranking_rows, time_rows = [], []
    for layout, spec in CASES.items():
        arms = (spec["compare"], REGIONAL)
        entry = pair_reports[layout]
        entry["motion_and_coverage_intervals"] = {}
        truth = base / "scenes/evaluator_references" / layout / "evaluator_truth.npz"
        with np.load(truth, allow_pickle=False) as fixed:
            reachable = fixed["reachable"].astype(bool)
        for arm in arms:
            run = base / "execution/formal" / layout / arm / "attempt_001/run"
            manifest_path = run / "manifest.json"
            manifest = read(manifest_path)
            episode = entry["arms"][arm]
            assert digest(manifest_path) == episode["source_manifest_sha256"]
            assert episode["actual_paid"] == 192 and episode["returned"]
            source_paths.append(manifest_path)
            names = ["steps.json", "observed_belief_trace.npz"]
            if arm == REGIONAL:
                names.extend(("regional_choices.json", "global_choices.json"))
            for name in names:
                artifact = run / name
                assert digest(artifact) == manifest["files_sha256"][name]
                source_paths.append(artifact)
            if arm == REGIONAL:
                rows, choices, truth_path = decision_rows(layout, spec, base, run, visible_mask)
                ranking_rows.extend(rows)
                entry["fixed_state_decisions"] = choices
                source_paths.append(truth_path)
            actual = {int(x["paid"]): x for x in coverage if (x["layout"], x["arm"]) == (layout, arm)}
            clocks = {int(x["actual_paid"]): x for x in checkpoints
                      if (x["layout"], x["arm"]) == (layout, arm) and x["checkpoint"] != "terminal"}
            for paid in spec["clocks"]:
                free = actual[paid]
                row = {**free, **clocks.get(paid, {}), "actual_paid": paid}
                time_rows.append({field: row.get(field, "") for field in FIELDS})
            ranges = ((60, 83), (60, 96), (132, 192)) if layout.endswith("_loop") else ((20, 48), (49, 60))
            correct = {paid: int(data["correct_free_cells"]) for paid, data in actual.items()}
            with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as record:
                raw = np.count_nonzero((record["belief"] == 0) & reachable, axis=(1, 2))
                assert dict(zip(record["paid"], raw, strict=True)) == correct
            start = read(base / "scenes/prepared" / layout / "initial_state.json")["anchor"]
            entry["motion_and_coverage_intervals"][arm] = [
                interval(read(run / "steps.json"), start, correct, lo, hi) for lo, hi in ranges
            ]
        entry["true_room_crossings"] = [
            row for row in crossings if row["layout"] == layout and row["arm"] in arms
        ]
        source_paths.append(base / "scenes/prepared" / layout / "initial_state.json")
        for arm, episode in entry["arms"].items():
            keep = ("actual_paid", "returned", "collisions", "correct_free_cells", "reachable_cells",
                    "C_free", "F1_scene_05cm", "J", "A95_paid", "rooms_visited",
                    "repeat_distance_m", "guarded_macros", "dispatched_macros", "arrived_macros", "elapsed_s")
            entry["arms"][arm] = {key: episode[key] for key in keep if key in episode}
    receipt = {
        "source_study": base.name,
        "formal_freeze_sha256": digest(freeze),
        "geometry_utility_sha256": geometry_hash,
        "checked_input_sha256": {str(path.relative_to(base)): digest(path) for path in source_paths},
        "calculation": "same recorded pool and other terms, no observation or action replay",
        "true_los": "offline optimistic scan mask clipped by independently archived solid obstacle occlusion "
                    "and reference reachable free cells, at the recorded 8 m / 90 degree setting; "
                    "these cells are not actual future observations",
        "cases": pair_reports,
    }
    return {
        "fixed_state_candidates.csv": csv_text(ranking_rows),
        "recorded_clocks.csv": csv_text(time_rows),
        "archive_diagnosis.json": json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    analyses = here.parent.parent
    parser.add_argument("--archive", type=Path, default=analyses / "region_structure_floor_20261122")
    parser.add_argument("--geometry-source", type=Path, default=analyses /
                        "reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source")
    parser.add_argument("--output", type=Path, default=here)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = make(args.archive, args.geometry_source)
    for name, version in files.items():
        path = args.output / name
        if args.check:
            assert path.read_text(encoding="utf-8") == version, f"{path} differs from the archived evidence"
        else:
            with path.open("x", encoding="utf-8", newline="") as output:
                output.write(version)
    print(f"{'Verified' if args.check else 'Wrote'} {len(files)} diagnostic files from frozen archives")


if __name__ == "__main__":
    main()
