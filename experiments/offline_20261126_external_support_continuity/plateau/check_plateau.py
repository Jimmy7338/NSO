"""Audit HM3D 00770 start 1 from sealed records without running an actor."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "external_support_repair_20261126"
LAYOUT = "b26_hm3d_00770_s1"
ARMS = ("region_S", "combined_S", "F_strict")
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
CROSS = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
ANCHOR = (39, 17)
FOOTPRINT_CELL = (46, 25)
BLOCKING_CELL = (47, 26)
FIRST_PLATEAU_PLAN = 38


def read_json(path):
    return json.loads(path.read_bytes())


def inflated(obstacles):
    """Frozen runtime raster inflation with radius equal to one cell."""
    padded = np.pad(~obstacles, 1, constant_values=False)
    return distance_transform_edt(padded)[1:-1, 1:-1] <= 1.0 + np.sqrt(2.0) / 2.0


def true_paths(edges, start):
    width = edges.shape[1]
    source, target = [], []
    for heading, (dr, dc) in enumerate(DIRECTIONS):
        for row, col in np.argwhere(edges[:, :, heading]):
            source.append(row * width + col)
            target.append((row + dr) * width + col + dc)
    graph = coo_matrix((np.ones(len(source)), (source, target)), shape=(edges.shape[0] * width,) * 2).tocsr()
    return dijkstra(graph, directed=True, indices=start[0] * width + start[1]).reshape(edges.shape[:2])


def frontier_counts(belief, position):
    traversable = (belief == 0) & ~inflated(belief == 1)
    traversable[position] = True
    observed, _ = label(traversable, structure=CROSS)
    component = observed == observed[position]
    frontiers = component & binary_dilation(belief == -1, structure=CROSS)
    return int(component.sum()), int(frontiers.sum())


def run_stats(arm, process, coverage, truth):
    steps = process["steps"]
    belief = process["belief"]
    only_changes = [
        row for j, row in enumerate(coverage)
        if j == 0 or row["correct_free_cells"] != coverage[j - 1]["correct_free_cells"]
    ]
    final = coverage[-1]
    summary = {
        "coverage_changes": only_changes, "terminal_coverage": final,
        "runtime_phase_counts": Counter(step["phase_before_action"] for step in steps),
        "collisions": sum(step["collision"] for step in steps),
        "matched_target_arrivals": sum(step["matched_target_pose_after_step"] is True for step in steps),
    }

    if arm == "F_strict":
        frontier_rows = []
        for paid in range(6, 16):
            position = tuple(steps[paid - 1]["position"])
            connected, frontiers = frontier_counts(belief[paid], position)
            frontier_rows.append({
                "paid": paid, "position": position, "connected_centres": connected,
                "frontier_cells": frontiers,
            })
        summary["connected_frontiers"] = frontier_rows
        summary["frontier_goal_poses"] = sorted({
            tuple(event["goal_pose"]) for event in process["frontier_events"] if "goal_pose" in event
        })
        summary["matched_target_arrivals"] = sum(
            (*step["position"], step["heading"]) in summary["frontier_goal_poses"] for step in steps
        )
        summary["frontier_stop"] = [event for event in process["frontier_events"] if event["event"] == "frontier_policy_stop"]
        summary["map_changes_when_frontier_component_disappears"] = [
            {
                "cell": [int(x) for x in cell], "before": int(belief[7][tuple(cell)]),
                "after": int(belief[8][tuple(cell)]),
                "evaluator_occupied": bool(truth["occupancy"][tuple(cell)]),
            }
            for cell in np.argwhere(belief[7] != belief[8])
        ]
        counterfactual = belief[8].copy()
        first_cut = (52, 24)
        counterfactual[first_cut] = belief[7][first_cut]
        connected, frontiers = frontier_counts(counterfactual, tuple(steps[7]["position"]))
        summary["offline_single_cell_counterfactual_at_paid_8"] = {
            "cell_restored_to_its_paid_7_label": first_cut, "connected_centres": connected,
            "frontier_cells": frontiers, "actor_executed": False,
        }
        return summary

    post_steps = [step for step in steps if step["step"] >= FIRST_PLATEAU_PLAN]
    late_steps = [step for step in post_steps if step["step"] > 52]
    regional = [item for item in process["regional_choices"] if item["paid_before"] >= FIRST_PLATEAU_PLAN]
    clocks = {item["paid_before"] for item in regional}
    native = [item for item in process["global_choices"] if item["action_id"] in clocks]
    score_audits = [score for choice in native for score in choice["outputs"]["score_audit"]]
    assert len(native) == len(regional)
    assert all(item["actual_target_pose"] == item["native_shadow_pose"] for item in regional)
    assert all(choice["outputs"]["scores"]["N"] == choice["outputs"]["scores"]["G"]
               == choice["outputs"]["scores"]["S"] for choice in native)
    first = regional[0]
    observed = belief[first["paid_before"]]
    safe = ~inflated(observed != 0)
    components, _ = label(safe, structure=CROSS)
    start = tuple(first["start_pose"][:2])
    occupied_only = (observed == 0) & ~inflated(observed == 1)
    occupied_components, _ = label(occupied_only, structure=CROSS)
    first_cell_only = observed.copy()
    first_cell_only[BLOCKING_CELL] = 0
    first_cell_safe = ~inflated(first_cell_only != 0)
    first_cell_components, _ = label(first_cell_safe, structure=CROSS)
    physical_cost = true_paths(truth["edge_valid"], start)
    proposals = []
    for item in first["certification"]:
        row = item["proposal"]
        pos = tuple(row["pose"][:2])
        proposals.append(row | {
            "certification_status": item["status"], "observed_component": int(components[pos]),
            "occupied_inflation_only_component": int(occupied_components[pos]),
            "simulator_swept_steps_lower_bound_excluding_turns": int(physical_cost[pos]),
        })
    summary.update({
        "last_covering_pose": [*steps[36]["position"], steps[36]["heading"]],
        "guard_blocks": process["regional_guards"],
        "plateau_action_counts": Counter(step["action"] for step in post_steps),
        "late_position_visit_counts": [
            [list(pos), n] for pos, n in Counter(tuple(step["position"]) for step in late_steps).most_common()
        ],
        "plateau_selected_target_counts": [
            [list(pose), n] for pose, n in Counter(tuple(item["actual_target_pose"]) for item in regional).most_common()
        ],
        "matched_target_arrivals_on_plateau": sum(step["matched_target_pose_after_step"] for step in post_steps),
        "regional_post_plateau_origins": Counter(item["decision_origin"] for item in regional),
        "regional_certification_statuses": Counter(row["status"] for item in regional for row in item["certification"]),
        "native_candidates": len(score_audits),
        "native_positive_quality_proxy_count": sum(row["quality_total_proxy"] > 0 for row in score_audits),
        "native_zero_radar_camera_coverage_prior_count": sum(
            row["predicted_radar_cells"] == row["predicted_camera_cells"] == row["coverage_only_proxy"] == 0
            and not any(row["prior_potentials"].values()) for row in score_audits
        ),
        "first_plateau_dispatch": {
            "paid_before": first["paid_before"], "start_pose": first["start_pose"],
            "actual_target_pose": first["actual_target_pose"], "remaining_paid": 192 - first["paid_before"],
            "start_component": int(components[start]), "anchor_component": int(components[ANCHOR]),
            "component_cells": int(np.count_nonzero(components == components[start])),
            "occupied_inflation_only_start_component": int(occupied_components[start]),
            "native_feasible_oriented_poses": native[0]["outputs"]["candidate_audit"]["feasible_states"],
            "rejected_proposals": proposals,
        },
        "first_forward_edge_outside_observed_component": {
            "position": start, "destination": FOOTPRINT_CELL,
            "destination_observed_label": int(observed[FOOTPRINT_CELL]),
            "destination_after_inflation_safe": bool(safe[FOOTPRINT_CELL]),
            "physical_swept_edge_bidirectionally_valid": bool(
                truth["edge_valid"][start + (2,)] and truth["edge_valid"][FOOTPRINT_CELL + (0,)]
            ),
            "one_cell_copy_makes_destination_safe": bool(first_cell_safe[FOOTPRINT_CELL]),
            "one_cell_copy_restores_proposals": sum(
                int(first_cell_components[tuple(row["pose"][:2])]) == int(first_cell_components[start])
                for row in proposals
            ),
            "blocking_diagonal_neighbor": BLOCKING_CELL,
            "blocking_diagonal_neighbor_observed_label": int(observed[BLOCKING_CELL]),
            "blocking_diagonal_neighbor_evaluator_occupancy": bool(truth["occupancy"][BLOCKING_CELL]),
            "blocking_diagonal_label_history": [
                [i, int(value)] for i, value in enumerate(belief[:, BLOCKING_CELL[0], BLOCKING_CELL[1]])
                if i == 0 or value != belief[i - 1, BLOCKING_CELL[0], BLOCKING_CELL[1]]
            ],
            "true_reachable_unknown": int(np.count_nonzero(truth["reachable"] & (observed == -1))),
        },
    })
    return summary


def main():
    audit = {"layout": LAYOUT, "source_sha256": {}, "arms": {}}
    source_paths = []
    coverage_path = ROOT / "results/formal/coverage_curve.csv"
    with coverage_path.open() as handle:
        coverage = [row for row in csv.DictReader(handle) if row["layout"] == LAYOUT]
    source_paths.append(coverage_path)
    truth_path = ROOT / "scenes/evaluator_references" / LAYOUT / "evaluator_truth.npz"
    source_paths.append(truth_path)
    traces = {}
    for arm in ARMS:
        path = ROOT / "execution/formal" / LAYOUT / arm / "attempt_001/run"
        keys = ("steps", "frontier_events") if arm == "F_strict" else (
            "steps", "regional_choices", "regional_guards", "global_choices",
        )
        process = {key: read_json(path / f"{key}.json") for key in keys}
        source_paths.extend(path / f"{key}.json" for key in keys)
        belief_path = path / "observed_belief_trace.npz"
        source_paths.append(belief_path)
        with np.load(belief_path, allow_pickle=False) as data:
            assert np.array_equal(data["paid"], np.arange(len(process["steps"]) + 1))
            process["belief"] = data["belief"]
        traces[arm] = [[step[key] for key in ("step", "position", "heading", "action", "packet_sha256")]
                      for step in process["steps"]]
        arm_coverage = [
            {"paid": int(row["paid"]), "correct_free_cells": int(row["correct_free_cells"]),
             "reachable_cells": int(row["reachable_cells"]), "coverage": float(row["C_free"])}
            for row in coverage if row["arm"] == arm
        ]
        with np.load(truth_path, allow_pickle=False) as truth:
            audit["arms"][arm] = run_stats(arm, process, arm_coverage, truth)
    audit["two_regional_arms_same_actions_poses_and_packets"] = traces["region_S"] == traces["combined_S"]
    assert audit["two_regional_arms_same_actions_poses_and_packets"]
    audit["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
    }
    (HERE / "hm3d_00770_s1_plateau.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
