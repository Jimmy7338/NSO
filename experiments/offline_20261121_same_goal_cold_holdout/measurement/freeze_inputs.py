"""Seal evaluator-only references from the published four-building cold-start design."""

import ast
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
SOURCE = STUDY.parent / "same_goal_route_choice_20261121"
SOURCE_WITNESSES = STUDY.parent / "same_goal_two_witness_20261121"
HOLDOUT = SOURCE / "holdout"
SOURCE_RECON = STUDY.parent / (
    "reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source"
)
SOURCE_LARGE = STUDY.parent / "semantic_large_scene_20261121"
SOURCE_NATURAL = STUDY.parent / "semantic_scene_redesign_20261119"
SOURCE_B11 = STUDY.parent / "paired_unseen_gain_20261120"
OUTPUT = Path(__file__).resolve().parent
FIELDS = ("layout", "topology", "rooms", "doors", "room_connections", "gross_floor_area_m2", "grid_resolution_m",
          "reachable_cells", "reachable_area_m2", "scan_range_m", "image_width_px", "image_height_px", "camera_fov_deg",
          "robot_radius_m", "initial_pose", "scene_seed", "baseline_native_budget", "reference_scene_samples",
          "reference_shelf_samples", "reference_cabinet_samples")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bind(path, ledger, required=None):
    checksum = sha(path)
    if required is not None:
        require(checksum == required, f"Source artifact changed: {path}")
    ledger[str(path)] = checksum
    return checksum


def published_geometry_specs(path):
    module = ast.parse(path.read_text())
    assignments = [node.value for node in module.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "ORDINARY" for t in node.targets)]
    require(len(assignments) == 1 and isinstance(assignments[0], ast.Dict),
            "Published ordinary rooms were not defined in the static scene contract")
    specs = {}
    for name, value in zip(assignments[0].keys, assignments[0].values, strict=True):
        require(isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "dict"
                and not value.args, "Ordinary layout scene definition changed")
        specs[ast.literal_eval(name)] = {field.arg: ast.literal_eval(field.value) for field in value.keywords}
    return specs


def topology_from_reference(reachable, spec, cell_m):
    height, width = reachable.shape
    yy = (height - np.arange(height) - .5) * cell_m
    xx = (np.arange(width) + .5) * cell_m
    room_codes = (2 * (yy >= spec["split_y"])[:, None] + (xx >= spec["split_x"])[None, :]).astype(int)
    room_names = ("A_southwest", "B_southeast", "C_northwest", "D_northeast")
    require({int(item) for item in room_codes[reachable]} == set(range(4)),
            "Frozen reference lacks start-reachable space in one room")
    pairs = set()
    for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        crossing = reachable[a] & reachable[b] & (room_codes[a] != room_codes[b])
        pairs.update(tuple(sorted((room_names[i], room_names[j])))
                     for i, j in zip(room_codes[a][crossing], room_codes[b][crossing], strict=True))
    expected = {("A_southwest", "B_southeast"), ("B_southeast", "D_northeast"), ("C_northwest", "D_northeast")}
    if spec["topology"] == "cycle":
        expected.add(("A_southwest", "C_northwest"))
    require(pairs == expected, f"Observed reference topology differs from published floor plan: {pairs}")
    return sorted(pairs)


def reference_metadata(layout, spec, protocol, evaluator, actor):
    initial_path = HOLDOUT / "prepared" / layout / "initial_state.json"
    initial = read(initial_path)
    directory = HOLDOUT / "evaluator_references" / layout
    truth_path = directory / "evaluator_truth.npz"
    receipt = read(directory / "RECEIPT.json")
    with np.load(truth_path, allow_pickle=False) as raw:
        reachable = raw["reachable"]
    cell = float(initial["grid_resolution_m"])
    config = initial["actual_config"]
    require(initial["policy_select_calls"] == initial["paid_actions"] == 0
            and initial["seed"] == protocol["scene_seed"]
            and initial["stop_budget"] == config["max_steps"] == protocol["native_budget"]
            and list(reachable.shape) == initial["shape"] and reachable.dtype == np.bool_
            and int(reachable.sum()) == receipt["reachable_cells"]
            and bool(reachable[tuple(initial["anchor"][:2])])
            and config["resolution_m"] == cell
            and not receipt["actor_executed"] and not receipt["actor_trajectory_read"],
            f"Reference and untouched actor cold start do not match for {layout}")
    require(str(truth_path) in evaluator["files_sha256"] and str(truth_path) not in actor["files_sha256"],
            f"Independent reference truth leaked into actor inputs for {layout}")
    room_connections = topology_from_reference(reachable, spec, cell)
    return {
        "layout": layout, "topology": spec["topology"], "rooms": 4, "doors": len(room_connections),
        "room_connections": room_connections, "gross_floor_area_m2": round(spec["width_m"] * spec["height_m"], 6),
        "grid_resolution_m": cell, "reachable_cells": int(reachable.sum()),
        "reachable_area_m2": round(int(reachable.sum()) * cell**2, 6), "scan_range_m": config["max_depth_m"],
        "image_width_px": config["width_px"], "image_height_px": config["height_px"], "camera_fov_deg": config["fov_deg"],
        "robot_radius_m": config["robot_radius_m"], "initial_pose": initial["anchor"], "scene_seed": initial["seed"],
        "baseline_native_budget": initial["stop_budget"], "reference_scene_samples": receipt["metadata"]["all"]["visible_samples"],
        "reference_shelf_samples": receipt["metadata"]["open31"]["visible_samples"],
        "reference_cabinet_samples": receipt["metadata"]["closed118"]["visible_samples"],
    }


def main():
    protocol_path = STUDY / "PROTOCOL.json"
    protocol = read(protocol_path)
    actor_path = HOLDOUT / "freeze.json"
    evaluator_path = HOLDOUT / "evaluator_freeze.json"
    files = {}
    bind(protocol_path, files)
    bind(Path(__file__).resolve(), files)
    bind(OUTPUT / "evaluate.py", files)
    bind(actor_path, files, protocol["source_actor_freeze_sha256"])
    bind(evaluator_path, files, protocol["source_evaluator_freeze_sha256"])
    actor, evaluator = read(actor_path), read(evaluator_path)
    require(actor["actors_run"] == evaluator["actors_run"] == 0 and actor["status"] == "actor_inputs_only"
            and not any("evaluator_references" in path or "evaluator_truth" in path for path in actor["files_sha256"]),
            "Archived source had test-policy outcomes or evaluator-only data in actor freeze")
    for path, checksum in (actor["files_sha256"] | evaluator["files_sha256"]).items():
        bind(Path(path), files, checksum)
    for path in (SOURCE / "measurement/evaluate.py", SOURCE / "PROTOCOL.json", SOURCE / "publication/receipt.json",
                 SOURCE_WITNESSES / "publication/receipt.json", SOURCE_WITNESSES / "measurement/results/engineering_gate.json",
                 SOURCE_LARGE / "evaluation/freeze.json", SOURCE_LARGE / "mechanism/scenes/freeze.json",
                 SOURCE_LARGE / "mechanism/runner/freeze.json",
                 SOURCE_LARGE / "evaluation/measure.py", SOURCE_NATURAL / "exploration_evaluation/surface_reference.py",
                 SOURCE_B11 / "execution/freeze.json", SOURCE_RECON / "utils/reconstruction_metrics.py",
                 STUDY.parent / "candidate_route_yield_20261121/measurement/measure.py",
                 STUDY.parent / "candidate_route_persistence_20261121/measurement/evaluate.py"):
        bind(path, files)
    require(read(SOURCE / "publication/receipt.json")["head"] == protocol["layout_source_commit"]
            and read(SOURCE_WITNESSES / "publication/receipt.json")["head"] == protocol["source_commit"]
            and read(SOURCE_WITNESSES / "measurement/results/engineering_gate.json")["engineering_gate"]["pass"],
            "Provenance of source layouts or two-witness release evidence differs from protocol")
    specs = published_geometry_specs(HOLDOUT / "scene.py")
    rows = [reference_metadata(layout, specs[layout], protocol, evaluator, actor) for layout in protocol["layouts"]]
    result = {"status": "frozen_evaluator_reference_inputs", "source_commit": protocol["source_commit"],
              "layout_source_commit": protocol["layout_source_commit"], "evaluator_truth_passed_to_actor": False,
              "heldout_policy_outcomes_read_for_freeze": False, "evaluator_inputs_not_for_actor_hashing": True,
              "source_actor_freeze_sha256": files[str(actor_path)], "source_evaluator_freeze_sha256": files[str(evaluator_path)],
              "inputs_sha256": files, "layouts": rows}
    with (OUTPUT / "EVALUATOR_FREEZE.json").open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    out_path = OUTPUT / "layouts.csv"
    with io.StringIO(newline="") as out:
        writer = csv.DictWriter(out, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        serialized = out.getvalue().encode()
    if out_path.is_file():
        require(out_path.read_bytes() == serialized, "Previously frozen reference area table changed")
    else:
        with out_path.open("xb") as out:
            out.write(serialized)


if __name__ == "__main__":
    main()
