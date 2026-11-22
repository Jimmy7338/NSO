#!/usr/bin/env python3
"""Localize held-out surface differences from sealed maps without running a policy."""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from open3d.core import Tensor

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
SOURCE = STUDY.parent / "region_global_revision_20261122/next/global_attribution"
ARMS = ("combined_S", "F_strict")
CLOCKS = ("paid_96", "paid_144", "terminal")
THRESHOLD = 0.05
BOUNDARY = "boundary_or_unassigned"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def surface_bins(xy, rooms, margin):
    """Conservatively assign only samples inside an eroded source-room rectangle."""
    bins = np.full(len(xy), -1, dtype=np.int16)
    x, y = xy.T
    for index, (x0, y0, x1, y1) in enumerate(rooms.values()):
        inside = (x > x0 + margin) & (x < x1 - margin) & (y > y0 + margin) & (y < y1 - margin)
        if np.any(bins[inside] != -1):
            raise ValueError("Source room interiors overlap.")
        bins[inside] = index
    return bins


def grid_bins(shape, resolution, rooms):
    rows, cols = np.indices(shape)
    x = (cols + 0.5) * resolution
    y = (shape[0] - rows - 0.5) * resolution
    bins = np.full(shape, -1, dtype=np.int16)
    for index, (x0, y0, x1, y1) in enumerate(rooms.values()):
        inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        if np.any(bins[inside] != -1):
            raise ValueError("Source room grid assignment overlaps.")
        bins[inside] = index
    if np.any(bins == -1):
        raise ValueError("Missing source room grid cells.")
    return bins


def match(left, right, description):
    if not math.isclose(left, right, abs_tol=1e-12, rel_tol=0):
        raise ValueError(f"{description} differs from published values: {left} vs {right}")


def history(layout, arm, episode, check, weighted, initial):
    directory = Path(episode["run_directory"])
    manifest, digest = weighted.engine.verify_manifest(directory, "manifest.json", episode["source_manifest_sha256"])
    trace = read(directory / "steps.json")
    weighted.native.verify_trace(trace, initial, 192, manifest)
    if (manifest["layout"], manifest["arm"], len(trace)) != (layout, arm, episode["actual_paid"]):
        raise ValueError(f"Episode identity drift: {directory}")
    with np.load(directory / "observed_belief_trace.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["paid"], np.arange(len(trace) + 1)):
            raise ValueError(f"Missing measured belief clock: {directory}")
        belief = archive["belief"]
    return dict(directory=directory, manifest=manifest, trace=trace, initial=initial, belief=belief,
                manifest_sha256=digest,
                steps_sha256=manifest["files_sha256"]["steps.json"],
                observed_belief_trace_sha256=manifest["files_sha256"]["observed_belief_trace.npz"])


def snapshot(run, clock, published, check, weighted, metric, reference, source_ray, surface_samples):
    paid = len(run["trace"]) if clock == "terminal" else int(clock.removeprefix("paid_"))
    result, matches = check.verify_measurement(weighted, metric, reference, run, clock, paid)
    for name in ("C_free", "p_scene_05cm", "r_scene_05cm", "F1_scene_05cm", "J"):
        match(result[name], float(published[name]), name)
    if int(published["actual_paid"]) != paid:
        raise ValueError("Published and actual measured action clocks differ.")
    mesh = weighted.files.read_mesh(result["mesh_path"])
    predicted, _ = surface_samples(mesh, 12_000, 812)
    predicted = predicted.astype(np.float32)
    distances = source_ray.compute_distance(Tensor(predicted), nthreads=1).numpy()
    precise = distances <= THRESHOLD
    match(float(precise.mean()), result["p_scene_05cm"], "precision partition")
    return dict(result=result, paid=paid, fixed_present=matches["scene"], predicted_xy=predicted[:, :2],
                predicted_precise=precise)


def fixed_sample_partition(fixed, snap, weight):
    present = {arm: value["fixed_present"] & fixed for arm, value in snap.items()}
    both = present["combined_S"] & present["F_strict"]
    group = dict(fixed_reference_samples=int(np.count_nonzero(fixed)),
                 fixed_reference_area_m2=float(np.count_nonzero(fixed) * weight))
    for key, mask in (("both", both), ("combined_only", present["combined_S"] & ~both),
                      ("strict_only", present["F_strict"] & ~both),
                      ("neither", fixed & ~(present["combined_S"] | present["F_strict"]))):
        count = int(np.count_nonzero(mask))
        group[f"{key}_samples"] = count
        group[f"{key}_m2"] = count * weight
    group["strict_minus_combined_m2"] = group["strict_only_m2"] - group["combined_only_m2"]
    if sum(group[f"{part}_samples"] for part in ("both", "combined_only", "strict_only", "neither")) != group["fixed_reference_samples"]:
        raise ValueError("Fixed surface sample partition does not conserve total.")
    return group


def report_room(layout, clock, room, grid_index, fixed, prediction, run, snap, truth_grid, weight):
    group = dict(layout=layout, checkpoint=clock, room=room) | fixed_sample_partition(fixed, snap, weight)
    for arm, prefix in (("combined_S", "combined"), ("F_strict", "strict")):
        sample = prediction[arm]
        value = snap[arm]
        found = group["both_samples"] + group[f"{prefix}_only_samples"]
        predicted = int(np.count_nonzero(sample))
        precise = int(np.count_nonzero(value["predicted_precise"] & sample))
        group.update({f"{prefix}_paid": value["paid"], f"{prefix}_represented_samples": found,
                      f"{prefix}_recall": fraction(found, group["fixed_reference_samples"]),
                      f"{prefix}_predicted_precision_samples": predicted,
                      f"{prefix}_precise_samples": precise, f"{prefix}_precision": fraction(precise, predicted)})
        if grid_index == -1:
            continue
        room_grid = truth_grid == grid_index
        states = poses(run[arm], value["paid"])
        local = [state for state in states if room_grid[state["cell"]]]
        group.update({f"{prefix}_first_visit_paid": min((s["paid"] for s in local), default=None),
                      f"{prefix}_observation_poses": len(local),
                      f"{prefix}_distinct_oriented_poses": len({(s["cell"], s["heading"]) for s in local}),
                      f"{prefix}_distinct_heading_bins": len({s["heading"] for s in local})})
    return group


def fraction(value, total):
    return value / total if total else None


def poses(run, paid):
    anchor = run["initial"]["anchor"]
    return [dict(cell=tuple(anchor[:2]), heading=anchor[2], paid=0)] + [
        dict(cell=tuple(row["position"]), heading=row["heading"], paid=row["step"]) for row in run["trace"][:paid]
    ]


def access(layout, arm, run, truth_grid, reachable, rooms):
    paid_end = len(run["trace"])
    start_return = next((s["step"] - 1 for s in run["trace"] if s["phase_before_action"] == "return"), None)
    anchor = tuple(run["initial"]["anchor"][:2])
    last_away = next((s["step"] for s in reversed(run["trace"]) if tuple(s["position"]) != anchor), None)
    reentry = last_away + 1 if run["manifest"]["returned"] and last_away is not None else None
    ticks = [("cold", 0), ("paid_96", 96), ("paid_144", 144), ("before_first_return_action", start_return),
             ("terminal_anchor_xy_entry", reentry), ("actual_terminal", paid_end)]
    records = []
    for clock, paid in ticks:
        if paid is None:
            records.append(dict(layout=layout, arm=arm, checkpoint=clock, actual_paid=None,
                                status="not_tagged" if clock == "before_first_return_action" else "unavailable"))
            continue
        states = poses(run, paid)
        seen = {int(truth_grid[s["cell"]]) for s in states}
        belief = run["belief"][paid]
        free = reachable & (belief == 0)
        covered_unvisited = int(np.count_nonzero(free & ~np.isin(truth_grid, list(seen))))
        records.append(dict(layout=layout, arm=arm, checkpoint=clock, actual_paid=paid, status="measured",
                            visited_rooms=len(seen), rooms="|".join(name for n, name in enumerate(rooms) if n in seen),
                            correctly_free_cells=int(free.sum()), reachable_cells=int(reachable.sum()),
                            C_free=fraction(int(free.sum()), int(reachable.sum())),
                            correct_free_cells_in_unvisited_rooms=covered_unvisited))
    return records


def csv_data(rows):
    buffer = io.StringIO(newline="")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(buffer, columns)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing products without writing.")
    args = parser.parse_args()
    sys.path.insert(0, str(STUDY / "measurement"))
    from historical import checks

    check, weighted, metric = checks()
    from surface_reference import surface_samples

    scenes = SOURCE / "scenes"
    weighted.SCENES = scenes
    freeze = SOURCE / "FORMAL_FREEZE.json"
    frozen = read(freeze)
    formal = SOURCE / "results/formal"
    summary_path = formal / "summary.json"
    summary = read(summary_path)
    if sha(freeze) != summary["run_freeze_sha256"]:
        raise ValueError("Publication measurement and formal freeze differ.")
    source_scene = scenes / "scene.py"
    geometry = load_module("residual_frozen_room_geometry", source_scene)
    margin = float(geometry.geometry.WALL_M / 2) + 1e-6  # Include source wall faces at the exact half-thickness.
    with (formal / "checkpoints.csv").open(newline="") as stream:
        published = {(row["layout"], row["arm"], row["checkpoint"]): row for row in csv.DictReader(stream)}
    episodes = {(row["layout"], row["arm"]): row for row in summary["episodes"]}
    by_room, by_access, by_scene, by_boundary = [], [], [], []
    evidence = dict(source_freeze_sha256=sha(freeze), source_summary_sha256=sha(summary_path),
                    source_geometry_sha256=sha(source_scene), source_evaluator_sha256=summary["evaluator_sha256"],
                    room_wall_generator_sha256=sha(geometry.geometry.__file__),
                    precision_sampler_sha256=sha(Path(sys.modules[surface_samples.__module__].__file__)),
                    evaluation_wrapper_sha256=sha(check.__file__),
                    source_checkpoints_csv_sha256=sha(formal/"checkpoints.csv"),
                    threshold_m=THRESHOLD, room_boundary_margin_m=margin, precision_samples_seed=812,
                    precision_samples_per_snapshot=12_000, actor_runs=0, layouts={})
    for layout, definition in geometry.HOLDOUT.items():
        rooms = definition["rooms"]
        initial_path = scenes / "prepared" / layout / "initial_state.json"
        initial = read(initial_path)
        reference_root = scenes / "evaluator_references" / layout
        reference_receipt = read(reference_root / "RECEIPT.json")
        reference_files = [reference_root / name for name in (
            "RECEIPT.json", "fixed_reference.npz", "source_mesh.ply", "evaluator_truth.npz")]
        actual_hashes = {str(path): sha(path) for path in (*reference_files, initial_path)}
        if any(frozen["evaluator_files"].get(str(path)) != actual_hashes[str(path)] for path in reference_files):
            raise ValueError(f"Frozen source geometry or all-scene reference changed: {layout}")
        if sha(source_scene) != reference_receipt["source_inputs_sha256"][str(source_scene)]:
            raise ValueError(f"Source room geometry is detached from the fixed reference: {layout}")
        reference = weighted.load_reference(layout, initial)
        points = reference["samples"]["scene"]
        weight = reference["weights"]["scene"]
        truth_grid = grid_bins(initial["shape"], initial["grid_resolution_m"], rooms)
        fixed_bins = surface_bins(points[:, :2], rooms, margin)
        ray = metric.ray_scene(reference["source_mesh"])
        normals_z = np.abs(ray.compute_closest_points(Tensor(points), nthreads=1)["primitive_normals"].numpy()[:, 2])
        vertical = normals_z <= 1e-3
        horizontal = normals_z >= 1 - 1e-3
        boundary_orientation = {"vertical": vertical, "horizontal": horizontal, "oblique": ~(vertical | horizontal)}
        runs = {arm: history(layout, arm, episodes[layout, arm], check, weighted, initial) for arm in ARMS}
        bundle = evidence["layouts"][layout] = dict(reference_files_sha256=actual_hashes, rooms=rooms,
                                                  area_per_reference_sample_m2=weight, arm_runs={})
        for arm, run in runs.items():
            bundle["arm_runs"][arm] = {k: v for k, v in run.items() if k.endswith("_sha256")}
            bundle["arm_runs"][arm]["snapshots"] = {}
            by_access.extend(access(layout, arm, run, truth_grid, reference["reachable"], rooms))
        for clock in CLOCKS:
            snap = {arm: snapshot(runs[arm], clock, published[layout, arm, clock], check, weighted, metric,
                                  reference, ray, surface_samples) for arm in ARMS}
            for arm, snapshot_data in snap.items():
                result = snapshot_data["result"]
                other = snap[ARMS[1] if arm == ARMS[0] else ARMS[0]]
                by_scene.append(dict(layout=layout, checkpoint=clock, arm=arm, paid=snapshot_data["paid"],
                                     equal_actual_clock=snapshot_data["paid"] == other["paid"],
                                     **{name: result[name] for name in ("C_free", "p_scene_05cm", "r_scene_05cm",
                                                                        "F1_scene_05cm", "J")}))
                bundle["arm_runs"][arm]["snapshots"][clock] = {
                    "actual_paid": snapshot_data["paid"], "mesh_sha256":result["mesh_sha256"],
                    "checkpoint_sha256":result["checkpoint_sha256"],
                }
            predicted_bins = {arm: surface_bins(row["predicted_xy"], rooms, margin) for arm, row in snap.items()}
            rows = [report_room(layout, clock, room, index, fixed_bins == index,
                                {arm: bin_ids == index for arm, bin_ids in predicted_bins.items()},
                                runs, snap, truth_grid, weight)
                    for room, index in zip((*rooms, BOUNDARY), (*range(len(rooms)), -1), strict=True)]
            if sum(row["fixed_reference_samples"] for row in rows) != len(points):
                raise ValueError("Region partition lost fixed samples.")
            boundary_rows = [
                dict(layout=layout, checkpoint=clock, source_face_orientation=orientation,
                     combined_paid=snap["combined_S"]["paid"], strict_paid=snap["F_strict"]["paid"])
                | fixed_sample_partition((fixed_bins == -1) & orientation_mask, snap, weight)
                for orientation, orientation_mask in boundary_orientation.items()]
            if sum(row["strict_only_samples"] for row in boundary_rows) != rows[-1]["strict_only_samples"]:
                raise ValueError("Orientation partition lost strict-only boundary surfaces.")
            for arm, prefix in (("combined_S", "combined"), ("F_strict", "strict")):
                represented = sum(row[f"{prefix}_represented_samples"] for row in rows)
                precise = sum(row[f"{prefix}_precise_samples"] for row in rows)
                prediction_size = sum(row[f"{prefix}_predicted_precision_samples"] for row in rows)
                match(represented / len(points), snap[arm]["result"]["r_scene_05cm"], "room recall aggregate")
                match(precise / prediction_size, snap[arm]["result"]["p_scene_05cm"], "room precision aggregate")
            by_room.extend(rows)
            by_boundary.extend(boundary_rows)
    files = {"room_surface.csv": csv_data(by_room), "phase_access.csv": csv_data(by_access),
             "scene_clocks.csv": csv_data(by_scene), "boundary_surface.csv": csv_data(by_boundary)}
    evidence["output_sha256"] = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}
    files["receipt.json"] = json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    for name, content in files.items():
        target = HERE / name
        if args.check:
            if target.read_bytes() != content.encode():
                raise ValueError(f"Recorded audit differs: {target}")
        else:
            target.write_bytes(content.encode())
    print(f"Verified {len(by_scene)} native mesh clocks across {len(geometry.HOLDOUT)} held-out floors and {len(by_room)} spatial rows.")


if __name__ == "__main__":
    main()
