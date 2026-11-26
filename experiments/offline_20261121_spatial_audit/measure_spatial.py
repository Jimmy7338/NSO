"""Locate registration-matched reference surface gains/losses in sealed NSO meshes."""

import argparse
from functools import lru_cache
import importlib.util
import math
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
from open3d.core import Tensor


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parent
COLD = ANALYSES / "same_goal_cold_holdout_20261121"
HOLDOUT = ANALYSES / "same_goal_route_choice_20261121/holdout"
SOURCE_BUILDER = ANALYSES / "semantic_large_scene_20261121/mechanism/scenes/prepare_reference.py"
INDEX_PATH = COLD / "publication/payload/EPISODE_INDEX.json"
PRIMARY = COLD / "measurement/results/checkpoints.json"
EVALUATOR_FREEZE = COLD / "measurement/EVALUATOR_FREEZE.json"
GEOMETRY_FREEZE = HOLDOUT / "evaluator_freeze.json"
PANELS = (
    ("chain13_terminal", "ordinary_chain_13", "terminal", 512),
    ("cycle14_terminal", "ordinary_cycle_14", "terminal", 512),
    ("chain15_terminal", "ordinary_chain_15", "terminal", 512),
    ("cycle16_terminal", "ordinary_cycle_16", "terminal", 512),
    ("cycle16_paid256", "ordinary_cycle_16", "paid_256", 256),
    ("cycle16_paid320", "ordinary_cycle_16", "paid_320", 320),
)
STATUSES = ("neither", "lost_G_only", "new_S_only", "both")
MEASURES = ("correct_free_cells", "reachable_cells", "C_free", "p_scene_05cm", "r_scene_05cm", "F1_scene_05cm", "J")
SOURCE_GROUPS = {"other": 0, "shelf": 31, "cabinet": 118}
DIAGNOSTIC_GROUPS = {"shelf": "open31", "cabinet": "closed118"}
PRECISION_DRAWS = 12000
PRECISION_SEED = 812


def load_local(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(COLD / "measurement"))
from evaluate import files, native, read, require, sha


def point_keys(points):
    encoded_points = np.ascontiguousarray(points, dtype=np.float32)
    return encoded_points.view(np.dtype(("V", 3 * encoded_points.dtype.itemsize))).reshape(-1)


def partition_reference(layout, reference, ref_receipt, builder):
    """Recover triangle IDs from the deterministic pre-visibility global sample draw."""
    raw_mesh, raw_source_ids, records = builder.source_mesh(layout)
    sealed_mesh = reference["source_mesh"]
    for field in ("vertices", "triangles"):
        raw_values = np.asarray(getattr(raw_mesh, field))
        sealed_values = np.asarray(getattr(sealed_mesh, field))
        require(raw_values.dtype == sealed_values.dtype and raw_values.shape == sealed_values.shape
                and raw_values.tobytes() == sealed_values.tobytes(),
                f"Geometry-only draw uses a different frozen source mesh for {layout}: {field}")
    require(records == ref_receipt["source_object_records"], f"Source fixture definitions changed: {layout}")
    unique_mesh, source_ids, dedup_receipt = builder.unique_sampling_faces(raw_mesh, raw_source_ids)
    sampling = ref_receipt["metadata"]["all"]
    require(dedup_receipt == ref_receipt["duplicate_sampling_handling"]
            and math.isclose(unique_mesh.get_surface_area(), sampling["sampled_source_area_m2"], rel_tol=0, abs_tol=1e-10),
            f"Frozen scene sampling geometry changed: {layout}")
    all_drawn, sampled_faces = builder.surface_samples(unique_mesh, sampling["drawn_samples"], sampling["seed"])
    draws = point_keys(all_drawn)
    visible = reference["samples"]["scene"]
    visible_keys = point_keys(visible)
    order = np.argsort(draws, kind="stable")
    sorted_draws = draws[order]
    require(len(np.unique(sorted_draws)) == len(draws), f"Ambiguous original draw key: {layout}")
    sorted_index = np.searchsorted(sorted_draws, visible_keys)
    require(not np.any(sorted_index >= len(draws)), f"Sealed visible point not present in original draw: {layout}")
    original_index = order[sorted_index]
    require(np.all(np.diff(original_index) > 0) and np.array_equal(draws[original_index], visible_keys),
            f"Sealed visible reference points do not exactly preserve the original draw: {layout}")
    labels = source_ids[sampled_faces[original_index]]
    require(set(labels) == set(SOURCE_GROUPS.values()), f"Unexpected source surface labels for {layout}")
    return labels


def snapshot(registry, published_index, layout, arm, checkpoint, expected_paid, measured_inputs):
    run = COLD / "runner/runs" / layout / arm
    index = published_index[str(run)]
    manifest_path = run / "manifest.json"
    files.verify(manifest_path, index["archived_files_sha256"]["manifest.json"])
    manifest = read(manifest_path)
    require(manifest["status"] == "completed_execution" and manifest["layout"] == layout and manifest["arm"] == arm,
            f"Snapshot ancestry does not have a completed frozen actor manifest: {run}")
    snap = run / checkpoint
    receipt_path, belief_path, mesh_path = (snap / key for key in ("checkpoint.json", "belief.npz", "native_mesh.ply"))
    mesh_entry = index["native_artifacts_on_host"][f"{checkpoint}/native_mesh.ply"]
    mesh_hash = mesh_entry["sha256"]
    files.verify(mesh_path, mesh_hash)
    require(mesh_path.stat().st_size == mesh_entry["bytes"], f"Published native mesh byte count mismatch: {mesh_path}")
    for path in (receipt_path, belief_path):
        key = str(path.relative_to(run))
        files.verify(path, index["archived_files_sha256"][key])
        require(index["archived_files_sha256"][key] == manifest["files_sha256"][key],
                f"Actor manifest and published checkpoint metadata checksums disagree: {path}")
    receipt = read(receipt_path)
    measured = registry[layout, arm, checkpoint]
    require(receipt == manifest["checkpoints"][checkpoint]
            and receipt["status"] == measured["status"] == "captured"
            and receipt["actual_paid"] == measured["actual_paid"] == expected_paid
            and sha(receipt_path) == measured["source_checkpoint_sha256"]
            and mesh_hash == manifest["files_sha256"][f"{checkpoint}/native_mesh.ply"]
            == receipt["files_sha256"]["native_mesh.ply"],
            f"Sealed references disagree on the used native checkpoint: {snap}")
    measured_inputs.update({str(path): sha(path) for path in (manifest_path, receipt_path, belief_path)})
    measured_inputs[str(mesh_path)] = mesh_hash
    return {
        "mesh_path": mesh_path, "belief_path": belief_path, "mesh_sha256": mesh_hash, "receipt": receipt,
        "receipt_sha256": sha(receipt_path), "published_actor_archive": index["archive"], "registered": measured,
    }


def confirm_same_paid_512(terminal, paid_512):
    exact_fields = ("pose", "packet_sha256", "packet_chain_sha256", "belief_sha256", "native_mesh_fields")
    require(terminal["mesh_sha256"] == paid_512["mesh_sha256"]
            and sha(terminal["belief_path"]) == sha(paid_512["belief_path"])
            and all(terminal["receipt"][key] == paid_512["receipt"][key] for key in exact_fields)
            and all(terminal["registered"][key] == paid_512["registered"][key] for key in MEASURES),
            f"Cycle16 terminal and paid512 are not the same native geometry: {terminal['mesh_path']}")


def evaluate_arm(arm, fixture_layers, reference, source_ray, frozen_metric, sampler):
    belief, pose = files.read_belief(arm["belief_path"])
    require(pose == arm["receipt"]["pose"], f"Archived belief has different coordinates: {arm['belief_path']}")
    mesh = files.read_mesh(arm["mesh_path"])
    observed = frozen_metric.ray_scene(mesh)
    predicted, _ = sampler(mesh, PRECISION_DRAWS, PRECISION_SEED)
    predicted_xyz = predicted.astype(np.float32)
    require(len(predicted_xyz) == PRECISION_DRAWS, f"Registered mesh could not reproduce 12,000 draws: {arm['mesh_path']}")
    accuracy = source_ray.compute_distance(Tensor(predicted_xyz), nthreads=1).numpy()
    all_layers = {"scene": reference["samples"]["scene"], **fixture_layers}
    presence = {
        name: observed.compute_distance(Tensor(points), nthreads=1).numpy() <= files.THRESHOLD_M
        for name, points in all_layers.items()
    }
    matches = accuracy <= files.THRESHOLD_M
    precision, recall = float(matches.mean()), float(presence["scene"].mean())
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    free, total = frozen_metric.count_free(belief, reference["reachable"])
    coverage = free / total
    measures = {"correct_free_cells": int(free), "reachable_cells": int(total), "C_free": coverage,
                "p_scene_05cm": precision, "r_scene_05cm": recall, "F1_scene_05cm": f1, "J": coverage * f1}
    for key, actual in measures.items():
        require(math.isclose(actual, arm["registered"][key], rel_tol=0, abs_tol=1e-12),
                f"Point/triangle distance failed to reproduce sealed {key}: {arm['mesh_path']}")
    return measures, presence, {
        "xyz": predicted_xyz, "matched": matches, "distance_m": accuracy,
        "weight_m2": np.float64(mesh.get_surface_area() / len(accuracy)),
    }


def surface_status(green, semantic):
    return green.astype(np.uint8) | (semantic.astype(np.uint8) << 1)


def add_areas(row, prefix, status, point_area):
    counts = np.bincount(status, minlength=len(STATUSES))
    require(int(counts.sum()) == len(status), f"Surface matching cannot conserve category counts: {prefix}")
    row[f"{prefix}_reference_count"] = len(status)
    row[f"{prefix}_area_per_point_m2"] = point_area
    row[f"{prefix}_reference_m2"] = len(status) * point_area
    for index, label in enumerate(STATUSES):
        row[f"{prefix}_{label}_count"] = int(counts[index])
        row[f"{prefix}_{label}_m2"] = int(counts[index]) * point_area
    row[f"{prefix}_net_S_minus_G_count"] = int(counts[2] - counts[1])
    row[f"{prefix}_net_S_minus_G_m2"] = int(counts[2] - counts[1]) * point_area


def write_panel(arrays, panel, points, label_ids, statuses, predicted, fixture_layers, weights):
    fixture_xyz = np.concatenate(list(fixture_layers.values()))
    fixture_status = np.concatenate([statuses[name] for name in fixture_layers])
    fixture_ids = np.concatenate([np.full(len(points), SOURCE_GROUPS[name], dtype=np.int16)
                                  for name, points in fixture_layers.items()])
    fixture_weights = np.concatenate([np.full(len(points), weights[name], dtype=np.float64)
                                      for name, points in fixture_layers.items()])
    slices = {
        "reference_xyz": points, "reference_status": statuses["scene"],
        "reference_source_id": label_ids, "reference_weight_m2": np.full(len(points), weights["scene"]),
        "fixture_xyz": fixture_xyz, "fixture_status": fixture_status,
        "fixture_source_id": fixture_ids, "fixture_weight_m2": fixture_weights,
    }
    slices.update({f"pred_{name}_{key}": value for name, group in predicted.items() for key, value in group.items()})
    arrays.update({f"{panel}__{key}": value for key, value in slices.items()})


def main(out):
    freeze = read(EVALUATOR_FREEZE)
    measured_inputs = {str(path): sha(path) for path in (
        EVALUATOR_FREEZE, GEOMETRY_FREEZE, INDEX_PATH, PRIMARY, PRIMARY.with_name("engineering_gate.json"),
    )}
    measured_inputs.update(freeze["inputs_sha256"])
    for source, digest in freeze["inputs_sha256"].items():
        files.verify(source, digest)
    registered = read(PRIMARY)
    final_receipt = read(PRIMARY.with_name("engineering_gate.json"))
    require(sha(EVALUATOR_FREEZE) == registered["evaluator_freeze_sha256"] == final_receipt["evaluator_freeze_sha256"]
            and sha(PRIMARY) == final_receipt["input_evidence_sha256"][PRIMARY.name],
            "Sealed measurement table and evaluator freeze do not match")
    raw_registry = registered["rows"]
    registry = {(item["layout"], item["mode"], item["checkpoint"]): item for item in raw_registry}
    require(len(registry) == len(raw_registry) == 160, "Published cold-run checkpoint registry changed")
    index_rows = read(INDEX_PATH)["bundles"]
    published_index = {item["run_directory"]: item for item in index_rows}
    require(len(published_index) == len(index_rows) == 16, "Episode index is missing a published run")
    metric = files.load_metric(read(files.SCENES / "freeze.json"), read(files.OLD_EVAL / "freeze.json"))
    source_scene = load_local("nso_spatial_frozen_holdout_scene", HOLDOUT / "scene.py")
    sys.path.insert(0, str(SOURCE_BUILDER.parent))
    builder = load_local("nso_spatial_frozen_source_builder", SOURCE_BUILDER)
    builder.build_scene = source_scene.build_scene
    reference_freeze = read(GEOMETRY_FREEZE)
    partition_counts = {}

    @lru_cache(maxsize=None)
    def reference_data(layout):
        reference = native.holdout_reference(layout, reference_freeze)
        ref_dir = HOLDOUT / "evaluator_references" / layout
        receipt = read(ref_dir / "RECEIPT.json")
        for source, digest in receipt["source_inputs_sha256"].items():
            files.verify(source, digest)
        measured_inputs.update(receipt["source_inputs_sha256"])
        measured_inputs.update(reference["files_sha256"])
        require(receipt["layout"] == layout and not receipt["actor_executed"] and not receipt["actor_trajectory_read"],
                f"Reference inputs are no longer independent of the actor: {layout}")
        labels = partition_reference(layout, reference, receipt, builder)
        partition_counts[layout] = {name: int(np.count_nonzero(labels == source_id))
                                    for name, source_id in SOURCE_GROUPS.items()}
        return reference, labels

    rows, arrays = [], {
        "panels": np.asarray([p[0] for p in PANELS]),
        "threshold_m": np.float64(files.THRESHOLD_M), "source_ids": np.asarray(list(SOURCE_GROUPS.values()), dtype=np.int16),
        "source_names": np.asarray(list(SOURCE_GROUPS)), "status_bitmask_labels": np.asarray(STATUSES),
        "precision_sample_count": np.int64(PRECISION_DRAWS), "precision_sample_seed": np.int64(PRECISION_SEED),
    }
    index_sha, table_sha = sha(INDEX_PATH), sha(PRIMARY)
    for panel, layout, checkpoint, paid in PANELS:
        reference, labels = reference_data(layout)
        points = reference["samples"]["scene"]
        weights = reference["weights"]
        fixture_layers = {name: reference["samples"][name] for name in DIAGNOSTIC_GROUPS}
        source_ray = metric.ray_scene(reference["source_mesh"])
        row = {
            "panel": panel, "layout": layout, "checkpoint": checkpoint, "actual_paid": paid,
            "threshold_m": files.THRESHOLD_M, "cycle16_terminal_paid512_identical": panel == "cycle16_terminal",
            "comparison_scope": "paired_5cm_triangle_distance",
            "fixture_partition": "exact_frozen_global_draw_labels_other0_shelf31_cabinet118",
            "targeted_fixture_samples": "separate_independent_draws_not_used_as_additive_partition",
            "episode_index_sha256": index_sha, "registered_checkpoint_table_sha256": table_sha,
            "fixed_reference_sha256": reference["files_sha256"][str(HOLDOUT / "evaluator_references" / layout / "fixed_reference.npz")],
        }
        measured, presence, predicted = {}, {}, {}
        for arm in ("G", "S"):
            pair_arm = snapshot(registry, published_index, layout, f"full{arm}", checkpoint, paid, measured_inputs)
            if panel == "cycle16_terminal":
                confirm_same_paid_512(pair_arm, snapshot(registry, published_index, layout, f"full{arm}", "paid_512", 512,
                                                       measured_inputs))
            measured[arm], presence[arm], predicted[arm] = evaluate_arm(
                pair_arm, fixture_layers, reference, source_ray, metric, builder.surface_samples,
            )
            accuracy = predicted[arm]["matched"]
            row.update({f"{key}_{arm}": value for key, value in measured[arm].items()})
            row.update({
                f"actor_archive_{arm}": pair_arm["published_actor_archive"], f"native_mesh_sha256_{arm}": pair_arm["mesh_sha256"],
                f"checkpoint_receipt_sha256_{arm}": pair_arm["receipt_sha256"], f"predicted_sample_count_{arm}": len(accuracy),
                f"predicted_sample_matched_count_{arm}": int(accuracy.sum()),
                f"predicted_sample_outside_5cm_count_{arm}": int((~accuracy).sum()),
                f"predicted_sample_weight_m2_{arm}": float(predicted[arm]["weight_m2"]),
            })
        for key in ("C_free", "p_scene_05cm", "r_scene_05cm", "F1_scene_05cm", "J"):
            row[f"delta_{key}_pp"] = 100 * (measured["S"][key] - measured["G"][key])
        statuses = {name: surface_status(presence["G"][name], presence["S"][name]) for name in presence["G"]}
        whole_status = statuses["scene"]
        add_areas(row, "scene", whole_status, weights["scene"])
        add_areas(row, "fixture_aligned", whole_status[labels != SOURCE_GROUPS["other"]], weights["scene"])
        for name, source_id in SOURCE_GROUPS.items():
            add_areas(row, name + "_aligned", whole_status[labels == source_id], weights["scene"])
        for name in fixture_layers:
            add_areas(row, name + "_targeted", statuses[name], weights[name])
        for label in STATUSES:
            count_key = f"{label}_count"
            require(row[f"scene_{count_key}"] == row[f"other_aligned_{count_key}"] + row[f"fixture_aligned_{count_key}"]
                    and row[f"fixture_aligned_{count_key}"] == row[f"shelf_aligned_{count_key}"] + row[f"cabinet_aligned_{count_key}"],
                    f"Named source partitions do not conserve whole-scene sample counts: {panel}")
        require(math.isclose(row["scene_net_S_minus_G_count"] / len(points), row["delta_r_scene_05cm_pp"] / 100,
                             abs_tol=1e-12, rel_tol=0), f"Recall change is not conserved by paired reference masks: {panel}")
        write_panel(arrays, panel, points, labels, statuses, predicted, fixture_layers, weights)
        rows.append(row)
        print(f"{panel} verified point-level precision, recall and scene area", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    files.write_csv(out / "spatial_pairs.csv", rows, list(rows[0]))
    destination = out / "spatial_pairs.npz"
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)
    files.write_json(out / "spatial_measurement_sources.json", {
        "status": "completed_read_only_triangle_distance_audit",
        "simulator_actions_generated": False, "sealed_input_files_modified": False,
        "threshold_distance_m": files.THRESHOLD_M, "fixed_reference_threshold_comparison": "distance <= threshold",
        "native_precision_draws_per_arm": PRECISION_DRAWS, "native_precision_seed": PRECISION_SEED,
        "registered_checkpoints_reproduced": 2 * len(rows),
        "native_ply_paths_verified_against_published_index": sum(path.endswith("/native_mesh.ply") for path in measured_inputs),
        "cycle16_terminal_and_paid512_mesh_belief_pose_metrics_identical_both_arms": True,
        "aligned_source_id_counts": partition_counts,
        "aligned_source_id_proof": "Recreated source mesh arrays equal sealed source PLY arrays, including vertex order. "
                                   "Original deterministic pre-visibility draw contains each sealed visible reference point "
                                   "as an exact, unique float32 key, in its original order; used its original triangle ID.",
        "separate_fixture_samples": "Denser independent fixed shelf/cabinet draws; do not subtract from scene totals.",
        "script_sha256": sha(Path(__file__).resolve()), "inputs_sha256": dict(sorted(measured_inputs.items())),
        "outputs_sha256": {name: sha(out / name) for name in ("spatial_pairs.csv", "spatial_pairs.npz")},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    main(parser.parse_args().out)
