#!/usr/bin/env python3
"""Post-score GT attribution of frozen counterfactual object hypotheses.

This is an evaluator-only diagnostic. It never updates a mapper, model, score,
candidate set or primary result using truth. A supervisor validates the finished
run, extracts its exact source archive, and runs one isolated serial worker.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"archive/manifest path escapes its root: {name}")
    return path


def validate_run(source):
    metadata = read_json(source / "metadata.json")
    if metadata.get("status") != "complete":
        raise ValueError("source run must be complete before post-score GT diagnosis")
    config = read_json(source / "config.json")
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if digest != metadata["config_sha256"]:
        raise ValueError("source configuration hash mismatch")
    artifacts = source / "artifact_hashes.json"
    if not artifacts.exists():
        raise ValueError("completed source artifact hash inventory is required")
    # Include full prefixes, records, prediction seals and outcomes, not merely
    # the few semantic mapper keyframes retained in the scoring hashes.
    for name, expected in read_json(artifacts).items():
        if file_hash(safe_member(source, name)) != expected:
            raise ValueError(f"source artifact changed: {name}")
    for seal_path in source.glob("*/pre_outcome_seal.json"):
        for name, expected in read_json(seal_path).items():
            if file_hash(safe_member(seal_path.parent, name)) != expected:
                raise ValueError(f"pre-outcome score seal mismatch: {name}")
    return metadata


def run(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(source):
        raise ValueError("diagnostic output must be separate from the frozen run")
    metadata = validate_run(source)
    output.mkdir(parents=True)
    audit = {
        "status": "running", "diagnostic_only": True, "feeds_planning": False,
        "source_run": str(source), "source_metadata_sha256": file_hash(source / "metadata.json"),
        "source_archive_sha256": file_hash(source / "sources.zip"),
        "script_sha256": file_hash(__file__), "workers": 1, "started_unix": time.time(),
        "source_replayed_claim": metadata.get("replayed", False),
        "warning": "This diagnosis replays prefix mapper state; it is not a full branch replay audit.",
    }
    write_json(output / "metadata.json", audit)
    try:
        with tempfile.TemporaryDirectory(prefix="nso_object_diagnosis_") as folder:
            snapshot = Path(folder)
            with zipfile.ZipFile(source / "sources.zip") as archive:
                for entry in archive.infolist():
                    target = safe_member(snapshot, entry.filename)
                    if entry.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(entry))
            for name, expected in metadata["source_sha256"].items():
                if file_hash(safe_member(snapshot, name)) != expected:
                    raise ValueError(f"archived source does not match run manifest: {name}")
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(snapshot))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                            "--snapshot", str(snapshot), "--source", str(source),
                            "--output", str(output)], cwd=snapshot, env=env, check=True)
        # Ensure no source evidence was changed by the post-hoc process.
        validate_run(source)
        audit.update(status="complete", elapsed_s=time.time() - audit["started_unix"])
    except Exception as error:
        audit.update(status="failed", error=repr(error))
        write_json(output / "metadata.json", audit)
        raise
    write_json(output / "metadata.json", audit)
    write_json(output / "artifact_hashes.json", {
        str(path.relative_to(output)): file_hash(path) for path in output.rglob("*") if path.is_file()
        and path.name != "artifact_hashes.json"})


def worker(source, output, snapshot):
    # Import every project dependency from the validated run snapshot. The
    # diagnostic itself is new and its own hash is recorded by the supervisor.
    sys.path.insert(0, str(snapshot))
    import numpy as np
    import open3d as o3d
    from scipy.ndimage import binary_closing, label
    from env.virtual3d_inspection_v4 import InspectionConfigV4, InspectionWorldV4
    from nso.semantic_completion_v3 import SemanticHistoryMapperV3, ObjectCompletionModel
    from nso.counterfactual_view_scoring import (
        _array_hash, _mapper_snapshot, _object_geometry_hash, _cluster_statistics)
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    from utils.grid_geometry import DIRECTIONS
    from utils.reconstruction_metrics import ray_scene

    import nso.counterfactual_view_scoring as frozen_scoring
    if not Path(frozen_scoring.__file__).resolve().is_relative_to(snapshot.resolve()):
        raise ValueError("worker imported live project code instead of the archived snapshot")
    config = read_json(source / "config.json")
    threshold, inside_tolerance, duplicate_tolerance = .05, 1e-6, 1e-6
    scorers = ("G", "O", "S", "X", "M")

    def measured_groups(mapper):
        q = mapper.quality_evidence(max_points=10000)
        if q is None:
            return []
        points = q["point"]
        rows = mapper.shape[0] - 1 - np.floor(points[:, 1] / .2).astype(int)
        cols = np.floor(points[:, 0] / .2).astype(int)
        inside = (rows >= 0) & (rows < mapper.shape[0]) & (cols >= 0) & (cols < mapper.shape[1])
        occupied = np.zeros(mapper.shape, bool)
        occupied[rows[inside], cols[inside]] = True
        groups, _ = label(binary_closing(occupied, structure=np.ones((3, 3))))
        ids = np.zeros(len(points), int)
        ids[inside] = groups[rows[inside], cols[inside]]
        accepted = []
        for group in sorted(set(ids) - {0}):
            selected = np.flatnonzero(ids == group)
            if len(selected) < 8:
                continue
            low, high = np.quantile(points[selected], [.05, .95], axis=0)
            span = high - low
            if max(span[:2]) > 2.2 or span[2] < .25:
                continue
            accepted.append({"group_id": int(group), "points": points[selected],
                             "labels": q["label"][selected]})
        return accepted

    def closest(ray, world, points):
        if not len(points):
            return np.empty(0), np.empty(0, np.uint8), np.empty((0, 3)), np.empty(0, np.int64)
        result = ray.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)), nthreads=1)
        hit = result["points"].numpy()
        triangle = result["primitive_ids"].numpy().astype(np.int64)
        if np.any(triangle < 0) or np.any(triangle >= len(world.triangle_classes)):
            raise ValueError("invalid closest GT triangle")
        distance = np.linalg.norm(points - hit, axis=1)
        return distance, world.triangle_classes[triangle], hit, triangle

    def solid_inside(points, distance, primitives):
        # Closed primitive membership plus distance from the *union exterior*
        # correctly classifies interior contact planes between touching solids.
        membership = np.zeros(len(points), bool)
        for primitive in primitives:
            low = np.asarray(primitive[:3]); high = low + np.asarray(primitive[3:6])
            membership |= np.all(points >= low - inside_tolerance, axis=1) & np.all(points <= high + inside_tolerance, axis=1)
        return membership & (distance > inside_tolerance)

    def fraction(weights, mask):
        total = float(weights.sum())
        return float(weights[mask].sum() / total) if total > 0 else None

    def dimensions_audit(obj, world, instance_counts):
        if not instance_counts or max(instance_counts.values()) == 0:
            return {"nearest_supported_instance": None}
        instance_id = max(instance_counts, key=lambda key: (instance_counts[key], -int(key)))
        target = world.objects[int(instance_id)]
        low = np.array(target["box"][:3]); dims = np.array(target["box"][3:6]); center = low + dims / 2
        return {"nearest_supported_instance": int(instance_id),
                "gt_instance_center": center.tolist(), "gt_instance_dims": dims.tolist(),
                "model_center_error_m": float(np.linalg.norm(obj["center"] - center)),
                "model_dims_error_m": (obj["dims"] - dims).tolist()}

    history_rows, object_rows, shape_rows = [], [], []
    for fixture in sorted(source.glob("*/fixture.json")):
        folder = fixture.parent
        fixture_info = read_json(fixture)
        c = InspectionConfigV4(**fixture_info["environment"])
        world = InspectionWorldV4(c, seed=fixture_info["entry"]["seed"])
        with np.load(folder / "reference.npz", allow_pickle=False) as reference:
            np.testing.assert_array_equal(np.asarray(world.mesh.vertices), reference["vertices"])
            np.testing.assert_array_equal(np.asarray(world.mesh.triangles), reference["triangles"])
        truth = ray_scene(world.mesh)
        records = read_json(folder / "prefix/records.json")
        histories = sorted(folder.glob("history_*/predictions.json"))
        for prediction_file in histories:
            history_folder = prediction_file.parent
            index = int(history_folder.name.split("_")[-1])
            predicted = read_json(prediction_file)
            seal = read_json(folder / "pre_outcome_seal.json")
            if file_hash(prediction_file) != seal[str(prediction_file.relative_to(folder))]:
                raise ValueError("prediction was not sealed before outcome evaluation")
            mappers = {mode: SemanticHistoryMapperV3(world.shape, c, c.truncation_m)
                       for mode in ("aligned", "shuffled", "absent")}
            from dataclasses import replace
            for row in records[:index + 1]:
                step = int(row["step"])
                frame = RGBDFrame.load(folder / "prefix/frames" / f"{step:04d}.npz")
                scan = PlanarScan.load(folder / "prefix/scans" / f"{step:04d}.npz")
                for mode, mapper in mappers.items():
                    labels = frame.semantic
                    if mode == "shuffled":
                        labels = np.where(labels == 2, 3, np.where(labels == 3, 2, labels)).astype(labels.dtype)
                    elif mode == "absent":
                        labels = np.zeros_like(labels)
                    mapper.update(replace(frame, semantic=labels), scan)
                    if row["collision"]:
                        dr, dc = DIRECTIONS[row["heading"]]
                        r, col = row["position"][0] + dr, row["position"][1] + dc
                        if 0 <= r < mapper.shape[0] and 0 <= col < mapper.shape[1]:
                            mapper.belief[r, col] = 1
            for mode, mapper in mappers.items():
                actual = _mapper_snapshot(mapper)["hashes"]
                if actual != predicted["invariants"]["mapper_hashes"][mode]:
                    raise ValueError(f"prefix replay differs from scoring snapshot: {history_folder}/{mode}")
            models = {"G": ObjectCompletionModel(mappers["aligned"], False),
                      "O": ObjectCompletionModel(mappers["aligned"], True, fine_categories=False),
                      "S": ObjectCompletionModel(mappers["aligned"], True),
                      "X": ObjectCompletionModel(mappers["shuffled"], True),
                      "M": ObjectCompletionModel(mappers["absent"], True)}
            for name, model in models.items():
                if _object_geometry_hash(model) != predicted["invariants"]["shape_geometry_sha256"][name]:
                    raise ValueError(f"hypothesis geometry changed: {name}")
                if _array_hash([("weights", model.weights)]) != predicted["invariants"]["shape_weights_sha256"][name]:
                    raise ValueError(f"hypothesis weights changed: {name}")
            base = models["G"]
            groups = measured_groups(mappers["aligned"])
            cluster = _cluster_statistics(mappers["aligned"])
            if len(groups) != len(base.objects):
                raise ValueError("measured cluster reconstruction and object model differ")
            points = base.points
            distance, classes, closest_points, triangles = closest(truth, world, points)
            interior = solid_inside(points, distance, world._solid_primitives)
            near, far = distance <= threshold, distance > threshold
            point_records = []
            local_objects, local_shapes = [], []
            common = {"fixture": folder.name, "history_step": index,
                      "seed": int(fixture_info["entry"]["seed"]), "depth_sigma_m": c.depth_sigma_m}
            for object_index, (obj, group) in enumerate(zip(base.objects, groups)):
                measured = group["points"]
                if len(measured) != obj["observed_points"]:
                    raise ValueError("measured point count differs from frozen hypothesis")
                d, cat, surface_points, tri = closest(truth, world, measured)
                near_measured = d <= threshold
                count = len(measured)
                class_fractions = {str(category): float(np.mean(cat == category)) for category in (1, 2, 3)}
                near_fractions = {str(category): float(np.mean((cat == category) & near_measured)) for category in (1, 2, 3)}
                near_conditional = {str(category): (float(np.mean(cat[near_measured] == category))
                                    if near_measured.any() else None) for category in (1, 2, 3)}
                instance_counts = {}
                for i, target in enumerate(world.objects):
                    low = np.array(target["box"][:3]); high = low + np.array(target["box"][3:6])
                    match = np.all(surface_points >= low - 1e-6, axis=1) & np.all(surface_points <= high + 1e-6, axis=1)
                    match &= (cat == target["category"]) & near_measured
                    instance_counts[str(i)] = int(match.sum())
                labels = group["labels"]
                marker = bool(np.any((labels == 2) | (labels == 3)))
                # Both the denominator and the threshold are explicit. A lack
                # of markers is never sufficient to assign this GT diagnosis.
                background = near_fractions["1"] > .5
                record = {**common, "object_index": object_index, "group_id": group["group_id"],
                          "observed_points": count, "marker_supported": marker,
                          "label_counts": {str(int(v)): int((labels == v).sum()) for v in np.unique(labels)},
                          "model_center": obj["center"].tolist(), "model_dims": obj["dims"].tolist(),
                          "measured_nearest_physical_class_fractions": class_fractions,
                          "measured_near_5cm_class_fractions_all_points": near_fractions,
                          "measured_near_5cm_class_fractions_conditional": near_conditional,
                          "measured_near_5cm_fraction": float(near_measured.mean()),
                          "measured_gt_distance_mean_m": float(d.mean()),
                          "measured_gt_distance_p95_m": float(np.percentile(d, 95)),
                          "instance_support_fractions_all_points": {i: n / count for i, n in instance_counts.items()},
                          "gt_background_majority_within_5cm": background,
                          **dimensions_audit(obj, world, instance_counts)}
                local_objects.append(record); object_rows.append(record)
                point_records.append((measured, np.full(count, object_index), d, cat, tri))
                for shape_number, shape_name in enumerate(("box", "shelf")):
                    selection = base.hypothesis_ids == 2 * object_index + shape_number
                    shape_points = points[selection]
                    quantized = np.rint(shape_points / duplicate_tolerance).astype(np.int64)
                    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
                    for scorer in scorers:
                        model = models[scorer]; weights = model.weights[selection]
                        total = float(weights.sum())
                        maxima = np.zeros(int(inverse.max()) + 1 if len(inverse) else 0)
                        np.maximum.at(maxima, inverse, weights)
                        row = {**common, "object_index": object_index, "shape": shape_name,
                               "scorer": scorer, "hypothesis_id": 2 * object_index + shape_number,
                               "marker_supported": marker, "gt_background_majority_within_5cm": background,
                               "point_count": int(selection.sum()), "potential_weighted_area_m2": total,
                               "shelf_prior_probability": model.objects[object_index]["shelf_prior_probability"],
                               "shelf_posterior_probability": model.objects[object_index]["shelf_probability"],
                               "object_probability": model.objects[object_index]["object_probability"],
                               "near_5cm_weight_fraction": fraction(weights, near[selection]),
                               "solid_interior_weight_fraction": fraction(weights, interior[selection]),
                               "far_from_exterior_weight_fraction": fraction(weights, far[selection]),
                               "far_inside_weight_fraction": fraction(weights, (far & interior)[selection]),
                               "far_outside_weight_fraction": fraction(weights, (far & ~interior)[selection]),
                               "gt_distance_weighted_mean_m": float(weights @ distance[selection] / total) if total > 0 else None,
                               "duplicate_sample_excess_weight_m2": max(0., total - float(maxima.sum())),
                               "nearest_physical_class_weight_fractions": {
                                   str(category): fraction(weights, classes[selection] == category) for category in (1, 2, 3)}}
                        local_shapes.append(row); shape_rows.append(row)
            target = output / folder.name / history_folder.name
            target.mkdir(parents=True)
            write_json(target / "objects.json", {"objects": local_objects, "hypotheses": local_shapes})
            arrays = {"hypothesis_points": points, "hypothesis_ids": base.hypothesis_ids,
                      "gt_distance_m": distance, "nearest_gt_class": classes, "nearest_gt_triangle": triangles,
                      "solid_interior": interior, "scorers": np.array(scorers),
                      "weights": np.stack([models[name].weights for name in scorers])}
            if point_records:
                for j, name in enumerate(("measured_points", "measured_object_index", "measured_gt_distance_m",
                                           "measured_nearest_gt_class", "measured_nearest_gt_triangle")):
                    arrays[name] = np.concatenate([row[j] for row in point_records])
            np.savez_compressed(target / "point_diagnostics.npz", **arrays)
            history_rows.append({**common, "objects": len(groups),
                                 "marker_supported_objects": sum(row["marker_supported"] for row in local_objects),
                                 "gt_background_supported_objects": sum(row["gt_background_majority_within_5cm"] for row in local_objects),
                                 "cluster_statistics": cluster, "prefix_replay_matches_scores": True,
                                 "hypothesis_geometry_and_weights_match_scores": True,
                                 "truth_union_surface_audit": world.surface_measure_audit})
            print("diagnosed", folder.name, history_folder.name, "objects", len(groups), flush=True)
            del mappers, models, base

    # Aggregate weighted areas, never average ratios of tiny and large objects.
    summary = {}
    ratio_fields = ("near_5cm_weight_fraction", "solid_interior_weight_fraction",
                    "far_from_exterior_weight_fraction", "far_inside_weight_fraction", "far_outside_weight_fraction")
    for scorer in scorers:
        rows = [row for row in shape_rows if row["scorer"] == scorer]
        total = sum(row["potential_weighted_area_m2"] for row in rows)
        summary[scorer] = {"potential_weighted_area_m2": total,
                          "unmarked_weighted_area_m2": sum(row["potential_weighted_area_m2"] for row in rows if not row["marker_supported"]),
                          "gt_background_supported_weighted_area_m2": sum(row["potential_weighted_area_m2"] for row in rows if row["gt_background_majority_within_5cm"]),
                          "duplicate_sample_excess_weight_m2": sum(row["duplicate_sample_excess_weight_m2"] for row in rows)}
        for name in ratio_fields:
            summary[scorer][name] = (sum(row["potential_weighted_area_m2"] * (row[name] or 0.) for row in rows) / total if total > 0 else None)
    write_json(output / "summary.json", {
        "histories": history_rows, "scorer_totals": summary, "threshold_m": threshold,
        "solid_interior_tolerance_m": inside_tolerance, "duplicate_coordinate_tolerance_m": duplicate_tolerance,
        "background_rule": "more than half of measured cluster points lie <=5cm from a GT class-1 exterior triangle",
        "scope": "post-score GT diagnostic; no scores, selections or primary metrics modified",
        "limitations": ["Nearest class is a geometric association; distance and ambiguous far support must be considered.",
                       "Measured cluster support counts are sample proportions, not physical area proportions.",
                       "Potential shape weights are model hypotheses, not unique physical exterior area.",
                       "Interior and <=5cm categories overlap near actual surfaces; far_inside/far_outside separate distant errors.",
                       "Duplicate-coordinate excess is a sampled diagnostic, not a complete coplanar overlap measure.",
                       "Historical/noise repetitions are not independent scenes; pooled totals are descriptive.",
                       "This script verifies prefix and shape replay only, not complete future branch execution."],
        "runtime": {"python": sys.version, "numpy": np.__version__, "open3d": o3d.__version__},
    })
    for name, rows in (("objects.csv", object_rows), ("hypotheses.csv", shape_rows)):
        with open(output / name, "w", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                                     for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.snapshot is None:
            parser.error("internal worker requires a validated snapshot")
        worker(args.source.resolve(), args.output.resolve(), args.snapshot.resolve())
    else:
        run(args.source, args.output)


if __name__ == "__main__":
    main()
