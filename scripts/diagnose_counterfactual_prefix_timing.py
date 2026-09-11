#!/usr/bin/env python3
"""Fixed-interval sensor-only audit of the four original 751/752 prefixes.

No world is constructed, GT/future branches are not read, and no physical
action is executed. The original archive supplies mapper, model, candidates
and rate scorer. Every scheduled checkpoint is kept, including unavailable
ones; original 80/160 predictions are reproduced exactly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("archive/input path escaped root")
    return path


def input_manifest(source):
    meta = read(source / "metadata.json")
    if meta.get("status") != "complete":
        raise ValueError("physical source must be complete")
    original = read(source / "artifact_hashes.json")
    paths = [source / name for name in ("metadata.json", "config.json", "sources.zip")]
    fixtures = sorted(source.glob("*/fixture.json"))
    entries = [read(path)["entry"] for path in fixtures]
    observed = sorted((entry["seed"], entry["environment"]["depth_sigma_m"]) for entry in entries)
    if observed != [(751, .01), (751, .03), (752, .01), (752, .03)]:
        raise ValueError("only the four original 751/752 fixtures are authorized")
    for fixture in fixtures:
        folder = fixture.parent
        paths += [fixture, folder / "prefix/records.json", folder / "pre_outcome_seal.json"]
        paths += list((folder / "prefix/frames").glob("*.npz"))
        paths += list((folder / "prefix/scans").glob("*.npz"))
        for prediction in folder.glob("history_*/predictions.json"):
            paths += [prediction, prediction.parent / "candidates.json"]
    result = {}
    for path in paths:
        name = str(path.relative_to(source)); actual = sha(path)
        if name in original and actual != original[name]:
            raise ValueError(f"frozen input changed: {name}")
        if name != "sources.zip" and name not in original:
            raise ValueError(f"input missing from original manifest: {name}")
        result[name] = actual
    return meta, result


def run(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("new separate output directory required")
    meta, inputs = input_manifest(source)
    output.mkdir(parents=True)
    write(output / "input_hashes.json", inputs)
    audit = {"status": "running", "scope": "sensor-only fixed prefix timing; development descriptive audit",
             "source": str(source), "script_sha256": sha(__file__), "source_archive_sha256": sha(source / "sources.zip"),
             "workers": 1, "checkpoints": list(range(0, 161, 10)), "requested_slots": 68,
             "GT_read": False, "future_branches_read": False, "new_physical_actions": 0,
             "score_objective": "original archived rate", "started_unix": time.time()}
    write(output / "metadata.json", audit)
    try:
        with tempfile.TemporaryDirectory(prefix="nso_prefix_timing_") as temporary:
            snapshot = Path(temporary)
            with zipfile.ZipFile(source / "sources.zip") as archive:
                for info in archive.infolist():
                    target = safe(snapshot, info.filename)
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(info))
            for name, expected in meta["source_sha256"].items():
                if sha(safe(snapshot, name)) != expected:
                    raise ValueError(f"archive hash mismatch: {name}")
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(snapshot))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--snapshot", str(snapshot),
                            "--source", str(source), "--output", str(output)], cwd=snapshot, env=env, check=True)
        if input_manifest(source)[1] != inputs:
            raise ValueError("frozen input changed during diagnosis")
        audit.update(status="complete", elapsed_s=time.time() - audit["started_unix"])
    except Exception as error:
        audit.update(status="failed", error=repr(error))
        write(output / "metadata.json", audit)
        raise
    write(output / "metadata.json", audit)
    write(output / "artifact_hashes.json", {str(p.relative_to(output)): sha(p) for p in output.rglob("*")
                                            if p.is_file() and p.name != "artifact_hashes.json"})


def worker(source, output, snapshot):
    sys.path.insert(0, str(snapshot))
    from dataclasses import replace
    import numpy as np
    from scipy.ndimage import binary_closing, label
    from env.virtual3d_inspection_v4 import InspectionConfigV4, InspectionWorldV4
    from nso.semantic_completion_v3 import SemanticHistoryMapperV3
    from nso.counterfactual_view_scoring import score_routes, CounterfactualScoreConfig
    from scripts.eval_counterfactual_views import candidate_routes, grid_config, update_collision
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    import nso.counterfactual_view_scoring as scoring
    if not Path(scoring.__file__).resolve().is_relative_to(snapshot):
        raise ValueError("live scorer imported")
    # Make an accidental simulator construction fail, including through the
    # imported candidate helper. Configuration and sensor history suffice.
    def no_world(*args, **kwargs):
        raise RuntimeError("GT world construction is forbidden in timing diagnosis")
    InspectionWorldV4.__init__ = no_world

    def cluster_details(mapper, first_quality_marker, prediction):
        q = mapper.quality_evidence(max_points=10000)
        if q is None:
            return []
        items = [(key, row) for key, row in mapper.quality.items() if .12 < row["point"][2] < 1.8]
        if len(items) > 10000:
            items = [items[i] for i in np.linspace(0, len(items)-1, 10000, dtype=int)]
        np.testing.assert_array_equal(np.array([value["point"] for _, value in items]), q["point"])
        points = q["point"]
        rows = mapper.shape[0]-1-np.floor(points[:, 1]/.2).astype(int)
        cols = np.floor(points[:, 0]/.2).astype(int)
        inside = (rows >= 0) & (rows < mapper.shape[0]) & (cols >= 0) & (cols < mapper.shape[1])
        occupied = np.zeros(mapper.shape, bool); occupied[rows[inside], cols[inside]] = True
        groups, _ = label(binary_closing(occupied, structure=np.ones((3, 3))))
        ids = np.zeros(len(points), int); ids[inside] = groups[rows[inside], cols[inside]]
        result = []; accepted_index = 0
        for group in sorted(set(ids) - {0}):
            selected = np.flatnonzero(ids == group)
            low, high = np.quantile(points[selected], [.05, .95], axis=0); span = high-low
            reasons = []
            if len(selected) < 8: reasons.append("fewer_than_8_quality_points")
            if max(span[:2]) > 2.2: reasons.append("horizontal_quantile_span_above_2.2m")
            if span[2] < .25: reasons.append("vertical_quantile_span_below_0.25m")
            labels = q["label"][selected]
            marked = np.isin(labels, [2, 3])
            marker_steps = [first_quality_marker[(items[i][0], int(q["label"][i]))]
                            for i in selected if q["label"][i] in (2, 3)]
            bits = q["bits"][selected]
            bit_count = [int(value).bit_count() for value in bits]
            row = {"group_id": int(group), "accepted": not reasons, "rejection_reasons": reasons,
                   "object_index": accepted_index if not reasons else None,
                   "measured_points": len(selected), "measured_center_mean_m": points[selected].mean(axis=0).tolist(),
                   "quantile_span_m": span.tolist(),
                   "label_counts": {str(int(v)): int(np.count_nonzero(labels == v)) for v in np.unique(labels)},
                   "marker_supported": bool(marked.any()),
                   "truth_status": "unknown; this diagnostic does not read GT",
                   "first_marker_support_step_from_current_member_keys": min(marker_steps) if marker_steps else None,
                   "member_keys_sha256": hashlib.sha256(np.array([items[i][0] for i in selected], dtype=np.int64).tobytes()).hexdigest(),
                   "azimuth_bits_fraction": [float(np.mean((bits & (1 << b)) != 0)) for b in range(8)],
                   "azimuth_distinct_bins_per_point_histogram": {str(n): bit_count.count(n) for n in sorted(set(bit_count))}}
            if not reasons:
                g = prediction["objects"]["G"][accepted_index]; sem = prediction["objects"]["S"][accepted_index]
                row.update(center_m=g["center"], dims_m=g["dims"], G=g, S=sem,
                           geometry_log_likelihood_ratio=(g["fit_errors_m2"][0]-g["fit_errors_m2"][1])/(2*.06**2),
                           posterior_absolute_separation=abs(g["shelf_probability"]-sem["shelf_probability"]))
                accepted_index += 1
            result.append(row)
        if accepted_index != len(prediction["objects"]["G"]):
            raise ValueError("cluster acceptance differs from archived model")
        return result

    all_rows = []; summaries = []; matched = []
    for fixture in sorted(source.glob("*/fixture.json")):
        folder = fixture.parent; info = read(fixture); c = InspectionConfigV4(**info["environment"])
        shape = (round(c.height_m/c.resolution_m), round(c.width_m/c.resolution_m))
        mappers = {mode: SemanticHistoryMapperV3(shape, c, c.truncation_m) for mode in ("aligned", "shuffled", "absent")}
        records = read(folder / "prefix/records.json")
        if [r["step"] for r in records] != list(range(len(records))):
            raise ValueError("nonconsecutive raw prefix")
        rows = []; first_raw = {}; first_quality = {}; raw_pixel_totals = {"2": 0, "3": 0}
        first_quality_marker = {}; first_accepted = None
        target = output / folder.name; target.mkdir()
        for record in records:
            step = record["step"]
            if step > 160: break
            frame = RGBDFrame.load(folder / "prefix/frames" / f"{step:04d}.npz")
            scan = PlanarScan.load(folder / "prefix/scans" / f"{step:04d}.npz")
            pixels = {str(cls): int(np.count_nonzero(frame.semantic == cls)) for cls in (2, 3)}
            for cls, count in pixels.items():
                raw_pixel_totals[cls] += count
                if count and cls not in first_raw:
                    first_raw[cls] = step
                    print("first_raw_marker", folder.name, "class", cls, "step", step, flush=True)
            for mode, mapper in mappers.items():
                labels = frame.semantic
                if mode == "shuffled": labels = np.where(labels == 2, 3, np.where(labels == 3, 2, labels)).astype(labels.dtype)
                elif mode == "absent": labels = np.zeros_like(labels)
                mapper.update(replace(frame, semantic=labels), scan)
                update_collision(mapper, record["position"], record["heading"], record["collision"])
            mapper = mappers["aligned"]
            for key, value in mapper.quality.items():
                cls = int(value["label"])
                if cls in (2, 3):
                    first_quality_marker.setdefault((key, cls), step)
                    first_quality.setdefault(str(cls), step)
            if step % 10: continue
            common = {"fixture": folder.name, "seed": info["entry"]["seed"], "depth_sigma_m": c.depth_sigma_m,
                      "step": step, "position": record["position"], "heading": record["heading"]}
            if record["done"]:
                rows.append({**common, "status": "unavailable_prefix_stopped", "last_step": records[-1]["step"]})
                continue
            obs = mapper.observation(tuple(record["position"]), record["heading"], step, record["collision"])
            routes, candidate_audit = candidate_routes(mapper, obs, max_actions=48, max_candidates=12)
            prediction = score_routes(mappers, routes, CounterfactualScoreConfig(grid_config(c), c))
            # JSON canonicalization removes tuples without changing numbers.
            prediction = json.loads(json.dumps(prediction)); routes_json = json.loads(json.dumps(routes))
            sealed_prediction = folder / f"history_{step:04d}" / "predictions.json"
            if sealed_prediction.exists():
                seal = read(folder / "pre_outcome_seal.json")
                if sha(sealed_prediction) != seal[str(sealed_prediction.relative_to(folder))]:
                    raise ValueError("original prediction seal changed")
                if prediction != read(sealed_prediction):
                    raise ValueError(f"original full prediction failed exact reproduction: {folder.name}/{step}")
                if routes_json != read(sealed_prediction.parent / "candidates.json"):
                    raise ValueError("original candidate routes failed exact reproduction")
                matched.append({"fixture": folder.name, "step": step, "original_prediction_sha256": sha(sealed_prediction),
                                "full_prediction_equal": True, "full_routes_equal": True})
            clusters = cluster_details(mapper, first_quality_marker, prediction)
            accepted_marked = sum(r["accepted"] and r["marker_supported"] for r in clusters)
            if accepted_marked and first_accepted is None: first_accepted = step
            candidates = []
            by_id = {r["candidate_id"]: r for r in prediction["candidates"]}
            for route in routes:
                entry = by_id[route["candidate_id"]]
                candidates.append({"candidate_id": route["candidate_id"], "group": route["group"],
                    "pose": route["pose"], "cost": route["cost"], "arrival_action": route["arrival_action"],
                    "is_coverage_anchor": route["is_coverage_anchor"],
                    "paid_actions": "".join({"forward": "F", "left": "L", "right": "R"}[a] for a in route["actions"]),
                    "states_sha256": entry["states_sha256"], "scores": entry["scores"]})
            rejected_counts = {group: sum(row["group"] == group for row in candidate_audit.get("rejected", []))
                               for group in ("frontier", "object", "uniform")}
            q = mapper.quality_evidence(max_points=10000)
            quality_labels = np.array([]) if q is None else q["label"]
            row = {**common, "status": "available", "current_marker_pixels": pixels,
                   "cumulative_marker_pixels": raw_pixel_totals.copy(), "first_raw_marker_step": first_raw.copy(),
                   "first_quality_marker_step": first_quality.copy(),
                   "quality_marker_point_counts": {str(cls): int(np.count_nonzero(quality_labels == cls)) for cls in (2, 3)},
                   "clusters": clusters, "accepted_marker_supported_clusters": accepted_marked,
                   "rejected_marker_supported_clusters": sum(not r["accepted"] and r["marker_supported"] for r in clusters),
                   "candidate_audit": {k: v for k, v in candidate_audit.items() if k != "rejected"},
                   "candidate_rejected_counts": rejected_counts, "candidates": candidates,
                   "selected": prediction["selected"], "rankings": prediction["rankings"],
                   "normalization": prediction["normalization"], "invariants": prediction["invariants"],
                   "scoring_diagnostics": prediction["diagnostics"]}
            rows.append(row)
            write(target / f"step_{step:04d}.json", row)
            print("checkpoint", folder.name, step, "marker_objects", accepted_marked, "candidates", len(candidates),
                  "selected_GS", prediction["selected"]["G"], prediction["selected"]["S"], flush=True)
        present = {r["step"] for r in rows}
        for step in range(0, 161, 10):
            if step not in present:
                rows.append({"fixture": folder.name, "seed": info["entry"]["seed"], "depth_sigma_m": c.depth_sigma_m,
                             "step": step, "status": "unavailable_prefix_stopped", "last_step": records[-1]["step"]})
        rows.sort(key=lambda r: r["step"])
        available = [r for r in rows if r["status"] == "available"]
        summary = {"fixture": folder.name, "seed": info["entry"]["seed"], "depth_sigma_m": c.depth_sigma_m,
                   "last_prefix_step": records[-1]["step"], "first_raw_marker_step": first_raw,
                   "first_quality_marker_step": first_quality, "first_sampled_accepted_marker_cluster_step": first_accepted,
                   "available_checkpoints": len(available), "unavailable_checkpoints": 17-len(available),
                   "GS_different_choice_steps": [r["step"] for r in available if r["selected"]["G"] != r["selected"]["S"]],
                   "raw_markers_but_no_accepted_marker_cluster_steps": [r["step"] for r in available
                       if sum(r["cumulative_marker_pixels"].values()) and not r["accepted_marker_supported_clusters"]],
                   "marker_objects_but_no_feasible_object_pose_steps": [r["step"] for r in available
                       if r["accepted_marker_supported_clusters"] and not r["candidate_audit"].get("pool_counts", {}).get("object", 0)]}
        summaries.append(summary); all_rows.extend(rows)
        write(target / "timeline.json", rows)
        print("fixture_summary", json.dumps(summary), flush=True)
        del mappers
    result = {"requested_checkpoints": 68, "available_checkpoints": sum(r["status"] == "available" for r in all_rows),
              "unavailable_checkpoints": sum(r["status"] != "available" for r in all_rows),
              "fixtures": summaries, "original_prediction_exact_matches": matched,
              "limits": ["Original coverage-policy prefixes only; no future counterfactual outcome is read or executed.",
                         "Unknown/unmarked clusters are not assigned true/background identities.",
                         "Marker first appearance is exact in saved frames/quality keys; accepted model first appearance is sampled every 10 steps.",
                         "Current cluster members determine per-cluster earliest support; geometry cluster IDs are not persistent object identities.",
                         "Pool feasibility and score differences are sensor-model diagnostics, not measured branch benefits.",
                         "All four fixtures and all scheduled slots remain in the report; no success-based subset."]}
    if len(all_rows) != 68 or len(matched) != 6:
        raise ValueError("fixed schedule or original checkpoint replay incomplete")
    write(output / "summary.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        if a.snapshot is None: p.error("internal worker requires snapshot")
        worker(a.source.resolve(), a.output.resolve(), a.snapshot.resolve())
    else:
        run(a.source, a.output)


if __name__ == "__main__":
    main()
