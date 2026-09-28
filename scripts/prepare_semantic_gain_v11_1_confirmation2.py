#!/usr/bin/env python3
"""Prepare sealed E prefixes and evaluator references without future actions."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.virtual3d_confirmation_v11_2 import (create_confirmation_world,
    collect_confirmation_prefix, load_confirmation_contexts)
from nso.competition_candidates_v8 import measured_assets
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from scripts.prepare_response_v7 import records_from_prefix, array_hash
from utils.counterfactual_surface_visibility import reference_visible
from utils.reconstruction_metrics import ReconstructionEvaluator


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n")


def mapper_from(raw, shape, config):
    mapper = ObservedRuntimeMapperV10(shape, config)
    for row in raw:
        mapper.update(row["frame"], row["scan"])
    return mapper


def main(output, freeze):
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((freeze / "receipt.json").read_text())
    frozen_hashes = json.loads((freeze / "artifact_hashes.json").read_text())
    if any(sha(freeze / name) != digest for name, digest in frozen_hashes.items()):
        raise RuntimeError("confirmation freeze artifact changed")
    if any(sha(ROOT / name) != digest for name, digest in receipt["source_sha256"].items()):
        raise RuntimeError("confirmation source differs from pre-world freeze")
    manifest = load_confirmation_contexts()
    output.mkdir(parents=True)
    metadata = {"schema_version": "semantic_gain_v11_1_confirmation2_preparation/1",
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_receipt_sha256": sha(freeze / "receipt.json"),
        "future_adaptive_actions_executed": 0, "future_outcomes_viewed": 0,
        "context_summaries": []}
    write(output / "metadata.json", metadata)
    try:
        for context_row in manifest["contexts"]:
            context_id = context_row["context_id"]
            paired = []
            for arrangement in manifest["arrangement_order"]:
                world = create_confirmation_world(context_id, arrangement)
                raw = collect_confirmation_prefix(world)
                expected = world.context.prefix_step + 1
                if len(raw) != expected or world.collisions:
                    raise RuntimeError("sealed prefix failed")
                actual_xy = raw[-1]["frame"].world_from_camera[:2, 3]
                if (not np.allclose(actual_xy, world.context.final_prefix_xy_m, rtol=0, atol=1e-12)
                        or world.heading != world.context.final_prefix_heading):
                    raise RuntimeError("sealed prefix endpoint mismatch")
                folder = output / context_id / arrangement
                (folder / "prefix" / "frames").mkdir(parents=True)
                (folder / "prefix" / "scans").mkdir()
                records = records_from_prefix(raw, world.shape, world.config)
                for index, row in enumerate(raw):
                    row["frame"].save(folder / "prefix" / "frames" / f"{index:04d}.npz")
                    row["scan"].save(folder / "prefix" / "scans" / f"{index:04d}.npz")
                write(folder / "prefix" / "records.json", records)
                write(folder / "fixture.json", {"context": asdict(world.context),
                    "arrangement": arrangement, "config": asdict(world.config)})
                mapper = mapper_from(raw, world.shape, world.config)
                assets = measured_assets(mapper)
                mesh = mapper.mesh()
                quality = mapper.quality_evidence(max_points=10**9)
                geometry_hash = array_hash([("belief", mapper.belief),
                    ("camera_seen", mapper.camera_seen), ("vertices", np.asarray(mesh.vertices)),
                    ("triangles", np.asarray(mesh.triangles))] + ([] if quality is None else
                    [(name, quality[name]) for name in sorted(quality) if name != "label"]))
                np.savez_compressed(folder / "prefix_map.npz", belief=mapper.belief,
                                    camera_seen=mapper.camera_seen)
                evaluator = ReconstructionEvaluator(world, count=32000, seed=2026)
                weight = float(world.mesh.get_surface_area()) / 32000
                seen = np.zeros(len(evaluator.reference), bool)
                for row in raw:
                    seen |= reference_visible(evaluator.reference, row["frame"], evaluator.truth,
                        row["frame"].world_from_camera, world.config.max_depth_m)
                np.savez_compressed(folder / "reference.npz", points=evaluator.reference,
                    classes=evaluator.classes, weights=np.full(len(evaluator.reference), weight),
                    prefix_seen=seen, reachable=world.reachable,
                    vertices=np.asarray(world.mesh.vertices), triangles=np.asarray(world.mesh.triangles))
                coverage = float(np.count_nonzero((mapper.belief == 0) & world.reachable)
                                 / np.count_nonzero(world.reachable))
                summary = {"context": context_id, "stratum": world.context.stratum,
                    "arrangement": arrangement, "prefix_paid_actions": world.context.prefix_step,
                    "frames": len(raw), "coverage_2d": coverage,
                    "measured_assets": len(assets),
                    "both_assets_have_marker_support": len(assets) == 2 and all(
                        asset["marked_points"] > 0 for asset in assets),
                    "geometry_sha256": geometry_hash, "collision_free_prefix": world.collisions == 0}
                write(folder / "structural_audit.json", summary)
                metadata["context_summaries"].append(summary)
                paired.append({"raw": raw, "records": records, "summary": summary,
                               "belief": mapper.belief.copy(), "camera_seen": mapper.camera_seen.copy()})
                print("prepared", context_id, arrangement, "coverage", coverage,
                      "assets", len(assets), flush=True)
            first, second = paired
            if first["records"] != second["records"]:
                raise AssertionError("paired motion records differ")
            for left, right in zip(first["raw"], second["raw"]):
                np.testing.assert_array_equal(left["frame"].depth_m, right["frame"].depth_m)
                np.testing.assert_array_equal(left["scan"].ranges_m, right["scan"].ranges_m)
                np.testing.assert_array_equal(left["frame"].semantic > 0,
                                              right["frame"].semantic > 0)
            np.testing.assert_array_equal(first["belief"], second["belief"])
            np.testing.assert_array_equal(first["camera_seen"], second["camera_seen"])
        efficacy = [r for r in metadata["context_summaries"] if r["stratum"] == "efficacy"]
        stress = [r for r in metadata["context_summaries"] if r["stratum"] == "coverage_stress"]
        gates = {"all_24_histories_prepared": len(metadata["context_summaries"]) == 24,
            "all_prefixes_collision_free": all(r["collision_free_prefix"]
                                                for r in metadata["context_summaries"]),
            "efficacy_has_two_marked_assets": all(r["both_assets_have_marker_support"]
                                                   for r in efficacy),
            "stress_starts_below_efficacy_coverage": max(r["coverage_2d"] for r in stress)
                < min(r["coverage_2d"] for r in efficacy)}
        metadata["gates"] = gates
        metadata["status"] = "complete_structural_pass" if all(gates.values()) else "halted_structural_failure"
        metadata["future_adaptive_actions_executed"] = 0
        metadata["future_outcomes_viewed"] = 0
    except Exception as exc:
        metadata["status"] = "failed_preparation"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write(output / "metadata.json", metadata)
        write(output / "artifact_hashes.json", {str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*")) if path.is_file()
            and path.name != "artifact_hashes.json"})
    if metadata["status"] != "complete_structural_pass":
        raise RuntimeError("E preparation structural gate failed; retain without future execution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve(), args.freeze.resolve())
