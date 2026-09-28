#!/usr/bin/env python3
"""Construct and audit V12 development worlds without running a planner."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from env.virtual3d_inspection_v4 import InspectionConfigV4, InspectionWorldV4


def digest(array):
    value = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path,
                        default=Path("configs/virtual3d/semantic_gain_v12_scale_development.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol.get("status") != "development_scenes_only_no_planner_outcomes_viewed":
        raise ValueError("V12 preparation accepts the development-only protocol")

    output = args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    rows = []
    for declared in protocol["contexts"]:
        settings = {**protocol["shared_conditions"],
                    **{key: value for key, value in declared.items() if key not in ("id", "seed")}}
        config = InspectionConfigV4(**settings)
        base = InspectionWorldV4(config, seed=declared["seed"], semantic_condition="aligned")
        base.position, base.heading = base.inspection_truth[0]["front_pose"]
        base_frame = base.sense()
        base_scan = base.scan()
        physical = {
            "vertices": digest(np.asarray(base.mesh.vertices)),
            "triangles": digest(np.asarray(base.mesh.triangles)),
            "triangle_classes": digest(base.triangle_classes),
            "occupancy": digest(base.occupancy),
            "depth": digest(base_frame.depth_m),
            "rgb": digest(base_frame.color_rgb),
            "laser": digest(base_scan.ranges_m),
        }
        semantics = {"aligned": digest(base_frame.semantic)}
        aligned_semantic = base_frame.semantic.copy()
        for intervention in ("shuffled", "absent"):
            world = InspectionWorldV4(config, seed=declared["seed"],
                                      semantic_condition=intervention)
            world.position, world.heading = world.inspection_truth[0]["front_pose"]
            frame, scan = world.sense(), world.scan()
            checks = {
                "vertices": digest(np.asarray(world.mesh.vertices)),
                "triangles": digest(np.asarray(world.mesh.triangles)),
                "triangle_classes": digest(world.triangle_classes),
                "occupancy": digest(world.occupancy),
                "depth": digest(frame.depth_m),
                "rgb": digest(frame.color_rgb),
                "laser": digest(scan.ranges_m),
            }
            if checks != physical:
                raise RuntimeError(f"physical intervention mismatch in {declared['id']}/{intervention}")
            if intervention == "shuffled":
                expected = np.where(aligned_semantic == 2, 3,
                            np.where(aligned_semantic == 3, 2, aligned_semantic)).astype(np.uint8)
                np.testing.assert_array_equal(frame.semantic, expected)
            else:
                if np.any(frame.semantic):
                    raise RuntimeError(f"absent semantics leaked in {declared['id']}")
            semantics[intervention] = digest(frame.semantic)

        true_categories = [row["true_category"] for row in base.inspection_truth]
        visible_categories = [row["visible_category"] for row in base.inspection_truth]
        rows.append({
            "id": declared["id"],
            "seed": declared["seed"],
            "config": asdict(config),
            "width_m": base.width,
            "height_m": base.height,
            "grid_shape": list(base.shape),
            "asset_count": len(base.inspection_truth),
            "complex_asset_count": int(np.count_nonzero(np.asarray(true_categories) == 3)),
            "visible_class_matches_truth": int(np.count_nonzero(
                np.asarray(true_categories) == np.asarray(visible_categories))),
            "reachable_free_cells": int(base.reachable.sum()),
            "all_free_cells_reachable": bool(np.array_equal(base.reachable, ~base._blocked)),
            "surface_area_m2": float(base.mesh.get_surface_area()),
            "physical_hashes": physical,
            "semantic_hashes": semantics,
            "physical_inputs_identical_across_interventions": True,
        })

    manifest = {
        "schema_version": "semantic_gain_v12_scale_preparation/1",
        "status": "development_world_construction_passed_no_planner_run",
        "protocol": str(args.protocol),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "contexts": rows,
        "context_count": len(rows),
        "interventions_per_context": 3,
        "planner_outcomes_generated": 0,
        "truth_used_for_planning": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "contexts": len(rows),
                      "assets": [row["asset_count"] for row in rows],
                      "widths_m": [row["width_m"] for row in rows]}))


if __name__ == "__main__":
    main()
