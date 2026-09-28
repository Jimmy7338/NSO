#!/usr/bin/env python3
"""Materialize six DEV mesh assets only; never construct a World or frames."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nso.scene_contract_v40 import canonical_json_bytes, load_json_strict, public_planner_spec
from nso.development_geometry_v40 import MAX_ASSET_BYTES, build_development_geometry, mesh_npz_bytes

DEFAULT_SOURCE = ROOT / "audit_results/v40_scene_contract_20260920"
DEFAULT_OUTPUT = ROOT / "audit_results/v40_p1_development_geometry_20260920"


def build_assets(source_dir=DEFAULT_SOURCE, output_dir=DEFAULT_OUTPUT):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite existing development geometry evidence")
    protocol_path = source_dir / "protocol.json"
    protocol = load_json_strict(protocol_path)
    source_hashes = {str(protocol_path): hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
    payloads, rows = {}, []
    for family in "ABCDEF":
        parent = f"DEV_{family}_00"
        spec_path = source_dir / "development" / f"{parent}.json"
        spec = load_json_strict(spec_path)
        source_hashes[str(spec_path)] = hashlib.sha256(spec_path.read_bytes()).hexdigest()
        geometry = build_development_geometry(spec, protocol)
        prefix = parent + "/"
        payloads[prefix + "renderer_private/geometry.npz"] = mesh_npz_bytes(geometry)
        payloads[prefix + "renderer_private/markers.json"] = canonical_json_bytes({
            "schema_version": "v40.development_marker_planes.v1", "marker_patches": geometry.marker_patches,
            "application": "color only actual first-hit points on these existing solid face patches",
            "whole_instance_color_semantics": False})
        payloads[prefix + "evaluation_private/instances.json"] = canonical_json_bytes({
            "schema_version": "v40.development_private_instances.v1", "parent_id": parent,
            "family": family, "private_instances": geometry.private_instances,
            "background_boxes": geometry.background_boxes, "public_workspace": geometry.public_workspace,
            "actual_geometry_gt": True, "observable_evaluation_target_set_status": "pending",
            "ground_contact_bottom_faces_need_visibility_filtering": True})
        payloads[prefix + "evaluation_private/mesh_audit.json"] = canonical_json_bytes(geometry.audit)
        payloads[prefix + "public_workspace.json"] = canonical_json_bytes(geometry.public_workspace)
        payloads[prefix + "public_planner_spec.json"] = canonical_json_bytes(public_planner_spec(spec, protocol=protocol))
        rows.append({"parent_id": parent, "family": family, "instances": len(geometry.private_instances),
            "vertices": len(geometry.vertices), "triangles": len(geometry.triangles),
            "mesh_logical_sha256": geometry.logical_sha256(),
            "geometry_relative_path": prefix + "renderer_private/geometry.npz",
            "private_instances_relative_path": prefix + "evaluation_private/instances.json"})
    sources = [ROOT / "nso/development_geometry_v40.py", Path(__file__).resolve(),
               ROOT / "tests/test_development_geometry_v40_p1.py", ROOT / "docs/research/V40_P1_GEOMETRY_20260920.md",
               ROOT / "nso/scene_contract_v40.py"]
    source_hashes.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    manifest = {"schema_version": "v40.development_geometry_manifest.v1", "parents": rows,
        "source_sha256": source_hashes, "artifact_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
        "development_scene_assets_created": 6, "development_facilities_created": 24,
        "formal_test_geometry_created": 0, "test_seed_material_read": False,
        "worlds_created": 0, "sensor_frames_created": 0, "policy_trajectories_created": 0, "quality_metric_runs": 0,
        "selection_uses_method_results": False, "runtime_integration_complete": False,
        "observable_target_sets_frozen": False, "public_navigation_graph_generated": False,
        "original_p0_specs_modified": False,
        "notes": ["Static development GT assets, not reconstructions or policy results.",
                  "Workspace metadata is separate from the unchanged P0 public_planner_spec schema.",
                  "Background and facilities are separately closed; contact faces require later visibility filtering."]}
    payloads["manifest.json"] = canonical_json_bytes(manifest)
    total = sum(map(len, payloads.values()))
    if total > MAX_ASSET_BYTES:
        raise ValueError("six-scene asset bundle exceeds 16 MiB cap")
    for name, data in payloads.items():
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    return {"output_dir": str(output_dir), "bytes_written": total, "files_written": len(payloads),
            "development_scenes": 6, "facilities": 24, "worlds_created": 0,
            "sensor_frames_created": 0, "runtime_integration_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_assets(args.source_dir, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
