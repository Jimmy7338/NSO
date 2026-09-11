#!/usr/bin/env python3
"""Small deterministic prefix audit; save hashes, never future reward values.

The full raw archive is the separate experiment runner's responsibility. This
construction audit saves source and per-frame identities for all four worlds,
without duplicate RGB-D/mesh storage or a candidate execution.
"""
import argparse
from dataclasses import fields
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import open3d as o3d
import scipy

from tests.virtual3d.test_competition_world_v8 import CompetitionWorldV8Tests
from utils.rgbd_contract import RGBDFrame, PlanarScan


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_identity(value):
    array = np.asarray(value)
    return dict(dtype=str(array.dtype), shape=list(array.shape), sha256=sha(array.tobytes()))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [
        "env/virtual3d_competition_v8.py", "env/virtual3d.py", "env/virtual3d_v2.py",
        "env/virtual3d_inspection_v4.py", "utils/grid_geometry.py", "utils/rgbd_contract.py",
        "tests/virtual3d/test_competition_world_v8.py", "scripts/audit_competition_world_v8.py",
        "configs/virtual3d/competition_v8_contexts.json",
    ]
    source_hashes = {path: sha((ROOT / path).read_bytes()) for path in source_paths}
    with zipfile.ZipFile(output / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.writestr(path, (ROOT / path).read_bytes())
    write_json(output / "pre_execution_seal.json", dict(
        scope="four fixed development worlds; sensors and paid prefix only",
        first_unit_test_already_passed_before_this_traceable_audit=True,
        future_candidate_executions=0, source_sha256=source_hashes,
        source_archive_sha256=sha((output / "sources.zip").read_bytes()),
        environment=dict(python=sys.version, numpy=np.__version__, open3d=o3d.__version__, scipy=scipy.__version__),
    ))
    started = time.monotonic()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CompetitionWorldV8Tests))
    (output / "tests.txt").write_text(stream.getvalue())
    worlds = []
    for key, world in getattr(CompetitionWorldV8Tests, "worlds", {}).items():
        rows = getattr(CompetitionWorldV8Tests, "prefixes", {}).get(key, [])
        frame_records = []
        for row in rows:
            frame, scan = row["frame"], row["scan"]
            marker_mask = frame.semantic > 0
            frame_records.append(dict(step=row["step"], action=row["action"],
                collision=bool(row["collision"]), done=bool(row["done"]),
                rgbd={field.name: array_identity(getattr(frame, field.name)) for field in fields(RGBDFrame)},
                scan={field.name: array_identity(getattr(scan, field.name)) for field in fields(PlanarScan)},
                marker_mask=array_identity(marker_mask), nonmarker_rgb=array_identity(frame.color_rgb[~marker_mask]),
                observed_marker_pixels={str(k): int(np.count_nonzero(frame.semantic == k)) for k in (2, 3)}))
        relative = f"{key[0]}_{key[1]}_prefix_identities.json"
        write_json(output / relative, frame_records)
        worlds.append(dict(context_id=key[0], arrangement=key[1], sensor_seed=world.seed,
            frames=len(rows), paid_actions=world.step_count, translations=world.moves,
            rotations=world.step_count-world.moves, collisions=world.collisions,
            final_grid_state=[*world.position, world.heading],
            final_camera_xy_m=rows[-1]["frame"].world_from_camera[:2, 3].tolist() if rows else None,
            observed_marker_pixels={str(k): sum(r["observed_marker_pixels"][str(k)] for r in frame_records) for k in (2, 3)},
            unique_gt_surface_area_m2=world.mesh.get_surface_area(),
            physical_surface_area_interpretation="construction identity, not a candidate response or quality score",
            occupancy=array_identity(world.occupancy), reachable=array_identity(world.reachable),
            prefix_identities_file=relative, prefix_identities_sha256=sha((output / relative).read_bytes())))
    after_hashes = {path: sha((ROOT / path).read_bytes()) for path in source_paths}
    source_unchanged = after_hashes == source_hashes
    passed = result.wasSuccessful() and source_unchanged and len(worlds) == 4
    summary = dict(status="passed_sensor_construction_only" if passed else "failed_construction_retained",
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        elapsed_s=time.monotonic()-started, source_unchanged=source_unchanged,
        future_candidate_executions=0, raw_or_mesh_files_saved=0, worlds=worlds,
        still_unverified=["prefix mapper nonlabel invariants", "fixed observable background coverage >=90%",
                          "both slots retain unseen observable area", "knownsafe single/double visit B48 feasibility",
                          "actual branch view exclusivity", "semantic effect on physical outcomes"],
        limits=["artificial asset-marker positive control; not natural semantics",
                "exact pairing is within each parent, not between mirrored P0/P1",
                "source and raw-array hashes permit later replay comparison; this is not a substitute for the runner raw archive"])
    write_json(output / "summary.json", summary)
    write_json(output / "artifact_sha256.json", {p.name: sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps(dict(status=summary["status"], tests=result.testsRun, worlds=len(worlds),
                          output=str(output), bytes=sum(p.stat().st_size for p in output.iterdir())), ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
