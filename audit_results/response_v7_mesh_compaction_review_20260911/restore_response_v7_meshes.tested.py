#!/usr/bin/env python3
"""Restore only the 20 witnessed V7 meshes from frozen mappers and raw sensors.

No world, ground-truth reference, evaluator or outcome value is read. Check-only
uses temporary files exclusively; original run directories remain unchanged.
"""
import argparse
from dataclasses import fields
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("storage_shelves", "ventilation_baffles")
TARGETS = {f"{family}/candidate_{candidate:03d}/{stage}_mesh.npz"
           for family in FAMILIES for candidate in range(1, 6) for stage in ("arrival", "final")}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def identity(array):
    import numpy as np
    contiguous = np.ascontiguousarray(array)
    return {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape), "nbytes": contiguous.nbytes,
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest()}


def frozen_sources(run, snapshot, expected):
    with zipfile.ZipFile(run / "sources.zip") as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(expected), "source archive inventory differs")
        for name in names:
            relative = Path(name)
            require(not relative.is_absolute() and ".." not in relative.parts, "unsafe archived source path")
            content = archive.read(name)
            require(hashlib.sha256(content).hexdigest() == expected[name], f"archived source hash differs: {name}")
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


def worker(args):
    # No project module is loaded until the archived package root is first.
    sys.path.insert(0, str(args.snapshot))
    import numpy as np
    import open3d
    import scipy
    from env.virtual3d_response_v7 import ResponseConfigV7
    from nso.semantic_completion_v3 import SemanticHistoryMapperV3
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    from utils.grid_geometry import DIRECTIONS

    run = args.run
    metadata = read(run / "metadata.json")
    manifest = read(run / "artifact_hashes.json")
    compact_path = run / "compaction_manifest.json"
    compact = read(compact_path) if compact_path.exists() else None
    expected_arrays = {} if compact is None else compact["meshes"]
    records = []

    def fuse(mapper, frame, scan, row):
        mapper.update(frame, scan)
        if row["collision"]:
            dr, dc = DIRECTIONS[row["heading"]]
            r, c = row["position"][0] + dr, row["position"][1] + dc
            if 0 <= r < mapper.shape[0] and 0 <= c < mapper.shape[1]:
                mapper.belief[r, c] = 1

    for family in FAMILIES:
        folder = run / family
        fixture = read(folder / "fixture.json")
        require(fixture["context"] == metadata["context"] and fixture["family"] == family, "family context mismatch")
        config = ResponseConfigV7(**fixture["config"])
        shape = (round(config.height_m / config.resolution_m), round(config.width_m / config.resolution_m))
        prefix_records = read(folder / "prefix/records.json")
        require(len(prefix_records) == 21 and [r["step"] for r in prefix_records] == list(range(21)), "prefix inventory differs")
        prefix_frames = [RGBDFrame.load(folder / "prefix/frames" / f"{i:04d}.npz") for i in range(21)]
        prefix_scans = [PlanarScan.load(folder / "prefix/scans" / f"{i:04d}.npz") for i in range(21)]
        routes = read(folder / "candidates.json")
        require([r["candidate_id"] for r in routes] == list(range(6)), "candidate inventory differs")
        for route in routes[1:]:
            candidate = route["candidate_id"]
            directory = folder / f"candidate_{candidate:03d}"
            actions = read(directory / "actions.json")
            require(len(actions) == route["cost"] == len(route["actions"]), "only complete paid branches can restore compacted meshes")
            require(0 < route["arrival_action"] < len(actions), "missing distinct arrival/final checkpoint")
            expected_files = {f"{i:04d}.npz" for i in range(1, len(actions) + 1)}
            for kind in ("frames", "scans"):
                require({p.name for p in (directory / kind).glob("*.npz")} == expected_files, "branch raw inventory differs")
            mapper = SemanticHistoryMapperV3(shape, config, config.truncation_m)
            for frame, scan, row in zip(prefix_frames, prefix_scans, prefix_records):
                fuse(mapper, frame, scan, row)
            for step, row in enumerate(actions, 1):
                require(row["action_index"] == step and row["absolute_step"] == 20 + step, "paid time ledger mismatch")
                require(row["action"] == route["actions"][step - 1] and not row["collision"], "failed/mismatched branch is not compactable")
                require([*row["position"], row["heading"]] == route["states"][step], "actual pose differs from sealed route")
                frame = RGBDFrame.load(directory / "frames" / f"{step:04d}.npz")
                scan = PlanarScan.load(directory / "scans" / f"{step:04d}.npz")
                fuse(mapper, frame, scan, row)
                if step not in (route["arrival_action"], len(actions)):
                    continue
                stage = "arrival" if step == route["arrival_action"] else "final"
                relative = f"{family}/candidate_{candidate:03d}/{stage}_mesh.npz"
                require(relative in TARGETS, "restoration escaped the fixed mesh whitelist")
                mesh = mapper.mesh()
                arrays = {"vertices": np.asarray(mesh.vertices), "triangles": np.asarray(mesh.triangles)}
                actual = {key: identity(value) for key, value in arrays.items()}
                if relative in expected_arrays:
                    require(actual == expected_arrays[relative]["arrays"], f"regenerated array identities differ: {relative}")
                else:
                    require((run / relative).is_file(), "missing mesh without compaction witness")
                    with np.load(run / relative, allow_pickle=False) as old:
                        require(set(old.files) == set(arrays), "original mesh schema differs")
                        require(actual == {key: identity(old[key]) for key in old.files}, "original/regenerated arrays differ")
                target = args.generated / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, **arrays)
                require(sha(target) == manifest[relative], f"regenerated NPZ bytes differ from original file: {relative}")
                records.append({"path": relative, "arrays": actual, "file_sha256": sha(target),
                                "file_bytes": target.stat().st_size, "status": "passed_exact_arrays_and_original_npz"})
                print("checked raw-to-mesh", relative, flush=True)
            del mapper
    require({r["path"] for r in records} == TARGETS and len(records) == 20, "not all twenty fixed checkpoints were restored")
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in ("env", "nso", "utils") and getattr(module, "__file__", None):
            require(Path(module.__file__).resolve().is_relative_to(args.snapshot), f"live project module loaded: {name}")
    write_json(args.result, {"status": "passed_all_twenty_raw_to_mesh", "meshes": records,
        "dependencies": {"numpy": np.__version__, "open3d": open3d.__version__, "scipy": scipy.__version__},
        "world_constructed": False, "gt_or_outcomes_read": False,
        "method": "isolated archived mapper; raw prefix rebuilt separately for each branch; all actual RGB-D/scans fused in order"})


def restore(args):
    started = time.monotonic()
    run = args.run.resolve()
    if args.check_only and args.report:
        require(not args.report.resolve().is_relative_to(run), "check-only report must be outside the original run")
    own_sha = sha(__file__)
    validator_path = ROOT / "scripts/compact_response_v7_meshes.py"
    validator_sha = sha(validator_path)
    spec = importlib.util.spec_from_file_location("response_mesh_asset_validator", validator_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    before_validation = validator.validate_run_assets(run)
    metadata = read(run / "metadata.json")
    require(metadata["status"] == "complete" and metadata["failures"] == 0, "restoration requires a complete, failure-free compactable run")
    manifest = read(run / "artifact_hashes.json")
    manifest_sha = sha(run / "artifact_hashes.json")
    source_sha = sha(run / "sources.zip")
    original_presence = {name: (run / name).is_file() for name in TARGETS}
    required_bytes = sum((run / name).stat().st_size if (run / name).exists() else
                         read(run / "compaction_manifest.json")["meshes"][name]["file_bytes"] for name in TARGETS)
    reserve = read(run / "config.json")["storage"]["reserve_bytes"]
    require(shutil.disk_usage(run).free >= reserve + required_bytes * (1 if args.check_only else 2),
            "not enough space for temporary reconstruction and the retained reserve")
    compact_path = run / "compaction_manifest.json"
    compact_sha = sha(compact_path) if compact_path.exists() else None
    report = {"schema_version": "response_v7_mesh_restore/1", "run": str(run), "check_only": args.check_only,
              "restore_script_sha256": own_sha, "asset_validator_sha256": validator_sha,
              "artifact_manifest_sha256": manifest_sha, "source_archive_sha256": source_sha,
              "compaction_manifest_sha256": compact_sha, "before_validation": before_validation,
              "original_files_written": [], "gt_or_outcomes_read": False, "workers": 1}
    with tempfile.TemporaryDirectory(prefix="response-v7-mesh-restore-") as temporary:
        work = Path(temporary)
        snapshot = work / "snapshot"
        frozen_sources(run, snapshot, metadata["source_sha256"])
        generated = work / "generated"
        result = work / "result.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--run", str(run),
                   "--snapshot", str(snapshot), "--generated", str(generated), "--result", str(result)]
        completed = subprocess.run(command, cwd=work, env=os.environ.copy())
        require(completed.returncode == 0 and result.exists(), "isolated raw-to-mesh reconstruction failed; original run unchanged")
        report["reconstruction"] = read(result)
        # Validate all originals/witnesses again before publishing any file.
        validator.validate_run_assets(run)
        require(sha(run / "artifact_hashes.json") == manifest_sha and sha(run / "sources.zip") == source_sha,
                "original manifests changed during restoration")
        require((sha(compact_path) if compact_path.exists() else None) == compact_sha, "compaction witness changed during restoration")
        if not args.check_only:
            for name in sorted(TARGETS):
                target = run / name
                if target.exists():
                    require(sha(target) == manifest[name], "existing mesh is not the original; refusing overwrite")
                    continue
                require(sha(generated / name) == manifest[name], "generated mesh changed before atomic publication")
                # Same-directory temporary file makes os.replace atomic.
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".restoring-mesh-", suffix=".npz", delete=False) as stream:
                    temporary_name = Path(stream.name)
                    with (generated / name).open("rb") as source:
                        shutil.copyfileobj(source, stream)
                    stream.flush(); os.fsync(stream.fileno())
                require(sha(temporary_name) == manifest[name], "temporary publication file differs")
                require(not target.exists(), "mesh appeared concurrently; refusing replacement")
                os.replace(temporary_name, target)
                report["original_files_written"].append(name)
        else:
            require(original_presence == {name: (run / name).is_file() for name in TARGETS}, "check-only changed original mesh presence")
    require(sha(__file__) == own_sha and sha(validator_path) == validator_sha, "restoration/validation source changed")
    report["after_validation"] = validator.validate_run_assets(run)
    require((sha(compact_path) if compact_path.exists() else None) == compact_sha, "compaction history was modified")
    report.update(status="passed_check_only" if args.check_only else "restored_and_verified",
                  meshes_checked=20, bytes_regenerated=required_bytes, elapsed_seconds=time.monotonic() - started)
    if args.report:
        write_json(args.report, report)
    elif not args.check_only:
        # Add a new witness; never replace the historical compaction manifest.
        witness = run / f"mesh_restoration_{time.time_ns()}.json"
        write_json(witness, report)
    print(json.dumps({k: report[k] for k in ("status", "meshes_checked", "bytes_regenerated", "elapsed_seconds", "original_files_written")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        restore(args)
