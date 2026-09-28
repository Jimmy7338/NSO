#!/usr/bin/env python3
"""Seal and audit T0's paired sensor prefix, never evaluate candidate outcomes."""
import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import open3d as o3d
import scipy
from env.virtual3d_response_v7 import (
    RESPONSE_FAMILIES, ResponseConfigV7, create_response_world,
    collect_response_prefix, get_response_context,
)
from env.virtual3d_inspection_v4 import read_inspection_markers_rgb
from nso.camera_mapping_v2 import CameraQualityMapperV2
from utils.rgbd_contract import RGBDFrame, PlanarScan


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_record(value):
    array = np.ascontiguousarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def named_hash(arrays):
    return hashlib.sha256(json.dumps({k: array_record(v) for k, v in arrays.items()},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def map_snapshot(mapper):
    mesh = mapper.mesh()
    points, bits, labels = mapper.evidence()
    common = {"belief": mapper.belief, "visible": mapper.visible,
              "camera_seen": mapper.camera_seen, "surface_points": points,
              "surface_bits": bits, "frames": np.asarray(mapper.frames),
              "surface_keys": np.asarray(list(mapper.surface), np.int64).reshape(-1, 3)}
    quality = {"keys": np.asarray(list(mapper.quality), np.int64).reshape(-1, 3)}
    if mapper.quality:
        for key in next(iter(mapper.quality.values())):
            if key != "label":
                quality[key] = np.asarray([row[key] for row in mapper.quality.values()])
    geometry = {key: np.asarray(getattr(mesh, key)) for key in ("vertices", "triangles", "vertex_normals")}
    colors = np.asarray(mesh.vertex_colors)
    hashes = {"map_nonlabel_sha256": named_hash(common),
              "quality_all_voxels_nonlabel_sha256": named_hash(quality),
              "tsdf_extracted_geometry_sha256": named_hash(geometry),
              "tsdf_vertex_colors_sha256": named_hash({"colors": colors}),
              "surface_labels_sha256": named_hash({"labels": labels})}
    arrays = {**{f"map_{k}": v for k, v in common.items()},
              **{f"quality_{k}": v for k, v in quality.items()},
              **geometry, "vertex_colors": colors, "surface_labels": labels}
    return hashes, arrays


def assert_sensor_equal(a, b):
    for contract in ("frame", "scan"):
        for field in fields(type(a[contract])):
            key = field.name
            if not np.array_equal(getattr(a[contract], key), getattr(b[contract], key)):
                raise AssertionError(f"independent replay differs at step {a['step']}/{contract}/{key}")
    for key in ("step", "action", "collision", "done"):
        if a[key] != b[key]:
            raise AssertionError(f"replay paid action mismatch: {key}")


def replay_only(output):
    """Separate process, frozen archive: manual actions, all saved observations."""
    seal = json.loads((output / "pre_sensor_seal.json").read_text())
    for relative, expected in seal["source_hashes"].items():
        if file_hash(ROOT / relative) != expected:
            raise AssertionError(f"archived source mismatch: {relative}")
    checks = []
    config = ResponseConfigV7(**seal["config"])
    for family in RESPONSE_FAMILIES:
        world = create_response_world("T0", family, config=config)
        mapper = CameraQualityMapperV2(world.shape, config, config.truncation_m)
        directory = output / family
        records = json.loads((directory / "records.json").read_text())
        for row in records:
            step, action = row["step"], row["action"]
            stored = {**{k: row[k] for k in ("step", "action", "collision", "done")},
                      "frame": RGBDFrame.load(directory / "raw" / f"frame_{step:04d}.npz"),
                      "scan": PlanarScan.load(directory / "raw" / f"scan_{step:04d}.npz")}
            if step == 0:
                frame, collision, done = world.sense(), False, False
            else:
                frame, collision, done = world.step(action)
            generated = dict(step=world.step_count, action=action, frame=frame,
                             scan=world.scan(), collision=collision, done=done)
            assert_sensor_equal(stored, generated)
            if list(world.position) != row["position"] or world.heading != row["heading"]:
                raise AssertionError("replayed physical pose differs")
            mapper.update(stored["frame"], stored["scan"])
        hashes, arrays = map_snapshot(mapper)
        saved_hashes = json.loads((directory / "map_hashes.json").read_text())
        if hashes != saved_hashes:
            raise AssertionError("replayed map hashes differ")
        with np.load(directory / "prefix_map.npz", allow_pickle=False) as archive:
            for key, array in arrays.items():
                if not np.array_equal(array, archive[key]):
                    raise AssertionError(f"replayed map arrays differ: {key}")
        checks.append({"family": family, "sensor_frames_exact": len(records),
                       "paid_actions_exact": world.step_count, "map_arrays_exact": len(arrays),
                       "status": "passed"})
    dump(output / "verification.json", {
        "status": "passed_frozen_archive_prefix_replay", "families": checks,
        "independence": "separate process; archived generator and mapper; manual action loop; original collection helper not called",
        "limits": "same physical sensor implementation, not a second ray-tracing engine; no candidate/outcome validation",
    })


def run(output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "pre_sensor_seal.json").exists():
        raise ValueError("audit already sealed; choose a new result directory, never overwrite evidence")
    if shutil.disk_usage(output).free < 420 * 1024**2:
        raise RuntimeError("insufficient reserve for the bounded audit")
    config = ResponseConfigV7()
    context = get_response_context("T0")
    # Freeze imported local implementation plus this audit and its tests.
    sources = set()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and ".venv" not in str(path.relative_to(ROOT)) and path.suffix == ".py":
                sources.add(path.relative_to(ROOT).as_posix())
    sources.update(("scripts/audit_response_world_v7.py", "tests/virtual3d/test_response_world_v7.py",
                    "configs/virtual3d/response_v7_contexts.json"))
    hashes = {name: file_hash(ROOT / name) for name in sorted(sources)}
    with zipfile.ZipFile(output / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(sources):
            archive.write(ROOT / name, name)
    seal = {"status": "sealed_before_this_audit_sensor_generation", "utc": datetime.now(timezone.utc).isoformat(),
            "context": asdict(context), "config": asdict(config), "source_hashes": hashes,
            "source_archive_sha256": file_hash(output / "sources.zip"),
            "construction_scope": ["T0"], "candidate_outcomes_generated": 0,
            "previous_checks": "T0-only implementation unit tests and an unarchived 21-frame equality check; no candidate outcomes accessed",
            "versions": {"python": sys.version, "numpy": np.__version__, "open3d": o3d.__version__, "scipy": scipy.__version__}}
    dump(output / "pre_sensor_seal.json", seal)
    worlds, prefixes, maps = [], [], []
    for family in RESPONSE_FAMILIES:
        world = create_response_world("T0", family, config=config)
        mapper = CameraQualityMapperV2(world.shape, config, config.truncation_m)
        rows = collect_response_prefix(world)
        directory = output / family
        (directory / "raw").mkdir(parents=True)
        records = []
        for row in rows:
            step, frame, scan = row["step"], row["frame"], row["scan"]
            frame.save(directory / "raw" / f"frame_{step:04d}.npz")
            scan.save(directory / "raw" / f"scan_{step:04d}.npz")
            mapper.update(frame, scan)
            origin = frame.world_from_camera[:3, 3]
            position = mapper.grid_cell(origin)
            direction = frame.world_from_camera[:3, 2]
            heading = int(np.argmax([direction[1], direction[0], -direction[1], -direction[0]]))
            records.append({**{k: row[k] for k in ("step", "action", "collision", "done")},
                            "position": list(position), "heading": heading,
                            "marker_pixels": int(np.count_nonzero(frame.semantic)),
                            "frame_arrays": {field.name: array_record(getattr(frame, field.name)) for field in fields(RGBDFrame)},
                            "scan_arrays": {field.name: array_record(getattr(scan, field.name)) for field in fields(PlanarScan)}})
        map_hashes, map_arrays = map_snapshot(mapper)
        np.savez_compressed(directory / "prefix_map.npz", **map_arrays)
        dump(directory / "map_hashes.json", map_hashes)
        dump(directory / "records.json", records)
        # Shape audit only: no surface sampling, view evaluator or reward label.
        np.savez_compressed(directory / "physical_shape_audit.npz", occupancy=world.occupancy,
                            reachable=world.reachable, blocked=world._blocked,
                            primitives=np.asarray(world._solid_primitives),
                            vertices=np.asarray(world.mesh.vertices), triangles=np.asarray(world.mesh.triangles),
                            triangle_classes=world.triangle_classes)
        worlds.append(world)
        prefixes.append(rows)
        maps.append(map_hashes)
    paired_checks = []
    for a, b in zip(*prefixes):
        af, bf = a["frame"], b["frame"]
        ma, mb = read_inspection_markers_rgb(af.color_rgb) > 0, read_inspection_markers_rgb(bf.color_rgb) > 0
        checks = {key: bool(np.array_equal(getattr(af, key), getattr(bf, key)))
                  for key in ("depth_m", "intrinsic", "world_from_camera", "timestamp_s")}
        checks.update({f"scan_{f.name}": bool(np.array_equal(getattr(a["scan"], f.name), getattr(b["scan"], f.name)))
                       for f in fields(PlanarScan)})
        checks["marker_support"] = bool(np.array_equal(ma, mb))
        checks["nonmarker_rgb"] = bool(np.array_equal(af.color_rgb[~ma], bf.color_rgb[~ma]))
        checks["step_and_action"] = a["step"] == b["step"] and a["action"] == b["action"]
        paired_checks.append({"step": a["step"], "checks": checks, "marker_pixels": int(ma.sum())})
    shared_map_keys = ("map_nonlabel_sha256", "quality_all_voxels_nonlabel_sha256", "tsdf_extracted_geometry_sha256")
    invariants = {key: maps[0][key] == maps[1][key] for key in shared_map_keys}
    invariants.update({key: bool(np.array_equal(getattr(worlds[0], key), getattr(worlds[1], key)))
                       for key in ("occupancy", "reachable", "_blocked")})
    all_passed = all(all(row["checks"].values()) for row in paired_checks) and all(invariants.values())
    dump(output / "paired_invariants.json", {"all_passed": all_passed, "frames": paired_checks,
          "map_and_ground": invariants, "map_hashes": dict(zip(RESPONSE_FAMILIES, maps)),
          "permitted_color_difference": "TSDF vertex colors and semantic labels encode the actual visible asset marker; geometry is compared separately",
          "tsdf_limit": "extracted geometry arrays checked exactly; internal ScalableTSDF voxel weights are not exposed/claimed checked"})
    if not all_passed:
        raise AssertionError("paired prefix violates a predeclared invariant; keep failure, do not substitute parent")
    with tempfile.TemporaryDirectory(prefix="response_v7_replay_") as temporary:
        source = Path(temporary)
        with zipfile.ZipFile(output / "sources.zip") as archive:
            archive.extractall(source)
        command = [sys.executable, str(source / "scripts/audit_response_world_v7.py"),
                   "--output", str(output.resolve()), "--replay-only"]
        completed = subprocess.run(command, text=True, capture_output=True, env=os.environ.copy())
        (output / "replay_process.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError("frozen-archive replay failed; see replay_process.log")
    for name, expected in hashes.items():
        if file_hash(ROOT / name) != expected:
            raise RuntimeError(f"source changed during audit: {name}")
    summary = {"status": "complete_prefix_audit_only", "context_id": "T0", "physical_worlds": 2,
               "frames_per_family": 21, "paid_actions_per_family": 20, "first_marker_step": 0,
               "terminal_marker_pixels_per_family": paired_checks[-1]["marker_pixels"],
               "prefix_invariants": "passed_exact", "independent_replay": "passed_frozen_archive_prefix_replay",
               "candidate_outcomes": 0, "other_contexts_constructed": [],
               "elapsed_seconds": time.monotonic() - started,
               "workers": 1, "source_archive_sha256": seal["source_archive_sha256"],
               "scientific_claim": "T0 satisfies the controlled information-gap input contract; semantic response and selection advantages remain untested"}
    dump(output / "summary.json", summary)
    report = f"""# V7首个配对父上下文前缀审计

仅构造预定训练父上下文T0的两类物理设备。两类各保存初态与20个付费动作后的21帧原始RGB-D、解释标签、雷达、位姿。未生成候选、未读取未来观测收益，也未构造其他T/C上下文或753/754及独立seed。

21帧深度、雷达全字段、相机位姿、内参、时间戳、marker支撑掩膜及非marker RGB逐字节相同。初态已见marker，末帧两族均有{paired_checks[-1]['marker_pixels']}个标记像素。两族地面占用、膨胀障碍、可达域及实测地图、全部quality体素非标签字段、TSDF提取顶点/三角形/法向完全一致。TSDF顶点颜色随真实标记改变，不把它错误计入几何不变量；未声称比较底层未公开的TSDF权重。

第二个进程从sources.zip还原生成器和mapper，独立手动执行全部动作并逐字段比对42帧已保存原始观测；重新融合后的所有归档地图数组一致。它仍使用同一物理传感器实现，并非第二套渲染器验证。

两族的隐藏实体配方不同，共同底座保证地面足迹一致，公共前板产生真实遮挡。语义2在本版本表示储物层板，3表示通风隔板；它们不沿用旧简单/复杂命名。类别仅是人工资产标记，合成器主动规定了类别与隐藏结构关联，尚未证明候选响应存在可利用差异，更未证明S优于G/O/N。

资源：单worker，耗时{summary['elapsed_seconds']:.2f}秒；原始观测全部留档。合同字段从current_authorized_construction透明更名initial_construction_probe，物理参数没有改变，旧JSON保存在contract_history。前期仅T0单元测试及一次未归档逐帧相等检查已在pre_sensor_seal明确记录。

关键文件：summary.json、paired_invariants.json、verification.json、pre_sensor_seal.json、sources.zip、两族raw/与prefix_map.npz。下一步应做共同候选结构和真实响应验证，本审计不提前判断其结果。
"""
    (output / "PREFIX_AUDIT.md").write_text(report)
    artifacts = {p.relative_to(output).as_posix(): file_hash(p)
                 for p in sorted(output.rglob("*")) if p.is_file()}
    dump(output / "artifact_hashes.json", artifacts)
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if size > 20 * 1024**2:
        raise RuntimeError(f"audit exceeded the 20MiB cap ({size} bytes); keep data and report it")
    print(json.dumps({**summary, "artifact_bytes": size}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "eval_results/response_v7_prefix_audit_first_20260911")
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    try:
        (replay_only if args.replay_only else run)(args.output)
    except Exception as error:
        dump(args.output / ("replay_failure.json" if args.replay_only else "audit_failure.json"),
             {"status": "failed", "error": repr(error), "utc": datetime.now(timezone.utc).isoformat()})
        raise
