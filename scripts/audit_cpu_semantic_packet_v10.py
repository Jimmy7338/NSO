#!/usr/bin/env python3
"""Actual Components/Runtime label intervention on one frozen paid raw prefix.

No simulator instance, new action, evaluation reference, outcome or F1 is read.
This is a development causality audit, not a policy performance experiment.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from copy import deepcopy
from dataclasses import asdict, replace, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.virtual3d_competition_v9 import CompetitionConfigV9
from nso.components import NSO_Components
from nso.runtime_integration import NSORuntimeIntegration
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, json_value
from nso.hierarchical_options_v10 import generate_options
import nso.cpu_four_modules_v10
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from utils.rgbd_contract import RGBDFrame, PlanarScan

PREPARATION = ROOT / "eval_results/competition_v9_preparation_20260911"
PREPARATION_MANIFEST_SHA256 = "765aa907dc8002db5bd11d5897fe31ca000374391de52ad742eea442aa92aa85"
HISTORY = "Q0/shelf_west"
MODES = ("original", "swap", "missing")
ABS_TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_digest(value):
    value = np.ascontiguousarray(np.asarray(value))
    require(not value.dtype.hasobject, "object arrays cannot identify physical inputs")
    header = json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":")).encode()
    digest = hashlib.sha256(len(header).to_bytes(8, "big") + header + value.tobytes()).hexdigest()
    return dict(dtype=value.dtype.str, shape=list(value.shape), sha256=digest)


def nonlabel_sensor_digest(frame, scan):
    records = {}
    for prefix, record in (("frame", frame), ("scan", scan)):
        for field in fields(record):
            if prefix == "frame" and field.name == "semantic":
                continue
            records[prefix + "." + field.name] = array_digest(getattr(record, field.name))
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def freeze_sources(output):
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.is_relative_to(ROOT):
            relative = path.relative_to(ROOT)
            if path.suffix == ".py" and not any(part.startswith(".venv") for part in relative.parts):
                paths.add(path)
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}
    with zipfile.ZipFile(output / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(ROOT / name, name)
    receipt = dict(schema_version="cpu_semantic_packet_v10_source_freeze/1",
                   frozen_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
                   archive_sha256=sha(output / "sources.zip"),
                   raw_prefix_replays_before_freeze=0, constructed_worlds=0,
                   candidate_capture="read-only Python return trace; no replacement scorer")
    write(output / "freeze.json", receipt)
    return receipt


def check_sources(freeze):
    changed = [name for name, expected in freeze["source_sha256"].items()
               if sha(ROOT / name) != expected]
    require(not changed, f"source changed during intervention audit: {changed}")


def load_raw_prefix(output):
    manifest_path = PREPARATION / "artifact_hashes.json"
    require(sha(manifest_path) == PREPARATION_MANIFEST_SHA256,
            "original preparation manifest differs from the independently bound version")
    manifest = json.loads(manifest_path.read_text())
    names = [HISTORY + "/prefix/records.json", HISTORY + "/prefix_map.npz"]
    names += [f"{HISTORY}/prefix/{kind}/{i:04d}.npz"
              for i in range(151) for kind in ("frames", "scans")]
    for name in names:
        require(name in manifest, f"raw prefix input absent from preparation manifest: {name}")
        require(sha(PREPARATION / name) == manifest[name], f"raw input SHA mismatch: {name}")
    records = json.loads((PREPARATION / names[0]).read_text())
    require(len(records) == 151 and [r["step"] for r in records] == list(range(151)),
            "this audit requires every initial/paid frame from 0 through 150")
    with np.load(PREPARATION / names[1], allow_pickle=False) as data:
        shape = tuple(map(int, data["belief"].shape))
    raw = []
    for i, record in enumerate(records):
        frame_name = f"{HISTORY}/prefix/frames/{i:04d}.npz"
        scan_name = f"{HISTORY}/prefix/scans/{i:04d}.npz"
        raw.append((record, RGBDFrame.load(PREPARATION / frame_name),
                    PlanarScan.load(PREPARATION / scan_name), frame_name, scan_name))
    receipt = dict(preparation=str(PREPARATION), preparation_manifest_sha256=sha(manifest_path),
                   history=HISTORY, actual_prefix_packets=151, paid_prefix_actions=150,
                   input_sha256={name: manifest[name] for name in names},
                   permitted_data="this history's raw RGB-D/scans, recorded poses/actions, observed map shape",
                   reference_outcome_f1_files_opened=[], shape=shape)
    write(output / "input_hashes.json", receipt)
    return raw, shape, receipt


def physical_state(mapper):
    arrays = {name: np.asarray(getattr(mapper, name))
              for name in ("belief", "visible", "camera_seen")}
    keys = sorted(mapper.quality)
    arrays["quality.keys"] = np.asarray(keys, dtype=np.int64).reshape(-1, 3)
    if keys:
        for field in sorted(set(mapper.quality[keys[0]]) - {"label"}):
            arrays["quality." + field] = np.asarray([mapper.quality[k][field] for k in keys])
    surface_keys = sorted(mapper.surface)
    arrays["surface.keys"] = np.asarray(surface_keys, dtype=np.int64).reshape(-1, 3)
    arrays["surface.points"] = np.asarray([mapper.surface[k][0] for k in surface_keys]).reshape(-1, 3)
    arrays["surface.bits"] = np.asarray([mapper.surface[k][1] for k in surface_keys], dtype=np.int64)
    mesh = mapper.mesh()
    for name in ("vertices", "triangles", "vertex_normals", "vertex_colors"):
        arrays["mesh." + name] = np.asarray(getattr(mesh, name))
    return {name: array_digest(value) for name, value in arrays.items()}


def run_variant(mode, raw, shape, config, output):
    args = SimpleNamespace(nso_backend="cpu_v10", eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode="S", cpu_disable_feedback=False, cpu_max_candidates=12,
        run_id=output.parent.name)
    components = NSO_Components(args)
    components.initialize("cpu", 1, *shape, *shape)
    runtime = NSORuntimeIntegration(components, 1, shape)
    transform = GridTransform(shape, config.resolution_m)
    packets, packet_records = [], []
    for record, original, scan, frame_name, scan_name in raw:
        semantic = original.semantic.copy()
        if mode == "swap":
            semantic[original.semantic == 2] = 3
            semantic[original.semantic == 3] = 2
        elif mode == "missing":
            semantic[:] = 0
        frame = replace(original, semantic=semantic)
        require(nonlabel_sensor_digest(frame, scan) == nonlabel_sensor_digest(original, scan),
                "intervention altered a non-label sensor field")
        step = int(record["step"])
        packet = SensorPacket("Q0_shelf_west", "label-intervention-" + mode, f"frame-{step}", step,
            frame, scan, tuple(map(int, record["position"])), int(record["heading"]),
            "recorded_rgb_marker_labels; declared label intervention=" + mode,
            "recorded_simulator_exact_discrete_odometry",
            None if step == 0 else record["action"], bool(record["collision"]), bool(record["done"]))
        packet.validate(transform, config)
        packets.append(packet)
        packet_records.append(dict(action_id=step, frame_id=packet.frame_id,
            original_frame_path=frame_name, original_scan_path=scan_name,
            packet_sha256=packet.sha256(), nonlabel_sensor_sha256=nonlabel_sensor_digest(frame, scan),
            semantic=array_digest(semantic), changed_label_pixels=int(np.count_nonzero(semantic != original.semantic))))
    anchor = (*packets[-1].position, packets[-1].heading)
    runtime.start_sensor_episode(0, config=config, transform=transform, packets=packets,
                                 total_budget=198, paid_prefix_actions=150, return_anchor=anchor)
    mapper = runtime.states[0]["mapper"]
    require(isinstance(mapper, ObservedRuntimeMapperV10), "runtime did not use the current V10 mapper")
    captured = []
    def record_return(frame, event, arg):
        if event == "return" and frame.f_code is generate_options.__code__:
            captured.append(deepcopy(arg))
    previous_profile = sys.getprofile()
    require(previous_profile is None, "do not replace another active profiling session")
    try:
        sys.setprofile(record_return)
        target = runtime.choose_goal(0, list(packets[-1].position), (0, shape[0], 0, shape[1]))
    finally:
        sys.setprofile(previous_profile)
    require(len(captured) == 1, "the real choose_goal must generate exactly one observed option pool")
    candidates, candidate_audit = captured[0]
    calls = components._cpu_backend.calls
    selections = [c for c in calls if c["method"] == "select_topo_target"]
    require(len(selections) == 1, "one real STGHP selection required per intervention")
    selection = selections[0]["outputs"]
    summary = runtime.sensor_episode_summary(0)
    state_hashes = physical_state(mapper)
    assets = components._cpu_backend.scenes[0]["assets"]
    asset_geometry = [{k: array_digest(v) if isinstance(v, np.ndarray) else v
                      for k, v in asset.items() if k not in ("marked_points", "class_vote")}
                     for asset in assets]
    result = dict(mode=mode, target_cell=target,
        selected_candidate_id=None if selection["selected"] is None else selection["selected"]["candidate_id"],
        scores=selection["scores"], packet_count=len(packets), paid_prefix_actions=150,
        newly_executed_actions=0, keyframe_count=len(mapper.keyframes),
        feedback=summary["modules"]["feedback"], map_frames=mapper.frames,
        runtime_summary=summary, physical_state=state_hashes, asset_geometry=asset_geometry,
        changed_label_pixels=sum(row["changed_label_pixels"] for row in packet_records),
        real_interface_calls=[dict(call_id=c["call_id"], module=c["module"], method=c["method"])
                              for c in calls], capabilities=components.capabilities)
    output.mkdir()
    write(output / "summary.json", result)
    write(output / "packets.json", packet_records)
    write(output / "candidates.json", candidates)
    write(output / "candidate_audit.json", candidate_audit)
    write(output / "selection.json", selection)
    write(output / "module_calls.json", calls)
    write(output / "runtime_events.json", runtime.audit)
    return result, json_value(candidates), json_value(candidate_audit), packet_records


def numeric_comparison(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    same_shape = a.shape == b.shape
    error = float(np.max(np.abs(a - b))) if same_shape and a.size else 0. if same_shape else None
    return dict(passed=bool(same_shape and np.isfinite(a).all() and np.isfinite(b).all()
                           and np.allclose(a, b, rtol=0, atol=ABS_TOL)),
                bit_exact=bool(same_shape and array_digest(a) == array_digest(b)),
                max_absolute_error=error, absolute_tolerance=ABS_TOL, relative_tolerance=0.)


def run(output):
    require(not output.exists(), "each source revision requires a new audit output directory")
    require(shutil.disk_usage(ROOT).free > 400 * 1024**2 + 8 * 1024**2,
            "insufficient space above the experiment reserve")
    output.mkdir(parents=True)
    freeze = freeze_sources(output)
    completed = []
    started = time.monotonic()
    try:
        raw, shape, inputs = load_raw_prefix(output)
        # A fixed sensor/calibration configuration only. No context loader or
        # world factory is invoked and no hidden asset geometry is supplied.
        config = CompetitionConfigV9()
        require(shape == (round(config.height_m / config.resolution_m),
                          round(config.width_m / config.resolution_m)), "observed map shape/config mismatch")
        write(output / "audit_contract.json", dict(schema_version="cpu_semantic_packet_v10/1",
            variants=MODES, history=HISTORY, total_budget=198, paid_prefix_actions=150,
            remaining_budget=48, max_candidates=12, score_mode="S", absolute_tolerance=ABS_TOL,
            relative_tolerance=0., config=asdict(config), new_world_actions=0,
            selection_calls_per_variant=1, missing_intervention="all semantic labels become zero",
            swap_intervention="only class 2 and class 3 exchange; RGB remains unchanged",
            role="development sensor-interface causality; not efficacy"))
        results, pools, audits, packets = {}, {}, {}, {}
        for mode in MODES:
            check_sources(freeze)
            result, pool, candidate_audit, packet_rows = run_variant(mode, raw, shape, config, output / mode)
            results[mode], pools[mode], audits[mode], packets[mode] = result, pool, candidate_audit, packet_rows
            completed.append(mode)
            check_sources(freeze)
            print(json.dumps(dict(mode=mode, status="complete", candidate_count=len(pool),
                selected_candidate_id=result["selected_candidate_id"],
                actual_pose_count=result["feedback"]["actual_camera_pose_count"],
                elapsed_s=round(time.monotonic()-started, 3))), flush=True)
        original = results["original"]
        gates, comparisons = {}, {}
        for mode in MODES:
            row = results[mode]
            gates[mode + "_151_actual_poses_and_150_paid_prefix"] = bool(
                row["map_frames"] == row["packet_count"] == 151
                and row["feedback"]["actual_camera_pose_count"] == 151
                and row["feedback"]["planning_camera_pose_count"] == 151
                and row["feedback"]["paid_actions"] == 150
                and row["feedback"]["remaining_budget"] == 48)
            calls = [(c["module"], c["method"]) for c in row["real_interface_calls"]]
            gates[mode + "_actual_four_required_calls"] = all(calls.count(pair) == 1 for pair in (
                ("IGCR", "bootstrap"), ("OV-SDF", "update_semantic"),
                ("STGHP", "update_topo"), ("STGHP", "select_topo_target")))
            gates[mode + "_nonempty_candidates"] = bool(pools[mode])
            gates[mode + "_physical_map_tsdf_quality_identical"] = row["physical_state"] == original["physical_state"]
            gates[mode + "_asset_geometry_identical"] = row["asset_geometry"] == original["asset_geometry"]
            gates[mode + "_candidate_pool_identical"] = pools[mode] == pools["original"]
            gates[mode + "_candidate_audit_identical"] = audits[mode] == audits["original"]
            gates[mode + "_nonlabel_raw_identical"] = [r["nonlabel_sensor_sha256"] for r in packets[mode]] == [r["nonlabel_sensor_sha256"] for r in packets["original"]]
            for baseline in ("G", "N"):
                name = mode + "_" + baseline + "_unchanged"
                comparisons[name] = numeric_comparison(row["scores"][baseline], original["scores"][baseline])
                gates[name] = comparisons[name]["bit_exact"]
        for name, a, b in (
            ("swapped_input_S_equals_original_X", results["swap"]["scores"]["S"], original["scores"]["X"]),
            ("missing_input_S_equals_original_G", results["missing"]["scores"]["S"], original["scores"]["G"]),
        ):
            comparisons[name] = numeric_comparison(a, b)
            gates[name] = comparisons[name]["passed"]
        comparisons["original_S_vs_swapped_input_S"] = numeric_comparison(original["scores"]["S"], results["swap"]["scores"]["S"])
        score_changed = not comparisons["original_S_vs_swapped_input_S"]["passed"]
        option_changed = original["selected_candidate_id"] != results["swap"]["selected_candidate_id"]
        gates["semantic_intervention_changes_S_score_or_selected_option"] = bool(score_changed or option_changed)
        gates["swap_and_missing_changed_input_labels"] = all(results[m]["changed_label_pixels"] > 0 for m in ("swap", "missing"))
        check_sources(freeze)
        for name, expected in inputs["input_sha256"].items():
            require(sha(PREPARATION / name) == expected, "original raw input changed during audit")
        summary = dict(schema_version="cpu_semantic_packet_v10_verification/1",
            status="passed_development_causal_checks" if all(gates.values()) else "failed_development_causal_gates",
            gates=gates, comparisons=comparisons,
            selected_candidate_ids={m: results[m]["selected_candidate_id"] for m in MODES},
            selected_option_changed=option_changed, S_score_changed=score_changed,
            source_sha256=freeze["source_sha256"], source_archive_sha256=freeze["archive_sha256"],
            preparation_manifest_sha256=PREPARATION_MANIFEST_SHA256,
            actual_prefix_replays=3, actual_prefix_packets_per_replay=151,
            new_world_actions=0, world_instances=0, future_or_gt_evaluation_inputs=[],
            score_parameter_adjustments=0, full_policy_performance_claim=False,
            elapsed_s=time.monotonic() - started)
        write(output / "verification.json", summary)
        lines = ["# V10 真实传感器接口语义干预审计", "", "结果：`" + summary["status"] + "`。",
            "", "同一 Q0/shelf_west 已付费前缀通过实际 Components/Runtime 各回放一次，分别保留、交换 2/3 类别、清空类别；RGB、深度、雷达及位姿保持不变。每次 151 帧、已付费 150 动作，总预算 198，剩余 48，固定锚点为前缀末位姿。没有执行新动作。",
            "", "|输入|S 选择 candidate ID|", "|---|---|"]
        lines += [f"|{m}|{results[m]['selected_candidate_id']}|" for m in MODES]
        lines += ["", f"检查通过 {sum(gates.values())}/{len(gates)}；原类别与交换类别的 S 分数变化：{score_changed}，选项变化：{option_changed}。",
            "", "逐项结果、严格数值误差、完整候选/评分/模块调用、物理数组哈希和原始输入 SHA 均已保存。实际相机历史按 151 帧核验，未用抽稀 keyframes 代替。",
            "", "这项结果只验证语义输入通过原接口影响同一历史上的选择；不证明轨迹性能、自然语义识别、开放词汇能力或完整方案优势。失败项保留，不调整参数或替换场景。"]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n")
        write(output / "artifact_hashes.json", {str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
        return summary
    except Exception as exc:
        write(output / "failure.json", dict(error_type=type(exc).__name__, message=str(exc),
            completed_variants=completed, retained_for_review=True, elapsed_s=time.monotonic()-started))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(json.dumps(dict(status=result["status"], checks_passed=sum(result["gates"].values()),
                         checks_total=len(result["gates"]), selected_candidate_ids=result["selected_candidate_ids"])))
