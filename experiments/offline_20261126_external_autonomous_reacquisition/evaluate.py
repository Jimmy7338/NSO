"""Score the sealed A/B reacquisition study against identical fixed scan references."""

from argparse import ArgumentParser
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

for thread_setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_setting] = "1"

import numpy as np
from open3d.core import Tensor


ROOT = Path(__file__).resolve().parent
ANALYSES = ROOT.parent
WEIGHTED = ANALYSES / "weighted_geometric_control_20261122/measurement/evaluate.py"
IMPLEMENTATION = (
    ANALYSES / "reproduction_20261104/simulation_publication_stage/experiments"
    / "unknown_exploration_20261104/source"
)
SCHEDULED = (96, 192)
ARMS = ("A", "B")
METRICS = ("C_free", "F1_scene_05cm", "J")
SENSOR_KEYS = ("depth_values_sha256", "rgb_values_sha256", "scan_ranges_sha256")
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
V3 = ANALYSES / "external_online_sweep_20261126"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, columns=()):
    fields = list(dict.fromkeys([*columns, *(key for row in rows for key in row)]))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measurement_engine():
    weighted = load_module("reacquisition_frozen_weighted_measurement", WEIGHTED)
    files = weighted.files
    metric = files.load_metric(read(files.SCENES / "freeze.json"), read(files.OLD_EVAL / "freeze.json"))
    return weighted, metric


def sealed_contract():
    freeze_path = ROOT / "FORMAL_FREEZE.json"
    freeze = read(freeze_path)
    protocol_path = ROOT / freeze["protocol_relative"]
    protocol = read(protocol_path)
    phase_path = ROOT / "execution/formal/phase_receipt.json"
    phase = read(phase_path)
    layout_specs = {row["layout"]: row for row in protocol["starts"]}
    layouts = list(layout_specs)
    jobs = {(row["layout"], row["arm"]): row for row in freeze["jobs"]}
    receipts = {(row["layout"], row["arm"]): row for row in phase["episodes"]}
    require(tuple(protocol["arms"]) == ARMS and freeze["budget"] == protocol["budget"] == 192
            and tuple(protocol["layouts"]) == tuple(layouts)
            and sorted(Counter(row["source_floor"] for row in layout_specs.values()).values()) == [2, 2]
            and set(jobs) == set(receipts) == {(layout, arm) for layout in layouts for arm in ARMS}
            and len(receipts) == len(phase["episodes"]) == len(jobs) == len(freeze["jobs"]) == 8,
            "Expected one complete registered A/B matrix, four starts nested in two source floors")
    require(phase["freeze_sha256"] == digest(freeze_path) and freeze["protocol_sha256"] == digest(protocol_path),
            "Formal run receipts and frozen protocol differ")
    for key, job in jobs.items():
        require(job["literal"] and job["regional_enabled"] and job["prior"] == "S"
                and job["native_policy"] == "combined_S"
                and job["repair"] == {"target": True, "outbound": True}
                and job["active_ground_reacquisition"] == (key[1] == "B"),
                f"Non-observer policy differs across the pair {key}")
        finish = receipts[key]
        require(finish["freeze_sha256"] == phase["freeze_sha256"]
                and all(finish.get(field) == value for field, value in job.items()),
                f"Job and receipt differ or describe different switches {key}")
    groups = ("actor_files", "actor_initial_files", "simulator_only_files", "evaluator_files", "qualification_files")
    for group in groups:
        for name, expected in freeze.get(group, {}).items():
            require(digest(name) == expected, f"Changed frozen input {name}")
    require(not any("evaluator_references" in Path(name).parts for name in freeze["actor_files"]),
            "An actor manifest includes evaluator geometry")
    provenance = {"freeze_sha256": digest(freeze_path), "protocol_sha256": digest(protocol_path),
                  "phase_receipt_sha256": digest(phase_path), "evaluator_sha256": digest(Path(__file__))}
    return freeze, protocol, layout_specs, receipts, provenance


def frozen_reference(scene_root, layout, freeze, weighted):
    initial_path = scene_root / "prepared" / layout / "initial_state.json"
    refs = scene_root / "evaluator_references" / layout
    truth_path, fixed_path, mesh_path, receipt_path = (refs / name for name in (
        "evaluator_truth.npz", "fixed_reference.npz", "source_mesh.ply", "RECEIPT.json",
    ))
    initial, receipt = read(initial_path), read(receipt_path)
    for path in (truth_path, fixed_path, mesh_path, receipt_path):
        require(digest(path) == freeze["evaluator_files"][str(path)], f"Unsealed scoring reference {path}")
    require(digest(initial_path) == freeze["actor_initial_files"][str(initial_path)], "Unsealed cold packet")
    with np.load(truth_path, allow_pickle=False) as truth:
        reachable = truth["reachable"]
        require(reachable.dtype == np.bool_ and not (reachable & truth["occupancy"]).any(),
                f"Reachable mask contains occupied source cells {layout}")
    with np.load(fixed_path, allow_pickle=False) as fixed:
        samples = fixed["all"].astype(np.float32, copy=False)
    sample_def = receipt["metadata"]["all"]
    require(reachable.shape == tuple(initial["shape"]) and int(reachable.sum()) == receipt["reachable_cells"]
            and int(reachable.sum()) > 0 and samples.shape == (sample_def["visible_samples"], 3)
            and len(samples) > 0 and np.isfinite(samples).all(), f"Frozen reference inconsistent for {layout}")
    empty = np.empty((0, 3), dtype=np.float32)
    return {
        "initial": initial, "reachable": reachable, "source_mesh": weighted.files.read_mesh(mesh_path),
        "samples": {"scene": samples, "shelf": empty, "cabinet": empty},
        "weights": {"scene": sample_def["area_per_sample_m2"], "shelf": 0, "cabinet": 0},
        "mask_file_sha256": digest(truth_path), "mask_array_sha256": array_digest(reachable),
        "reference_samples_sha256": digest(fixed_path), "source_reference_sha256": digest(mesh_path),
    }


def checked_run(layout, arm, receipt, freeze, source, weighted, sealed, phase="formal"):
    run = Path(receipt["run_path"]).resolve()
    require(run.is_relative_to(ROOT / "execution" / phase), f"Run outside {phase} directory {run}")
    result_path = receipt["result_path"]
    if result_path:
        require(digest(result_path) == receipt["result_sha256"], f"Native result hash differs {run}")
    if receipt["exit_code"] != 0:
        return {"status": "technical_failure", "run": run, "exit_code": receipt["exit_code"],
                "stop_reason": read(result_path).get("error") if result_path else "missing_native_result"}
    manifest, manifest_sha = weighted.engine.verify_manifest(run, "manifest.json")
    initial = source["initial"]
    require(manifest_sha == receipt["result_sha256"]
            and manifest["layout"] == layout and manifest["arm"] == arm
            and manifest["initial_packet_sha256"] == initial["initial_packet_sha256"]
            and manifest["initial_owner_sha256"] == initial["initial_owner_sha256"]
            and manifest["runner_freeze_sha256"] == sealed["freeze_sha256"]
            and manifest["source_protocol_sha256"] == sealed["protocol_sha256"]
            and manifest["cold_started_at_paid_zero"] and manifest["actual_closed_loop"]
            and not manifest["evaluator_inputs_read_by_actor"]
            and (manifest["global_mode"], manifest["local_mode"]) == ("G", "S")
            and manifest["native_budget"] == freeze["budget"], f"Changed runtime registration {run}")
    trace = read(run / "steps.json")
    weighted.native.verify_trace(trace, initial, freeze["budget"], manifest)
    weighted.native.verify_action_guard(trace, freeze["budget"])
    weighted.verify_pool(run, manifest, initial, trace)
    require(all(call["ranking"]["attribution_path_repair"]["score_repair"] == {"target": True, "outbound": True}
                for call in read(run / "regional_choices.json") if call["ranking"]["ranking"]),
            f"Native combined_S score changed {run}")
    with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as archive:
        ticks, beliefs = archive["paid"], archive["belief"]
    require(np.array_equal(ticks, np.arange(len(trace)+1)) and beliefs.shape == (len(ticks), *initial["shape"])
            and np.isin(beliefs, (-1, 0, 1)).all(), f"Wrong trace clock or coordinate frame {run}")
    counts = np.count_nonzero((beliefs == 0) & source["reachable"], axis=(1, 2))
    threshold = (95 * int(source["reachable"].sum()) + 99) // 100
    visits = np.flatnonzero(counts >= threshold)
    summary = read(run / "runtime_summary.json")["termination"]
    final = [*trace[-1]["position"], trace[-1]["heading"]] if trace else initial["anchor"]
    require(summary["final_action_id"] == len(trace) == manifest["paid_actions"]
            and manifest["final_pose"] == final and manifest["observed_native_termination"] == summary
            and manifest["returned"] == (final == initial["anchor"]),
            f"Actual native closure, endpoint, and anchor disagree {run}")
    raw = read(run / "initial_raw_rgbd_scan_sha256.json")
    require(raw["step"] == 0 and raw["packet_sha256"] == initial["initial_packet_sha256"],
            f"Cold packet proof absent {run}")
    return {
        "status": "completed_execution", "run": run, "manifest": manifest, "manifest_sha256": manifest_sha,
        "trace": trace, "beliefs": beliefs, "counts": counts, "initial_raw": raw,
        "events": read(run / "audit.json"), "actual_paid": len(trace),
        "A95_paid": int(visits[0]) if len(visits) else None,
        "stop_reason": summary["reason"], "end_pose": final,
        "ended_at_anchor_cell": final[:2] == initial["anchor"][:2],
        "ended_at_anchor_pose": final == initial["anchor"],
        "declared_return_steps": sum(row["phase_before_action"] == "return" for row in trace),
        "collisions": manifest["collisions"],
    }


def snapshot_row(data, name, source, weighted, metric):
    row, masks = weighted.native.evaluate_snapshot(
        data["run"], name, data["manifest"], source["initial"], data["trace"], source, metric,
    )
    if row["status"] != "captured":
        return None
    require(row["correct_free_cells"] == int(data["counts"][row["actual_paid"]]),
            f"Snapshot and independent observed trace disagree {data['run']}/{name}")
    relevant = {key: row[key] for key in (*METRICS, "correct_free_cells", "actual_paid",
                "p_scene_05cm", "r_scene_05cm", "native_mesh_triangles", "packet_sha256", "source_checkpoint_sha256")}
    return relevant | {"provenance": "original_native_checkpoint"}, masks


def restore_observation_clock(data, wanted, source, scene, layout, weighted, metric, raw_validation=None):
    """Rerender only recorded sensor poses to obtain exact unscheduled native maps."""
    from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10

    world = scene.make_world(layout)
    cfg, initial = world.config, source["initial"]
    mapper = ObservedRuntimeMapperV10(tuple(initial["shape"]), cfg, truncation_m=cfg.truncation_m)
    transform = GridTransform(tuple(initial["shape"]), cfg.resolution_m)
    first = data["initial_raw"] | {
        "position": initial["anchor"][:2], "heading": initial["anchor"][2],
        "action": None, "collision": False, "done": False,
    }
    checks = {}
    for name in data["manifest"]["checkpoints"]:
        receipt = read(data["run"] / name / "checkpoint.json")
        if receipt["status"] == "captured":
            checks[receipt["actual_paid"]] = receipt["native_mesh_fields"]
    restored = {}
    tested = 0
    last = max(wanted | (set(raw_validation.requests) if raw_validation is not None else set()))
    for raw in (first, *data["trace"][:last]):
        paid = raw["step"]
        world.position, world.heading, world.step_count = tuple(raw["position"]), raw["heading"], paid
        frame, scan = world.sense(), world.scan()
        for key, actual in zip(SENSOR_KEYS, (frame.depth_m, frame.color_rgb, scan.ranges_m)):
            require(array_digest(actual) == raw[key], f"Sensor values disagree during offline restoration {data['run']}:{paid} {key}")
        packet = SensorPacket(initial["scene_id"], "single_observation_cold_start", f"frame-{paid}", paid, frame, scan,
                              world.position, world.heading, "current_first_hit_RGBD_scan_visible_ideal_structure_labels",
                              "exact_discrete_simulator_odometry", raw["action"], raw["collision"], raw["done"]).validate(transform, cfg)
        require(packet.sha256() == raw["packet_sha256"], f"Wrong replay packet at {data['run']}:{paid}")
        mapper.update(frame, scan)
        require(np.array_equal(mapper.belief, data["beliefs"][paid]),
                f"Wrong replay occupancy at {data['run']}:{paid}")
        if raw_validation is not None:
            raw_validation.observe(packet, mapper.belief)
        if paid not in wanted and paid not in checks:
            continue
        mesh = mapper.mesh()
        geometry = {f"{field}_sha256": array_digest(getattr(mesh, field)) for field in ("vertices", "triangles")}
        if paid in checks:
            require(geometry == checks[paid], f"Wrong replay geometry at archived checkpoint {data['run']}:{paid}")
            tested += 1
        if paid in wanted:
            measured, present = weighted.files.measure_geometry(mesh, mapper.belief, source, metric)
            scene_points = source["samples"]["scene"]
            present["scene"] = (metric.ray_scene(mesh).compute_distance(Tensor(scene_points), nthreads=1).numpy() <= .05
                                if measured["sampled_predicted_surface"] else np.zeros(len(scene_points), dtype=bool))
            present["correct_free"] = source["reachable"] & (mapper.belief == 0)
            require(measured["correct_free_cells"] == int(data["counts"][paid])
                    and math.isclose(float(present["scene"].mean()), measured["r_scene_05cm"], abs_tol=1e-12, rel_tol=0),
                    "Exact-clock measures disagree with independent counts")
            restored[paid] = (measured | {"actual_paid": paid, "packet_sha256": raw["packet_sha256"],
                                           "provenance": "actual_sensor_prefix_reconstructed", **geometry}, present)
    require(tested >= 1, f"No native geometry checkpoint verified during restoration {data['run']}")
    return restored


def paired_change(a, b, reference):
    left, left_masks = a
    right, right_masks = b
    result = {f"delta_{key}_pp": 100 * (right[key] - left[key]) for key in METRICS}
    for label, scale, unit in (("scene", reference["weights"]["scene"], "m2"), ("correct_free", 1, "cells")):
        earlier, later = left_masks[label], right_masks[label]
        gained = int(np.count_nonzero(later & ~earlier))
        lost = int(np.count_nonzero(earlier & ~later))
        for kind, value in (("gross", gained), ("lost", lost), ("net", gained-lost)):
            result[f"{label}_{kind}_{unit}"] = value * scale
    return result


def sparse_aggregate(rows, count):
    summaries = {}
    for label in ("paid_96", "paid_192", "last_common_paid", "terminal"):
        selected = [r for r in rows if r["comparison"] == label and r["paired_available"]]
        pairs = len(selected)
        floors = sorted({row["source_floor"] for row in selected})
        measures = {}
        for metric in METRICS:
            key = f"delta_{metric}_pp"
            values = [row[key] for row in selected]
            measures[metric] = {
                "positive": sum(v > 1e-9 for v in values), "tied": sum(abs(v) <= 1e-9 for v in values),
                "negative": sum(v < -1e-9 for v in values),
                "paired_start_mean_pp": statistics.fmean(values) if values else None,
                "equal_floor_mean_pp": statistics.fmean(
                    statistics.fmean(row[key] for row in selected if row["source_floor"] == floor) for floor in floors
                ) if floors else None,
            }
        summaries[label] = {"registered_pairs": count, "observed_pairs": pairs, "observed_source_floors": len(floors),
                            "same_paid_is_required": label != "terminal", "metrics": measures}
    return summaries


def evaluate(output):
    from audit_reacquisition import (RawObserverValidation, attach_event_measures, inspect_run,
                                     matched_outcomes, requested_event_clocks)

    freeze, protocol, layouts, receipts, provenance = sealed_contract()
    scene_root = Path(freeze["scene_root"])
    sys.path[:0] = [str(IMPLEMENTATION), str(scene_root), str(V3)]
    scene = load_module("reacquisition_measured_source", scene_root / "scene.py")
    from observed_sweep import SweepEvidence
    weighted, metric = measurement_engine()
    require(not output.exists(), f"Use a new output directory for the frozen evaluation {output}")
    episodes, checkpoints, paired, curve, audits, event_rows = [], [], [], [], [], []
    for layout, start in layouts.items():
        source = frozen_reference(scene_root, layout, freeze, weighted)
        context = {"layout": layout, "source_floor": start["source_floor"], "start_index": start["start_index"]}
        data = {arm: checked_run(layout, arm, receipts[layout, arm], freeze, source, weighted, provenance) for arm in ARMS}
        last = min(entry["actual_paid"] for entry in data.values()) if all("actual_paid" in d for d in data.values()) else None
        if last is not None:
            require(np.array_equal(data["A"]["beliefs"][0], data["B"]["beliefs"][0]),
                    f"A/B initial observations differ despite the same cold packet {layout}")
        states, audits_here = {}, {}
        for arm, run in data.items():
            info = context | {"arm": arm}
            fixed = dict.fromkeys(SCHEDULED)
            raw_summary = {field: run.get(field) for field in (
                "status", "actual_paid", "A95_paid", "stop_reason", "end_pose", "ended_at_anchor_cell",
                "ended_at_anchor_pose", "declared_return_steps", "collisions", "manifest_sha256", "exit_code",
            )}
            episode = info | raw_summary | {
                "reach_mask_file_sha256": source["mask_file_sha256"], "reach_mask_array_sha256": source["mask_array_sha256"],
                "source_reference_mesh_sha256": source["source_reference_sha256"],
                "surface_samples_sha256": source["reference_samples_sha256"],
                "reachable_cells": int(source["reachable"].sum()), "terminal_budget": freeze["budget"],
                "A95_status": ("unknown_after_technical_failure" if run["status"] != "completed_execution"
                               else "reached" if run["A95_paid"] is not None else "not_reached_before_stop"),
            }
            if run["status"] == "completed_execution":
                actual = run["actual_paid"]
                terminal = snapshot_row(run, "terminal", source, weighted, metric)
                require(terminal is not None, f"No original terminal checkpoint {run['run']}")
                fixed = {tick: snapshot_row(run, f"paid_{tick}", source, weighted, metric) for tick in SCHEDULED}
                for tick, payload in fixed.items():
                    require((payload is not None) == (actual >= tick), f"Padded or missing native checkpoint {layout}/{arm}:{tick}")
                recorded = {tick: value for tick, value in fixed.items() if value is not None} | {actual: terminal}
                independent = inspect_run(run, source["initial"], arm)
                wanted = requested_event_clocks(independent)
                if last is not None:
                    wanted.add(last)
                missing = wanted - recorded.keys()
                config = source["initial"]["actual_config"]
                auditor = None
                if independent["attempts"]:
                    evidence = SweepEvidence(source["reachable"].shape, config["resolution_m"], config["robot_radius_m"])
                    auditor = RawObserverValidation(evidence, independent, source["initial"])
                if missing or auditor is not None:
                    recorded |= restore_observation_clock(run, missing, source, scene, layout, weighted, metric, auditor)
                independent["raw_certificate_recomputations"] = auditor.finish() if auditor is not None else {}
                states[arm] = {"terminal": terminal, "last_common_paid": recorded.get(last),
                               **{f"paid_{clock}": payload for clock, payload in fixed.items()}}
                attach_event_measures(independent, recorded, source, paired_change)
                audits_here[arm] = independent
                audits.append(info | independent)
                event_rows.extend(info | item for item in independent["attempts"])
                episode.update({key: terminal[0][key] for key in METRICS})
                episode["correct_free_cells_terminal"] = terminal[0]["correct_free_cells"]
                episode["peak_correct_free_cells"] = int(run["counts"].max())
                total = episode["reachable_cells"]
                curve.extend(info | {"paid": paid, "correct_free_cells": int(count), "reachable_cells": total,
                                     "C_free": int(count) / total} for paid, count in enumerate(run["counts"]))
            else:
                states[arm] = dict.fromkeys(("terminal", "last_common_paid", "paid_96", "paid_192"))
            episodes.append(episode)
            for label, payload in states[arm].items():
                requested = int(label.removeprefix("paid_")) if label.startswith("paid_") else last if label == "last_common_paid" else None
                checkpoints.append(info | {"checkpoint": label, "requested_paid": requested,
                    "available": payload is not None} | (payload[0] if payload is not None else {}))

        pair_context = matched_outcomes(data, audits_here, last)
        for label in states["A"]:
            control, observer = (states[arm][label] for arm in ARMS)
            row = context | pair_context | {
                "comparison": label, "paired_available": control is not None and observer is not None,
                "A_actual_paid": control[0]["actual_paid"] if control else None,
                "B_actual_paid": observer[0]["actual_paid"] if observer else None,
            }
            row["same_paid"] = bool(row["paired_available"] and row["A_actual_paid"] == row["B_actual_paid"])
            for key in METRICS:
                for name, payload in zip(ARMS, (control, observer)):
                    row[f"{name}_{key}"] = payload[0][key] if payload else None
            for key in METRICS:
                row[f"delta_{key}_pp"] = 100*(observer[0][key]-control[0][key]) if row["paired_available"] else None
            if label != "terminal" and row["paired_available"]:
                require(row["same_paid"], "A fixed-clock comparison paired different paid action counts")
            paired.append(row)
        print(f"{layout}: " + ", ".join(f"{a} {r['status']} at {r.get('actual_paid')}" for a, r in data.items()),
              flush=True)

    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("episodes", episodes), ("checkpoints", checkpoints), ("coverage_curve", curve),
                       ("paired_metrics", paired), ("inspection_attempts", event_rows)):
        write_csv(output / f"{name}.csv", rows, columns=("layout", "source_floor", "start_index"))
    write_json(output / "AUTONOMOUS_EVIDENCE_AUDIT.json", {"schema": "nso_autonomous_ground_observation_audit_v1",
        "meaning": "Receipt-to-command checks; the measured surface increment uses an unchanged metric over actual sensor prefixes.",
        "evaluation_provenance": provenance, "runs": audits, "evaluated_start_count": len(layouts)})
    summary = {
        "schema": "nso_autonomous_reacquisition_comparison_v1", "scope": "registered-start A/B comparison",
        "registered_pairs": len(layouts), "distinct_source_floor_count": 2, "formal_runs": len(receipts),
        "source_scope": protocol.get("source_scope"),
        "measurement_provenance": provenance, "reference": "same frozen V2-style reachable mask and surface samples within each start pair",
        "clock_policy": "96 and 192 require native capture at exactly that paid time. Last common time is min(terminal A, terminal B). Terminal rows may have unequal paid counts.",
        "arms": {"A": "combined_S no autonomous inspection", "B": "combined_S with autonomous inspection"},
        "comparisons": sparse_aggregate(paired, len(layouts)),
        "source_floor_comparisons": {floor: sparse_aggregate(
            [row for row in paired if row["source_floor"] == floor],
            sum(row["source_floor"] == floor for row in layouts.values()),
        ) for floor in sorted({row["source_floor"] for row in layouts.values()})},
        "per_arm": {arm: {
            "registered_runs": len(layouts),
            "completed_native_runs": sum(e["status"] == "completed_execution" and e["arm"] == arm for e in episodes),
            "technical_failures": sum(e["status"] == "technical_failure" and e["arm"] == arm for e in episodes),
            "native_collision_count": sum((e["collisions"] or 0) for e in episodes if e["arm"] == arm),
            "stopped_at_anchor": sum(bool(e["ended_at_anchor_pose"]) for e in episodes if e["arm"] == arm),
            "A95_reached": sum(e["A95_status"] == "reached" and e["arm"] == arm for e in episodes),
            "event_counts": dict(sum((Counter(row["event_counts"]) for row in audits if row["arm"] == arm), Counter())),
        } for arm in ARMS},
        "attempt_conditioned_comparisons": {
            kind: sparse_aggregate([r for r in paired if r["B_inspection_state"] == kind],
                                   len({r["layout"] for r in paired if r["B_inspection_state"] == kind}))
            for kind in sorted({r["B_inspection_state"] for r in paired})
        },
        "result_sha256": {file.name: digest(file) for file in sorted(output.iterdir())},
    }
    write_json(output / "summary.json", summary)
    return summary


def evaluate_development(output):
    """Record the operational evidence from the single separately sealed known-floor run."""
    from audit_reacquisition import RawObserverValidation, attach_event_measures, inspect_run, requested_event_clocks

    freeze_path = ROOT / "DEVELOPMENT_FREEZE.json"
    phase_path = ROOT / "execution/development/phase_receipt.json"
    freeze, phase = read(freeze_path), read(phase_path)
    receipt, = phase["episodes"]
    require(phase["freeze_sha256"] == digest(freeze_path), "Development receipt does not bind the freeze")
    layout, arm = receipt["layout"], receipt["arm"]
    scene_root = Path(freeze["scene_root"])
    sys.path[:0] = [str(IMPLEMENTATION), str(scene_root), str(V3)]
    scene = load_module("reacquisition_development_geometry", scene_root / "scene.py")
    from observed_sweep import SweepEvidence

    weighted, metric = measurement_engine()
    source = frozen_reference(scene_root, layout, freeze, weighted)
    provenance = {"freeze_sha256": digest(freeze_path), "protocol_sha256": freeze["protocol_sha256"],
                  "phase_receipt_sha256": digest(phase_path)}
    run = checked_run(layout, arm, receipt, freeze, source, weighted, provenance, phase="development")
    require(run["status"] == "completed_execution", "Development run lacks terminal native records")
    audit = inspect_run(run, source["initial"], arm)
    config = source["initial"]["actual_config"]
    evidence = SweepEvidence(source["reachable"].shape, config["resolution_m"], config["robot_radius_m"])
    bound = RawObserverValidation(evidence, audit, source["initial"])
    clocks = requested_event_clocks(audit)
    if clocks:
        states = restore_observation_clock(run, clocks, source, scene, layout, weighted, metric, bound)
        attach_event_measures(audit, states, source, paired_change)
    audit["raw_certificate_recomputations"] = bound.finish()
    terminal, _ = snapshot_row(run, "terminal", source, weighted, metric)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "DEVELOPMENT_MEASUREMENT_AUDIT.json", {
        "schema": "nso_autonomous_reacquisition_development_measurement_v1",
        "scope": "One known development start, A/B inference requires formal fresh-start pairs",
        "layout": layout, "arm": arm, "provenance": provenance,
        "native_manifest_sha256": run["manifest_sha256"], "scoring_mask_sha256": source["mask_file_sha256"],
        "reachable_cells": int(source["reachable"].sum()), "terminal_metrics": terminal,
        "terminal_actual_paid": run["actual_paid"], "terminal_reason": run["stop_reason"],
        "terminal_anchor_pose": run["ended_at_anchor_pose"], "audit": audit,
    })


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("formal", "development"), default="formal")
    parser.add_argument("--out", type=Path)
    command = parser.parse_args()
    destination = (command.out or ROOT / "results" / command.phase).resolve()
    evaluate(destination) if command.phase == "formal" else evaluate_development(destination)
