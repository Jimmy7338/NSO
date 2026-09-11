#!/usr/bin/env python3
"""Four fixed V7 failure regressions with a live observed-map execution gate.

Source/route sealing precedes execution. This diagnostic does not load previous
outcomes, evaluate reconstruction utility, or claim general navigation safety.
"""
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

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
CASES = (("T1", "storage_shelves"), ("T1", "ventilation_baffles"),
         ("T4", "storage_shelves"), ("T4", "ventilation_baffles"))
BUDGET = 48


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def array_hash(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    return hashlib.sha256(json.dumps([array.dtype.str, array.shape]).encode() + array.tobytes()).hexdigest()


def worker(args):
    sys.path.insert(0, str(args.snapshot))
    import numpy as np
    import open3d
    import scipy
    from env.virtual3d_response_v7 import ResponseContextV7, ResponseConfigV7, ResponseWorldV7
    from nso.execution_guard_v8 import ObservedExecutionGuard
    from nso.semantic_completion_v3 import SemanticHistoryMapperV3
    from nso.counterfactual_view_scoring import _mapper_snapshot
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    from utils.grid_geometry import DIRECTIONS

    def sensor_equal(actual, expected):
        for field in fields(type(actual)):
            np.testing.assert_array_equal(getattr(actual, field.name), getattr(expected, field.name))

    def collision_update(mapper, position, heading, collision):
        if collision:
            dr, dc = DIRECTIONS[heading]
            r, c = position[0] + dr, position[1] + dc
            if 0 <= r < mapper.shape[0] and 0 <= c < mapper.shape[1]:
                mapper.belief[r, c] = 1

    def successor(state, action):
        r, c, heading = state
        if action == "forward":
            dr, dc = DIRECTIONS[heading]
            return (r + int(dr), c + int(dc), heading)
        return (r, c, (heading + (1 if action == "right" else -1)) % 4)

    output = args.output
    seal = read(output / "pre_execution_seal.json")
    for name, expected in seal["source_sha256"].items():
        require(sha(args.snapshot / name) == expected, f"source snapshot mismatch: {name}")
    summaries = []
    for context_id, family in CASES:
        folder = output / context_id / family
        fixture = read(folder / "fixture.json")
        context_data = fixture["context"]
        require(context_data["context_id"] == context_id and fixture["family"] == family, "fixed case changed")
        context = ResponseContextV7(**{**context_data, "offset_xy_m": tuple(context_data["offset_xy_m"])})
        config = ResponseConfigV7(**fixture["config"])
        world = ResponseWorldV7(context, family, config)
        mapper = SemanticHistoryMapperV3(world.shape, config, config.truncation_m)
        prefix_records = read(folder / "prefix/records.json")
        require(len(prefix_records) == 21, "prefix must contain initial plus twenty paid frames")
        prefix_checks = []
        # Independently execute original prefix, never restore/teleport its end.
        for index, record in enumerate(prefix_records):
            if index == 0:
                frame, collision, done = world.sense(), False, False
            else:
                require(record["action"] == world.prefix_actions[index - 1], "prefix action differs")
                frame, collision, done = world.step(record["action"])
            scan = world.scan()
            expected_frame = RGBDFrame.load(folder / "prefix/frames" / f"{index:04d}.npz")
            expected_scan = PlanarScan.load(folder / "prefix/scans" / f"{index:04d}.npz")
            sensor_equal(frame, expected_frame); sensor_equal(scan, expected_scan)
            require(record["step"] == world.step_count == index and tuple(record["position"]) == world.position
                    and record["heading"] == world.heading and record["collision"] == collision
                    and record["moves"] == world.moves and record["collisions"] == world.collisions
                    and record["done"] == done, "prefix physical ledger differs")
            require(not collision and not done, "fixed prefix cannot fail silently")
            mapper.update(frame, scan)
            prefix_checks.append({"step": index, "sensor_fields_exact": True, "state_exact": True})
        dump(folder / "prefix_replay.json", {"status": "passed_exact", "frames": prefix_checks})
        route = read(folder / "route.json")
        anchor = (*world.position, world.heading)
        require(route["candidate_id"] == 5 and tuple(route["states"][0]) == anchor
                and tuple(route["states"][-1]) == anchor
                and len(route["actions"]) == route["cost"] <= BUDGET, "fixed route/budget changed")
        for a, b, action in zip(route["states"], route["states"][1:], route["actions"]):
            require(successor(tuple(a), action) == tuple(b), "sealed route contains an unpaid state jump")
        guard = ObservedExecutionGuard(config.resolution_m, config.robot_radius_m)
        result_dir = folder / "guarded_candidate_005"
        (result_dir / "frames").mkdir(parents=True)
        (result_dir / "scans").mkdir()
        (result_dir / "actions.jsonl").touch()
        dump(result_dir / "initial_mapper_hashes.json", _mapper_snapshot(mapper)["hashes"])
        actions = []; states = [list(anchor)]; decisions = []; transitions = []
        mode = "route"; route_cursor = 0; terminal = None; first_denial = None; return_triggered = False

        def decision(kind, **detail):
            row = {"decision_id": len(decisions), "kind": kind, "mode": mode,
                   "paid_actions_before": len(actions), "absolute_step": world.step_count,
                   "position": list(world.position), "heading": world.heading,
                   "remaining_actions": BUDGET - len(actions), "belief_sha256": array_hash(mapper.belief), **detail}
            decisions.append(row)
            with (result_dir / "decisions.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            return row["decision_id"]

        while terminal is None:
            current = (*world.position, world.heading)
            remaining = BUDGET - len(actions)
            if mode == "return" and current == anchor:
                # Location equality must not disguise an unsafe anchor or a
                # zero-movement denial as a successful executed return.
                at_anchor_plan = guard.return_plan(mapper.belief, world.position, world.heading, anchor, remaining)
                decision("return_plan_at_anchor", anchor=list(anchor), **asdict(at_anchor_plan))
                if not at_anchor_plan.available:
                    terminal = "return_unavailable:" + at_anchor_plan.reason
                else:
                    terminal = ("returned_to_prefix_anchor" if any(a["mode"] == "return" for a in actions)
                                else "already_at_prefix_anchor_after_denial")
                break
            if mode == "route" and route_cursor == len(route["actions"]):
                terminal = "original_route_completed" if current == anchor else "route_ended_away_from_anchor"
                break
            if remaining == 0:
                terminal = "budget_exhausted_at_anchor" if current == anchor else "budget_exhausted_away_from_anchor"
                break
            if mode == "route":
                action = route["actions"][route_cursor]
            else:
                # Deterministic replanning before every return action uses only
                # the latest observed map, including any new return observation.
                plan = guard.return_plan(mapper.belief, world.position, world.heading, anchor, remaining)
                decision("return_plan", anchor=list(anchor), **asdict(plan))
                if not plan.available:
                    terminal = "return_unavailable:" + plan.reason
                    break
                require(plan.paid_cost == len(plan.actions) and plan.actions, "available nonterminal return plan is empty")
                action = plan.actions[0]
            assessment = guard.assess(mapper.belief, world.position, world.heading, action, remaining)
            assessment_id = decision("assessment", action=action, route_cursor=route_cursor, **asdict(assessment))
            if not assessment.allowed:
                if first_denial is None:
                    first_denial = {"decision_id": assessment_id, "reason": assessment.reason,
                                    "paid_actions_before": len(actions), "action": action}
                if mode == "route":
                    transitions.append({"from": "route", "to": "return", "paid_actions": len(actions),
                                        "cause_decision_id": assessment_id, "reason": assessment.reason})
                    mode = "return"; return_triggered = True
                    decision("mode_switch", previous_mode="route", reason="first_route_action_denied")
                    continue
                # A returned first edge should agree with assess on the same
                # unchanged map. Recompute once, preserve any inconsistency.
                retry = guard.return_plan(mapper.belief, world.position, world.heading, anchor, remaining)
                decision("return_replan_after_denial", anchor=list(anchor), **asdict(retry))
                terminal = ("return_unavailable:" + retry.reason) if not retry.available else "return_plan_assessment_disagreement"
                break
            require(shutil.disk_usage(output).free >= seal["protocol"]["reserve_bytes"], "storage reserve reached; retain partial case")
            before_state = (*world.position, world.heading)
            frame, collision, done = world.step(action)
            scan = world.scan()
            paid = len(actions) + 1
            frame.save(result_dir / "frames" / f"{paid:04d}.npz")
            scan.save(result_dir / "scans" / f"{paid:04d}.npz")
            mapper.update(frame, scan); collision_update(mapper, world.position, world.heading, collision)
            actual = (*world.position, world.heading)
            expected = successor(before_state, action)
            row = {"action_index": paid, "absolute_step": world.step_count, "mode": mode,
                   "action": action, "assessment_decision_id": assessment_id,
                   "position": list(world.position), "heading": world.heading,
                   "collision": collision, "done": done, "expected_state": list(expected),
                   "belief_after_sha256": array_hash(mapper.belief)}
            actions.append(row); states.append(list(actual))
            with (result_dir / "actions.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if mode == "route":
                route_cursor += 1
            if collision:
                terminal = "physical_collision_after_allowed_action"
            elif actual != expected:
                terminal = "physical_state_mismatch"
            elif mode == "route" and actual != tuple(route["states"][route_cursor]):
                terminal = "original_route_state_mismatch"
            elif done:
                terminal = "world_terminal_state"
        decision("terminal_stop", reason=terminal, stop_is_control_decision_not_free_sensor_frame=True)
        final_state = (*world.position, world.heading)
        final = _mapper_snapshot(mapper)
        dump(result_dir / "final_mapper_hashes.json", final["hashes"])
        np.savez_compressed(result_dir / "final_mesh.npz", vertices=np.asarray(final["mesh"].vertices),
                            triangles=np.asarray(final["mesh"].triangles))
        np.savez_compressed(result_dir / "final_map.npz", belief=mapper.belief, visible=mapper.visible,
                            camera_seen=mapper.camera_seen)
        dump(result_dir / "actions.json", actions)
        dump(result_dir / "states.json", states)
        dump(result_dir / "decisions.json", decisions)
        dump(result_dir / "mode_transitions.json", transitions)
        result = {"context_id": context_id, "family": family, "candidate_id": 5,
                  "prefix_paid_actions": 20, "branch_budget": BUDGET, "paid_actions": len(actions),
                  "route_paid_actions": sum(a["mode"] == "route" for a in actions),
                  "return_paid_actions": sum(a["mode"] == "return" for a in actions),
                  "collisions": sum(a["collision"] for a in actions), "terminal_reason": terminal,
                  "first_denial": first_denial, "return_triggered": return_triggered,
                  "anchor": list(anchor), "final_state": list(final_state),
                  "at_anchor_including_heading": final_state == anchor,
                  "final_footprint_known_safe": bool(guard.safe_grid(mapper.belief)[world.position]),
                  "guarded_return_completed": terminal == "returned_to_prefix_anchor",
                  "original_route_completed": route_cursor == len(route["actions"]) and final_state == anchor,
                  "scope": "all four previously known failed-route development regressions; no system safety or semantic efficacy conclusion"}
        dump(result_dir / "result.json", result); summaries.append(result)
        print("guard diagnostic", context_id, family, "paid", len(actions), "terminal", terminal, flush=True)
        del mapper, world
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in ("env", "nso", "utils") and getattr(module, "__file__", None):
            require(Path(module.__file__).resolve().is_relative_to(args.snapshot), f"nonarchived module imported: {name}")
    dump(output / "results.json", {"cases": summaries, "paid_actions": sum(r["paid_actions"] for r in summaries),
         "max_paid_actions": 4 * BUDGET, "dependencies": {"numpy": np.__version__, "open3d": open3d.__version__, "scipy": scipy.__version__},
         "old_outcome_values_read": False, "gt_used_by_guard_or_return_plan": False,
         "evaluation_scope": "execution feasibility only; no reconstruction/semantic reward evaluated"})


def run(args):
    output = args.output.resolve()
    require(not output.exists(), "output already exists; preserve previous diagnostic evidence")
    output.mkdir(parents=True)
    metadata = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
                "cases": [dict(context_id=c, family=f, candidate_id=5) for c, f in CASES],
                "independently_replayed": False}
    dump(output / "metadata.json", metadata)
    started = time.monotonic()
    inputs = {}; base_sources = None; contents = None
    for context_id, family in CASES:
        source_run = args.training_root.resolve() / context_id
        old_meta = read(source_run / "metadata.json")
        old_manifest = read(source_run / "artifact_hashes.json")
        require(old_meta["status"] == "complete", "source acquisition is incomplete")
        if base_sources is None:
            base_sources = old_meta["source_sha256"]
            with zipfile.ZipFile(source_run / "sources.zip") as archive:
                require(set(archive.namelist()) == set(base_sources), "source archive inventory differs")
                contents = {name: archive.read(name) for name in archive.namelist()}
                require(all(hashlib.sha256(content).hexdigest() == base_sources[name] for name, content in contents.items()), "old source archive differs")
        require(old_meta["source_sha256"] == base_sources, "fixed source runs use different old simulator/mapper versions")
        source_folder = source_run / family
        names = ["fixture.json", "candidates.json", "prefix/records.json"]
        names += [f"prefix/{kind}/{i:04d}.npz" for kind in ("frames", "scans") for i in range(21)]
        folder = output / context_id / family
        for name in names:
            original = source_folder / name
            require(sha(original) == old_manifest[f"{family}/{name}"], f"original decision/prefix file changed: {original}")
            target = folder / name; target.parent.mkdir(parents=True, exist_ok=True)
            os.link(original, target)
            inputs[str(target.relative_to(output))] = {"source": str(original), "sha256": sha(original)}
        routes = read(folder / "candidates.json")
        require([r["candidate_id"] for r in routes] == list(range(6)), "original candidate inventory changed")
        dump(folder / "route.json", routes[5])
    overlays = ("nso/execution_guard_v8.py", "scripts/eval_execution_guard_v8.py", "tests/virtual3d/test_execution_guard_v8.py")
    for name in overlays:
        contents[name] = (ROOT / name).read_bytes()
    source_hashes = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    with zipfile.ZipFile(output / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(contents):
            archive.writestr(name, contents[name])
    decision_assets = {str(p.relative_to(output)): sha(p)
                       for context_id, family in CASES for p in (output / context_id / family).rglob("*") if p.is_file()}
    seal = {"schema_version": "execution_guard_v8_fixed_failures/1", "sealed_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_hashes, "source_archive_sha256": sha(output / "sources.zip"),
            "original_inputs": inputs, "decision_assets": decision_assets,
            "cases": metadata["cases"], "protocol": {"branch_budget": BUDGET, "workers": 1,
            "reserve_bytes": 400 * 1024**2, "first_denial": "switch to latest-map paid return",
            "return_policy": "recompute return_plan and assess first action before every return step",
            "physical_collision": "terminate immediately, retain evidence; no invented escape",
            "unavailable_return": "terminal stop without calling world.step(stop) or receiving new sensor data",
            "zero_outcome_selection": "all fixed T1/T4 both-family candidate5 failures; no old outcome file read"}}
    dump(output / "pre_execution_seal.json", seal)
    metadata.update(status="sealed_before_execution", source_archive_sha256=seal["source_archive_sha256"],
                    pre_execution_seal_sha256=sha(output / "pre_execution_seal.json"))
    dump(output / "metadata.json", metadata)
    try:
        with tempfile.TemporaryDirectory(prefix="execution-guard-v8-") as temporary:
            snapshot = Path(temporary)
            for name, content in contents.items():
                relative = Path(name)
                require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
                target = snapshot / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
            command = [sys.executable, str(snapshot / "scripts/eval_execution_guard_v8.py"), "--worker",
                       "--snapshot", str(snapshot), "--output", str(output)]
            process = subprocess.run(command, cwd=snapshot, env=os.environ.copy())
            require(process.returncode == 0, "guard execution subprocess failed; retain incomplete case evidence")
        for name, expected in decision_assets.items():
            require(sha(output / name) == expected, "sealed input modified during diagnostic")
        for item in inputs.values():
            require(sha(item["source"]) == item["sha256"], "original V7 asset was modified")
        for name in overlays:
            require(sha(ROOT / name) == source_hashes[name], "guard/runner source changed during diagnostic")
        result = read(output / "results.json")
        require(len(result["cases"]) == 4 and result["paid_actions"] <= 4 * BUDGET, "fixed diagnostic matrix exceeded")
        metadata.update(status="complete_execution_pending_independent_replay", completed_cases=4,
                        branch_paid_actions=result["paid_actions"], elapsed_seconds=time.monotonic() - started)
    except Exception as error:
        metadata.update(status="failed_execution", error=repr(error), elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        dump(output / "metadata.json", metadata)
        dump(output / "artifact_hashes.json", {str(p.relative_to(output)): sha(p) for p in output.rglob("*")
                                               if p.is_file() and p.name != "artifact_hashes.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=ROOT / "eval_results/response_v7_training_20260911")
    parser.add_argument("--output", type=Path, default=ROOT / "eval_results/execution_guard_v8_failures_20260911")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        run(args)
