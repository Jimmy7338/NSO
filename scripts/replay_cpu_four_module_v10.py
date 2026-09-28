#!/usr/bin/env python3
"""Independent physical/accounting replay of the V10 CPU integration fixture.

Only the archived world, RGB-D records and measured mapper are reused. This
verifier owns its footprint stencil, orientation BFS, action state machine,
quality-feedback arithmetic, option path checks and log consistency checks.
It never calls Components, Runtime, guard, ledger, candidate or scoring methods.
It does not calculate efficacy, GT reconstruction scores or baseline superiority.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from collections import deque
from dataclasses import fields
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SCHEMA = "cpu_four_module_v10_independent_replay/1"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def relative(root, name):
    require(isinstance(name, str) and bool(name), "missing artifact path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
            "unsafe relative artifact path: " + name)
    result = root.joinpath(*path.parts)
    require(result.resolve().is_relative_to(root.resolve()) and not result.is_symlink(), "escaped artifact")
    return result


def equal(actual, expected, name):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.dtype == b.dtype, name + ": dtype/shape mismatch")
    np.testing.assert_array_equal(a, b, err_msg=name)


def number(actual, expected, name, tolerance=1e-12):
    require(not isinstance(actual, bool) and np.isfinite(actual)
            and abs(float(actual) - float(expected)) <= tolerance,
            f"{name}: {actual!r} != {expected!r}")


def inside(safe, state):
    return 0 <= state[0] < safe.shape[0] and 0 <= state[1] < safe.shape[1]


def successor(state, action):
    r, c, heading = state
    require(heading in range(4), "invalid pose heading")
    if action == "forward":
        dr, dc = DIRECTIONS[heading]
        return r + dr, c + dc, heading
    require(action in ("left", "right"), "unknown/free primitive action")
    return r, c, (heading + (1 if action == "right" else -1)) % 4


def footprint(belief, config):
    """Independent discrete obstacle-cell distance stencil, including border.

    At the fixture's radius/resolution=1, this is exactly a 3 by 3 all-free
    neighborhood. No production inflation or EDT function is called.
    """
    require(belief.ndim == 2 and np.isin(belief, (-1, 0, 1)).all(), "invalid observed belief")
    radius = float(config.robot_radius_m) / float(config.resolution_m) + np.sqrt(2) / 2
    span = int(np.ceil(radius))
    free = np.pad(belief == 0, span, constant_values=False)
    safe = np.ones(belief.shape, bool)
    for dr in range(-span, span + 1):
        for dc in range(-span, span + 1):
            if np.hypot(dr, dc) <= radius:
                safe &= free[span + dr:span + dr + belief.shape[0],
                             span + dc:span + dc + belief.shape[1]]
    return safe


def distances(safe, start, reverse=False):
    start = tuple(start)
    if not inside(safe, start) or not safe[start[:2]]:
        return {}
    values, queue = {start: 0}, deque([start])
    while queue:
        state = queue.popleft()
        r, c, h = state
        dr, dc = DIRECTIONS[h]
        neighbors = ((r, c, (h - 1) % 4), (r, c, (h + 1) % 4),
                     (r - dr, c - dc, h) if reverse else (r + dr, c + dc, h))
        for target in neighbors:
            if target not in values and inside(safe, target) and safe[target[:2]]:
                values[target] = values[state] + 1
                queue.append(target)
    return values


def return_spec(safe, state, anchor, budget, home):
    if not inside(safe, state) or not safe[state[:2]]:
        return False, "current_footprint_not_known_safe", 0
    if not inside(safe, anchor) or not safe[anchor[:2]]:
        return False, "anchor_footprint_not_known_safe", 0
    if state not in home:
        return False, "anchor_disconnected_in_latest_map", 0
    if home[state] > budget:
        return False, "known_return_exceeds_remaining_budget", home[state]
    return True, "known_safe_return_within_budget", home[state]


def assessment(safe, state, action, anchor, budget, home):
    target = successor(state, action)
    initial_reason = ("no_action_budget" if budget < 1 else
                      "current_footprint_not_known_safe" if not inside(safe, state) or not safe[state[:2]] else
                      "next_footprint_not_known_safe" if not inside(safe, target) or not safe[target[:2]] else None)
    if initial_reason is not None:
        return dict(allowed=False, reason=initial_reason, action=action,
                    next_pose=list(target), remaining_budget=budget,
                    reserved_return_cost=None, learned_probability=None, calibrated_uncertainty=None)
    allowed, reason, cost = return_spec(safe, target, anchor, budget - 1, home)
    return dict(allowed=allowed, reason=reason, action=action, next_pose=list(target),
                remaining_budget=budget, reserved_return_cost=cost,
                learned_probability=None, calibrated_uncertainty=None)


def path_check(states, actions, start, end, safe, name):
    require(len(states) == len(actions) + 1, name + ": state/action lengths")
    require(tuple(states[0]) == start and tuple(states[-1]) == end, name + ": endpoints/heading")
    state = start
    require(inside(safe, state) and safe[state[:2]], name + ": unsafe initial footprint")
    for index, action in enumerate(actions):
        state = successor(state, action)
        require(tuple(states[index + 1]) == state, name + ": action/pose mismatch")
        require(inside(safe, state) and safe[state[:2]], name + ": unknown/occupied footprint")


def option_check(route, safe, state, anchor, budget, outward, home):
    target = tuple(route["pose"])
    outgoing, returning = route["outbound_actions"], route["return_actions"]
    require(bool(outgoing), "free option is not an executable paid view")
    path_check(route["outbound_states"], outgoing, state, target, safe, "outbound")
    path_check(route["return_states"], returning, target, anchor, safe, "return")
    require(route["states"] == route["outbound_states"] + route["return_states"][1:], "full states splice")
    require(route["actions"] == outgoing + returning, "full actions splice")
    require(route["outbound_cost"] == len(outgoing) == outward.get(target), "shortest outbound cost")
    require(route["return_cost"] == len(returning) == home.get(target), "shortest return-heading cost")
    require(route["arrival_action"] == len(outgoing), "arrival action")
    require(route["cost"] == len(outgoing) + len(returning) <= budget, "full reserved option budget")
    require(tuple(route["return_anchor"]) == anchor, "fixed task anchor")


def evidence(mapper):
    known = mapper.belief != -1
    qualities, support = {}, set()
    for raw_key, row in mapper.quality.items():
        key = tuple(map(int, raw_key))
        info, residual = float(row["information"]), float(row["residual"])
        qualities[key] = info / (info + .25 + 100 * residual)
        p = row["point"]
        r = known.shape[0] - 1 - int(np.floor(p[1] / mapper.config.resolution_m))
        c = int(np.floor(p[0] / mapper.config.resolution_m))
        if .12 < p[2] < 1.8 and 0 <= r < known.shape[0] and 0 <= c < known.shape[1] and known[r, c]:
            support.add(key)
    return known.copy(), (np.asarray(mapper.camera_seen) & known).copy(), qualities, support


def record_arrays(record):
    return {field.name: np.asarray(getattr(record, field.name)) for field in fields(record)}


def raw_check(path, expected, name):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == set(expected), name + ": raw field inventory")
        for key, array in expected.items():
            equal(data[key], array, name + "/" + key)


def packet_sha(meta, frame, scan):
    """Independent reproduction of the length-delimited packet identity."""
    out = hashlib.sha256()
    def append(value):
        out.update(len(value).to_bytes(8, "big")); out.update(value)
    names = ("scene_id", "episode_id", "frame_id", "action_id", "frame", "scan", "position",
             "heading", "sensor_source", "pose_source", "action", "collision", "done")
    for name in names:
        append(name.encode())
        if name in ("frame", "scan"):
            for inner, value in record_arrays(frame if name == "frame" else scan).items():
                array = np.ascontiguousarray(value)
                append(inner.encode()); append(array.dtype.str.encode())
                append(str(array.shape).encode()); append(array.tobytes())
        else:
            append(json_sha(meta[name]).encode())
    return out.hexdigest()


def audit_frame_calls(calls, mapper, raw, anchor, budget, feedback_expected, totals):
    state = (*raw["position"], raw["heading"])
    safe = footprint(mapper.belief, mapper.config)
    home = distances(safe, anchor, reverse=True)
    outward = None
    for call in calls:
        method, inputs, outputs = call["method"], call["inputs"], call["outputs"]
        require(call["frame_id"] == raw["frame_id"] and call["action_id"] == raw["action_id"], "module frame/action binding")
        require(call["map_version"] == mapper.frames, "module map version")
        require(call["input_sha256"] == json_sha(inputs) and call["output_sha256"] == json_sha(outputs), "module digest binding")
        require(call["trained"] is False and call["calibrated"] is False and call["truth_used"] is False,
                "fixture capability claim changed")
        if method == "assess_local_action":
            require(outputs == assessment(safe, state, outputs["action"], anchor, budget, home), "independent RPN assessment")
            require(inputs["remaining_budget"] == budget and tuple(inputs["anchor"]) == anchor, "RPN current budget/anchor")
            require(inputs["belief_sha256"] == json_sha(mapper.belief.tolist()), "RPN observed map digest")
            totals["assessments_checked"] += 1
        elif method == "plan_return":
            available, reason, cost = return_spec(safe, state, anchor, budget, home)
            require((outputs["available"], outputs["reason"], outputs["paid_cost"]) == (available, reason, cost),
                    "independent return availability/cost")
            if available:
                walked = [list(state)]
                for action in outputs["actions"]:
                    walked.append(list(successor(tuple(walked[-1]), action)))
                path_check(walked, outputs["actions"], state, anchor, safe, "runtime return")
                require(len(outputs["actions"]) == cost, "runtime return shortest cost")
            else:
                require(outputs["actions"] == [], "unavailable return carries actions")
            totals["returns_checked"] += 1
        elif method == "select_topo_target":
            require(inputs["remaining_budget"] == budget, "global selection used stale budget")
            require(inputs["actual_feedback_version"] == feedback_expected["feedback_version"], "global selection used stale feedback")
            require(inputs["semantic_version"] == inputs["graph_version"] == mapper.frames, "global selection used stale modules")
            candidates = outputs["candidates"]
            outward = distances(safe, state) if outward is None else outward
            require(len({r["candidate_id"] for r in candidates}) == len(candidates), "duplicate candidate ID")
            for route in candidates:
                option_check(route, safe, state, anchor, budget, outward, home)
                totals["candidate_paths_checked"] += 1
            scores = outputs["scores"]
            require(set(scores) == {"N", "G", "O", "S", "X", "M"}, "missing score control")
            require(all(len(values) == len(candidates) and np.isfinite(values).all() for values in scores.values()), "score lengths/finite")
            require(scores["M"] == scores["G"], "missing-class fallback is not exact")
            for i, row in enumerate(outputs["score_audit"]):
                require(row["candidate_id"] == candidates[i]["candidate_id"], "score row candidate binding")
                for mode in scores:
                    number(scores[mode][i], row["common_total_proxy"] + row["prior_potentials"][mode], "score decomposition")
            selected = outputs["selected"]
            index = min(range(len(candidates)), key=lambda i: (-scores[outputs["mode"]][i],
                        candidates[i]["cost"], candidates[i]["candidate_id"])) if candidates else None
            if index is None or scores[outputs["mode"]][index] <= 0:
                require(selected is None, "nonpositive candidate selected")
            else:
                require(selected is not None, "positive best candidate omitted")
                for key, value in candidates[index].items():
                    require(selected[key] == value, "selected candidate content differs")
                require(selected["selection_call_id"] == call["call_id"], "selection call ID")
                require(selected["selected_feedback_version"] == feedback_expected["feedback_version"], "option feedback version")
            totals["global_choices_checked"] += 1
        elif method in ("bootstrap", "compute_reward"):
            require(outputs["feedback_version"] == feedback_expected["feedback_version"], "IGCR version")
            require(outputs["paid_actions"] == raw["action_id"] and outputs["remaining_budget"] == budget, "IGCR paid accounting")
            if method == "compute_reward":
                for key, value in feedback_expected["parts"].items():
                    number(outputs["parts"][key], value, "independent IGCR " + key)
                number(outputs["old_quality_proxy"], feedback_expected["old_quality_proxy"], "fixed old quality")
                require(outputs["actual_camera_pose_count"] == mapper.frames, "IGCR actual camera history count")
                require(outputs["fixed_old_support_count"] == feedback_expected["fixed_old_support_count"], "fixed quality denominator")
                require(outputs["done"] == raw["done"] and outputs["collision"] == raw["collision"], "IGCR terminal/collision sensor binding")
                totals["feedback_events_checked"] += 1
    return safe, home


def worker(run, output, frozen):
    started = time.perf_counter()
    # These imports must resolve exclusively to the checked source archive.
    from env.virtual3d_competition_v9 import CompetitionConfigV9, CompetitionWorldV9, get_competition_context
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in ("env", "nso", "utils") and getattr(module, "__file__", None):
            require(Path(module.__file__).resolve().is_relative_to(frozen.resolve()), "source import escaped archive: " + name)
    forbidden_fragments = ("planner", "scoring", "prior", "objective", "candidates", "options",
                           "response_features", "components", "runtime_integration", "observed_feedback",
                           "execution_guard", "cpu_four_modules")
    def no_production_decisions(frame, event, argument):
        if event == "call":
            path = Path(frame.f_code.co_filename)
            if path.is_relative_to(frozen / "nso"):
                require(not any(part in path.name for part in forbidden_fragments),
                        "production decision/accounting method called by independent replay: " + str(path))
    sys.setprofile(no_production_decisions)
    manifest = read(run / "manifest.json")
    raws = read(relative(run, manifest["raw_manifest"]))
    config = CompetitionConfigV9(**manifest["config"])
    world = CompetitionWorldV9(get_competition_context(manifest["context_id"]), manifest["arrangement"], config=config)
    anchor, budget = tuple(manifest["initial_anchor"]), manifest["total_budget"]
    require(raws and len(raws) <= budget + 1 and type(budget) is int and budget > 0, "invalid fixture budget/raw inventory")
    require(tuple((*world.position, world.heading)) == anchor, "fixture initial anchor differs from physical start")
    calls, events = read(run / "module_calls.json"), read(run / "runtime_events.json")
    require([c["call_id"] for c in calls] == list(range(1, len(calls) + 1)), "module call ordering")
    for index, call in enumerate(calls):
        require(call["parent_call_id"] == (None if index == 0 else calls[index - 1]["call_id"]), "broken module parent chain")
        if index:
            require(call["action_id"] >= calls[index - 1]["action_id"], "module calls moved backwards in sensor history")
    call_groups = {}
    for call in calls:
        call_groups.setdefault(call["frame_id"], []).append(call)
    require(set(call_groups) == {raw["frame_id"] for raw in raws}, "unbound or missing module frames")
    require({call["module"] for call in calls} == {"OV-SDF", "STGHP", "RPN-UQ", "IGCR"}, "missing four-module participation")
    require(len({raw["frame_id"] for raw in raws}) == len(raws), "duplicate raw sensor frame")
    authorizations = [event for event in events if event["event"] == "paid_action_authorized"]
    observations = [event for event in events if event["event"] == "observation"]
    require(len(authorizations) == len(observations) == len(raws) - 1, "paid authorization/observation/raw counts")
    mapper = ObservedRuntimeMapperV10(world.shape, config)
    totals = dict(raw_frames_checked=0, raw_scans_checked=0, paid_actions_checked=0,
                  assessments_checked=0, returns_checked=0, candidate_paths_checked=0,
                  global_choices_checked=0, feedback_events_checked=0)
    histories, previous_safe, previous_home = {}, None, None
    ever_known = ever_camera = old_keys = ever_support = None
    old_quality = 0.
    raw_hashes = {}
    for i, raw in enumerate(raws):
        require(raw["action_id"] == i, "nonconsecutive raw paid action ID")
        require(raw["scene_id"] == raws[0]["scene_id"] and raw["episode_id"] == raws[0]["episode_id"], "raw episode isolation")
        if i == 0:
            require(raw["action"] is None and raw["done"] is False and raw["collision"] is False, "invalid initial frame metadata")
            actual_frame, actual_collision, actual_done = world.sense(), False, False
        else:
            previous = raws[i - 1]
            state = (*previous["position"], previous["heading"])
            next_state = successor(state, raw["action"])
            require(inside(previous_safe, state) and previous_safe[state[:2]], "paid action starts with unknown/unsafe footprint")
            require(inside(previous_safe, next_state) and previous_safe[next_state[:2]], "paid successor has unknown/unsafe footprint")
            require(next_state in previous_home and previous_home[next_state] <= budget - i, "paid action loses return-heading budget")
            authorization, observation = authorizations[i - 1], observations[i - 1]
            expected = assessment(previous_safe, state, raw["action"], anchor, budget - i + 1, previous_home)
            require(authorization["assessment"] == expected and expected["allowed"], "authorization mismatch with independent gate")
            require(authorization["frame_id"] == previous["frame_id"] and authorization["next_action_id"] == i,
                    "authorization bound to wrong frame/action")
            require(observation["action_id"] == i and observation["frame_id"] == raw["frame_id"], "observation event binding")
            require(events.index(authorization) < events.index(observation), "sensor frame precedes paid authorization")
            if i > 1:
                require(events.index(observations[i - 2]) < events.index(authorization), "next action before prior sensor observation")
            actual_frame, actual_collision, actual_done = world.step(raw["action"])
            require(bool(actual_collision) == raw["collision"] and bool(actual_done) == raw["done"], "physical collision/terminal differs")
            totals["paid_actions_checked"] += 1
        actual_scan = world.scan()
        require(tuple(raw["position"]) == tuple(world.position) and raw["heading"] == world.heading, "physical pose differs")
        frame_path, scan_path = relative(run, raw["frame_path"]), relative(run, raw["scan_path"])
        raw_check(frame_path, record_arrays(actual_frame), f"frame {i}")
        raw_check(scan_path, record_arrays(actual_scan), f"scan {i}")
        raw_hashes[raw["frame_path"]], raw_hashes[raw["scan_path"]] = sha(frame_path), sha(scan_path)
        require(raw["frame_sha256"] == raw_hashes[raw["frame_path"]]
                and raw["scan_sha256"] == raw_hashes[raw["scan_path"]], "raw manifest file hash")
        saved_frame, saved_scan = RGBDFrame.load(frame_path), PlanarScan.load(scan_path)
        # Metadata hash includes done/collision/provenance, independent of production SensorPacket.
        packet_digest = packet_sha(raw, saved_frame, saved_scan)
        require(raw["packet_sha256"] == packet_digest, "raw manifest packet hash")
        if i:
            require(observations[i - 1]["packet_sha256"] == packet_digest, "runtime packet digest")
        mapper.update(saved_frame, saved_scan)
        known, camera, quality, support = evidence(mapper)
        if i == 0:
            ever_known, ever_camera = known.copy(), camera.copy()
            old_keys, ever_support = frozenset(support), set(support)
            old_quality = float(sum(quality[key] for key in sorted(old_keys)) / len(old_keys)) if old_keys else 0.
            expected_feedback = dict(feedback_version=1)
        else:
            current_quality = float(sum(quality.get(key, 0.) for key in sorted(old_keys)) / len(old_keys)) if old_keys else 0.
            parts = dict(new_observed_2d_area_m2=float(np.count_nonzero(known & ~ever_known)) * config.resolution_m**2,
                         new_camera_observed_2d_area_m2=float(np.count_nonzero(camera & ~ever_camera)) * config.resolution_m**2,
                         measured_support_proxy_m2=len(support - ever_support) * .15**2,
                         fixed_old_quality_proxy_change=current_quality - old_quality, action_cost=1)
            expected_feedback = dict(feedback_version=i + 1, parts=parts, old_quality_proxy=current_quality,
                                     fixed_old_support_count=len(old_keys))
            ever_known |= known; ever_camera |= camera; ever_support |= support; old_quality = current_quality
        frame_calls = call_groups[raw["frame_id"]]
        methods = [call["method"] for call in frame_calls]
        require(methods.count("update_semantic") == 1 and methods.count("update_topo") == 1,
                "each actual frame must update both semantic and topology interfaces exactly once")
        require(methods.count("bootstrap") == int(i == 0) and methods.count("compute_reward") == int(i > 0),
                "feedback is not once per actual paid frame")
        if i:
            require(methods.index("update_semantic") < methods.index("compute_reward") < methods.index("update_topo"),
                    "observation module order")
        previous_safe, previous_home = audit_frame_calls(frame_calls, mapper, raw, anchor, budget - i, expected_feedback, totals)
        histories[raw["frame_id"]] = dict(action_id=i, packet_sha256=packet_digest)
        totals["raw_frames_checked"] += 1; totals["raw_scans_checked"] += 1
    # Full mapper support inventory, not a sampled quality_evidence projection.
    with np.load(run / "final_map.npz", allow_pickle=False) as saved:
        expected_names = {"belief", "camera_seen", "quality_keys", "quality_points", "quality_normals", "quality_n",
                          "quality_bits", "quality_label", "quality_information", "quality_residual",
                          "quality_best_range", "quality_normal_dispersion"}
        require(set(saved.files) == expected_names, "final map field inventory")
        equal(saved["belief"], mapper.belief, "final belief")
        equal(saved["camera_seen"], mapper.camera_seen, "final camera footprint")
        keys = [tuple(map(int, key)) for key in saved["quality_keys"]]
        require(len(set(keys)) == len(keys) and set(keys) == set(mapper.quality), "final full quality support keys")
        equal(saved["quality_keys"], np.asarray(sorted(mapper.quality), dtype=np.int64).reshape(-1, 3), "quality keys and deterministic order")
        for dest, source in (("quality_points", "point"), ("quality_normals", "normal"), ("quality_n", "n"), ("quality_bits", "bits"),
                             ("quality_label", "label"), ("quality_information", "information"), ("quality_residual", "residual"),
                             ("quality_best_range", "best_range"), ("quality_normal_dispersion", "normal_dispersion")):
            expected = np.asarray([mapper.quality[key][source] for key in keys],
                                  dtype=np.int64 if source in ("n", "bits", "label") else float)
            if source in ("point", "normal"):
                expected = expected.reshape(-1, 3)
            equal(saved[dest], expected, dest)
    mesh = mapper.mesh()
    with np.load(run / "final_mesh.npz", allow_pickle=False) as saved:
        mesh_fields = ("vertices", "triangles", "vertex_normals", "vertex_colors")
        require(set(saved.files) == set(mesh_fields), "final TSDF mesh field inventory")
        for name in mesh_fields:
            equal(saved[name], np.asarray(getattr(mesh, name)), "final TSDF mesh " + name)
    # Check source chain from selected option to actual outbound motion.
    selects = {call["call_id"]: call for call in calls if call["method"] == "select_topo_target"}
    global_events = [event for event in events if event["event"] == "global_choice"]
    require(len(global_events) == len(selects) >= 2, "fixture requires at least two actual global choices")
    progress_by_option = {}
    for event in authorizations:
        option_id = event["option_id"]
        if option_id is None:
            require(event["phase"] == "return", "unbound outbound action")
            continue
        call = selects[event["selection_call_id"]]
        option = call["outputs"]["selected"]
        require(option is not None and option["option_id"] == option_id, "executed option provenance")
        offset = progress_by_option.get(option_id, 0)
        require(offset < len(option["outbound_actions"]) and option["outbound_actions"][offset] == event["assessment"]["action"],
                "actual action does not consume selected outbound plan")
        progress_by_option[option_id] = offset + 1
    for event, call in zip(global_events, selects.values()):
        require(event["option"] == call["outputs"]["selected"] and event["frame_id"] == call["frame_id"], "runtime choice differs from module choice")
    second_step = histories[global_events[1]["frame_id"]]["action_id"]
    second_call = list(selects)[1]
    require(any(event["next_action_id"] > second_step and event["selection_call_id"] == second_call
                for event in authorizations), "no real motion from the second global choice")
    closes = [event for event in events if event["event"] == "close_after_final_observation"]
    require(len(closes) == 1, "missing or duplicate episode closure")
    close = closes[0]
    require(close["final_frame_id"] == raws[-1]["frame_id"] and close["final_action_id"] == len(raws) - 1,
            "closure lost actual terminal frame")
    require(not observations or events.index(close) > events.index(observations[-1]), "closure precedes terminal observation")
    finishes = [call for call in calls if call["method"] == "finish"]
    require(len(finishes) == 1 and finishes[0]["frame_id"] == raws[-1]["frame_id"], "terminal feedback binding")
    terminal = finishes[0]["outputs"]
    require(terminal["done"] and terminal["paid_actions"] == len(raws) - 1 and terminal["remaining_budget"] == budget - len(raws) + 1,
            "terminal feedback budget")
    if terminal["event_type"] == "logical_termination":
        require(terminal["action_cost"] == terminal["reward"] == 0 and terminal["actual_camera_pose_count"] == len(raws),
                "logical closure minted free observation/reward")
    reset_events = [event for event in events if event["event"] == "reset_after_close"]
    require(len(reset_events) == 1 and events.index(reset_events[0]) > events.index(close), "missing reset or reset before closure")
    reset = reset_events[0]
    require(reset["scene_state_is_none"] and reset["component_state_is_none"] and reset["completed_episode_count"] == 1,
            "reset did not clear scene/module state or retain completed episode")
    require(reset["final_frame_id"] == raws[-1]["frame_id"] and reset["final_action_id"] == len(raws) - 1,
            "reset lost terminal identity")
    summary = read(run / "summary.json")
    episode = summary.get("episode", summary.get("episode_summary", summary))
    require(episode["mapper_frames"] == episode["unique_packet_count"] == len(raws), "summary fused/unique frames")
    require(episode["closed"] and episode["externally_scripted_prefix_actions"] == 0, "fresh CPU episode scope")
    require(episode["modules"]["feedback"]["paid_actions"] == len(raws) - 1, "summary cost")
    require(episode["modules"]["feedback"]["consumed_action_ids"] == list(range(1, len(raws))), "summary action IDs")
    actual_return = tuple((*world.position, world.heading)) == anchor
    final_footprint_safe = bool(previous_safe[tuple(world.position)])
    failed_expected = not actual_return or world.collisions > 0 or not final_footprint_safe
    require(close["returned_to_anchor"] == actual_return and close["failed"] == failed_expected,
            "geometric return and independent failure status")
    require(close["current_footprint_known_safe"] == final_footprint_safe and terminal["failed"] == failed_expected,
            "terminal feedback lost final footprint/collision failure")
    require(actual_return and world.collisions == 0 and final_footprint_safe, "integration fixture did not safely return")
    report = dict(schema_version=SCHEMA, status="passed_full_cpu_integration_replay", **totals,
                  collisions=int(world.collisions), returned_to_anchor=actual_return,
                  final_footprint_known_safe=final_footprint_safe, failed=failed_expected,
                  actual_pose=list((*world.position, world.heading)), remaining_budget=budget - len(raws) + 1,
                  real_motion_after_second_global_choice=True,
                  terminal_before_close_verified=True, reset_after_terminal_verified=bool(reset_events),
                  raw_artifact_sha256=raw_hashes, wall_time_s=time.perf_counter() - started,
                  scope=dict(development_fixture=True, holdout=False, independent_physical_raw_replay=True,
                             independent_grid_safety_and_return_bfs=True, independent_sensor_feedback_arithmetic=True,
                             archived_mapper_and_tsdf_numerics_reused=True, tsdf_mesh_exact=True,
                             independent_public_candidate_path_accounting=True, score_optimality_is_logged_pool_only=True,
                             production_runtime_components_guard_planner_scorer_called=False,
                             full_trained_ans_verified=False, open_vocabulary_semantics_verified=False,
                             calibrated_uncertainty_verified=False, efficacy_or_mainstream_superiority_verified=False,
                             full_gt_reconstruction_metrics_evaluated=False))
    sys.setprofile(None)
    write(output / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--frozen", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output.resolve()
    if args.worker:
        try:
            worker(run, output, args.frozen)
        except Exception as error:
            write(output / "verification.json", dict(schema_version=SCHEMA, status="failed", error=str(error),
                                                     traceback=traceback.format_exc()))
            raise
        return
    require(not output.exists(), "refusing to overwrite an independent review")
    output.mkdir(parents=True)
    try:
        manifest, freeze = read(run / "manifest.json"), read(run / "freeze.json")
        archive = run / "sources.zip"
        sources = freeze["source_sha256"]
        require(isinstance(sources, dict) and sources, "missing frozen source inventory")
        require(sha(archive) == freeze["archive_sha256"], "source archive hash differs")
        if "source_sha256" in manifest:
            require(manifest["source_sha256"] == sources, "manifest/freeze source inventory differs")
        if "archive_sha256" in manifest:
            require(manifest["archive_sha256"] == sha(archive), "manifest/archive hash differs")
        self_name = "scripts/replay_cpu_four_module_v10.py"
        require(sources.get(self_name) == sha(Path(__file__)), "independent verifier was not frozen before execution")
        artifacts = read(run / "artifact_hashes.json")
        require(artifacts and all(sha(relative(run, name)) == value for name, value in artifacts.items()),
                "immutable run artifact hash mismatch")
        actual_files = {str(path.relative_to(run)) for path in run.rglob("*") if path.is_file() and path.name != "artifact_hashes.json"}
        require(set(artifacts) == actual_files, "immutable run artifact inventory mismatch")
        bindings = {name: sha(run / name) for name in ("manifest.json", "freeze.json", "sources.zip", "module_calls.json",
                                                       "runtime_events.json", "summary.json", "final_map.npz", "final_mesh.npz",
                                                       "raw_manifest.json", "artifact_hashes.json")}
        with tempfile.TemporaryDirectory(prefix="nso-v10-replay-") as temp:
            frozen = Path(temp) / "source"; frozen.mkdir()
            with zipfile.ZipFile(archive) as zipped:
                require(set(zipped.namelist()) == set(sources), "source archive inventory mismatch")
                for info in zipped.infolist():
                    require(not info.is_dir() and ((info.external_attr >> 16) & 0o170000) != 0o120000, "archive contains directory/symlink")
                    target = relative(frozen, info.filename)
                    data = zipped.read(info.filename)
                    require(hashlib.sha256(data).hexdigest() == sources[info.filename], "archived source hash mismatch")
                    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
            isolated = Path(temp) / "verifier.py"
            shutil.copyfile(Path(__file__), isolated)
            env = dict(os.environ); env["PYTHONPATH"] = str(frozen); env["PYTHONNOUSERSITE"] = "1"
            result = subprocess.run([sys.executable, str(isolated), str(run), "--output", str(output),
                                     "--worker", "--frozen", str(frozen)], cwd=frozen, env=env,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (output / "worker_stdout.txt").write_text(result.stdout)
            (output / "worker_stderr.txt").write_text(result.stderr)
            require(result.returncode == 0, "isolated replay failed; see verification.json and worker_stderr.txt")
        require(all(sha(run / name) == value for name, value in bindings.items()), "run changed during independent replay")
        write(output / "binding.json", dict(input_sha256=bindings, verifier_sha256=sha(Path(__file__)),
                                           source_sha256=sources, archive_sha256=sha(archive)))
        shutil.copyfile(Path(__file__), output / "verifier_source.py")
        report = read(output / "verification.json")
        (output / "REPORT.md").write_text(
            "# V10 CPU 四模块闭环独立回放\n\n"
            f"状态：{report['status']}。实际 {report['paid_actions_checked']} 个付费动作、"
            f"{report['raw_frames_checked']} 个 RGB-D 帧及对应雷达帧逐字段完全相同。\n\n"
            f"独立检查 {report['candidate_paths_checked']} 条候选路径、{report['global_choices_checked']} 次全局选择、"
            f"{report['assessments_checked']} 次执行门禁及 {report['feedback_events_checked']} 次传感器反馈。"
            "第二次选择后有真实运动，实际零碰撞并返回固定锚点及朝向。\n\n"
            "这是开发场景的接线、复现和预算审计；不证明原训练 ANS、开放词汇能力、校准 UQ、"
            "整轨迹语义增益或主流基线优势。安全结论限定于实际记录轨迹和当时观测地图。\n")
        print(json.dumps({key: report[key] for key in ("status", "paid_actions_checked", "raw_frames_checked",
                                                       "global_choices_checked", "candidate_paths_checked")}, ensure_ascii=False))
    except Exception as error:
        if not (output / "verification.json").exists():
            write(output / "verification.json", dict(schema_version=SCHEMA, status="failed", error=str(error)))
        raise


if __name__ == "__main__":
    main()
