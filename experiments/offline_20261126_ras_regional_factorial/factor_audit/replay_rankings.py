"""Replay frozen D0G0/D1G1 regional score and selection on saved observed states.

Renders no frames and does not instantiate the native actor. Only previously
acquired belief, poses and option/cost/measurement logs enter regional replay.
"""

from collections import defaultdict
from hashlib import file_digest as hash_stream, sha256
import json
from math import isclose
from pathlib import Path
import sys

import numpy as np


ANALYSES = Path(__file__).resolve().parents[2]
BASE = ANALYSES / "region_independent_floors_20261123"
REGIONAL = ANALYSES / "region_structure_floor_20261122"
ATTRIBUTION = ANALYSES / "region_global_revision_20261122/next/global_attribution"
NATIVE = ANALYSES / "reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source"
sys.path[:0] = [str(REGIONAL), str(ATTRIBUTION), str(NATIVE)]
from policy.observed_regions import RegionalPlanner
from score import rerank


def read(path):
    return json.loads(path.read_text())


def file_digest(path):
    with path.open("rb") as stream:
        return hash_stream(stream, "sha256").hexdigest()


def normalize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize(entry) for key, entry in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(entry) for entry in value]
    return value


def content_digest(row):
    content = json.dumps(normalize(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(content.encode()).hexdigest()


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return isclose(a, b, abs_tol=1e-10, rel_tol=1e-12)


def check_candidates(global_row, regional, proposals, resolution, context):
    """Check native candidate arrays, recorded scores, and original fallback."""
    output = global_row["outputs"]
    routes = output["candidates"]
    entries = output["score_audit"]
    ensure(len(routes) == len(entries) and len({r["candidate_id"] for r in routes}) == len(routes),
           f"{context} candidate alignment")
    ensure([r["candidate_id"] for r in routes] == [a["candidate_id"] for a in entries],
           f"{context} score/candidate order")
    ensure(global_row["input_sha256"] == content_digest(global_row["inputs"])
           and global_row["output_sha256"] == content_digest(output),
           f"{context} native call bytes")
    ensure(global_row["inputs"]["remaining_budget"] + regional["paid_before"] == 192,
           f"{context} budget clock mismatch")
    ensure(output["planner_revision"] == "v10_3_1"
           and output["policy_variant"] == "exploration_joint_v1" and output["mode"] == "G",
           f"{context} version or base geometry mode")
    for entry in entries:
        coverage = resolution**2 * (
            entry["predicted_radar_cells"] * entry["gain_posterior"]["radar"]
            + entry["predicted_camera_cells"] * entry["gain_posterior"]["camera"])
        ensure(close(coverage, entry["coverage_only_proxy"])
               and close(coverage + entry["quality_total_proxy"], entry["common_total_proxy"]),
               f"{context} outbound sensor/quality components do not add for {entry['candidate_id']}")
    proposal_by_pose = {tuple(item["pose"]): item for item in proposals}
    admission = {item["candidate_id"]: item for item in regional["certification"] if item["status"] == "admitted"}
    ensure(len(admission) == regional["new_native_certified_count"], f"{context} added candidate count")
    native = [route for route in routes if "regional_proposal" not in route]
    added = [route for route in routes if "regional_proposal" in route]
    ensure(len(added) == len(admission), f"{context} archived proposal set incomplete")
    for route in added:
        proposal = proposal_by_pose[tuple(route["pose"])]
        certified = admission[route["candidate_id"]]
        ensure(certified["proposal"] == route["regional_proposal"] == proposal
               and certified["outbound_actions"] == route["outbound_cost"]
               and certified["return_actions"] == route["return_cost"], f"{context} certification mismatch")
    # Original selector evaluates only native candidates before regional insertion.
    shadow_id = None
    if native:
        scores = output["scores"]["G"]
        best = min(range(len(native)), key=lambda i: (-scores[i], native[i]["cost"], native[i]["candidate_id"]))
        if scores[best] > 0:
            shadow_id = native[best]["candidate_id"]
    shadow = next((route for route in native if route["candidate_id"] == shadow_id), None)
    ensure(regional["native_shadow_pose"] == (shadow["pose"] if shadow else None),
           f"{context} native fallback target")
    return output, native, added, shadow


def replay(layout, arm, run, manifest, floor):
    regional = read(run / "regional_choices.json")
    global_choices = read(run / "global_choices.json")
    guards = read(run / "regional_guards.json")
    steps = read(run / "steps.json")
    with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as trace:
        ensure(trace["paid"].tolist() == list(range(len(steps)+1)), f"{layout}/{arm} observed clock")
        beliefs = trace["belief"].copy()
    ensure(len(steps) == manifest["paid_actions"] == 192 and len(regional) == len(global_choices),
           f"{layout}/{arm} complete native episode")
    planner = RegionalPlanner()
    config = floor["actual_config"]
    resolution = config["resolution_m"]
    events = defaultdict(list)
    for row, native in zip(regional, global_choices, strict=True):
        ensure(row["selection_call_id"] == native["call_id"] and row["paid_before"] == native["action_id"],
               f"{layout}/{arm} async native call log mismatch")
        events[row["paid_before"]].append((row, native))
    rejections = defaultdict(list)
    for guard in guards:
        rejections[guard["paid"]].append(guard)

    issued = {}
    dispatched = set()
    count = defaultdict(int)
    for paid, grid in enumerate(beliefs):
        pose = floor["anchor"] if paid == 0 else steps[paid-1]["position"] + [steps[paid-1]["heading"]]
        planner.observe(grid, pose, paid)
        current_rejections = rejections[paid].copy()
        for regional, native_call in events[paid]:
            context = f"{layout}/{arm} @ {paid}, call {regional['selection_call_id']}"
            while current_rejections and current_rejections[0]["selection_call_id"] < regional["selection_call_id"]:
                guard = current_rejections.pop(0)
                planner.note_attempt(issued[guard["option_id"]], guard["reason"], paid)
                count["known_guard_rejections_replayed"] += 1
            ensure(pose == regional["start_pose"], f"{context} mismatched actual pose")
            proposals = planner.proposals(grid, pose, paid, resolution, max_depth_m=config["max_depth_m"],
                                          fov_deg=config["fov_deg"], robot_radius_m=config["robot_radius_m"])
            recorded_proposals = [item["proposal"] for item in regional["certification"]]
            ensure(normalize(proposals) == recorded_proposals
                   and len(proposals) == regional["raw_proposal_count"], f"{context} regenerated proposals")
            output, native, added, shadow = check_candidates(native_call, regional, proposals, resolution, context)
            base_chosen, audit = planner.rank(grid, pose, paid, resolution, native, added)
            if arm == "combined_S" and audit["ranking"]:
                chosen, ranking = rerank(planner, audit, [*native, *added], output["score_audit"],
                                         resolution, True, True)
                detail = {"score_repair": {"target": True, "outbound": True},
                          "source_candidate_count": len(audit["ranking"]),
                          "v1_chosen": audit["chosen"], "actual_paid": paid, "ranking": ranking}
                audit = audit | {"ranking": ranking, "chosen": ranking[0], "attribution_path_repair": detail}
            else:
                chosen = base_chosen
            ensure(normalize(audit) == regional["ranking"], f"{context} complete regional audit differs")
            decision = chosen if chosen is not None else shadow
            selected = output["selected"]
            if decision is None:
                ensure(selected is None and regional["actual_target_pose"] is None, f"{context} null response")
            else:
                ensure((decision["candidate_id"], decision["pose"]) == (selected["candidate_id"], selected["pose"])
                       and selected["pose"] == regional["actual_target_pose"], f"{context} actual selection")
                issued[selected["option_id"]] = selected
            count["calls"] += 1
            count["regions_ranked"] += bool(audit["ranking"])
            count["candidate_scores"] += len(audit["ranking"])
            count["native_outbound_component_rows"] += len(output["score_audit"])
            count["native_fallbacks"] += chosen is None
        for guard in current_rejections:
            planner.note_attempt(issued[guard["option_id"]], guard["reason"], paid)
            count["known_guard_rejections_replayed"] += 1
        if paid < len(steps):
            option_id = steps[paid]["active_option_id"]
            if option_id is not None and option_id not in dispatched:
                planner.note_attempt(issued[option_id], "dispatched", paid)
                dispatched.add(option_id)
                count["dispatched_options_replayed"] += 1
    ensure(count["known_guard_rejections_replayed"] == len(guards), f"{layout}/{arm} guard replay missing")
    return {"layout": layout, "original_arm": arm, "arm": "D0G0" if arm == "region_S" else "D1G1",
            "trajectory_manifest": str(run / "manifest.json"),
            "trajectory_manifest_sha256": file_digest(run / "manifest.json"), **dict(count)}


def main():
    protocol = read(BASE / "PROTOCOL.json")
    frozen = read(BASE / "FORMAL_FREEZE.json")
    scene_files = read(BASE / "scenes/actor_freeze_holdout.json")
    receipt = read(BASE / "execution/formal/phase_receipt.json")
    source = read(BASE / "SOURCE_RECEIPT.json")
    ensure(file_digest(BASE / "FORMAL_FREEZE.json") == receipt["freeze_sha256"]
           and source["base_commit"] == frozen["base_commit"] == protocol["base_commit"]
           and frozen["budget"] == scene_files["budget"] == 192, "frozen comparison identity mismatch")
    for row in source["pinned_files"]:
        ensure(file_digest(ANALYSES / row["local"]) == row["sha256"], f"Changed pinned source {row['local']}")
    floors = {row["layout"]: row for row in scene_files["layouts"]}
    complete = {(row["layout"], row["arm"]): row for row in receipt["episodes"]}
    cohorts = []
    for layout in protocol["layouts"]:
        for arm in ("region_S", "combined_S"):
            entry = complete[layout, arm]
            run = Path(entry["run_path"])
            manifest = read(run / "manifest.json")
            ensure(entry["exit_code"] == 0 and entry["result_sha256"] == file_digest(run / "manifest.json")
                   and manifest["initial_packet_sha256"] == floors[layout]["initial_packet_sha256"]
                   and manifest["initial_owner_sha256"] == floors[layout]["initial_owner_sha256"]
                   and manifest["initial_pose"] == floors[layout]["anchor"]
                   and manifest["runner_freeze_sha256"] == receipt["freeze_sha256"]
                   and manifest["study_prior"] == "S" and manifest["regional_enabled"],
                   f"{layout}/{arm} reused attempt does not match frozen action or sensor contract")
            for filename in ("regional_choices.json", "global_choices.json", "regional_guards.json",
                             "observed_belief_trace.npz", "steps.json"):
                ensure(manifest["files_sha256"][filename] == file_digest(run / filename),
                       f"{layout}/{arm} altered native evidence {filename}")
            evidence = replay(layout, arm, run, manifest, floors[layout])
            cohorts.append(evidence)
    keys = ("calls", "regions_ranked", "candidate_scores", "native_outbound_component_rows",
            "native_fallbacks", "known_guard_rejections_replayed", "dispatched_options_replayed")
    report = {"status": "verified", "archive_reference_commit": "74ecf883503d5a117d1e566417aff0213f76472a",
              "published_actor_source_baseline": frozen["base_commit"], "old_floor_freeze_sha256": receipt["freeze_sha256"],
              "scope": "All original two-arm regional proposals, state, score ordering and actual target selections "
                       "replayed from acquired belief and native candidate logs. Native route coverage components "
                       "were independently re-added, without rebuilding native route prediction or running a new actor.",
              "totals": {k: sum(row.get(k, 0) for row in cohorts) for k in keys}, "runs": cohorts}
    print(json.dumps(report, indent=2) + "\n", end="")


if __name__ == "__main__":
    main()
