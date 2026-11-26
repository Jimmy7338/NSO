#!/usr/bin/env python3
"""Run score attribution interventions and the independent follow-up."""

import argparse
import csv
from pathlib import Path
import time

from runner import PREVIOUS, ROOT, read, sha, write
import study as native_study


CONFIG = ROOT / "PROTOCOL.json"
AUDIT = ROOT.parent / "access_audit"


def development(spec):
    observed = {r["state_uid"]: r for r in csv.DictReader((AUDIT/"dispatches.csv").open(newline=""))}
    histories = set()
    jobs = []
    for layout, uid in spec["source_states"].items():
        expected = observed[uid]
        assert expected["layout"] == layout and expected["decision_reason"] == "selected_observed_region_view"
        source = (PREVIOUS / expected["steps_archive"]).parent
        histories.update(source / name for name in (
            "manifest.json", "steps.json", "global_choices.json", "local_choices.json", "regional_choices.json",
            "initial_raw_rgbd_scan_sha256.json",
        ))
        assert expected["both_pose"] in (expected["target_region_only_pose"], expected["outbound_measured_only_pose"])
        for arm, field, target, outbound in (
            ("attribution_once", "target_region_only_pose", True, False),
            ("outbound_once", "outbound_measured_only_pose", False, True),
        ):
            assert expected[field] != expected["old_pose"]
            branch = {
                "state_uid": uid, "source_run": str(source), "paid_before": int(expected["paid_before"]),
                "selection_call_id": int(expected["selection_call_id"]),
                "expected_pose": list(map(int, expected[field].split(","))),
            }
            jobs.append(dict(layout=layout, arm=arm, prior="S", regional_enabled=True, literal=True,
                             repair=dict(target=target, outbound=outbound), branch=branch))
    return jobs, histories


def formal(spec, scenes, budget):
    actor = read(scenes/"actor_freeze_holdout.json")
    evaluator = read(scenes/"evaluator_freeze_holdout.json")
    assert actor["layouts"] == evaluator["layouts"]
    assert actor["budget"] == evaluator["budget"] == budget and actor["actors_run"] == evaluator["actors_run"] == 0
    jobs = []
    layouts = [r["layout"] if isinstance(r, dict) else r for r in actor["layouts"]]
    for layout in layouts:
        for arm in spec["arms"]:
            treatment = {"repair": {"target": True, "outbound": True}} if arm == "combined_S" else {}
            jobs.append(dict(layout=layout, arm=arm, prior=None if arm == "F_strict" else "S",
                             regional_enabled=arm != "F_strict", literal=True, **treatment))
    return layouts, jobs, actor["files_sha256"] | actor.get("simulator_only_collision_cache_sha256", {}), evaluator["files_sha256"]


def seal(phase):
    assert not (ROOT / "execution" / phase).exists()
    original_freeze = PREVIOUS / "FORMAL_FREEZE.json"
    previous, config = read(original_freeze), read(CONFIG)
    assert sha(original_freeze) == config["source_original_freeze_sha256"]
    common = {key:config[key] for key in ("budget", "capture_paid", "local_auxiliary_weights", "local_route_invariants")}
    assert all(previous[k] == v for k,v in common.items())
    spec = config[phase]
    source_scene = PREVIOUS / "scenes"
    scenes = source_scene if phase == "development" else ROOT / "scenes"
    sources = {key:value for key,value in previous["actor_files"].items()
               if not Path(key).is_relative_to(source_scene)}
    evaluator = {key:value for key,value in previous["evaluator_files"].items()
                 if not Path(key).is_relative_to(source_scene)}
    extra = [ROOT/"runner.py", ROOT/"score.py", Path(__file__), CONFIG, original_freeze, scenes/"scene.py"]
    if phase == "development":
        jobs, archived = development(spec)
        layouts = spec["layouts"]
        extra.extend(archived)
        extra.append(AUDIT/"dispatches.csv")
        evaluator |= native_study.hashes(p for layout in layouts
                                         for p in (scenes/"evaluator_references"/layout).iterdir())
    else:
        layouts,jobs,actor_ledger,evaluator_ledger = formal(spec,scenes,common["budget"])
        sources |= actor_ledger
        evaluator |= evaluator_ledger
        extra.append(scenes/"actor_freeze_holdout.json")
    initial = [scenes/"scene.py"]
    for layout in layouts:
        path = scenes/"prepared"/layout
        initial.extend(path/name for name in ("initial_state.json", "initial_frame.npz", "initial_scan.npz"))
        cold = read(path/"initial_state.json")
        assert cold["stop_budget"] == config["budget"] and cold["paid_actions"] == cold["policy_select_calls"] == 0
    assert all(p.is_file() for p in initial)
    actor_initial = native_study.hashes(initial)
    required_protocol = ROOT/f"{phase.upper()}_PROTOCOL.json"
    write(required_protocol, common | spec | {"layouts": layouts, "joint_protocol_sha256": sha(CONFIG)})
    extra.extend([required_protocol, *sorted((scenes/"precomputed_assets").glob("*"))])
    sources |= native_study.hashes(extra) | actor_initial
    evaluator |= native_study.hashes([ROOT/"measure.py"])
    native_study.verify(sources)
    native_study.verify(evaluator)
    assert not any("evaluator_references" in path or "evaluator_truth" in path for path in sources)
    path = ROOT/f"{phase.upper()}_FREEZE.json"
    write(path, common | dict(schema=config["schema"], phase=phase, frozen_unix_s=time.time(),
         protocol_relative=required_protocol.name, protocol_sha256=sha(required_protocol),
         actor_files=sources, evaluator_files=evaluator, actor_initial_files=actor_initial,
         jobs=jobs, scene_root=str(scenes)))
    print(f"Sealed {len(jobs)} {phase} actors {sha(path)}",flush=True)


def run(phase, workers, resume):
    assert 1 <= workers <= 6
    native_study.ROOT, native_study.EXECUTION = ROOT, ROOT / "execution"
    native_study.dispatch(phase, workers, resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "run"))
    parser.add_argument("phase", choices=("development", "formal"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "seal":
        seal(args.phase)
    else:
        run(args.phase, args.workers, args.resume)
