#!/usr/bin/env python3
"""Seal, execute and appraise the one-factor cold-start comparison."""

import argparse
from pathlib import Path
import sys
import time

from runner import ARM_MODES, PREVIOUS, ROOT


sys.path.append(str(PREVIOUS))
from regression import read, sha, write
import study as native_study


PROTOCOL = ROOT / "PROTOCOL.json"
SOURCE_FREEZE = PREVIOUS / "FORMAL_FREEZE.json"


def seal(phase):
    assert not (ROOT / "execution" / phase).exists(), "This phase already ran."
    protocol = read(PROTOCOL)
    previous = read(SOURCE_FREEZE)
    assert sha(SOURCE_FREEZE) == protocol["source_formal_freeze_sha256"]
    scenes = PREVIOUS / "scenes" if phase == "development" else ROOT / "scenes"
    source_scene = PREVIOUS / "scenes"
    spec = protocol[phase]
    layouts = spec["layouts"]
    jobs = []
    for layout in layouts:
        for arm in spec["arms"]:
            prior, regional = ARM_MODES[arm]
            jobs.append(dict(layout=layout, arm=arm, prior=prior, regional_enabled=regional, literal=True))
    common = ("budget", "capture_paid", "local_auxiliary_weights", "local_route_invariants")
    for key in common:
        assert protocol[key] == previous[key], key
    actor = {name: digest for name, digest in previous["actor_files"].items()
             if not Path(name).is_relative_to(source_scene)}
    meter = {name: digest for name, digest in previous["evaluator_files"].items()
             if not Path(name).is_relative_to(source_scene)}
    initial = [scenes / "scene.py"]
    for layout in layouts:
        directory = scenes / "prepared" / layout
        initial.extend(directory / name for name in ("initial_state.json", "initial_frame.npz", "initial_scan.npz"))
        start = read(directory / "initial_state.json")
        assert start["policy_select_calls"] == start["paid_actions"] == 0
        assert start["stop_budget"] == protocol["budget"]
    assert all(path.is_file() for path in initial)
    actor_initial = native_study.hashes(initial)
    phase_protocol = ROOT / f"{phase.upper()}_PROTOCOL.json"
    phase_settings = {key: protocol[key] for key in common} | spec | {
        "protocol_source_sha256": sha(PROTOCOL), "experiment_role": phase,
    }
    if phase_protocol.exists():
        assert read(phase_protocol) == phase_settings
    else:
        write(phase_protocol, phase_settings)
    extra_actor = [ROOT / "runner.py", Path(__file__), PROTOCOL, SOURCE_FREEZE, phase_protocol,
                   *sorted((scenes / "precomputed_assets").glob("*"))]
    references = [path for layout in layouts
                  for path in (scenes / "evaluator_references" / layout).iterdir() if path.is_file()]
    assert all((scenes / "evaluator_references" / l / "RECEIPT.json").is_file() for l in layouts)
    if phase == "formal":
        for filename, files in (("actor_freeze_holdout.json", extra_actor),
                                ("evaluator_freeze_holdout.json", references)):
            seal_path = scenes / filename
            ledger = read(seal_path)
            assert ledger["actors_run"] == 0 and ledger["budget"] == protocol["budget"]
            native_study.verify(ledger["files_sha256"])
            files.append(seal_path)
        references.append(scenes / "formal_geometry_audit.json")
    actor |= actor_initial | native_study.hashes(extra_actor)
    meter |= native_study.hashes([*references, ROOT / "evaluate_revision.py"])
    native_study.verify(actor)
    native_study.verify(meter)
    assert not any("evaluator_references" in path or "evaluator_truth" in path for path in actor)
    result = {"schema": protocol["schema"], "phase": phase, "frozen_unix_s": time.time(),
              "protocol_relative": phase_protocol.name, "protocol_sha256": sha(phase_protocol),
              "scene_root": str(scenes), "actor_files": actor, "evaluator_files": meter,
              "actor_initial_files": actor_initial, "jobs": jobs} | {key: protocol[key] for key in common}
    path = ROOT / f"{phase.upper()}_FREEZE.json"
    write(path, result)
    print(f"Sealed {len(jobs)} runs {sha(path)}", flush=True)


def run(phase, workers, resume):
    assert 1 <= workers <= 6
    native_study.ROOT = ROOT
    native_study.EXECUTION = ROOT / "execution"
    native_study.dispatch(phase, workers, resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "run"))
    parser.add_argument("phase", choices=("development", "formal"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "seal":
        seal(args.phase)
    else:
        run(args.phase, args.workers, args.resume)


if __name__ == "__main__":
    main()
