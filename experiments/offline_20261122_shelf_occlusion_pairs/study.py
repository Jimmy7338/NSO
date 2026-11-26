#!/usr/bin/env python3
"""Freeze inputs and execute the registered shelf-occlusion trajectory matrix."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
ANALYSES = ROOT.parent
OLD_LOCAL = ANALYSES / "region_structure_floor_20261122/local"
INCREMENT = ANALYSES / "region_global_revision_20261122/next/local_increment"
SCENES = ROOT / "scenes"
ACTOR = ROOT / "runtime/run.py"
RULES = ROOT / "PROTOCOL.json"
ARMS = ("G", "S", "D", "D+")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, record):
    path = Path(path)
    assert not path.exists(), path
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def hashes(paths):
    return {str(p.resolve()): sha(p) for p in sorted(set(paths))}


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Sealed input drift: {path}")


def merge(*parts):
    result = {}
    for sources in parts:
        for path, digest in sources.items():
            if result.setdefault(path, digest) != digest:
                raise ValueError(f"Disagreeing source binding for {path}")
    return result


def freeze_path(phase):
    return ROOT / f"{phase.upper()}_FREEZE.json"


def seal(phase):
    spec = read(RULES)
    setup = read(ROOT / "source_receipt.json")
    assert setup["base_commit"] == spec["base_commit"]
    assert not (ROOT / "execution" / phase).exists()
    scene_seal = SCENES / f"{phase.upper()}_SCENE_FREEZE.json"
    scenes = read(scene_seal)
    parents = spec["schedule"]["development_parents" if phase == "development" else "test_parents"]
    conditions = spec["schedule"]["conditions"]
    placements = [f"{parent}_{condition}" for parent in parents for condition in conditions]
    assert scenes["placements"] == placements
    assert scenes["native_budget"] == spec["native_budget"] and scenes["policy_calls"] == 0
    development_tolerance_path = ROOT/"measurement/development_repeatability.json"
    tolerance = 1e-9
    if phase == "formal":
        prior = read(development_tolerance_path)
        assert prior["development_freeze_sha256"] == sha(freeze_path("development"))
        assert prior["repeat_snapshots"] >= 2
        tolerance = prior["tie_tolerance_m2"]
        assert tolerance == max(1e-9, prior["max_abs_delta_m2"])
    permitted_prior = {}
    for ledger in (OLD_LOCAL/"FROZEN_INPUTS.json", INCREMENT/"INPUT_FREEZE.json"):
        subset = {name: digest for name, digest in read(ledger)["actor_files_sha256"].items()
                  if not any(Path(name).is_relative_to(prior/"scenes") for prior in (OLD_LOCAL, INCREMENT))}
        permitted_prior = merge(permitted_prior, subset)
    actor_files = merge(
        permitted_prior, scenes["actor_files_sha256"], hashes(
            [RULES, ACTOR, ROOT/"source_receipt.json", scene_seal, Path(__file__),
             *ACTOR.parent.glob("*.py"), OLD_LOCAL/"calibration.json"]),
    )
    meter_files = []
    if phase == "formal":
        meter_files = list((ROOT/"measurement").glob("*.py"))
        assert ROOT/"measurement/evaluate.py" in meter_files
        meter_files.append(development_tolerance_path)
    meter = merge(scenes["evaluator_files_sha256"], hashes(meter_files))
    caches = scenes["simulator_only_cache_sha256"]
    assert not any("evaluator_references" in p or "evaluator_truth" in p for p in actor_files | caches)
    common_initial = {str(SCENES/"prepared"/name/"initial_state.json") for name in placements}
    assert all(Path(p).suffix in (".py", ".so") or p in common_initial
               for p in actor_files.keys() & meter.keys())
    verify(actor_files)
    verify(caches)
    verify(meter)
    record = {key: spec[key] for key in (
        "base_commit", "native_budget", "capture_paid", "max_trigger_paid", "weights",
        "local_route", "direct_scale", "hidden_scale", "arm_order",
    )}
    record |= {
        "phase": phase, "protocol_sha256": sha(RULES), "scene_seal_sha256": sha(scene_seal),
        "training_scope": "development" if phase == "development" else "unseen_test",
        "placements": placements, "parents": parents, "scene_root": str(SCENES),
        "stage_1": placements[:len(conditions)*2], "stage_2": placements[len(conditions)*2:],
        "actor_files_sha256": actor_files, "simulator_only_cache_sha256": caches,
        "evaluator_files_sha256": meter, "tie_tolerance_m2": tolerance, "frozen_unix_s": time.time(),
    }
    path = freeze_path(phase)
    write(path, record)
    print(f"{phase}: {len(placements)} placements, {sha(path)}", flush=True)


def bind_run(result, freeze_digest):
    manifest_path = Path(result["run_directory"]) / "manifest.json"
    if sha(manifest_path) != result["manifest_sha256"]:
        raise ValueError(f"Run manifest drift: {manifest_path}")
    manifest = read(manifest_path)
    assert result["phase_freeze_sha256"] == freeze_digest
    assert manifest["study_freeze_sha256"] == freeze_digest
    assert manifest["experimental_local_arm"] == result["execution_arm"] and manifest["placement"] == result["placement"]
    verify({str(manifest_path.parent / name): digest
            for name, digest in manifest["files_sha256"].items()})
    bank = manifest_path.parent / "fork_bank.json"
    if result["fork_bank_sha256"] != sha(bank):
        raise ValueError(f"Fork bank drift: {bank}")
    return read(bank)


def execute(case_root, placement, arm, seal_path, bank_path=None):
    arm_root = case_root / arm.replace("+", "plus")
    arm_root.mkdir(exist_ok=True, parents=True)
    freeze_digest = sha(seal_path)
    input_bank_sha = sha(bank_path) if bank_path else None
    previous = sorted(arm_root.glob("attempt_*"))
    failures = []
    for attempt in previous:
        exit_path = attempt / "exit.json"
        if not exit_path.is_file():
            raise RuntimeError(f"Attempt attached to no exit receipt: inspect {attempt}")
        record = read(exit_path)
        assert record["phase_freeze_sha256"] == freeze_digest and record["source_bank_sha256"] == input_bank_sha
        if record["status"] == "verified_native_execution":
            bind_run(record, freeze_digest)
            return record
        failures.append(record)
    for index in range(len(previous), 2):
        attempt = arm_root / f"attempt_{index+1:03d}"
        attempt.mkdir()
        out = attempt / "run"
        command = [sys.executable, "-X", "faulthandler", str(ACTOR), placement, arm,
                   "--out", str(out), "--freeze", str(seal_path)]
        if bank_path is not None:
            command.extend(["--fork-bank", str(bank_path)])
        elif arm != "G":
            command.append("--allow-unbound-fallback")
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1",
                           PYTHONDONTWRITEBYTECODE="1")
        log = attempt / "worker.log"
        with log.open("x") as stream:
            worker = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            write(attempt / "launch.json", {"placement": placement, "arm": arm, "command": command,
                  "pid": worker.pid, "start_unix_s": time.time(), "phase_freeze_sha256": freeze_digest,
                  "source_bank_sha256": input_bank_sha})
            code = worker.wait()
        manifest, bank = out / "manifest.json", out / "fork_bank.json"
        verified = code == 0 and manifest.is_file() and bank.is_file()
        outcome = dict(placement=placement, status="verified_native_execution" if verified else "technical_failure",
                       execution_arm=arm, run_directory=str(out), manifest_sha256=sha(manifest) if verified else None,
                       fork_bank_sha256=sha(bank) if bank.is_file() else None, exit_code=code,
                       source_bank_sha256=input_bank_sha, observed_exit_unix_s=time.time(),
                       phase_freeze_sha256=freeze_digest, log_sha256=sha(log), attempt=index+1,
                       reason=None if verified else "native_process_failed" if code else "missing_manifest_or_fork_receipt")
        if verified:
            bind_run(outcome, freeze_digest)
        write(attempt / "exit.json", outcome)
        print(f"{placement} {arm} attempt {index+1}: {outcome['status']} ({code})", flush=True)
        if verified:
            return outcome
        failures.append(outcome)
    return failures[-1]


def plan_branches(anchor):
    """Only use digest equivalence within one completed, verified cold placement."""
    if anchor["status"] != "verified_native_execution":
        return {arm: arm for arm in ARMS}, {}, None
    bank = bind_run(anchor, anchor["phase_freeze_sha256"])
    path = Path(anchor["run_directory"]) / "fork_bank.json"
    if bank["status"] == "not_triggered":
        return {arm: "G" for arm in ARMS}, {}, path
    assert bank["status"] == "triggered"
    choices = bank["arms"]
    assert set(choices) == set(ARMS)
    identities = {arm: choices[arm]["route_identity_sha256"] for arm in ARMS}
    assert all(len(digest) == 64 for digest in identities.values())
    group, assignment = {}, {}
    for arm in ARMS:
        key = (bank["shared_identity_sha256"], bank["continuation_id"], identities[arm])
        assignment[arm] = group.setdefault(key, arm)
    return assignment, identities, path


def finish_placement(path, case, outcomes, assignment, identities, seal_path):
    freeze_digest = sha(seal_path)
    arms = {}
    for key in ARMS:
        chosen = assignment[key]
        run = outcomes[chosen]
        if run["status"] == "verified_native_execution":
            bank = bind_run(run, freeze_digest)
            if chosen != "G" and identities:
                assert bank["arms"][chosen]["route_identity_sha256"] == identities[chosen]
        arms[key] = {"status": run["status"], "execution_arm": chosen,
                     "run_directory": run["run_directory"], "manifest_sha256": run["manifest_sha256"],
                     "fork_bank_sha256": run["fork_bank_sha256"], "shared_execution": key != chosen,
                     "route_identity_sha256": identities.get(key), "technical_exit_code": run["exit_code"],
                     "technical_reason": run["reason"]}
    receipt = {"placement": case, "phase_freeze_sha256": freeze_digest, "arms": arms,
               "unique_attempted_assignments": len(outcomes),
               "unique_completed_executions": sum(row["status"] == "verified_native_execution" for row in outcomes.values())}
    write(path, receipt)
    return receipt


@contextmanager
def lock(directory):
    with (directory / "supervisor.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(stage, workers):
    assert 1 <= workers <= 6
    phase = "development" if stage == "development" else "formal"
    seal_path = freeze_path(phase)
    frozen = read(seal_path)
    verify(frozen["actor_files_sha256"])
    verify(frozen["simulator_only_cache_sha256"])
    verify(frozen["evaluator_files_sha256"])
    cases = frozen["stage_2" if stage == "formal_rest" else "stage_1"]
    assert cases
    if stage == "formal_rest":
        interim = read(ROOT/"results/formal_first/release.json")
        assert interim["phase_freeze_sha256"] == sha(seal_path)
        assert interim["placements"] == frozen["stage_1"]
        assert sha(ROOT/"results/formal_first/summary.json") == interim["summary_sha256"]
    directory = ROOT / "execution" / phase
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{stage}_receipt.json"
    with lock(directory), ThreadPoolExecutor(max_workers=workers) as pool:
        if final.exists():
            print(f"Existing registered stage receipt: {final}", flush=True)
            return
        finished, plan, active, observed = {}, {}, {}, {}
        for case in cases:
            current = directory / case
            current.mkdir(exist_ok=True)
            receipt = current / "placement_receipt.json"
            if receipt.exists():
                finished[case] = read(receipt)
            else:
                future = pool.submit(execute, current, case, "G", seal_path)
                active[future] = (case, "G")
                observed[case] = {}
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                case, arm = active.pop(future)
                outcomes = observed[case]
                outcomes[arm] = future.result()
                path = directory / case
                if arm == "G":
                    assignment, identities, bank_path = plan_branches(outcomes["G"])
                    plan[case] = assignment, identities
                    for next_arm in ARMS[1:]:
                        if assignment[next_arm] == next_arm:
                            fresh = pool.submit(execute, path, case, next_arm, seal_path, bank_path)
                            active[fresh] = (case, next_arm)
                assignment, identities = plan[case]
                if set(outcomes) == set(assignment.values()):
                    finished[case] = finish_placement(path/"placement_receipt.json", case, outcomes, assignment,
                                                       identities, seal_path)
                    print(f"{case}: {finished[case]['unique_completed_executions']} unique executed trajectories", flush=True)
        assert set(finished) == set(cases)
        rows = []
        for case in cases:
            file = directory / case / "placement_receipt.json"
            rows.append(dict(path=str(file), sha256=sha(file), placement=case))
        write(final, {"stage": stage, "phase_freeze_sha256": sha(seal_path), "placements": cases,
                      "unique_completed_executions": sum(x["unique_completed_executions"] for x in finished.values()),
                      "registered_policy_outcomes": len(cases)*len(ARMS), "placement_receipts": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("phase", choices=("development", "formal"))
    run_parser = commands.add_parser("run")
    run_parser.add_argument("stage", choices=("development", "formal_first", "formal_rest"))
    run_parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    seal(args.phase) if args.command == "seal" else run(args.stage, args.workers)
