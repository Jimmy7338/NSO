"""Run a separate active-observation gate and conditionally seal the twelve paired jobs."""

import argparse
import fcntl
import importlib.util
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "region_independent_floors_20261123"
PREVIOUS = ROOT.parent / "external_online_sweep_20261126"
V2 = ROOT.parent / "external_support_continuity_20261126"
V1 = ROOT.parent / "external_support_repair_20261126"
SOURCE = ROOT.parent / "region_global_revision_20261122/next/global_attribution"
ELIGIBLE_LAYOUT = "b26_hm3d_00770_s1"
GATE_ARM = "region_S"
MASK_FILE = V1 / "scenes/evaluator_references" / ELIGIBLE_LAYOUT / "evaluator_truth.npz"
BASELINE_CELLS = 522
MASK_CELLS = 1640
SCHEMA = "nso_external_active_sweep_observation_v4"
FREEZE = ROOT / "FORMAL_FREEZE.json"
GATE_FREEZE = ROOT / "GATE_FREEZE.json"
GATE_ROOT = ROOT / "execution/gate"
GATE_RECEIPT = GATE_ROOT / "phase_receipt.json"
GATE_VALIDATION = GATE_ROOT / "GATE_VALIDATION.json"


def supervisor(phase="formal"):
    spec = importlib.util.spec_from_file_location("active_observation_supervisor", BASE / "study.py")
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    shared.ROOT = ROOT
    shared.FREEZE = GATE_FREEZE if phase == "gate" else FREEZE
    shared.RUNS = ROOT / "execution" / phase
    return shared


def inherited(shared):
    path = PREVIOUS / "FORMAL_FREEZE.json"
    previous = shared.read(path)
    protocol_path = PREVIOUS / "FORMAL_PROTOCOL.json"
    protocol = shared.read(protocol_path)
    v2 = shared.read(V2 / "FORMAL_FREEZE.json")
    unchanged = ("scene_root", "simulator_only_files", "evaluator_files", "actor_initial_files", "jobs")
    if (previous["protocol_sha256"] != shared.sha(protocol_path) or protocol["budget"] != 192
            or len(previous["jobs"]) != 12 or len(protocol["starts"]) != 4
            or any(previous[key] != v2[key] for key in unchanged)):
        raise ValueError("Active observation requires the same registered V2 scenes, evaluator and 12 jobs")
    for key in ("actor_files", "actor_initial_files", "simulator_only_files", "evaluator_files", "qualification_files"):
        shared.verify(previous[key])
    return previous, protocol, shared.sha(path)


def seal(shared, phase):
    if shared.RUNS.exists() or shared.FREEZE.exists():
        raise ValueError(f"The {phase} phase is already registered or attempted")
    previous, protocol, original_sha = inherited(shared)
    qualification = dict(previous["qualification_files"])
    jobs = previous["jobs"]
    if phase == "gate":
        jobs = [row for row in jobs if (row["layout"], row["arm"]) == (ELIGIBLE_LAYOUT, GATE_ARM)]
        if len(jobs) != 1:
            raise ValueError("The registered HM3D gate job is missing")
    else:
        gate = verify_gate(supervisor("gate"), write=False)
        if not gate["eligible_for_formal"] or gate != shared.read(GATE_VALIDATION):
            raise ValueError("The separate active-observation gate is missing, stale, or failed")
        qualification |= shared.fingerprints([GATE_FREEZE, GATE_RECEIPT, GATE_VALIDATION, MASK_FILE])
        qualification |= gate["gate_files_sha256"]
    protocol_path = ROOT / ("GATE_PROTOCOL.json" if phase == "gate" else "FORMAL_PROTOCOL.json")
    source = protocol | {
        "schema": SCHEMA, "study_phase": phase,
        "archive_before_repair": "15962cd0d0e6fd6c3693a21b21524a61c5cb0f5b",
        "changed_component": "One targeted active observation of a blocked swept transition at HM3D start 1",
        "scope": "The intervention is available to all three arms on HM3D start 1 only; other starts follow V3",
        "gate_hypothesis": "New measured scan, complete sweep certificate and successful forward execution, with terminal coverage exceeding 522/1640 on the archived common mask",
        "primary_reference": "Identical V2 simulator and evaluator references",
        "paired_sensitivity": "V3 comparator, matching paid clocks on the common 1,640-cell V1 HM3D start-1 mask",
        "reference_phase_is_not_formal_evidence": phase == "gate",
    }
    shared.write(protocol_path, source)
    added = shared.fingerprints([ROOT / name for name in ("runner.py", "study.py", "active_observation.py")]
                                + [protocol_path])
    frozen = previous | {"schema": SCHEMA, "phase": phase, "frozen_unix_s": time.time(),
                         "previous_online_sweep_freeze_sha256": original_sha,
                         "protocol_relative": protocol_path.name, "protocol_sha256": shared.sha(protocol_path),
                         "actor_files": previous["actor_files"] | added, "qualification_files": qualification,
                         "jobs": jobs}
    shared.write(shared.FREEZE, frozen)
    print(f"Frozen {len(jobs)} fresh {phase} job(s): {shared.sha(shared.FREEZE)}", flush=True)
    return frozen


def execute_gate(shared):
    gate = seal(shared, "gate")
    shared.RUNS.mkdir(parents=True)
    with (shared.RUNS / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        planned, = gate["jobs"]
        attempt = shared.RUNS / planned["layout"] / planned["arm"] / "attempt_001"
        attempt.parent.mkdir(parents=True)
        worker = shared.attempt(planned, attempt)
        shared.write(GATE_RECEIPT, {"freeze_sha256": shared.sha(shared.FREEZE), "phase": "gate", "episodes": [worker]})
    return verify_gate(shared)


def trace_event(event, start, steps):
    if event.get("event") != "active_observed_edge_executed" or event.get("complete_sweep") is not True:
        return False
    scan, edge = event.get("scan_paid"), event.get("edge_paid")
    if not (type(scan) is int and type(edge) is int and 1 <= scan < edge <= len(steps)):
        return False
    before_scan, observed = (steps[scan-2] if scan > 1 else start), steps[scan-1]
    before_motion, executed = steps[edge-2], steps[edge-1]
    certificate = event.get("target_sweep_certificate")
    return bool(certificate and before_scan["scan_ranges_sha256"] != observed["scan_ranges_sha256"]
                and before_scan["packet_sha256"] != observed["packet_sha256"]
                and observed["packet_sha256"] == event.get("scan_packet_sha256")
                and executed["packet_sha256"] == event.get("edge_packet_sha256")
                and executed["action"] == "forward" and executed["guard_allowed"] is True
                and executed["collision"] is False and before_motion["position"] == event.get("from_cell")
                and executed["position"] == event.get("to_cell")
                and sum(abs(a-b) for a, b in zip(before_motion["position"], executed["position"], strict=True)) == 1)


def verify_gate(shared, write=True):
    import numpy as np

    seal_hash = shared.sha(GATE_FREEZE)
    frozen, receipt = shared.read(GATE_FREEZE), shared.read(GATE_RECEIPT)
    shared.verify(frozen["actor_files"])
    if (receipt["phase"] != "gate" or receipt["freeze_sha256"] != seal_hash
            or len(frozen["jobs"]) != 1 or len(receipt["episodes"]) != 1):
        raise ValueError("The validation gate must contain one separately sealed execution")
    job, = frozen["jobs"]
    worker, = receipt["episodes"]
    run = Path(worker["run_path"])
    desired = GATE_ROOT / ELIGIBLE_LAYOUT / GATE_ARM / "attempt_001" / "run"
    if (run != desired or (worker["layout"], worker["arm"]) != (ELIGIBLE_LAYOUT, GATE_ARM)
            or any(worker.get(k) != v for k, v in job.items()) or worker["started_unix_s"] < frozen["frozen_unix_s"]
            or worker != shared.read(run.parent / "exit.json") or worker["freeze_sha256"] != seal_hash):
        raise ValueError("Gate process is not the single, separately sealed worker")
    files = [GATE_FREEZE, GATE_RECEIPT, run.parent / "launch.json", run.parent / "exit.json", run.parent / "worker.log"]
    result = {"schema": "nso_active_sweep_separate_gate_v1", "layout": ELIGIBLE_LAYOUT, "arm": GATE_ARM,
              "eligible_for_formal": False, "worker_exit_code": worker["exit_code"],
              "old_cover_threshold_cells": BASELINE_CELLS, "v1_reachable_mask_cells": MASK_CELLS,
              "actual_paid": None, "terminal_correct_free_on_v1_mask": None, "verified_event_chain": []}
    if worker["exit_code"] == 0:
        manifest_path = run / "manifest.json"
        if worker["result_sha256"] != shared.sha(manifest_path):
            raise ValueError("Gate manifest differs from its native worker receipt")
        manifest = shared.read(manifest_path)
        paths = {name: run / name for name in ("audit.json", "steps.json", "observed_belief_trace.npz",
                                             "start/checkpoint.json", "initial_raw_rgbd_scan_sha256.json")}
        for name, path in paths.items():
            if name != "start/checkpoint.json" and manifest["files_sha256"].get(name) != shared.sha(path):
                raise ValueError(f"Gate evidence is not bound by the native manifest: {name}")
        counts_file = paths["observed_belief_trace.npz"]
        steps = shared.read(paths["steps.json"])
        start = shared.read(paths["start/checkpoint.json"])
        zero = shared.read(paths["initial_raw_rgbd_scan_sha256.json"])
        paid = int(manifest["paid_actions"])
        approved = next(row for row in shared.read(ROOT / "GATE_PROTOCOL.json")["starts"] if row["layout"] == ELIGIBLE_LAYOUT)
        if (paid != len(steps) or not 0 <= paid <= frozen["budget"] or manifest["cold_started_at_paid_zero"] is not True
                or manifest["actual_closed_loop"] is not True or start["actual_paid"] != 0 or zero["step"] != 0
                or start["packet_sha256"] != zero["packet_sha256"] or start["packet_sha256"] != approved["initial_packet_sha256"]
                or start["pose"] != approved["anchor"] or manifest["runner_freeze_sha256"] != seal_hash
                or [row["step"] for row in steps] != list(range(1, paid+1))):
            raise ValueError("Gate did not start from the frozen paid-zero observation and record every charged state")
        with np.load(MASK_FILE, allow_pickle=False) as values, np.load(counts_file, allow_pickle=False) as trace:
            mask, clock, belief = values["reachable"], trace["paid"], trace["belief"]
        if int(mask.sum()) != MASK_CELLS or not np.array_equal(clock, np.arange(paid+1)) or belief.shape != (paid+1, *mask.shape):
            raise ValueError("Gate coverage does not align with the frozen common mask and paid clock")
        series = (belief[:, mask] == 0).sum(axis=1)
        checks = [event for event in shared.read(paths["audit.json"]) if trace_event(event, zero, steps)]
        cover = int(series[-1])
        result.update(actual_paid=paid, terminal_correct_free_on_v1_mask=cover,
                      peak_correct_free_on_v1_mask=int(series.max()), verified_event_chain=checks,
                      eligible_for_formal=(bool(checks) and cover > BASELINE_CELLS))
        files += [manifest_path, MASK_FILE, *paths.values()]
    result["gate_files_sha256"] = shared.fingerprints(files)
    if write:
        if GATE_VALIDATION.exists():
            if result != shared.read(GATE_VALIDATION):
                raise ValueError("Gate evidence changed after the eligibility receipt was written")
        else:
            shared.write(GATE_VALIDATION, result)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("gate", "verify-gate", "seal", "run"))
    cli.add_argument("--workers", type=int, default=6)
    args = cli.parse_args()
    if args.command in ("gate", "verify-gate"):
        shared = supervisor("gate")
        result = execute_gate(shared) if args.command == "gate" else verify_gate(shared)
        print(f"Gate eligibility: {result['eligible_for_formal']}; {GATE_VALIDATION}", flush=True)
        raise SystemExit(0 if result["eligible_for_formal"] else 1)
    shared = supervisor()
    if args.command == "seal":
        seal(shared, "formal")
    else:
        formal = shared.read(FREEZE)
        if formal["schema"] != SCHEMA or formal["phase"] != "formal" or len(formal["jobs"]) != 12:
            raise ValueError("Formal run requires the combined, gate-qualified twelve-run freeze")
        shared.verify(formal["qualification_files"])
        shared.run(args.workers)


if __name__ == "__main__":
    main()
