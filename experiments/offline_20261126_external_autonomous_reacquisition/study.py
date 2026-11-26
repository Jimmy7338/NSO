"""Prospectively register development and paired autonomous-reacquisition runs."""

import argparse
import importlib.util
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "region_independent_floors_20261123"
PREVIOUS = ROOT.parent / "external_online_sweep_20261126"
V4 = ROOT.parent / "external_active_sweep_observation_20261126"
V2 = ROOT.parent / "external_support_continuity_20261126"
V1 = ROOT.parent / "external_support_repair_20261126"
SOURCE = ROOT.parent / "region_global_revision_20261122/next/global_attribution"
SCENES = ROOT / "scenes_work"
FREEZE = ROOT / "FORMAL_FREEZE.json"
DEV_FREEZE = ROOT / "DEVELOPMENT_FREEZE.json"
DEV_LAYOUT = "b26_hm3d_00770_s1"
DEVELOPMENT_RECEIPT = ROOT / "execution/development/phase_receipt.json"
DEVELOPMENT_VALIDATION = ROOT / "validation/DEVELOPMENT_VALIDATION.json"
PARAMETERS = ROOT / "PARAMETERS_FROZEN.json"
POLICY = ROOT / "autonomous_observation.py"
ADAPTER = ROOT / "reacquisition_adapter.py"
ARMS = ("A", "B")
BUDGET = 192
SCHEMA = "nso_external_autonomous_reacquisition_v5"


def supervisor(phase="formal"):
    spec = importlib.util.spec_from_file_location("autonomous_reacquisition_supervisor", BASE / "study.py")
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    shared.ROOT = ROOT
    shared.FREEZE = DEV_FREEZE if phase == "development" else FREEZE
    shared.RUNS = ROOT / "execution" / phase
    return shared


def baseline(shared):
    path = PREVIOUS / "FORMAL_FREEZE.json"
    inherited = shared.read(path)
    protocol = shared.read(PREVIOUS / "FORMAL_PROTOCOL.json")
    v2 = shared.read(V2 / "FORMAL_FREEZE.json")
    official = ("scene_root", "actor_initial_files", "simulator_only_files", "evaluator_files")
    if (protocol["budget"] != BUDGET or inherited["budget"] != BUDGET
            or any(inherited[key] != v2[key] for key in official)
            or inherited["protocol_sha256"] != shared.sha(PREVIOUS / "FORMAL_PROTOCOL.json")):
        raise ValueError("The archived V3 baseline does not match the frozen V2 scene and sensor contract")
    for key in (*official[1:], "actor_files", "qualification_files"):
        shared.verify(inherited[key])
    return inherited, protocol, shared.sha(path)


def job(starts, arc):
    return {"layout": starts["layout"], "arm": arc, "native_policy": "combined_S", "prior": "S",
            "regional_enabled": True, "literal": True, "repair": {"target": True, "outbound": True},
            "active_ground_reacquisition": arc == "B"}


def prepared_starts(shared, protocol):
    actor_path = SCENES / "actor_freeze_holdout.json"
    eval_path = SCENES / "evaluator_freeze_holdout.json"
    index_path = SCENES / "STARTS.json"
    actor, evaluator, index = (shared.read(path) for path in (actor_path, eval_path, index_path))
    starts = actor["layouts"]
    names = [row["layout"] for row in starts]
    poses = {(row["source_floor"], tuple(row["anchor"])) for row in starts}
    old_poses = {(row["source_floor"], tuple(row["anchor"])) for row in protocol["starts"]}
    if (len(starts) != 4 or len(set(names)) != 4 or set(names) != set(index["layouts"])
            or starts != evaluator["layouts"] or any(receipt["actors_run"] != 0 or receipt["policy_calls"] != 0
                                                      or receipt["budget"] != BUDGET for receipt in (actor, evaluator))
            or len(poses) != 4 or poses & old_poses or any(row["layout"] in protocol["layouts"] for row in starts)):
        raise ValueError("Formal study requires four distinct, prospectively registered fresh starts shared across A and B")
    actor_files, eval_files = (dict(receipt["files_sha256"]) for receipt in (actor, evaluator))
    caches = dict(actor["simulator_only_collision_cache_sha256"])
    cold_files = [SCENES / "scene.py", index_path]
    reference_files = []
    native_spec = protocol["starts"][0]["actual_config"]
    for row in starts:
        layout = row["layout"]
        cold_root = SCENES / "prepared" / layout
        reference = SCENES / "evaluator_references" / layout
        cold = shared.read(cold_root / "initial_state.json")
        if (cold["paid_actions"] != 0 or cold["policy_select_calls"] != 0 or row["stop_budget"] != BUDGET
                or row["actual_config"] != native_spec
                or any(cold[key] != row[key] for key in ("anchor", "seed", "actual_config", "initial_packet_sha256",
                                                         "initial_owner_sha256", "stop_budget"))):
            raise ValueError(f"Fresh start {layout} changes sensor/budget configuration or initial packet")
        cold_files += [cold_root / name for name in ("initial_state.json", "initial_frame.npz", "initial_scan.npz")]
        reference_files += [reference / name for name in ("evaluator_truth.npz", "fixed_reference.npz", "source_mesh.ply", "RECEIPT.json")]
        cache = SCENES / "precomputed_assets" / f"{row['source_floor']}.npz"
        if str(cache) not in caches:
            raise ValueError(f"Missing registered simulator cache for {layout}")
    if set(actor_files) & set(eval_files) or any("evaluator_references" in Path(file).parts or "evaluator_truth" in Path(file).stem
                                               for file in actor_files):
        raise ValueError("Fresh actor scope leaks evaluator-only state")
    for files in (actor_files, eval_files, caches):
        shared.verify(files)
    inputs = shared.fingerprints(cold_files)
    return starts, inputs, actor_files | inputs | shared.fingerprints([actor_path]), (
        eval_files | shared.fingerprints([eval_path, *reference_files])), caches


def development_binding(shared):
    receipt = shared.read(DEVELOPMENT_RECEIPT)
    freeze = shared.read(DEV_FREEZE)
    if (receipt["phase"] != "development" or receipt["freeze_sha256"] != shared.sha(DEV_FREEZE)
            or len(freeze["jobs"]) != 1 or len(receipt["episodes"]) != 1):
        raise ValueError("Formal registration requires exactly one separately sealed development worker")
    worker, = receipt["episodes"]
    run = Path(worker["run_path"])
    if (worker["layout"] != DEV_LAYOUT or worker["arm"] != "B" or worker["exit_code"] != 0
            or not run.is_relative_to(ROOT / "execution/development") or worker["result_sha256"] != shared.sha(run / "manifest.json")):
        raise ValueError("The completed development worker does not match its registered treatment and output")
    manifest = shared.read(run / "manifest.json")
    if manifest["cold_started_at_paid_zero"] is not True or manifest["runner_freeze_sha256"] != receipt["freeze_sha256"]:
        raise ValueError("The development worker did not bind its declared paid-zero start")
    native = [run / name for name in ("manifest.json", "audit.json", "steps.json", "observed_belief_trace.npz",
                                     "initial_raw_rgbd_scan_sha256.json")]
    for path in native[1:]:
        if manifest["files_sha256"].get(path.name) != shared.sha(path):
            raise ValueError(f"Missing or altered development evidence: {path.name}")
    return freeze, run, shared.fingerprints([DEV_FREEZE, DEVELOPMENT_RECEIPT, *native])


def freeze_parameters(shared):
    frozen, run, native = development_binding(shared)
    validation = shared.read(DEVELOPMENT_VALIDATION)
    if (not all(validation.get(key) is True for key in ("autonomous_evidence_gain", "actual_certified_crossing", "safety_rejection"))
            or Path(validation["run"]).resolve() != run.resolve()
            or not validation["evidence_sha256"]):
        raise ValueError("Development must show autonomous evidence gain, a certified actual crossing, and a safety rejection")
    shared.verify(validation["evidence_sha256"])
    current = shared.fingerprints([POLICY, ADAPTER])
    if any(frozen["actor_files"].get(path) != digest for path, digest in current.items()):
        raise ValueError("Formal parameters must match the implementation independently validated in development")
    reviewed = {"schema": "nso_autonomous_reacquisition_parameters_v1", "observer_sha256": current[str(POLICY)],
                "adapter_sha256": current[str(ADAPTER)],
                "observer_filename": POLICY.name, "readable_parameters": validation.get("parameters"),
                "development_evidence_sha256": native | validation["evidence_sha256"] | shared.fingerprints([DEVELOPMENT_VALIDATION]),
                "frozen_unix_s": time.time()}
    shared.write(PARAMETERS, reviewed)
    print(f"Frozen validated observer and development evidence: {shared.sha(PARAMETERS)}", flush=True)


def seal(shared, phase):
    if shared.RUNS.exists() or shared.FREEZE.exists():
        raise ValueError(f"The {phase} phase already has a registration or attempts")
    inherited, old_protocol, parent_hash = baseline(shared)
    policy_files = shared.fingerprints([POLICY, ADAPTER, ROOT / "runner.py", ROOT / "study.py"])
    qualification = dict(inherited["qualification_files"])
    if phase == "development":
        starts = [row for row in old_protocol["starts"] if row["layout"] == DEV_LAYOUT]
        jobs = [job(starts[0], "B")]
        actor_inputs, new_actors, evaluation, simulator = (inherited[key] for key in (
            "actor_initial_files", "actor_files", "evaluator_files", "simulator_only_files"))
        scene_root = inherited["scene_root"]
    else:
        params = shared.read(PARAMETERS)
        if params["observer_sha256"] != shared.sha(POLICY) or params["adapter_sha256"] != shared.sha(ADAPTER):
            raise ValueError("Frozen observer parameters or adapter differ from the current modules")
        shared.verify(params["development_evidence_sha256"])
        qualification |= params["development_evidence_sha256"] | shared.fingerprints([PARAMETERS])
        starts, actor_inputs, new_actors, evaluation, simulator = prepared_starts(shared, old_protocol)
        scene_root = str(SCENES)
        jobs = [job(start, arc) for start in starts for arc in ARMS]
    protocol_path = ROOT / ("DEVELOPMENT_PROTOCOL.json" if phase == "development" else "FORMAL_PROTOCOL.json")
    protocol = old_protocol | {"schema": SCHEMA, "study_phase": phase,
                              "archive_before_repair": "15962cd0d0e6fd6c3693a21b21524a61c5cb0f5b",
                              "layouts": [row["layout"] for row in starts],
                              "starts": starts, "arms": ["B"] if phase == "development" else list(ARMS),
                              "fixed_native_policy": "combined_S V3 with identical local/global settings",
                              "single_treatment": "B enables autonomous ground reacquisition; A disables it",
                              "reference_earlier_targeted_v4": "8917a30fd82c1550c240e3d31748dd8f4332c73d",
                              "development_reference_sha256": parent_hash, "comparison_unit": "paired start",
                              "source_scope": {"distinct_source_floors": sorted({row["source_floor"] for row in starts}),
                                               "previously_used_source_floors": sorted({row["source_floor"] for row in starts}
                                                       & {row["source_floor"] for row in old_protocol["starts"]})}}
    shared.write(protocol_path, protocol)
    actors = inherited["actor_files"] | new_actors | policy_files | shared.fingerprints([protocol_path])
    comparison = inherited | {"schema": SCHEMA, "phase": phase, "frozen_unix_s": time.time(), "scene_root": scene_root,
                              "protocol_relative": protocol_path.name, "protocol_sha256": shared.sha(protocol_path),
                              "actor_files": actors, "actor_initial_files": actor_inputs, "evaluator_files": evaluation,
                              "simulator_only_files": simulator, "qualification_files": qualification, "jobs": jobs,
                              "before_autonomous_reacquisition_freeze_sha256": parent_hash}
    shared.write(shared.FREEZE, comparison)
    print(f"Frozen {len(jobs)} {phase} run(s): {shared.sha(shared.FREEZE)}", flush=True)


def run(shared, phase, workers):
    frozen = shared.read(shared.FREEZE)
    if frozen["schema"] != SCHEMA or frozen["phase"] != phase or len(frozen["jobs"]) != (1 if phase == "development" else 8):
        raise ValueError(f"Unexpected {phase} run matrix")
    shared.verify(frozen["qualification_files"])
    if phase == "formal":
        shared.run(workers)
        return
    shared.RUNS.mkdir(parents=True)
    planned, = frozen["jobs"]
    target = shared.RUNS / planned["layout"] / planned["arm"] / "attempt_001"
    target.parent.mkdir(parents=True)
    worker = shared.attempt(planned, target)
    shared.write(DEVELOPMENT_RECEIPT, {"phase": phase, "freeze_sha256": shared.sha(shared.FREEZE), "episodes": [worker]})
    if worker["exit_code"]:
        raise SystemExit(worker["exit_code"])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("seal-development", "run-development", "development", "freeze-parameters", "seal", "run"))
    cli.add_argument("--workers", type=int, default=6)
    args = cli.parse_args()
    phase = "development" if args.command.endswith("development") else "formal"
    shared = supervisor(phase)
    if args.command in ("seal-development", "development", "seal"):
        seal(shared, phase)
    if args.command in ("run-development", "development", "run"):
        run(shared, phase, args.workers)
    if args.command == "freeze-parameters":
        freeze_parameters(shared)


if __name__ == "__main__":
    main()
