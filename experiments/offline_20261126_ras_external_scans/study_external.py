"""Seal and supervise the externally sourced floors without altering their geometry."""

import argparse
import importlib.util
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1] / "region_independent_floors_20261123"
SCENES = ROOT / "scenes"
FREEZE = ROOT / "FORMAL_FREEZE.json"
ARMS = ("region_S", "combined_S", "F_strict")


def supervisor():
    spec = importlib.util.spec_from_file_location("external_supervisor", BASE / "study.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native.ROOT, native.FREEZE, native.RUNS = ROOT, FREEZE, ROOT / "execution/formal"
    return native


def seal(native):
    original_freeze = BASE / "FORMAL_FREEZE.json"
    original = native.read(original_freeze)
    actor_path = SCENES / "actor_freeze_holdout.json"
    evaluator_path = SCENES / "evaluator_freeze_holdout.json"
    actor, evaluator = native.read(actor_path), native.read(evaluator_path)
    if actor["layouts"] != evaluator["layouts"] or actor["actors_run"] or evaluator["actors_run"]:
        raise ValueError("The actor and evaluator require matching prospective registrations.")
    starts = actor["layouts"]
    registered = sorted({row["source_floor"] for row in starts})
    if len(registered) not in (1, 2) or any(
            sorted(row["start_index"] for row in starts if row["source_floor"] == floor) != [1, 2]
            for floor in registered):
        raise ValueError("Register two starts for each eligible external floor")
    layouts = [row["layout"] for row in starts]
    if len(layouts) != len(set(layouts)):
        raise ValueError("Duplicate registered external-start identifier")
    protocol = {"schema": "nso_ras_external_starts_v1",
                "archive_head": "74ecf883503d5a117d1e566417aff0213f76472a",
                "base_commit": original["base_commit"], "layouts": layouts,
                "starts": starts, "arms": list(ARMS), "budget": original["budget"],
                "capture_paid": original["capture_paid"],
                "comparison_unit": "source floor with paired starts",
                "semantics": "Supplied category masks are absent when the external source provides no compatible classes. Both regional arms retain the same built-in local S authoring and common geometric fallback, with zero semantic evidence in that condition."}
    jobs = []
    for layout in layouts:
        for arm in ARMS:
            job = {"layout": layout, "arm": arm, "prior": None if arm == "F_strict" else "S",
                   "regional_enabled": arm != "F_strict", "literal": True}
            if arm == "combined_S":
                job["repair"] = {"target": True, "outbound": True}
            jobs.append(job)
    new_initials = [SCENES / "scene.py"]
    for layout in layouts:
        cold_dir = SCENES / "prepared" / layout
        new_initials.extend(cold_dir / name for name in ("initial_state.json", "initial_frame.npz", "initial_scan.npz"))
        cold = native.read(cold_dir / "initial_state.json")
        if cold["paid_actions"] != 0 or cold["policy_select_calls"] != 0 or cold["stop_budget"] != original["budget"]:
            raise ValueError(f"Invalid external cold state: {layout}")
    actor_initial_files = native.fingerprints(new_initials)
    def independent(files):
        return {name: digest for name, digest in files.items() if not Path(name).is_relative_to(BASE / "scenes")}
    actor_files = independent(original["actor_files"]) | actor["files_sha256"] | actor_initial_files
    eval_files = independent(original["evaluator_files"]) | evaluator["files_sha256"]
    cache = actor.get("simulator_only_collision_cache_sha256", {})
    for files in (actor_files, eval_files, cache):
        native.verify(files)
    if any("evaluator_references" in name or "precomputed_assets" in name or "evaluator_truth" in name
           for name in actor_files):
        raise ValueError("External actor scope contains scene truth")
    protocol_file = ROOT / "FORMAL_PROTOCOL.json"
    native.write(protocol_file, protocol)
    actor_files |= native.fingerprints([actor_path, protocol_file, ROOT / "runner.py", Path(__file__)])
    frozen = original | {"schema": protocol["schema"], "frozen_unix_s": time.time(),
                         "protocol_relative": protocol_file.name, "protocol_sha256": native.sha(protocol_file),
                         "scene_root": str(SCENES), "actor_initial_files": actor_initial_files,
                         "actor_files": actor_files, "evaluator_files": eval_files, "simulator_only_files": cache,
                         "jobs": jobs, "source_archive_head": protocol["archive_head"]}
    native.write(FREEZE, frozen)
    print(f"Frozen {len(jobs)} external-start jobs at {native.sha(FREEZE)}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("seal", "run"))
    cli.add_argument("--workers", type=int, default=6)
    args = cli.parse_args()
    native = supervisor()
    if args.action == "seal":
        seal(native)
    else:
        native.run(args.workers)
