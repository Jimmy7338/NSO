"""Run the frozen three-arm plan while replacing only simulator support data."""

import argparse
import importlib.util
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "region_independent_floors_20261123"
PREVIOUS = ROOT.parent / "external_support_repair_20261126"
SOURCE = ROOT.parent / "region_global_revision_20261122/next/global_attribution"
FREEZE = ROOT / "FORMAL_FREEZE.json"
SCENES = ROOT / "scenes"
OFFLINE = PREVIOUS / "scenes"


def supervisor():
    spec = importlib.util.spec_from_file_location("support_repair_supervisor", BASE / "study.py")
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    shared.ROOT, shared.FREEZE, shared.RUNS = ROOT, FREEZE, ROOT / "execution/formal"
    return shared


def seal(shared):
    if shared.RUNS.exists():
        raise ValueError("Seal the comparison before creating any formal run")
    old_path = PREVIOUS / "FORMAL_FREEZE.json"
    old = shared.read(old_path)
    protocol = shared.read(PREVIOUS / "FORMAL_PROTOCOL.json")
    tests = shared.read(ROOT / "VALIDATION.json")
    if tests["status"] != "verified" or tests["formal_planner_invocations_before_freeze"] != 0:
        raise ValueError("Pass independent geometry validation before sealing the formal rerun")
    shared.verify(tests["evidence_sha256"])
    for name in ("actor_files", "simulator_only_files", "evaluator_files"):
        shared.verify(old[name])
    initial, refs, caches = [], [], []
    for start in protocol["starts"]:
        layout = start["layout"]
        cold = SCENES / "prepared" / layout
        before = OFFLINE / "prepared" / layout
        for name in ("initial_frame.npz", "initial_scan.npz", "initial_state.json"):
            current, prior = cold / name, before / name
            if shared.sha(current) != shared.sha(prior):
                raise ValueError(f"The repaired simulator changed a frozen cold input: {layout}/{name}")
            initial.append(current)
        reference = SCENES / "evaluator_references" / layout
        refs.extend(reference / name for name in ("evaluator_truth.npz", "fixed_reference.npz",
                                                    "source_mesh.ply", "RECEIPT.json"))
        caches.append(SCENES / "precomputed_assets" / f"{start['source_floor']}.npz")
    data = protocol | {"schema": "nso_scan_support_continuity_comparison_v2",
                       "archive_before_repair": "cbdeeafc48dfad7b60772c8e9bff1d10f18b4831",
                       "changed_component": "Simulation floor support continuity/cache only",
                       "priors_actions_goals_budget": "Literal jobs, starts, labels, motion footprint and weights inherited",
                       "reference_scope": "Same source PLY and frozen source-visible surface samples; primary free grid from repaired capsule graph",
                       "original_free_grid_sensitivity": "Compare v1 and v2 on the common v1 reachable mask as secondary",
                       "diagnostic_sources_sha256": shared.sha(ROOT / "VALIDATION.json")}
    shared.write(ROOT / "FORMAL_PROTOCOL.json", data)
    new_actors = old["actor_files"] | shared.fingerprints([*initial, ROOT / "runner.py", Path(__file__),
                       SCENES / "scene.py", ROOT / "FORMAL_PROTOCOL.json"])
    evaluators = old["evaluator_files"] | shared.fingerprints(refs)
    repaired = old | {"schema": data["schema"], "frozen_unix_s": time.time(),
                      "scene_root": str(SCENES), "actor_initial_files": shared.fingerprints(initial),
                      "actor_files": new_actors, "evaluator_files": evaluators,
                      "simulator_only_files": shared.fingerprints([*caches, ROOT / "repair_geometry.py",
                          ROOT / "prepare_study.py", ROOT / "VALIDATION.json"]) | tests["evidence_sha256"],
                      "protocol_relative": "FORMAL_PROTOCOL.json",
                      "protocol_sha256": shared.sha(ROOT / "FORMAL_PROTOCOL.json"),
                      "before_repair_freeze": shared.sha(old_path)}
    shared.write(FREEZE, repaired)
    print(f"Frozen {len(repaired['jobs'])} scheduled simulations: {shared.sha(FREEZE)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "run"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    native = supervisor()
    seal(native) if args.command == "seal" else native.run(args.workers)
