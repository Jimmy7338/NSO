"""Separate actor, simulator and reference dependencies for each frozen split."""

import argparse
import time

from prepare import read, write
from scene import BUDGET, HERE, LAYOUTS, MODEL, REGISTRATION, SPECS, sha
from sources import SOURCES


def files_sha(paths):
    return {str(value): sha(value) for value in sorted(set(paths))}


def freeze(split):
    names = [name for name in LAYOUTS if
             (SPECS[name]["scene_role"] == "unrelated_development") == (split == "development")]
    registration = read(REGISTRATION)
    now = time.time()
    if not names or now < registration["frozen_unix_s"] or registration["actor_select_calls"]:
        raise ValueError("Scene admission must complete without consulting a planner")
    actor_files = [HERE / "scene.py", HERE / "sources.py", REGISTRATION,
                   MODEL / "scene_adapter/replicacad_world.py", MODEL / "assets/textured_raycast.py"]
    evaluator_files = [HERE / "prepare.py", HERE / "preflight.py", HERE / "seal_scenes.py", HERE / "register.py",
                       HERE / "reference_protocol.json", *sorted((HERE / "admission_v1/source_preflights").glob("*.json")),
                       *sorted((HERE / "source_preflights").glob("*.json"))]
    simulator, descriptions = [], []
    for name in names:
        actor_input = HERE / "prepared" / name
        reference = HERE / "evaluator_references" / name
        actor_files.extend(actor_input / filename for filename in
                           ("initial_state.json", "initial_frame.npz", "initial_scan.npz"))
        evaluator_files.extend(reference / filename for filename in
                               ("RECEIPT.json", "cold_start_preflight.json", "evaluator_truth.npz",
                                "fixed_reference.npz", "source_mesh.ply"))
        source_floor = SPECS[name]["source_floor"]
        simulator.extend((HERE / "precomputed_assets" / f"{source_floor}.npz", SOURCES[source_floor]["path"]))
        cold = read(actor_input / "initial_state.json")
        receipt = read(reference / "RECEIPT.json")
        if (cold["paid_actions"] or cold["policy_select_calls"] or receipt["actor_executed"]
                or receipt["actor_trajectory_read"] or receipt.get("visible_labels_used", False)):
            raise ValueError("Scene setup depended on policy execution or category tags")
        if (receipt["source_mesh_sha256"] != sha(reference / "source_mesh.ply")
                or receipt["reference_npz_sha256"] != sha(reference / "fixed_reference.npz")):
            raise ValueError(f"Independent reference changed for {name}")
        descriptions.append({key: cold[key] for key in (
            "layout", "source_floor", "start_index", "seed", "anchor", "shape", "stop_budget",
            "initial_packet_sha256", "initial_owner_sha256", "actual_config", "scene_role")})
    metadata = dict(split=split, layouts=descriptions, budget=BUDGET, actors_run=0, policy_calls=0,
                    frozen_unix_s=now, registry_sha256=sha(REGISTRATION),
                    evaluated_source_floor_count=len({item["source_floor"] for item in descriptions}))
    readable = files_sha(actor_files)
    if any("evaluator_reference" in path or "precomputed_assets" in path or path.endswith(".glb")
           for path in readable):
        raise ValueError("Hidden geometry or reference was exposed in the actor-input ledger")
    write(HERE / f"actor_freeze_{split}.json", metadata | {
        "status": "actor_inputs_only", "files_sha256": readable,
        "simulator_only_collision_cache_sha256": files_sha(simulator),
        "runtime_interface": "Planner receives current acquired sensor packet, arrays, history, and action feedback."})
    write(HERE / f"evaluator_freeze_{split}.json", metadata | {
        "status": "reference_hidden_from_actor", "files_sha256": files_sha(evaluator_files)})
    print(f"Frozen {split}: {', '.join(names)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("holdout", "development"))
    freeze(parser.parse_args().split)
