"""Bind the four unseen starts separately from independent evaluation geometry."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
SOURCE = ANALYSES / "paired_unseen_gain_20261120/scene/unseen_world.py"
ACTOR_INPUTS = ("initial_frame.npz", "initial_scan.npz", "initial_state.json")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(paths):
    return {str(path): sha(path) for path in sorted(paths)}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    protocol = read(HERE.parent / "PROTOCOL.json")
    layouts = protocol["heldout"]["layouts"]
    if not (HERE / "preparation.complete").exists() or (HERE.parent / "runner/runs/holdout").exists():
        raise ValueError("Complete source-only preparation before any held-out policy starts")
    actor = [HERE / "scene.py", SOURCE]
    evaluator = [HERE / "prepare.py", Path(__file__), HERE / "reference_protocol.json"]
    descriptions = []
    for layout in layouts:
        origin = HERE / "prepared" / layout
        if sorted(path.name for path in origin.iterdir()) != list(ACTOR_INPUTS):
            raise ValueError(f"Actor-facing input whitelist changed for {layout}")
        actor.extend(origin / name for name in ACTOR_INPUTS)
        evaluator.append(origin / "initial_state.json")
        actual = read(origin / "initial_state.json")
        ref = HERE / "evaluator_references" / layout
        rules = read(ref / "RECEIPT.json")
        preflight = read(ref / "cold_start_preflight.json")
        if (rules["actor_executed"] or rules["actor_trajectory_read"] or preflight["actor_select_calls"]
                or actual["paid_actions"] != 0 or actual["stop_budget"] != protocol["heldout"]["paid_action_budget"]
                or preflight["initial_unknown_reachable_cells"] <= 0):
            raise ValueError(f"Held-out preflight used prior outcomes or failed cold start: {layout}")
        by_hash = {"fixed_reference.npz": rules["reference_npz_sha256"], "source_mesh.ply": rules["source_mesh_sha256"]}
        for name, expected in by_hash.items():
            if sha(ref / name) != expected:
                raise ValueError(f"Independent reference changed for {layout}/{name}")
        evaluator.extend(ref.iterdir())
        descriptions.append({key: actual[key] for key in ("layout", "shape", "anchor", "seed", "stop_budget")})
    mixed = {"protocol_sha256": sha(HERE.parent / "PROTOCOL.json"), "layouts": descriptions, "actors_run": 0}
    write(HERE / "freeze.json", mixed | {"status": "actor_inputs_only", "files_sha256": bind(actor)})
    write(HERE / "evaluator_freeze.json", mixed | {"status": "reference_hidden_from_actor", "files_sha256": bind(evaluator)})


if __name__ == "__main__":
    main()
