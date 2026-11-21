"""Apply the frozen parent-layout gate to independently measured route endpoints."""

import argparse
import hashlib
import json
from pathlib import Path


MODES = ("G", "S", "X")
OBJECT = "object_net_m2"
GROSS = "object_gross_m2"


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign(value, tolerance):
    if abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def strict_observation(first, second, protocol):
    if any(row["checkpoints"]["arrival"]["status"] != "captured" for row in (first, second)):
        return None
    costs = [(row["outbound_cost"], row["reserved_return_cost"]) for row in (first, second)]
    observed_steps = [row["checkpoints"]["arrival"]["actual_paid"] for row in (first, second)]
    if costs[0] != costs[1] or observed_steps[0] != observed_steps[1]:
        return None
    left = first["checkpoints"]["arrival"]
    right = second["checkpoints"]["arrival"]
    thresholds = protocol["metrics"]
    delta = {key: left[key] - right[key] for key in (OBJECT, GROSS, "J")}
    expected = {
        mode: (first["scores"][mode] - first["common_score"]) - (second["scores"][mode] - second["common_score"])
        for mode in MODES
    }
    return {
        "layout": first["layout"], "candidate_ids": [first["candidate_id"], second["candidate_id"]],
        "native_outbound_return": costs[0], "actual_paid": observed_steps[0], "delta": delta,
        "actual_direction": {
            key: sign(value, thresholds["material_J_fraction" if key == "J" else "material_object_area_m2"])
            for key, value in delta.items()
        },
        "predicted_object_score_delta": expected,
        "predicted_direction": {
            mode: sign(value, thresholds["numeric_score_tie"]) for mode, value in expected.items()
        },
    }


def flipped_and_predicted(pair, key):
    if None in pair:
        return False
    left, right = pair
    actual = left["actual_direction"][key]
    return (
        actual != 0 and actual == -right["actual_direction"][key]
        and all(row["predicted_direction"]["S"] == row["actual_direction"][key] for row in pair)
        and (key != OBJECT or all(
            row["actual_direction"][GROSS] in (0, row["actual_direction"][OBJECT]) for row in pair
        ))
    )


def tally(observations, key):
    material = [r for r in observations if r is not None and r["actual_direction"][key] != 0]
    return {
        "material_count": len(material),
        "agreements": {
            mode: sum(r["predicted_direction"][mode] == r["actual_direction"][key] for r in material)
            for mode in MODES
        },
    }


def leads(tallies):
    values = tallies["agreements"]
    return values["S"] > max(values["G"], values["X"])


def decide(rows, protocol):
    by_id = {(row["layout"], row["candidate_id"]): row for row in rows}
    layouts = protocol["layouts_in_order"]
    if len(rows) != len(by_id) or set(by_id) != {
        (layout, candidate_id) for layout in layouts for candidate_id in protocol["candidates_in_order"]
    }:
        raise ValueError("Measurements must contain exactly one row per frozen placement and candidate.")
    parents = {}
    for parent, ids in protocol["strict_pairs_by_parent"].items():
        positions = [layout for layout in layouts if by_id[layout, 0]["parent"] == parent]
        if len(positions) != 2:
            raise ValueError(f"Expected two frozen placements for {parent}.")
        families = [[strict_observation(by_id[layout, a], by_id[layout, b], protocol)
                     for layout in positions] for a, b in ids]
        observations = [row for pair in families for row in pair]
        object_tally = tally(observations, OBJECT)
        j_tally = tally(observations, "J")
        object_flips = [ids[i] for i, pair in enumerate(families) if flipped_and_predicted(pair, OBJECT)]
        j_flips = [ids[i] for i, pair in enumerate(families) if flipped_and_predicted(pair, "J")]
        object_gate = bool(object_flips) and leads(object_tally)
        parents[parent] = {
            "strict_pairs_by_swap": families, "object_qualified": object_gate,
            "A_qualified": object_gate and bool(j_flips) and leads(j_tally),
            "object_net_ranking": object_tally, "whole_scene_J_ranking": j_tally,
            "material_object_flips_correctly_ranked_by_S": object_flips,
            "material_J_flips_correctly_ranked_by_S": j_flips,
        }
    if any(value["A_qualified"] for value in parents.values()):
        return {"branch": "A", "parents": parents, "continuation_pair": None}
    if not any(value["object_qualified"] for value in parents.values()):
        return {"branch": "C", "parents": parents, "continuation_pair": None}
    candidates = [
        row for parent in parents.values() if parent["object_qualified"]
        for pair in parent["strict_pairs_by_swap"] for row in pair
        if row is not None and row["actual_direction"][OBJECT] != 0
        and row["predicted_direction"]["S"] == row["actual_direction"][OBJECT]
    ]
    candidates.sort(key=lambda row: (
        row["actual_direction"]["J"] == row["actual_direction"][OBJECT], -abs(row["delta"][OBJECT]),
        layouts.index(row["layout"]), row["candidate_ids"],
    ))
    return {"branch": "B", "parents": parents, "continuation_pair": candidates[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--protocol", type=Path, default=root / "PROTOCOL.json")
    parser.add_argument("--measurement", type=Path, default=root / "measurement/results/candidate_rows.json")
    parser.add_argument("--out", type=Path, default=root / "decision.json")
    args = parser.parse_args()
    outcome = decide(read_json(args.measurement)["rows"], read_json(args.protocol))
    outcome["input_sha256"] = {"protocol": digest(args.protocol), "measurement": digest(args.measurement)}
    result = json.dumps(outcome, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with args.out.open("x") as stream:
        stream.write(result)
    print(outcome["branch"])


if __name__ == "__main__":
    main()
