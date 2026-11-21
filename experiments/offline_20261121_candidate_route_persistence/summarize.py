"""Pair measured checkpoints and apply the frozen terminal interpretation."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS = ("C_free", "F1_scene_05cm", "J", "object_net_m2")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_csv(path, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def practical_label(value, band):
    if value > band:
        return "lead"
    if value < -band:
        return "trail"
    return "equivalent"


def is_measured(row):
    return all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) for key in METRICS)


def select_checkpoints(checkpoints, tolerance):
    by_route = {}
    for row in checkpoints:
        identity = (row["layout"], int(row["candidate_id"]))
        if not is_measured(row):
            continue
        budget = row["actual_paid"]
        if abs(row["C_free"] * row["F1_scene_05cm"] - row["J"]) > tolerance:
            raise ValueError(f"J product mismatch at {identity} paid {budget}")
        points = by_route.setdefault(identity, {})
        old = points.get(budget)
        if old is not None and any(abs(row[key] - old[key]) > tolerance for key in METRICS):
            raise ValueError(f"Two different measurements at one clock for {identity} paid {budget}")
        if old is None or row["checkpoint"] == "terminal":
            points[budget] = row
    return by_route


def compare(kind, label, left, right, band):
    d_c = left["C_free"] - right["C_free"]
    d_f = left["F1_scene_05cm"] - right["F1_scene_05cm"]
    d_j = left["J"] - right["J"]
    d_c_term = d_c * (left["F1_scene_05cm"] + right["F1_scene_05cm"]) / 2
    d_f_term = d_f * (left["C_free"] + right["C_free"]) / 2
    identical_clock = left["actual_paid"] == right["actual_paid"]
    result = {
        "layout": left["layout"], "contrast": kind, "comparison": label,
        "candidate_shelf_side": left["candidate_id"], "candidate_comparator": right["candidate_id"],
        "comparison_clock": "same_paid" if identical_clock else "different_actual_stops",
        "common_paid": left["actual_paid"] if identical_clock else None,
    }
    for prefix, row in (("shelf", left), ("comparator", right)):
        result[f"{prefix}_actual_paid"] = row["actual_paid"]
        result.update({f"{prefix}_{key}": row[key] for key in METRICS})
        if label == "terminal":
            result.update({f"{prefix}_{field}": row[field] for field in ("returned", "collisions", "stop_reason")})
    result.update({
        "delta_C_free": d_c, "delta_F1_scene_05cm": d_f, "delta_J": d_j,
        "delta_J_percentage_points": 100 * d_j, "J_coverage_contribution": d_c_term,
        "J_F1_contribution": d_f_term, "delta_object_net_m2": left["object_net_m2"] - right["object_net_m2"],
        "practical_result": practical_label(d_j, band),
    })
    return result


def lead_erosion(pairs, reference_paid):
    baseline = next((row for row in pairs if row["common_paid"] == reference_paid), None)
    if baseline is None or baseline["practical_result"] != "lead":
        return None
    previous = baseline
    for index, row in enumerate(pairs):
        if row["common_paid"] <= reference_paid:
            continue
        if row["practical_result"] != "lead":
            return {
                "last_observed_lead_paid": previous["common_paid"], "first_observed_erosion_paid": row["common_paid"],
                "first_observed_erosion_label": row["practical_result"],
                "at_last_observed_lead_delta_J": previous["delta_J"], "at_first_observed_erosion_delta_J": row["delta_J"],
                "regained_lead_at_later_measured_clock": next((next_row["common_paid"] for next_row in pairs[index + 1:]
                    if next_row["practical_result"] == "lead"), None),
            }
        previous = row
    return None


def summarize(protocol, checkpoints, terminals):
    expected_new = {(layout, cid) for layout, ids in protocol["new_routes"].items() for cid in ids}
    expected_old = {(layout, cid) for layout, ids in protocol["archived_full_routes"].items() for cid in ids}
    endpoints = {(row["layout"], int(row["candidate_id"])): row for row in terminals}
    if len(terminals) != len(endpoints) or endpoints.keys() != expected_new | expected_old:
        raise ValueError("Terminal files must have exactly the ten declared route outcomes")
    if any(not is_measured(row) or not isinstance(row["returned"], bool) for row in endpoints.values()):
        raise ValueError("Terminal geometry and return outcomes are needed for each declared route")
    band = protocol["measurement"]["terminal_equivalence_J_fraction"]
    tolerance = protocol["measurement"]["numerical_tolerance"]
    by_route = select_checkpoints(checkpoints + terminals, tolerance)
    joined = []
    outcomes = {}
    for name, pairs in protocol["decision"].items():
        if name == "interpretation":
            continue
        for layout, (shelf, other) in pairs.items():
            pair = ((layout, shelf), (layout, other))
            a, b = (by_route[key] for key in pair)
            paired_checks = [compare(name, "checkpoint", a[paid], b[paid], band)
                             for paid in sorted(a.keys() & b.keys())]
            terminal_a, terminal_b = (endpoints[key] for key in pair)
            terminal = compare(name, "terminal", terminal_a, terminal_b, band)
            terminal["return_noninferior"] = terminal["shelf_returned"] or not terminal["comparator_returned"]
            terminal["collision_noninferior"] = terminal["shelf_collisions"] <= terminal["comparator_collisions"]
            terminal["sustained_practical_advantage"] = (
                terminal["return_noninferior"] and terminal["collision_noninferior"]
                and terminal["practical_result"] == "lead"
            )
            joined.extend(paired_checks + [terminal])
            outcomes.setdefault(layout, {})[name] = {
                "terminal": terminal, "common_checkpoint_values": paired_checks,
                "first_observed_erosion": lead_erosion(paired_checks, 30 if layout.startswith("warehouse") else 58),
            }
    gallery = {layout: all(r["terminal"]["sustained_practical_advantage"] for r in contrasts.values())
               for layout, contrasts in outcomes.items() if layout.startswith("gallery")}
    return joined, {"practical_band_J_fraction": band, "new_routes": len(expected_new),
                    "archived_routes": len(expected_old), "per_layout": outcomes,
                    "gallery_sustained_by_placement": gallery,
                    "preregistered_online_score_proposal_triggered": all(gallery.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "PROTOCOL.json")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "measurement/results/checkpoints.json")
    parser.add_argument("--terminal", type=Path, default=ROOT / "measurement/results/terminal.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    rows, decision = summarize(read(args.protocol), read(args.checkpoints)["rows"], read(args.terminal)["rows"])
    args.output.mkdir(parents=True, exist_ok=True)
    encode_csv(args.output / "paired_layouts.csv", rows)
    encode_csv(args.output / "paired_terminal.csv", [row for row in rows if row["comparison"] == "terminal"])
    decision["inputs_sha256"] = {p.name: sha(p) for p in (args.protocol, args.checkpoints, args.terminal)}
    (args.output / "decision.json").write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
