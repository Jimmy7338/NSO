"""Join deterministic rollouts to frozen native predictions and audit pairwise rankings."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from archive_predictions import MODES, ROOT, close, digest, read_json, require, write_json, write_rows

RANKING = Path(__file__).resolve().parent
OUTCOMES = ("object_net_m2", "object_gross_m2", "object_lost_m2", "C_free", "F1_scene_05cm", "J")
BY_CLASS = tuple(f"{name}_{kind}_m2" for name in ("shelf", "cabinet") for kind in ("net", "gross", "lost"))
SCORES = ("N", *(f"total_{mode}" for mode in MODES),
          *(f"object_{mode}_minus_N" for mode in MODES), "category_S_minus_G", "category_X_minus_G")
TERMS = {
    "object": {mode: f"object_{mode}_minus_N" for mode in MODES},
    "total": {mode: f"total_{mode}" for mode in MODES},
    "category": {mode: f"category_{mode}_minus_G" for mode in ("S", "X")},
}
MEASURED = ("object_net_m2", "object_gross_m2", "J")


def sign(delta, bound):
    if delta is None:
        return None
    if abs(delta) <= bound:
        return 0
    return 1 if delta > 0 else -1


def difference(first, second):
    if first is None or second is None:
        return None
    return first - second


def collect(archive_csv, measurements, protocol):
    with archive_csv.open(newline="") as stream:
        frozen = list(csv.DictReader(stream))
    measured = {(row["layout"], row["candidate_id"]): row for row in measurements}
    require(len(frozen) == len(measured) == len(measurements) == 16, "Expected 16 unique measured archived candidate rows")
    joined = {}
    wide_rows = []
    baseline_metrics = {}
    for old in frozen:
        layout, cid = old["layout"], int(old["candidate_id"])
        row = measured[layout, cid]
        require(old["parent"] == row["parent"] and old["native_side"] == row["side"]
                and int(old["nominal_outbound"]) == row["outbound_cost"]
                and int(old["nominal_reserved_return"]) == row["reserved_return_cost"],
                f"Archive and actual execution metadata disagree at {layout}/{cid}")
        require(close(float(old["N"]), row["common_score"]), f"Common score mismatch at {layout}/{cid}")
        for mode in MODES:
            require(close(float(old[f"total_{mode}"]), row["scores"][mode])
                    and int(old[f"native_rank_{mode}"]) == row["ranks"][mode]
                    and close(float(old[f"template_area_{mode}_m2"]), row["template_area_m2"][mode]),
                    f"Archive and measurement native scoring disagree at {layout}/{cid}/{mode}")
        common_paid = protocol["common_paid_checkpoints_by_parent"][row["parent"]]
        requested = {"start": 0, "arrival": row["outbound_cost"], f"paid_{common_paid}": common_paid}
        states = row["checkpoints"]
        require(set(requested) <= set(states), f"Missing endpoint statuses at {layout}/{cid}")
        start = states["start"]
        if start["status"] == "captured":
            require(start["actual_paid"] == 0, f"Wrong common start step at {layout}/{cid}")
            baseline_anchor = baseline_metrics.setdefault(layout, start)
            require(all(close(baseline_anchor[key], start[key]) for key in OUTCOMES),
                    f"Initial outcome metrics are not shared at {layout}/{cid}")
        else:
            require(all(states[stage]["status"] == "unavailable" for stage in requested),
                    f"Missing common start for a candidate with recorded outcomes at {layout}/{cid}")
        wide = {
            **old, "arrival_status": row["arrival_status"], "termination": json.dumps(row["termination"], ensure_ascii=False),
            **{key: row.get(key) for key in (
                "status", "arrival", "actual_paid", "paid_at_arrival_or_interruption", "actual_collisions", "stop",
            )},
        }
        for stage, planned in requested.items():
            state = states[stage]
            require(state["status"] in ("captured", "unavailable"), f"Unexpected checkpoint status at {layout}/{cid}/{stage}")
            if state["status"] == "captured":
                require(state["actual_paid"] >= 0 and close(state["C_free"] * state["F1_scene_05cm"], state["J"]),
                        f"J mismatch at {layout}/{cid}/{stage}")
                require(close(state["object_gross_m2"] - state["object_lost_m2"], state["object_net_m2"])
                        and close(state["shelf_net_m2"] + state["cabinet_net_m2"], state["object_net_m2"]),
                        f"Inconsistent area model at {layout}/{cid}/{stage}")
                if stage.startswith("paid_"):
                    require(state["actual_paid"] == planned, f"Incorrect matched paid step at {layout}/{cid}")
            prefix = "common_paid" if stage.startswith("paid_") else stage
            wide.update({
                f"{prefix}_status": state["status"], f"{prefix}_unavailable_reason": state.get("reason", state.get("outcome")),
                f"{prefix}_planned_steps": planned, f"{prefix}_paid_steps": state.get("actual_paid"),
                **{f"{prefix}_{key}": state.get(key) for key in (
                    *OUTCOMES, *BY_CLASS, "delta_C_free", "delta_F1_scene_05cm", "delta_J",
                    "correct_free_cells", "reachable_cells", "p_scene_05cm", "r_scene_05cm",
                )},
            })
        joined[layout, cid] = {"archive": old, "actual": row}
        wide_rows.append(wide)
    return joined, wide_rows


def metric_thresholds(protocol):
    metrics = protocol["metrics"]
    numeric = metrics["numeric_J_tie"]
    return {
        "numeric": dict.fromkeys(MEASURED, numeric),
        "material": {
            "object_net_m2": metrics["material_object_area_m2"],
            "object_gross_m2": metrics["material_object_area_m2"],
            "J": metrics["material_J_fraction"],
        },
    }


def pair_row(layout, parent, cids, stage, design, joined, protocol):
    first, second = (joined[layout, cid] for cid in cids)
    old_a, old_b = first["archive"], second["archive"]
    a, b = first["actual"]["checkpoints"][stage], second["actual"]["checkpoints"][stage]
    has_pair = a["status"] == b["status"] == "captured"
    equal_paid = has_pair and a["actual_paid"] == b["actual_paid"]
    comparable = bool(equal_paid and design != "additional_asynchronous_arrival")
    record = {
        "parent": parent, "layout": layout, "candidate_a": cids[0], "candidate_b": cids[1],
        "comparison": design, "checkpoint": stage,
        "original_candidate_id": protocol["original_candidate"][layout],
        "a_native_outbound": int(old_a["nominal_outbound"]), "b_native_outbound": int(old_b["nominal_outbound"]),
        "a_native_reserved_return": int(old_a["nominal_reserved_return"]),
        "b_native_reserved_return": int(old_b["nominal_reserved_return"]),
        "a_native_side": old_a["native_side"], "b_native_side": old_b["native_side"],
        "informative_optical_side": old_a["informative_optical_side"],
        "a_optical_gate_crossing_side": old_a["optical_gate_crossing_side"],
        "b_optical_gate_crossing_side": old_b["optical_gate_crossing_side"],
        "a_paid": a.get("actual_paid"), "b_paid": b.get("actual_paid"),
        "a_status": a["status"], "b_status": b["status"],
        "a_unavailable_reason": a.get("reason", a.get("outcome")),
        "b_unavailable_reason": b.get("reason", b.get("outcome")),
        "equal_actual_paid": bool(equal_paid), "comparable_direction": comparable,
        "a_termination": json.dumps(first["actual"]["termination"], ensure_ascii=False),
        "b_termination": json.dumps(second["actual"]["termination"], ensure_ascii=False),
        "gross_material_contradiction_to_net": None,
    }
    for name in (*SCORES, *OUTCOMES, *BY_CLASS, "delta_J"):
        if name in SCORES:
            first_value, second_value = float(old_a[name]), float(old_b[name])
        else:
            first_value, second_value = a.get(name), b.get(name)
        record.update({f"a_{name}": first_value, f"b_{name}": second_value,
                       f"diff_{name}": difference(first_value, second_value)})
    epsilon = protocol["metrics"]["numeric_score_tie"]
    for name in SCORES:
        record[f"sign_{name}"] = sign(record[f"diff_{name}"], epsilon)
    for tier, bands in metric_thresholds(protocol).items():
        for name, band in bands.items():
            record[f"{tier}_sign_{name}"] = sign(record[f"diff_{name}"], band)
    net, gross = (record[f"material_sign_{name}"] for name in MEASURED[:2])
    if net is not None and gross is not None:
        record["gross_material_contradiction_to_net"] = net * gross == -1
    return record


def make_pairs(joined, protocol):
    pairs = []
    for layout in protocol["layouts_in_order"]:
        parent = layout.split("_")[0]
        paid_stage = f"paid_{protocol['common_paid_checkpoints_by_parent'][parent]}"
        for family, design in (("strict_pairs_by_parent", "strict"), ("additional_pairs_by_parent", "additional")):
            for cids in protocol[family].get(parent, []):
                a, b = cids
                old_costs = protocol["actual_annotated_outbound_return_costs"][parent]
                if design == "strict":
                    require(old_costs[a] == old_costs[b], f"Declared strict costs do not match at {layout}/{cids}")
                    comparisons = (("arrival", "strict_arrival"), (paid_stage, "strict_common_paid"))
                else:
                    comparisons = (("arrival", "additional_asynchronous_arrival"), (paid_stage, "additional_common_paid"))
                for stage, label in comparisons:
                    pairs.append(pair_row(layout, parent, cids, stage, label, joined, protocol))
    return pairs


def parent_ranks(pairs):
    """Describe pairwise direction by parent without interpreting the count as independent replication."""
    groups = [
        ("strict_arrival", {"strict_arrival"}),
        ("strict_common_paid", {"strict_common_paid"}),
        ("additional_common_paid", {"additional_common_paid"}),
        ("matched_main", {"strict_arrival", "additional_common_paid"}),
    ]
    tallies = []
    for parent in ("warehouse", "gallery"):
        for layer, labels in groups:
            nominated = [row for row in pairs if row["parent"] == parent and row["comparison"] in labels]
            for tier in ("numeric", "material"):
                for metric in MEASURED:
                    comparable = [row for row in nominated if row["comparable_direction"]]
                    directional = [row for row in comparable if row[f"{tier}_sign_{metric}"] in (-1, 1)]
                    for score_type, names in TERMS.items():
                        for mode, key in names.items():
                            judged = Counter()
                            for row in directional:
                                predicted = row[f"sign_{key}"]
                                real = row[f"{tier}_sign_{metric}"]
                                judged["correct" if predicted == real else "tied" if predicted == 0 else "wrong"] += 1
                            tallies.append({
                                "parent": parent, "comparison_layer": layer, "metric_tie_band": tier,
                                "observed_metric": metric, "score_type": score_type, "mode": mode,
                                "n_nominated_pairs": len(nominated), "n_comparable": len(comparable),
                                "n_incomparable": len(nominated) - len(comparable),
                                "n_nondirectional": len(comparable) - len(directional),
                                "n_directional": len(directional),
                                "n_correct": judged["correct"], "n_prediction_tied": judged["tied"],
                                "n_wrong": judged["wrong"],
                            })
    return tallies


def swap_rows(pairs):
    """Keep the two placements of each pair family together for the gate audit."""
    result = []
    options = {(row["parent"], row["layout"], row["candidate_a"], row["candidate_b"], row["comparison"]): row
               for row in pairs}
    for left in pairs:
        parent = left["parent"]
        if left["layout"] != f"{parent}_left" or left["comparison"] == "additional_asynchronous_arrival":
            continue
        lo, hi = left["candidate_a"], left["candidate_b"]
        right = options[parent, f"{parent}_right", lo, hi, left["comparison"]]
        record = {"parent": parent, "candidate_a": lo, "candidate_b": hi, "comparison": left["comparison"]}
        comparable = left["comparable_direction"] and right["comparable_direction"]
        record["both_placements_comparable"] = comparable
        for metric in MEASURED:
            direction_left, direction_right = (row[f"material_sign_{metric}"] for row in (left, right))
            reversed_direction = bool(comparable and direction_left in (-1, 1) and direction_left == -direction_right)
            record[f"{metric}_material_direction_left_placement"] = direction_left
            record[f"{metric}_material_direction_right_placement"] = direction_right
            record[f"{metric}_both_material_and_swap"] = reversed_direction
            for mode in MODES:
                predictions = [row[f"sign_object_{mode}_minus_N"] for row in (left, right)]
                record[f"{metric}_object_{mode}_both_correct"] = (
                    bool(comparable and direction_left in (-1, 1) and direction_right in (-1, 1)
                         and [direction_left, direction_right] == predictions)
                )
        record["either_gross_material_contradiction"] = any(
            row["gross_material_contradiction_to_net"] is True for row in (left, right)
        )
        result.append(record)
    return result


def compare_oracle(joined, pairs, protocol):
    records = []
    eps = protocol["metrics"]["numeric_J_tie"]
    for layout in protocol["layouts_in_order"]:
        parent = layout.split("_")[0]
        original = protocol["original_candidate"][layout]
        common_step = f"paid_{protocol['common_paid_checkpoints_by_parent'][parent]}"
        strict_peer = None
        for row in pairs:
            if row["layout"] != layout or row["comparison"] != "strict_arrival":
                continue
            pair = (row["candidate_a"], row["candidate_b"])
            if original in pair:
                strict_peer = pair[1] if original == pair[0] else pair[0]
        for stage in (common_step, "arrival"):
            rows = {cid: joined[layout, cid]["actual"]["checkpoints"][stage] for cid in protocol["candidates_in_order"]}
            measured = {cid: record["J"] for cid, record in rows.items() if record["status"] == "captured"}
            paid_steps = sorted({record["actual_paid"] for record in rows.values() if record["status"] == "captured"})
            comparability = "shared_paid_step" if len(paid_steps) == 1 else "asynchronous_diagnostic"
            best_j = max(measured.values(), default=None)
            best_ids = [cid for cid, value in measured.items() if abs(value - best_j) <= eps]
            all_four = len(measured) == 4
            records.append({
                "parent": parent, "layout": layout, "checkpoint": stage,
                "comparability": comparability if measured else "unavailable",
                "all_four_captured": all_four,
                "candidate_set_upper_bound": comparability == "shared_paid_step" and all_four,
                "original_candidate_id": original, "original_J": measured.get(original),
                "best_observed_ids": ",".join(map(str, best_ids)), "best_observed_J": best_j,
                "best_observed_minus_original_J": difference(best_j, measured.get(original)),
                "strict_matched_peer_id": strict_peer,
                "strict_matched_peer_minus_original_J": difference(measured.get(strict_peer), measured.get(original)),
                "strict_matched_peer_equal_paid": (
                    strict_peer is not None and original in measured and strict_peer in measured
                    and rows[strict_peer]["actual_paid"] == rows[original]["actual_paid"]
                ),
                **{f"id_{cid}_J": measured.get(cid) for cid in rows},
                **{f"id_{cid}_actual_paid": rows[cid].get("actual_paid") for cid in rows},
            })
    return records


def main(args):
    protocol = read_json(args.protocol)
    archive_receipt = read_json(args.archive_receipt)
    require(archive_receipt["source_commit"] == protocol["source_commit"]
            and digest(args.archive_csv) == archive_receipt["output_sha256"][args.archive_csv.name],
            "Frozen archive prediction hash mismatch")
    measured = read_json(args.measurement)
    joined, wide_rows = collect(args.archive_csv, measured["rows"], protocol)
    pairs = make_pairs(joined, protocol)
    summaries = parent_ranks(pairs)
    swaps = swap_rows(pairs)
    oracle = compare_oracle(joined, pairs, protocol)
    outputs = {
        "per_candidate.csv": wide_rows, "pairwise.csv": pairs, "rank_agreement_by_parent.csv": summaries,
        "swap_audit.csv": swaps, "candidate_oracle.csv": oracle,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_rows(args.output / name, rows)
    write_json(args.output / "ranking_summary.json", {
        "n_parent_layouts": len({row["parent"] for row in wide_rows}),
        "n_placements": len({row["layout"] for row in wide_rows}),
        "n_distinct_branches": len(wide_rows),
        "pairwise": pairs, "rank_agreement_by_parent": summaries, "swap_audit": swaps, "oracle": oracle,
    })
    write_json(args.output / "analysis_receipt.json", {
        "status": "completed_descriptive_rank_diagnostics_no_independent_16_assumption",
        "inputs_sha256": {
            f"rank_script/{Path(__file__).name}": digest(Path(__file__)),
            "rank_script/archive_predictions.py": digest(RANKING / "archive_predictions.py"),
            "protocol": digest(args.protocol), "archived_predictions_csv": digest(args.archive_csv),
            "archived_prediction_receipt": digest(args.archive_receipt), "measurement_json": digest(args.measurement),
        },
        "outputs_sha256": {name: digest(args.output / name) for name in (*outputs, "ranking_summary.json")},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "PROTOCOL.json")
    parser.add_argument("--archive-csv", type=Path, default=RANKING / "archive_predictions.csv")
    parser.add_argument("--archive-receipt", type=Path, default=RANKING / "archive_prediction_receipt.json")
    parser.add_argument("--measurement", type=Path, default=ROOT / "measurement/results/candidate_rows.json")
    parser.add_argument("--output", type=Path, default=RANKING / "results")
    main(parser.parse_args())
