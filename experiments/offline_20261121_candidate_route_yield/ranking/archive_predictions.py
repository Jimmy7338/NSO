"""Extract and verify the archived predictions without reading any new measurements."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODES = ("G", "S", "X")


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")


def write_rows(path, rows):
    columns = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def close(a, b):
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)


def get_intervention(attempt, arm, expected, weights):
    manifest = read_json(attempt / "artifact_manifest.json")
    require(manifest["status"] == "completed_execution", str(attempt))
    for name in ("intervention.json", "choices.json"):
        require(digest(attempt / name) == manifest["files_sha256"][name], f"Manifest mismatch for {attempt / name}")
    record = read_json(attempt / "intervention.json")["intervention"]
    choices = read_json(attempt / "choices.json")
    plan, = (
        choice["outputs"] for choice in choices
        if choice["outputs"]["plan_number"] == record["native_plan_number"]
    )
    candidates, scores = plan["candidates"], plan["scores"]
    require(canonical(candidates) == record["candidate_set_sha256"], f"Candidate hash mismatch in {attempt}")
    require(canonical(scores) == record["score_set_sha256"], f"Score hash mismatch in {attempt}")
    require(
        record["qualified"] and record["paid_before"] == 0
        and record["mode"] == arm == plan["mode"]
        and record["chosen_candidate_id"] == expected,
        f"Original selection contract mismatch in {attempt}",
    )
    require(
        all(row["X"] == row["S"][::-1] and row["G"] == [0.5, 0.5] for row in record["shape_priors"]),
        f"Swapped priors mismatch in {attempt}",
    )
    require(plan["selected"] == record["chosen_native_option"], f"Archived selected option mismatch in {attempt}")
    summaries = {value["candidate_id"]: value for value in record["candidate_summaries"]}
    require(len(summaries) == len(candidates) == 4, f"Candidate count mismatch in {attempt}")
    audit = {row["candidate_id"]: row for row in plan["score_audit"]}
    for mode in MODES:
        order = sorted(range(4), key=lambda i: (-scores[mode][i], candidates[i]["cost"], candidates[i]["candidate_id"]))
        require(candidates[order[0]]["candidate_id"] == expected, f"Original rank 1 mismatch in {attempt}")
        for rank, index in enumerate(order, 1):
            raw = candidates[index]
            cid = raw["candidate_id"]
            row = summaries[cid]
            require(row["ranks"][mode] == rank and row["scores"][mode] == scores[mode][index],
                    f"Rank/score mismatch for {attempt}/{cid}/{mode}")
            require(row["common_score"] == scores["N"][index] == audit[cid]["common_total_proxy"],
                    f"Common score mismatch for {attempt}/{cid}")
            require(
                row["outbound_cost"] == raw["outbound_cost"] and row["reserved_return_cost"] == raw["return_cost"]
                and row["cost"] == raw["cost"] == raw["outbound_cost"] + raw["return_cost"]
                and raw["outbound_cost"] == len(raw["outbound_actions"])
                and raw["return_cost"] == len(raw["return_actions"]),
                f"Route costs disagree for {attempt}/{cid}",
            )
            require(
                audit[cid]["template_gain_coefficient"] == weights
                and row["template_area_m2"][mode] == audit[cid]["template_expected_visible_area_m2"][mode]
                and close(row["scores"][mode] - row["common_score"], weights * row["template_area_m2"][mode]),
                f"Template component mismatch for {attempt}/{cid}/{mode}",
            )
    return record, {raw["candidate_id"]: raw for raw in candidates}, audit


def extract(archive, protocol, output):
    source_commit = protocol["source_commit"]
    published = read_json(archive / "publication/receipt.json")
    require(published["head"] == source_commit, "Archive publication receipt points to a different source commit")
    completion = read_json(archive / "evaluation/completion_audit.json")
    require(completion["supported_parents"] == 0 and completion["controlled_attempts"] == 12,
            "Prior controlled experiment summary mismatch")
    verified_paths = {"publication/receipt.json", "evaluation/completion_audit.json"}
    for path, expected in completion["frozen_hashes"].items():
        require(digest(archive / path) == expected, f"Prior completion audit mismatch at {path}")
        verified_paths.add(path)
    source_runner = archive / "mechanism/runner"
    optical_index = read_json(source_runner / "posthoc_diagnostic_index.json")
    require(optical_index["actor_freeze_sha256"] == completion["frozen_hashes"]["mechanism/runner/freeze.json"],
            "Optical diagnostics belong to a different actor freeze")
    verified_paths.add("mechanism/runner/posthoc_diagnostic_index.json")
    for path, expected in optical_index["source_sha256"].items():
        require(digest(source_runner / path) == expected, f"Frozen optical source mismatch at {path}")
        verified_paths.add(f"mechanism/runner/{path}")
    with (source_runner / "first_decision_diagnostics.csv").open(newline="") as stream:
        lookup = {(row["layout"], int(row["candidate_id"])): row for row in csv.DictReader(stream)}
    rows = []
    unique_records = {}
    for layout in protocol["layouts_in_order"]:
        parent, placement = layout.split("_")
        expected = protocol["original_candidate"][layout]
        arms = {}
        for arm in MODES:
            attempt = source_runner / "layouts" / layout / "runs" / arm / "attempt_001"
            for name in ("intervention.json", "choices.json", "artifact_manifest.json"):
                verified_paths.add(str((attempt / name).relative_to(archive)))
            arms[arm] = get_intervention(attempt, arm, expected, weights=1.0)
        record, native, audit = arms["G"]
        shared = ("packet_sha256", "belief_sha256", "candidate_set_sha256", "score_set_sha256", "candidate_summaries")
        for arm in MODES:
            require(all(record[key] == arms[arm][0][key] for key in shared),
                    f"The original arms did not see the same first decision in {layout}")
        unique_records[layout] = record
        for row in sorted(record["candidate_summaries"], key=lambda value: value["candidate_id"]):
            cid = row["candidate_id"]
            optical = lookup[layout, cid]
            costs = protocol["actual_annotated_outbound_return_costs"][parent][cid]
            require(
                costs == [row["outbound_cost"], row["reserved_return_cost"]]
                == [int(optical["outbound"]), int(optical["reserved_return"])],
                f"Diagnostic/predeclared costs mismatch for {layout}/{cid}",
            )
            require(optical["informative_slot"] == placement and optical["native_role"] == native[cid]["group"]
                    and (optical["S_selected"] == "True") == (cid == expected)
                    and float(optical["N"]) == row["common_score"]
                    and all(float(optical[mode]) == row["scores"][mode] for mode in MODES),
                    f"Optical archives disagree with score receipt for {layout}/{cid}")
            scores = row["scores"]
            common = row["common_score"]
            rows.append({
                "parent": parent, "layout": layout, "placement": placement, "candidate_id": cid,
                "native_role": native[cid]["group"], "native_side": row["side"],
                "informative_optical_side": optical["informative_slot"],
                "optical_gate_crossing_side": optical["candidate_crossing"],
                "new_interior_optical_bins": int(optical["new_interior_optical_bins"]),
                "max_north_m": float(optical["max_north_m"]),
                "observed_safe_left_crossings": int(optical["observed_safe_left_crossings"]),
                "observed_safe_right_crossings": int(optical["observed_safe_right_crossings"]),
                "nominal_outbound": costs[0], "nominal_reserved_return": costs[1], "nominal_total": row["cost"],
                "original_selected": cid == expected, "native_plan_number": record["native_plan_number"],
                "nominal_target_pose": json.dumps(native[cid]["pose"], separators=(",", ":")),
                "N": common, **{f"total_{mode}": scores[mode] for mode in MODES},
                **{f"native_rank_{mode}": row["ranks"][mode] for mode in MODES},
                **{f"object_{mode}_minus_N": scores[mode] - common for mode in MODES},
                **{f"template_area_{mode}_m2": row["template_area_m2"][mode] for mode in MODES},
                "category_S_minus_G": scores["S"] - scores["G"], "category_X_minus_G": scores["X"] - scores["G"],
                "template_gain_coefficient": audit[cid]["template_gain_coefficient"],
                "common_coverage_proxy": audit[cid]["coverage_only_proxy"],
                "common_quality_proxy": audit[cid]["quality_total_proxy"],
                "candidate_set_sha256": record["candidate_set_sha256"], "score_set_sha256": record["score_set_sha256"],
                "original_packet_sha256": record["packet_sha256"], "original_belief_sha256": record["belief_sha256"],
            })
    require(len(rows) == len(lookup) == 16, "Expected 16 distinct archival candidates")
    parents = sorted({row["parent"] for row in rows})
    for parent in parents:
        left, right = (unique_records[f"{parent}_{side}"] for side in ("left", "right"))
        require(left["candidate_set_sha256"] == right["candidate_set_sha256"], f"Candidate bank changed in {parent}")
        for lhs, rhs in zip(left["candidate_summaries"], right["candidate_summaries"]):
            require(lhs["candidate_id"] == rhs["candidate_id"] and lhs["common_score"] == rhs["common_score"]
                    and lhs["scores"]["G"] == rhs["scores"]["G"]
                    and lhs["scores"]["S"] == rhs["scores"]["X"] and lhs["scores"]["X"] == rhs["scores"]["S"],
                    f"Score parity failed for swapped placement in {parent}")
    output.mkdir(parents=True, exist_ok=True)
    out = output / "archive_predictions.csv"
    write_rows(out, rows)
    write_json(output / "archive_prediction_receipt.json", {
        "status": "verified_archived_first_decision_without_new_outcomes",
        "source_commit": source_commit, "published_evidence_tree": published["evidence_tree"],
        "archive_source_files_sha256": {path: digest(archive / path) for path in sorted(verified_paths)},
        "output_sha256": {out.name: digest(out)},
        "n_archived_controlled_runs": 12, "n_native_candidates": len(rows), "n_parent_layouts": len(parents),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "PROTOCOL.json")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    extract(args.archive or Path(protocol["previous_local_archive"]), protocol, args.output)
