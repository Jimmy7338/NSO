"""Verify the sealed short study, selected continuation and preserved B11 evidence."""

import csv
import hashlib
import json
import math
from pathlib import Path

from decide import decide


ROOT = Path(__file__).resolve().parent
OLD_B11 = ROOT.parent / "paired_unseen_gain_20261120"
FOLLOWUP_B11 = ROOT.parent / "paired_unseen_followup_20261120"


def read(path):
    return json.loads(path.read_text())


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def verify(mapping, relative_to):
    for filename, expected in mapping.items():
        path = relative_to / filename
        require(hash_file(path) == expected, f"Hash mismatch: {path}")
    return len(mapping)


def verify_prior_b11():
    count = verify(read(OLD_B11 / "FINAL_EVIDENCE_RECEIPT.json")["source_files_sha256"], OLD_B11)
    with (FOLLOWUP_B11 / "evaluation/results/layouts.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    layouts = {row["layout"] for row in rows}
    require(len(layouts) == 4, "Expected four B11 paired layouts")
    arms = {}
    for arm in ("G_visibility", "S_visibility", "F_nearest", "F_strict"):
        pairs = [row for row in rows if row["arm"] == arm]
        require(len(pairs) == 4 and {row["layout"] for row in pairs} == layouts, f"Incomplete B11 arm {arm}")
        require(all(row["status"] == "completed_execution" and int(row["budget"]) == 256 for row in pairs),
                f"Unsealed execution or incompatible budget for {arm}")
        arms[arm] = len(pairs)
    heldout = read(FOLLOWUP_B11 / "gain_calibration/heldout_summary.json")
    require(heldout["independent_layout_count"] == 4 and not heldout["changed_route_layouts"],
            "Held-out B11 evidence changed")
    publication = read(FOLLOWUP_B11 / "publication/receipt.json")
    require(publication["original_B11_tree_unchanged"], "Prior B11 tree preservation unverified")
    return {"verified_original_bound_files": count, "four_layout_arm_counts": arms,
            "heldout_layout_count": heldout["independent_layout_count"], "previous_draft_head": publication["head"]}


def main():
    source = read(ROOT / "PROTOCOL_FREEZE.json")
    protocol = read(ROOT / "PROTOCOL.json")
    count = verify(source["files_sha256"], ROOT)
    require(source["source_commit"] == protocol["source_commit"], "Frozen study commit changed")
    short_freeze_path = ROOT / "runner/freeze.json"
    short_runner = read(ROOT / "runner/supervisor_receipt.json")
    require(short_runner["status"] == "completed" and short_runner["freeze_sha256"] == hash_file(short_freeze_path),
            "Short-run supervisor did not validate")
    jobs = short_runner["jobs"]
    require(len(jobs) == 16 and all(job["process_returncode"] == 0 for job in jobs), "Short-run job failure")
    candidates = ROOT / "measurement/results/candidate_rows.json"
    rows = read(candidates)["rows"]
    checkpoints = [point for row in rows for point in row["checkpoints"].values()]
    require(len(rows) == 16 and len(checkpoints) == 48 and all(
        item["status"] == "captured" and item["actual_paid"] == item["requested_paid"] for item in checkpoints
    ), "Missing or invalid short checkpoints")
    require(sum(row["actual_collisions"] for row in rows) == 0, "Short-run collision status changed")
    decision_path = ROOT / "decision.json"
    decision = read(decision_path)
    expected = decide(rows, protocol)
    require(all(decision[key] == value for key, value in json.loads(json.dumps(expected)).items()),
            "Independent gate replay differs")
    require(decision["input_sha256"] == {"protocol": hash_file(ROOT / "PROTOCOL.json"),
                                        "measurement": hash_file(candidates)}, "Decision input changed")
    require(decision["branch"] == "B", "This study audit expects the selected continuation branch B")
    branch_path = ROOT / "runner/branch_b/freeze.json"
    branch_files = verify(read(branch_path)["files_sha256"], ROOT)
    terminal_path = ROOT / "measurement/results/branch_b_continuation.json"
    terminal = read(terminal_path)
    require(terminal["source_b_decision_sha256"] == hash_file(decision_path)
            and terminal["source_short_measurements_sha256"] == hash_file(candidates)
            and terminal["branch_freeze_sha256"] == hash_file(branch_path), "Continuation provenance changed")
    require(terminal["common_paid_terminal_comparison"] and terminal["declared_cap"] == 256
            and len(terminal["rows"]) == 2 and all(row["actual_paid"] == terminal["declared_cap"]
                                                   for row in terminal["rows"]), "Terminal budget mismatch")
    ranking = ROOT / "ranking/results"
    rank_count = verify(read(ranking / "analysis_receipt.json")["outputs_sha256"], ranking)
    figures = verify(read(ranking / "figure_receipt.json")["outputs_sha256"], ranking)
    require(all(math.isclose(row["terminal"]["J"], row["terminal"]["C_free"] * row["terminal"]["F1_scene_05cm"],
                             rel_tol=0.0, abs_tol=1e-12) for row in terminal["rows"]), "Terminal J identity changed")
    audit = {
        "status": "verified", "source_commit": source["source_commit"], "frozen_source_files": count,
        "source_protocol_sha256": hash_file(ROOT / "PROTOCOL.json"), "parent_layouts": 2,
        "short_rollouts": len(rows), "exact_snapshots": len(checkpoints),
        "short_actions": sum(row["actual_paid"] for row in rows), "decision": decision["branch"],
        "decision_sha256": hash_file(decision_path), "branch_frozen_files": branch_files,
        "branch_terminal_data_sha256": hash_file(terminal_path),
        "branch_terminal_paid": [row["actual_paid"] for row in terminal["rows"]],
        "verified_ranking_outputs": rank_count, "verified_figure_outputs": figures,
        "earlier_user_followup": verify_prior_b11(),
        "new_weight_calibration_runs": 0, "new_confirmatory_large_scene_runs": 0,
    }
    (ROOT / "completion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
