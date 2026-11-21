"""Verify the optional implementation's autonomous development stop and freeze receipts."""

import csv
import json
from pathlib import Path

from report import ROOT, read, sha, report_rows


def require(condition, cause):
    if not condition:
        raise ValueError(cause)


def verify(entries, parent):
    for name, expected in entries.items():
        require(sha(parent / name) == expected, f"Artifact changed: {parent / name}")
    return len(entries)


def replayed_actions(trace):
    return [(r["step"], r["action"], r["position"], r["heading"], r["packet_sha256"], r["packet_chain_sha256"])
            for r in trace]


def verify_batch(protocol, freeze_path):
    freeze = read(freeze_path)
    jobs = read(ROOT / "runner/development_supervisor_receipt.json")
    config = protocol["development"]
    schedule = [(layout, arm) for layout in config["layouts"] for arm in config["modes"]]
    require(jobs["status"] == "completed" and jobs["runner_freeze_sha256"] == sha(freeze_path) and
            [(r["layout"], r["arm"]) for r in freeze["schedule"]] == schedule and
            [(r["layout"], r["arm"]) for r in jobs["jobs"]] == schedule,
            "Executed schedule does not match the frozen eight declared free-choice episodes")
    sealed = verify(freeze["files_sha256"], ROOT)
    records, index = {}, []
    for row in jobs["jobs"]:
        layout, arm = row["layout"], row["arm"]
        require(row["process_returncode"] == 0 and row["log_sha256"] == sha(ROOT / f"runner/logs/development/{layout}_{arm}.log"),
                f"Supervisor/motion failed in {layout}/{arm}")
        path = ROOT / f"runner/runs/development/{layout}/{arm}/manifest.json"
        data = read(path)
        verify(data["files_sha256"], path.parent)
        trace = read(path.parent / "steps.json")
        require(data["actual_closed_loop"] and not data["forced_route_or_goal"] and not data["evaluator_inputs_read_by_actor"] and
                data["global_mode"] == "G" and data["local_mode"] == arm[-1] and
                data["returned"] and data["collisions"] == 0 and
                len(trace) == data["paid_actions"] <= config["paid_action_budget"],
                f"Run did not deliver the declared safe autonomous experiment at {layout}/{arm}")
        records[layout, arm] = (data, replayed_actions(trace))
        index.append({"manifest": str(path.relative_to(ROOT)), "manifest_sha256": sha(path),
                      "files_sha256": data["files_sha256"]})
    for layout in config["layouts"]:
        (g, g_actions), (s, s_actions) = (records[layout, arm] for arm in config["modes"])
        require(g_actions == s_actions and g["initial_packet_sha256"] == s["initial_packet_sha256"] and
                g["files_sha256"]["terminal/native_mesh.ply"] == s["files_sha256"]["terminal/native_mesh.ply"],
                f"G/S pair differs in an executed action, actual observation or terminal mesh at {layout}")
    return index, sealed, sum(len(trace) for _, trace in records.values())


def main():
    protocol_path = ROOT / "PROTOCOL.json"
    protocol = read(protocol_path)
    source = Path(protocol["source_local"])
    require(read(source / "publication/receipt.json")["head"] == protocol["source_commit"], "Wrong evidence source")
    freeze_path = ROOT / "runner/development_freeze.json"
    raw, sealed, paid = verify_batch(protocol, freeze_path)
    analysis = ROOT / "measurement/results/development"
    episodes, choice = analysis / "episodes.json", analysis / "choice_audit.json"
    decision = read(ROOT / "measurement/development_decision.json")
    require(decision["protocol_sha256"] == sha(protocol_path) and decision["runner_freeze_sha256"] == sha(freeze_path)
            and decision["episodes_sha256"] == sha(episodes) and decision["actual_choice_audit_sha256"] == sha(choice),
            "Decision no longer binds the reproduced inputs")
    require(decision["all_eight_native_safe_return"] and decision["admitted_non_native_route"]
            and not decision["advance_holdout"] and not decision["category_caused_executed_difference"],
            "Claimed development stop differs from independent release decision")
    require(not (ROOT / "runner/runs/holdout").exists() and not (ROOT / "runner/holdout_freeze.json").exists(),
            "A held-out policy appeared despite the stop condition")
    scenery = []
    for name in ("freeze.json", "evaluator_freeze.json"):
        path = ROOT / "holdout" / name
        receipt = read(path)
        require(receipt["actors_run"] == 0, "A preflight used policy outcomes")
        verify(receipt["files_sha256"], ROOT)
        scenery.append({"receipt": str(path.relative_to(ROOT)), "receipt_sha256": sha(path)})
    primary, _ = report_rows(read(episodes)["rows"], read(choice)["rows"], protocol["development"])
    require(all(row["newS_minus_newG_J_pp"] == 0.0 for row in primary), "Terminal J differs within a paired layout")
    diagnostic_path = ROOT / "diagnostics/development_availability_receipt.json"
    diagnostic = read(diagnostic_path)
    verify(diagnostic["sha256_inputs"], ROOT)
    verify(diagnostic["sha256_outputs"], diagnostic_path.parent)
    with (diagnostic_path.parent / "development_availability_by_arm.csv").open(newline="") as stream:
        events = [row for row in csv.DictReader(stream) if row["arm"] == "newG"]
    key_counts = ("local_decisions", "alternate_admitted", "admitted_and_category_changed_pair_margin",
                  "actual_paid_change_from_native", "causal_flip_in_same_recorded_state")
    counts = {name: sum(int(row[name]) for row in events) for name in key_counts}
    require(counts["alternate_admitted"] > 0 and counts["causal_flip_in_same_recorded_state"] == 0,
            "Availability and independent category stop disagree")
    stage = ROOT / "results/development"
    report = read(stage / "REPORT_RECEIPT.json")
    verify(report["input_sha256"], ROOT)
    verify(report["output_sha256"], stage)
    audit = {"status": "verified_development_stop_no_actual_category_effect", "source_commit": protocol["source_commit"],
             "native_frozen_actor_inputs": sealed, "runs": len(raw), "paid_actions": paid, "returned_runs": len(raw),
             "simulated_collisions": 0, "identical_action_observation_and_mesh_pairs": len(primary), "holdout_actor_runs": 0,
             "observed_counts_per_paired_sequence": counts, "protocol_sha256": sha(protocol_path),
             "frozen_development_decision_sha256": sha(ROOT / "measurement/development_decision.json"),
             "paired_figure_and_csv_receipt_sha256": sha(stage / "REPORT_RECEIPT.json"),
             "manuscript_passage_sha256": sha(ROOT / "RESULTS_ZH.md")}
    evidence = {"status": "verified", "payload_scope": "Eight development runs, four static held-out scenes and result receipts",
                "run_files_rooted_at_each_manifest": raw, "source_only_scene_receipts": scenery,
                "report_receipt": "results/development/REPORT_RECEIPT.json",
                "raw_native_mesh_and_observed_belief_location": str(ROOT / "runner/runs/development")}
    for filename, content in (("completion_audit.json", audit), ("EVIDENCE_INDEX.json", evidence)):
        (ROOT / filename).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
