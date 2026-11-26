#!/usr/bin/env python3
"""Verify the first two registered parent pairs before releasing the remainder."""

import csv
from pathlib import Path
import time

from study import ARMS, ROOT, read, sha, verify, write


def main():
    freeze = ROOT / "FORMAL_FREEZE.json"
    sealed = read(freeze)
    phase = ROOT / "execution/formal"
    first = read(phase / "formal_first_receipt.json")
    results = ROOT / "results/formal_first"
    summary_path = results / "summary.json"
    summary = read(summary_path)
    placements = sealed["stage_1"]
    assert first["phase_freeze_sha256"] == sha(freeze) and first["placements"] == placements
    verify(sealed["evaluator_files_sha256"])
    with (results / "episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert {(row["placement"], row["arm"]) for row in rows} == {(p, a) for p in placements for a in ARMS}
    stage_audit = []
    for proof in first["placement_receipts"]:
        assert sha(proof["path"]) == proof["sha256"]
        receipt = read(proof["path"])
        p = proof["placement"]
        opt = read(ROOT/"scenes/pair_audits"/f"{p.rsplit('_',1)[0]}.json")
        assert opt["same_reachable_and_swept_graph"]
        assert opt["front_visible_screen_fraction"] < opt["front_visible_open_fraction"]
        assert opt["hidden_points_seen_by_side_pose"] > 0
        controls = {}
        for name in ARMS:
            arm = receipt["arms"][name]
            if arm["status"] != "verified_native_execution":
                controls[name] = {"status": arm["status"], "technical_reason": arm["technical_reason"]}
                continue
            path = Path(arm["run_directory"])
            assert arm["manifest_sha256"] == sha(path / "manifest.json")
            assert arm["fork_bank_sha256"] == sha(path / "fork_bank.json")
            manifest, bank = (read(path/name) for name in ("manifest.json", "fork_bank.json"))
            controls[name] = {
                "status": arm["status"], "actual_run": arm["execution_arm"],
                "trigger": bank["status"], "trigger_paid": bank.get("paid_before"),
                "selected_route": bank["arms"][name]["route_id"] if bank["arms"] else None,
                "actual_terminal_paid": manifest["paid_actions"], "returned": manifest["returned"],
                "checkpoint_96": manifest["checkpoints"]["paid_96"]["status"],
                "actual_route_event": manifest["route_event"],
            }
        stage_audit.append({"placement":p, "front_visible_open":opt["front_visible_open_fraction"],
                            "front_visible_screen":opt["front_visible_screen_fraction"],
                            "lateral_recoverable_reference_m2":opt["hidden_area_seen_by_side_pose_m2"],
                            "unique_trajectories":receipt["unique_completed_executions"], "arms":controls})
    report = {
        "phase_freeze_sha256": sha(freeze), "stage_receipt_sha256": sha(phase/"formal_first_receipt.json"),
        "placements": placements, "summary_sha256": sha(summary_path),
        "verified_outputs": hashes_for(results), "first_two_units": stage_audit,
        "observed_metric_summary": summary, "parameters_released_unchanged": True,
        "release_unix_s": time.time(),
    }
    write(results/"release.json",report)
    print(f"Stage 1 released with {len(rows)} independently audited assignments and {first['unique_completed_executions']} unique executions.")


def hashes_for(results):
    return {name:sha(results/name) for name in ("episodes.csv","pairs.csv","comparisons.csv","failures.csv","summary.json")}


if __name__ == "__main__":
    main()
