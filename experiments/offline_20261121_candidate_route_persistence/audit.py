"""Check the five-route execution, archived comparators and paired publication evidence."""

import csv
import json
import math
from pathlib import Path

from summarize import ROOT, METRICS, read, sha, summarize


def require(condition, failure):
    if not condition:
        raise ValueError(failure)


def verify_files(mapping, parent):
    for name, expected in mapping.items():
        require(sha(parent / name) == expected, f"Evidence changed: {parent / name}")
    return len(mapping)


def sealed_artifact(path):
    manifest = read(path)
    verify_files(manifest["files_sha256"], path.parent)
    return manifest, {"manifest": str(path.relative_to(ROOT)), "manifest_sha256": sha(path),
                      "files_sha256": manifest["files_sha256"]}


def verify_new(protocol):
    freeze_path = ROOT / "runner/freeze.json"
    freeze = read(freeze_path)
    receipt = read(ROOT / "runner/supervisor_receipt.json")
    require(receipt["runner_freeze_sha256"] == sha(freeze_path) and receipt["status"] == "completed",
            "Runner has no completed freeze-bound receipt")
    count = verify_files(freeze["files_sha256"], ROOT)
    expected = {(layout, route) for layout, ids in protocol["new_routes"].items() for route in ids}
    jobs = {(item["layout"], item["candidate_id"]): item for item in receipt["jobs"]}
    require(len(receipt["jobs"]) == 5 and jobs.keys() == expected, "The exact five jobs were not run")
    manifests = {}
    artifacts = []
    for (layout, route), job in jobs.items():
        path = ROOT / f"runner/runs/{layout}/candidate_{route}"
        require(job["process_returncode"] == 0 and sha(ROOT / f"runner/logs/{layout}_{route}.log") == job["log_sha256"],
                f"Invalid route job {layout}/{route}")
        work, index = sealed_artifact(path / "manifest.json")
        recovered = read(path / "checkpoint_recovery.json")
        prefix = [38, 58] if layout.startswith("gallery") else [30]
        recovered_paid = {item["actual_paid"] for item in recovered["snapshot_references"].values()
                          if item["status"] == "restored_exact"}
        require(recovered["before_first_new_action"] and set(prefix) <= recovered_paid
                and recovered["matched_short_prefix_actions"] == prefix[-1], "Missing exact recovery before release")
        require(work["source_commit"] == protocol["source_commit"] and work["neutral_g_continuation"]
                and not work["reused_exact_archive"] and not work["evaluator_inputs_read_by_actor"]
                and work["native_budget"] == protocol["native_total_paid_cap"]
                and work["actual_new_actions"] == work["paid_actions"] - prefix[-1], "Execution departed from protocol")
        require(len(read(path / "steps.json")) == work["paid_actions"], f"Incomplete paid action trace: {path}")
        manifests[layout, route] = work
        artifacts.append(index)
    return manifests, artifacts, count


def verify_archive_recovery():
    index = []
    previous = ROOT.parent / "semantic_large_scene_20261121/mechanism/runner/layouts"
    for layout in ("gallery_left", "gallery_right"):
        manifest, artifact = sealed_artifact(ROOT / f"measurement/archive_recovery/{layout}/candidate_1/manifest.json")
        origin = previous / layout / "runs/G/attempt_001"
        require(manifest["new_policy_actions"] == manifest["planning_calls"] == manifest["scoring_calls"] == 0,
                f"Archive recovery ran a new policy in {layout}")
        require(manifest["archived_actions_replayed"] == 256 and
                all(value in manifest["captured"] for value in ("paid_38", "paid_58", "paid_128")),
                f"Required archived checkpoints missing in {layout}")
        require(manifest["terminal_mesh_sha256"] == sha(origin / "final_tsdf_mesh.ply"),
                f"Archive terminal mesh mismatch in {layout}")
        index.append(artifact)
    return index


def main():
    protocol = read(ROOT / "PROTOCOL.json")
    frozen = read(ROOT / "PROTOCOL_FREEZE.json")
    bound_files = verify_files(frozen["files_sha256"], ROOT)
    manifests, raw, run_frozen = verify_new(protocol)
    recovered = verify_archive_recovery()
    result = ROOT / "measurement/results"
    checkpoints, terminals = (read(result / f"{name}.json")["rows"] for name in ("checkpoints", "terminal"))
    captured = {(row["layout"], row["candidate_id"], row["checkpoint"]): row for row in checkpoints
                if row["status"] == "captured"}
    for terminal in terminals:
        key = terminal["layout"], terminal["candidate_id"]
        checkpoints_needed = [38, 58, 128] if key[0].startswith("gallery") else [30]
        for paid in checkpoints_needed:
            require((*key, f"paid_{paid}") in captured, f"A required snapshot is missing at {key}/{paid}")
        if key in manifests:
            source = manifests[key]
            require((source["paid_actions"], source["returned"], source["collisions"]) ==
                    (terminal["actual_paid"], terminal["returned"], terminal["collisions"]), f"Terminal execution mismatch at {key}")
    old = Path(protocol["source_local"]) / "measurement/results/candidate_rows.json"
    archived = {(row["layout"], row["candidate_id"]): row["checkpoints"] for row in read(old)["rows"]}
    for key in manifests:
        if key[0].startswith("gallery"):
            for paid, archived_name in ((38, "arrival"), (58, "paid_58")):
                current = captured[*key, f"paid_{paid}"]
                earlier = archived[key][archived_name]
                require(all(math.isclose(current[m], earlier[m], abs_tol=1e-12, rel_tol=0) for m in METRICS),
                        f"Recomputed archive metrics changed at {key}/{paid}")
    interval = read(ROOT / "interval/interval_receipt.json")
    interval_count = verify_files(interval["sha256_outputs"], ROOT / "interval")
    require(not interval["actor_started"] and interval["branch_count"] == 4, "Wrong archived-interval scope")
    decision = read(ROOT / "results/decision.json")
    _, recomputed = summarize(protocol, checkpoints, terminals)
    require(all(decision[key] == value for key, value in recomputed.items()), "Pair comparison does not reproduce")
    with (ROOT / "interval/neutral_move_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 4 and all(row["matches_archived_neutral_outbound_prefix"] == "True" for row in rows),
            "Missing verified gallery trajectories")
    masks = {row["presence_path"]: row["presence_sha256"] for row in captured.values()}
    verify_files(masks, result)
    evidence = {"status": "verified", "protocol_input": "PROTOCOL_FREEZE.json", "new_runs": raw,
                "archived_gallery_sensor_recoveries": recovered,
                "measurement_presence_masks_relative_to_measurement_results": masks,
                "note": "Relative file hashes are rooted at each manifest's parent. Raw meshes, maps and mask arrays are retained on the original host."}
    audit = {
        "status": "verified", "source_commit": protocol["source_commit"], "source_files_verified": bound_files,
        "execution_input_files_verified": run_frozen, "new_runs": len(manifests),
        "new_actions": sum(row["actual_new_actions"] for row in manifests.values()),
        "verified_recovered_prefix_actions": sum(row["matched_short_prefix_actions"] for row in manifests.values()),
        "archived_full_runs": len(terminals) - len(manifests), "archive_no_planning_recoveries": len(recovered),
        "captured_checkpoints": len(captured), "unavailable_optional_checkpoints": len(checkpoints) - len(captured),
        "gallery_38_58_outputs_verified": interval_count, "terminal_paid": [row["actual_paid"] for row in terminals],
        "returned_count": sum(row["returned"] for row in terminals),
        "collisions": sum(row["collisions"] for row in terminals),
        "gallery_two_sides_trigger": recomputed["preregistered_online_score_proposal_triggered"],
        "new_weight_sweep_or_scene_runs": 0,
        "result_hashes": {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / "results").iterdir())},
        "checkpoint_measurements_sha256": sha(result / "checkpoints.json"),
        "terminal_measurements_sha256": sha(result / "terminal.json"), "manuscript_passage_sha256": sha(ROOT / "RESULTS_ZH.md"),
    }
    for name, data in (("EVIDENCE_INDEX.json", evidence), ("completion_audit.json", audit)):
        (ROOT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
