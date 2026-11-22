#!/usr/bin/env python3
"""Evaluate independent branches and cold-start pairs with the unchanged meter."""

from pathlib import Path

from runner import PREVIOUS, ROOT, read, sha
import evaluate as native


def main():
    native.ROOT = ROOT
    summarize = native.contrast_summary
    measure = native.measure_episode

    def evaluate(ev,directory,job,worker,frozen,digest,scene_root,world,metric):
        output = measure(ev,directory,job,worker,frozen,digest,scene_root,world,metric)
        if job.get("branch") and output[0]["status"] == "completed_execution":
            match, proof = (read(directory/name) for name in ("matched_intervention.json", "verified_action_prefix.json"))
            expect = job["branch"]
            assert match["selection_call_id"] == expect["selection_call_id"] and match["source_candidate_pool_exact_match"]
            assert proof["paid"] == expect["paid_before"] and proof["complete_prefix_exact"]
            output[0].update(branch_common_paid=proof["paid"], branch_target=match["selected_pose"],
                             branch_receipt_sha256=sha(directory/"matched_intervention.json"))
        return output

    def contrasts(spec, episodes, phase):
        if phase == "development":
            source = read(PREVIOUS/"results/formal/summary.json")
            historical = {(row["layout"],row["arm"]):row for row in source["episodes"]
                          if row["arm"]=="region_S" and row["layout"] in spec["layouts"]}
            for row in historical.values():
                assert sha(Path(row["run_directory"])/"manifest.json") == row["source_manifest_sha256"]
            episodes |= historical
            native.FACTORIAL = {
                "attribution_once_vs_recorded": {"attribution_once":1,"region_S":-1},
                "outbound_once_vs_recorded": {"outbound_once":1,"region_S":-1},
                "outbound_vs_attribution": {"outbound_once":1,"attribution_once":-1},
            }
        else:
            native.FACTORIAL = {
                "repair_vs_region_v1": {"combined_S":1,"region_S":-1},
                "repair_vs_strict": {"combined_S":1,"F_strict":-1},
                "region_v1_vs_strict": {"region_S":1,"F_strict":-1},
            }
        return summarize(spec,episodes,"formal")

    native.measure_episode, native.contrast_summary = evaluate, contrasts
    native.main()


if __name__ == "__main__":
    main()
