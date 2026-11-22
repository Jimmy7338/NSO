#!/usr/bin/env python3
"""Run an old-prefix intervention or the repaired global selector end to end."""

import argparse
from pathlib import Path
import sys

from score import rerank


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parents[2] / "region_structure_floor_20261122"
ARM_MODES = {"attribution_once": ("S", True), "outbound_once": ("S", True),
             "region_S": ("S", True), "combined_S": ("S", True), "F_strict": (None, False)}
sys.path.insert(0, str(PREVIOUS))
from regression import read, sha, write


def main():
    import formal_runner
    import regional_adapter
    native = formal_runner.runner
    original_run, original_install = native.run, regional_adapter.install

    def run(args, frozen):
        job = next(j for j in frozen["jobs"] if j["layout"] == args.layout and j["arm"] == args.arm)
        if not job.get("repair"):
            return original_run(args, frozen)
        branch = job.get("branch")
        old_file = Path(branch["source_run"]) if branch else None
        old_calls = read(old_file / "regional_choices.json") if old_file else []
        old_global_rows = {row["call_id"]: row for row in read(old_file / "global_choices.json")} if old_file else {}
        old_local_rows = {row["selection_call_id"]: row for row in read(old_file / "local_choices.json")} if old_file else {}
        triggered = []

        def install(components, runtime, planner):
            from nso.cpu_sensor_contract_v10 import json_value

            original_rank = planner.rank
            backend = components._cpu_backend
            original_native_scores = backend._scores_v10_1
            pending = None

            def native_scores(state, routes, attempted):
                nonlocal pending
                if pending is not None and pending[0] is state and pending[1] is routes:
                    output = pending[2]
                    pending = None
                    return output
                return original_native_scores(state, routes, attempted)

            def rank(belief, pose, paid, resolution, original, added):
                nonlocal pending
                chosen, audit = original_rank(belief, pose, paid, resolution, original, added)
                if not audit["ranking"] or branch and (paid != branch["paid_before"] or triggered):
                    return chosen, audit
                call = backend.calls[-1]
                state = backend.scenes[0]
                observed = call["outputs"]
                if branch:
                    old = next(row for row in old_calls if row["selection_call_id"] == branch["selection_call_id"])
                    assert call["call_id"] == old["selection_call_id"] and list(pose) == old["start_pose"]
                    assert json_value(audit) == old["ranking"], "Archived regional candidate pool/state has not been reproduced."
                    assert call["output_sha256"] == old["native_output_sha256"]
                    local = old_local_rows[call["call_id"]]
                    assert state["packet"].sha256() == local["packet_sha256"]
                fresh = list(observed["score_audit"])
                if added:
                    attempted = state["gain"].attempted_camera_mask(state["mapper"], state["ledger"].planning_camera_poses())
                    new_scores = original_native_scores(state, added, attempted)
                    fresh.extend(new_scores[1])
                    pending = (state, added, new_scores)
                flags = job["repair"]
                selected, ranking = rerank(planner, audit, [*original, *added], fresh, resolution, **flags)
                explanation = {"score_repair": flags, "source_candidate_count": len(audit["ranking"]),
                               "v1_chosen": audit["chosen"], "actual_paid": paid, "ranking": ranking}
                if branch:
                    assert len(fresh) == len(old_global_rows[call["call_id"]]["outputs"]["score_audit"])
                    assert selected["pose"] == branch["expected_pose"] and selected["pose"] != chosen["pose"]
                    write(args.out / "matched_intervention.json", explanation | {
                        "selection_call_id": call["call_id"], "source_run": str(old_file), "selected_pose": selected["pose"],
                        "source_candidate_pool_exact_match": True, "source_native_score_sha256": call["output_sha256"],
                        "common_fork_packet_sha256": state["packet"].sha256(),
                    })
                    triggered.append(call["call_id"])
                return selected, audit | {"ranking": ranking, "chosen": ranking[0],
                                          "attribution_path_repair": explanation}

            backend._scores_v10_1 = native_scores
            planner.rank = rank
            return original_install(components, runtime, planner)

        regional_adapter.install = install
        try:
            original_run(args, frozen)
        finally:
            regional_adapter.install = original_install
        if branch:
            assert len(triggered) == 1
            prefix = branch["paid_before"]
            original_steps = read(old_file / "steps.json")[:prefix]
            new_steps = read(args.out / "steps.json")[:prefix]
            assert len(original_steps) == len(new_steps) == prefix
            for old, new in zip(original_steps, new_steps, strict=True):
                assert {k: v for k, v in old.items() if k != "decision_wall_s"} == {
                    k: v for k, v in new.items() if k != "decision_wall_s"}
            assert sha(old_file / "initial_raw_rgbd_scan_sha256.json") == sha(args.out / "initial_raw_rgbd_scan_sha256.json")
            write(args.out / "verified_action_prefix.json", {"paid": prefix, "complete_prefix_exact": True,
                  "original_manifest_sha256": sha(old_file / "manifest.json"), "source_run": str(old_file)})

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("layout")
    parser.add_argument("arm")
    parser.add_argument("--freeze", required=True, type=Path)
    args, _ = parser.parse_known_args()
    frozen = read(args.freeze)
    native.ROOT = ROOT
    native.ARM_MODES = {j["arm"]: (j["prior"], j["regional_enabled"]) for j in frozen["jobs"]}
    native.run = run
    formal_runner.start()


if __name__ == "__main__":
    main()
