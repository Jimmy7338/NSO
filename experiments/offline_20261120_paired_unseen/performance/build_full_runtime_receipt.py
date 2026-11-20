"""Post hoc runtime accounting; never imports an actor or executes observations."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ("chain_a", "cycle_a", "chain_b", "cycle_b")
ARMS = ("S_visibility", "S_acquisition")


def read(path):
    return json.loads(Path(path).read_text())


def resource(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def inspect_run(layout, arm):
    folder = ROOT / "execution/layouts" / layout / "runs" / arm / "attempt_001"
    if not (folder / "result.json").exists():
        failure = folder / "failure.json"
        assert failure.exists(), f"Run has not reached a terminal outcome: {folder}"
        return dict(layout=layout, arm=arm, status="failed_or_interrupted",
                    failure=read(failure), sources=[resource(failure)])
    result = read(folder / "result.json")
    assert result["source_and_adapter_hashes_unchanged"]
    binding = read(folder / "B11_execution_binding.json")
    manifest = read(folder / "artifact_manifest.json")
    assert binding["full_posterior_same_quadrature"]
    assert binding["short_cost_band_used"] is False
    assert binding["execution_freeze"] == resource(ROOT / "execution/freeze.json")
    for name, expected in manifest["files_sha256"].items():
        assert resource(folder / name)["sha256"] == expected, (folder, name)
    steps = read(folder / "steps.json")
    choices = read(folder / "choices.json")
    audits = read(folder / "template_audit.json")
    assert len(audits) == len(choices) == result["global_planning_calls"]
    counts = {}
    decisions = []
    fallback_records = []
    for index, (choice, audit) in enumerate(zip(choices, audits)):
        candidates = choice["outputs"]["candidates"]
        ids = [c["candidate_id"] for c in candidates]
        assert ids == [r["candidate_id"] for r in audit["routes"]]
        assert len(ids) == audit["candidate_count"]
        assert audit["original_candidate_provider"]
        assert audit["full_policy_cost_band_applied"] is False
        assert audit["conditional_mean_or_fine_basis"] is False
        assert audit["shared_G_S_full_basis"]
        assert audit["extent_mode"] == "uncertainty"
        assert audit["sigma_m"] == .01 and audit["structural_temperature"] == 16.
        g = audit["model_audits"]["G"]
        s = audit["model_audits"]["S"]
        assert g["raw_template_points"] == s["raw_template_points"]
        assert g["applied_global_point_cap"] is None
        assert s["applied_global_point_cap"] is None
        row = dict(decision_index=index, mapper_frames=audit["mapper_frames"],
                   candidate_ids=ids, original_point_count=g["raw_template_points"])
        acquisition = audit["acquisition_diagnostics"]
        if arm == "S_visibility":
            assert acquisition is None
        elif acquisition is None:
            # The frozen runtime need not instantiate C when there is no route.
            assert not candidates
            row.update(unique_outbound_routes=0,
                       evaluated_positive_hypothesis_route_terms=0,
                       acquisition_not_instantiated_reason="no_candidates")
        else:
            assert acquisition["original_basis_and_posterior_unchanged"]
            assert acquisition["all_past_fused"]
            assert acquisition["no_positive_probability_pruning"]
            assert acquisition["geometry_collapse"] is False
            assert acquisition["source_geometry_read"] is False
            assert acquisition["policy_candidates_changed"] is False
            assert acquisition["acquired_frames"] == audit["mapper_frames"]
            assert len(g["objects"]) == len(s["objects"])
            positive = zero = 0
            for go, so in zip(g["objects"], s["objects"]):
                assert go["support_track"] == so["support_track"]
                ga, sa = go["joint_probability_mass"], so["joint_probability_mass"]
                assert len(ga) == len(sa)
                for grow, srow in zip(ga, sa):
                    assert len(grow) == len(srow) == 2
                    for gm, sm in zip(grow, srow):
                        assert math.isfinite(gm) and math.isfinite(sm)
                        assert gm >= 0 and sm >= 0
                        if gm > 0 or sm > 0:
                            positive += 1
                        else:
                            zero += 1
            unique_routes = {tuple(tuple(p) for p in c["outbound_states"][1:])
                             for c in candidates}
            current = acquisition["counts"]
            assert positive + zero == acquisition["full_particle_terms"]
            assert current["hypothesis_route_terms"] == positive * len(unique_routes)
            assert current["zero_mass_terms_skipped"] == zero * len(unique_routes)
            assert current["computational_fallbacks"] == len(acquisition["computational_fallback_records"])
            for k, value in current.items():
                counts[k] = counts.get(k, 0) + value
            fallback_records.extend(dict(decision_index=index, **r)
                                    for r in acquisition["computational_fallback_records"])
            row.update(unique_outbound_routes=len(unique_routes), positive_hypotheses=positive,
                       exact_zero_hypotheses=zero,
                       evaluated_positive_hypothesis_route_terms=current["hypothesis_route_terms"],
                       counts=current)
        decisions.append(row)
    longest = max(steps, key=lambda s: s["decision_wall_s"], default=None)
    source_names = ("result.json", "B11_execution_binding.json", "artifact_manifest.json",
                    "choices.json", "template_audit.json", "steps.json")
    return dict(layout=layout, arm=arm, status=result["status"],
                paid_actions=result["paid_actions"], budget=result["paid_action_budget"],
                coverage_fraction=result["final_coverage_fraction"],
                correct_free_fraction=result["free_correct_fraction"],
                first_correct_free_action=result["first_free_correct_action"],
                collisions=result["collisions"], returned=result["final_pose_equals_initial_anchor"],
                termination=result["termination"], trajectory_sha256=result["trajectory_sha256"],
                global_planning_calls=len(choices), all_original_candidates_retained=True,
                all_positive_hypothesis_terms_accounted=True if arm == "S_acquisition" else None,
                positive_term_accounting_scope="Acquisition forecast terms only; visibility uses the shared full-posterior point basis without per-hypothesis future forecasts.",
                decisions=decisions,
                acquisition_counts=counts, computational_fallback_records=fallback_records,
                elapsed_wall_s=result["elapsed_s"], decision_wall_s_total=result["decision_wall_s_total"],
                longest_exported_decision=None if longest is None else dict(step=longest["step"], seconds=longest["decision_wall_s"]),
                sources=[resource(folder / name) for name in source_names])


def main():
    status_path = ROOT / "execution/batches/full/attempt_001/status.json"
    status = read(status_path)
    assert status["status"] == "finished" and not status["pending"] and not status["running"]
    assert len(status["completed"]) == 8
    rows = [inspect_run(layout, arm) for layout in LAYOUTS for arm in ARMS]
    frozen = read(ROOT / "execution/freeze.json")["files_sha256"]
    for filename, expected in frozen.items():
        assert resource(filename)["sha256"] == expected, filename
    attempts = [dict(path=str(p.parent), failure=read(p), source=resource(p))
                for p in sorted((ROOT / "execution/layouts").glob("*/runs/*/attempt_*/failure.json"))]
    completed = [r for r in rows if r["status"] == "completed_execution"]
    summary = dict(status="completed_eight_terminal_runtime_audit", runs=rows,
                   completed_runs=len(completed), failed_or_interrupted_runs=8-len(completed),
                   all_attempt_failures=attempts,
                   failure_inventory_scope="All full-policy attempts under execution/layouts/*/runs/*/attempt_*; earlier cost-pool preparation failures are separate preserved preparation evidence.",
                   actor_process_exit_codes=[dict(layout=r["layout"], arm=r["arm"], exit_code=r["exit_code"])
                                             for r in status["completed"]],
                   total_computational_fallbacks=sum(r.get("acquisition_counts", {}).get("computational_fallbacks", 0) for r in rows),
                   positive_probability_pruning_permitted=False,
                   runtime_source_changes_during_batch=False,
                   frozen_source_resources_verified_at_final_audit=len(frozen),
                   new_simulator_actions_for_this_audit=0,
                   max_concurrent_runs=3, numerical_library_thread_limit=1,
                   timing_note="Observed wall time includes three concurrent runs on the shared eight-CPU host; it is not a real-robot real-time result. Longest decision is the maximum exported action-producing decision call in steps.json.",
                   resources=[resource(Path(__file__)), resource(status_path),
                              resource(ROOT / "execution/freeze.json"),
                              resource(ROOT / "full_authorization.json"),
                              resource(ROOT / "performance/cpu_accounting_during_full.json")])
    destination = ROOT / "performance/full_runtime_receipt.json"
    assert not destination.exists(), "Preserve an existing final runtime receipt"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(resource(destination)))


if __name__ == "__main__":
    main()
