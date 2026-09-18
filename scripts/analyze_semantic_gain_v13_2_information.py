#!/usr/bin/env python3
"""Enforce the outbound contract, apply the frozen screen, compare commitment horizons."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_semantic_gain_v13_history import sha, write
from scripts.collect_semantic_gain_v13_2_information import verify, safe, seal
from scripts.analyze_semantic_gain_v13_paired_information import main as analyze


def main(source, output, original, comparison):
    new_hashes = verify(source)
    old_hashes = verify(original)
    new_manifest = json.loads((source / "manifest.json").read_text())
    old_manifest = json.loads((original / "manifest.json").read_text())
    new_rows = json.loads((source / "partial.json").read_text())
    old_rows = json.loads((original / "partial.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    if (new_manifest["status"] != "complete" or new_manifest["mode"] != "paired"
        or old_manifest["status"] != "complete" or not safe(new_rows)
        or not summary["physical_screen_passed"]
        or summary["first_targets_reached"] != len(new_rows)):
        raise ValueError("Completed paired experiment with all outbound/return guards required")
    for context in new_manifest["contexts"]:
        child = json.loads((source / context / "manifest.json").read_text())
        if child["protocol"]["first_option_execution"] != "complete_outbound_then_common_geometry_continuation":
            raise ValueError("Unexpected execution contract")
    for key in ("parents", "arrangements", "eligible_pairs", "expected_candidate_outcomes",
                "total_budget", "primary_reward", "pair_value", "parent_aggregation", "aggregate",
                "numerical_positive_tolerance", "development_training_screen"):
        if new_manifest["protocol"][key] != old_manifest["protocol"][key]:
            raise ValueError("Frozen comparison condition changed: " + key)
    shared_runtime = sorted(set(new_manifest["source_sha256"]) & set(old_manifest["source_sha256"]))
    for name in shared_runtime:
        if new_manifest["source_sha256"][name] != old_manifest["source_sha256"][name]:
            raise ValueError("Shared runtime/source changed: " + name)
    def key(row):
        return row["parent"], row["arrangement"], row["checkpoint_action"], row["candidate_id"]
    new = {key(row): row for row in new_rows}
    old = {key(row): row for row in old_rows}
    if set(new) != set(old) or len(new) != len(new_rows) or len(old) != len(old_rows):
        raise ValueError("Comparison candidate universe differs")
    differences = []
    for identity, row in sorted(new.items()):
        previous = old[identity]
        for field in ("state_sha256", "candidate_pool_sha256", "total_budget", "before"):
            if row[field] != previous[field]:
                raise ValueError("Decision identity or prefix metrics differ: " + field)
        differences.append(dict(parent=identity[0], arrangement=identity[1], action_id=identity[2],
            candidate_id=identity[3], outbound_joint_gain=row["joint_gain"], roundtrip_joint_gain=previous["joint_gain"],
            difference=row["joint_gain"] - previous["joint_gain"],
            coverage_difference=row["after"]["coverage_2d"] - previous["after"]["coverage_2d"],
            f1_difference=row["after"]["f1_05cm"] - previous["after"]["f1_05cm"]))
    # Evaluation formula, reference sampling and all original numerical gates are unchanged.
    analyze(source, output)
    information = json.loads((output / "summary.json").read_text())
    states = []
    for parent, arrangement, step in sorted({key(row)[:3] for row in new_rows}):
        subset = [r for r in differences if (r["parent"], r["arrangement"], r["action_id"]) == (parent, arrangement, step)]
        states.append(dict(parent=parent, arrangement=arrangement, action_id=step,
            mean_candidate_joint_difference=statistics.mean(r["difference"] for r in subset),
            mean_candidate_coverage_difference=statistics.mean(r["coverage_difference"] for r in subset),
            mean_candidate_f1_difference=statistics.mean(r["f1_difference"] for r in subset)))
    parents = [dict(parent=p, **{field: statistics.mean(r[field] for r in states if r["parent"] == p)
                for field in ("mean_candidate_joint_difference", "mean_candidate_coverage_difference", "mean_candidate_f1_difference")})
               for p in new_manifest["protocol"]["parents"]]
    comparison.mkdir(parents=True, exist_ok=False)
    write(comparison / "result.json", dict(status="complete", post_hoc_mechanism_comparison=True,
        only_intervention_commitment_changed=True, matched_candidate_outcomes=len(differences),
        candidate_differences=differences, states=states, parents=parents,
        equal_parent_mean_candidate_joint_difference=statistics.mean(r["mean_candidate_joint_difference"] for r in parents),
        gains=sum(r["difference"] > 1e-12 for r in differences),
        losses=sum(r["difference"] < -1e-12 for r in differences),
        ties=sum(abs(r["difference"]) <= 1e-12 for r in differences),
        development_information_screen_passed=information["development_information_screen_passed"],
        trained_semantic_policy_efficacy_proven=False,
        interpretation="Same initial candidate and budget, different first commitment horizon. Mean candidate gains are not semantic-policy gains."))
    write(comparison / "manifest.json", dict(status="complete", new_source=str(source), old_source=str(original),
        new_input_artifact_hashes=new_hashes, old_input_artifact_hashes=old_hashes,
        information_summary_sha256=sha(output / "summary.json"), shared_source_files=shared_runtime,
        analysis_wrapper_sha256=sha(Path(__file__).resolve())))
    (comparison / "analysis_wrapper.py").write_bytes(Path(__file__).read_bytes())
    seal(comparison)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("source", "output", "original", "comparison"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    main(args.source.resolve(), args.output.resolve(), args.original.resolve(), args.comparison.resolve())
