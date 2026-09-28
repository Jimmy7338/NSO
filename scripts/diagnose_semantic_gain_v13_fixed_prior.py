#!/usr/bin/env python3
"""Post-hoc fixed-prior and route-cost diagnosis; never changes the frozen screen."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(source, output):
    import numpy as np

    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Complete frozen acquisition required")
    artifacts = json.loads((source / "artifact_hashes.json").read_text())
    rows = json.loads((source / "partial.json").read_text())
    if len(rows) != manifest["protocol"]["expected_candidate_outcomes"]:
        raise ValueError("Incomplete outcome universe")
    checked = {}

    def read_frozen(path):
        name = str(path.relative_to(source))
        actual = sha(path)
        if artifacts[name] != actual:
            raise ValueError("Frozen artifact changed: " + name)
        checked[name] = actual
        return json.loads(path.read_text())

    read_frozen(source / "partial.json")
    states = []
    paired_features = {}
    for pair in manifest["protocol"]["eligible_pairs"]:
        for arrangement in manifest["protocol"]["arrangements"]:
            parent, step = pair["parent"], pair["action_id"]
            features = read_frozen(source / (parent + "_" + arrangement) / f"features_step_{step}.json")
            paired_features[(parent, step, arrangement)] = features
            outcomes = {r["candidate_id"]: r for r in rows if
                (r["parent"], r["arrangement"], r["checkpoint_action"]) == (parent, arrangement, step)}
            if {f["candidate_id"] for f in features} != set(outcomes):
                raise ValueError("Feature/outcome candidates do not match")
            choices = {}
            for mode in ("N", "G", "S", "X"):
                def key(f):
                    audit = f["fixed_prediction_audit"]
                    score = audit["common_proxy_per_action"] + audit["potential_proxies_before_cost"][mode] / f["roundtrip_cost"]
                    return -score, f["roundtrip_cost"], f["candidate_id"]
                chosen = min(features, key=key)
                result = outcomes[chosen["candidate_id"]]
                choices[mode] = dict(candidate_id=chosen["candidate_id"], joint_gain=result["joint_gain"],
                    coverage_2d=result["after"]["coverage_2d"], f1_05cm=result["after"]["f1_05cm"])
            states.append(dict(parent=parent, arrangement=arrangement, action_id=step,
                candidate_count=len(features),
                observed_asset_count=len(features[0]["observed_assets"]),
                nonzero_confidence_assets=sum(a["semantic_confidence"] > 0 for a in features[0]["observed_assets"]),
                all_candidate_apertures_zero=all(not any(f["fixed_prediction_audit"]["aperture_factors"]) for f in features),
                candidates_with_nonzero_apertures=sum(any(f["fixed_prediction_audit"]["aperture_factors"]) for f in features),
                mean_planned_first_return_fraction_of_remaining_budget=statistics.mean(f["return_cost"] / f["remaining_budget"] for f in features),
                mean_planned_first_roundtrip_fraction_of_remaining_budget=statistics.mean(f["roundtrip_cost"] / f["remaining_budget"] for f in features),
                choices=choices))
    parents = []
    for parent in manifest["protocol"]["parents"]:
        subset = [s for s in states if s["parent"] == parent]
        means = {mode: statistics.mean(s["choices"][mode]["joint_gain"] for s in subset) for mode in ("N", "G", "S", "X")}
        parents.append(dict(parent=parent, fixed_policy_mean_joint_gain=means,
                            S_minus_G=means["S"] - means["G"], S_minus_X=means["S"] - means["X"]))
    def geometry_only(features):
        result = []
        for feature in features:
            value = {k: v for k, v in feature.items() if k not in
                     ("observed_assets", "fixed_prediction_audit", "feature_boundary")}
            value["observed_assets"] = [{k: v for k, v in asset.items() if k not in
                ("class_vote", "semantic_confidence")} for asset in feature["observed_assets"]]
            value["fixed_prediction_audit"] = {k: feature["fixed_prediction_audit"][k] for k in
                ("common_proxy_per_action", "aperture_factors", "generic_area_proxies", "prefix_aperture_support")}
            result.append(value)
        return result

    geometry_pair_checks = []
    for pair in manifest["protocol"]["eligible_pairs"]:
        pair_features = [paired_features[(pair["parent"], pair["action_id"], arrangement)]
                         for arrangement in manifest["protocol"]["arrangements"]]
        geometry_pair_checks.append(dict(parent=pair["parent"], action_id=pair["action_id"],
            geometry_descriptor_pair_equal=all(geometry_only(f) == geometry_only(pair_features[0]) for f in pair_features),
            observed_class_votes_differ=any([a["class_vote"] for a in f[0]["observed_assets"]] !=
                [a["class_vote"] for a in pair_features[0][0]["observed_assets"]] for f in pair_features[1:])))
    reference_composition = []
    for context in manifest["contexts"]:
        path = source / context / "reference.npz"
        name = str(path.relative_to(source))
        actual = sha(path)
        if artifacts[name] != actual:
            raise ValueError("Frozen reference changed")
        checked[name] = actual
        with np.load(path, allow_pickle=False) as data:
            labels = data["classes"]
            ids, counts = np.unique(labels, return_counts=True)
            reference_composition.append(dict(context=context, reference_samples=len(labels),
                counts_by_class=dict(zip(map(str, ids.tolist()), counts.tolist())),
                facility_reference_fraction=float(np.mean(np.isin(labels, [2, 3]))),
                category_names={"2": "storage_shelf", "3": "closed_equipment_cabinet"}))
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="complete", post_hoc=True, trained=False, protocol_or_threshold_changed=False,
        source=str(source), diagnostic_source_sha256=sha(Path(__file__)), checked_input_sha256=checked,
        states=states, parents=parents, geometry_pair_checks=geometry_pair_checks,
        reference_composition=reference_composition,
        all_apertures_zero_states=sum(s["all_candidate_apertures_zero"] for s in states),
        S_G_same_choice_states=sum(s["choices"]["S"]["candidate_id"] == s["choices"]["G"]["candidate_id"] for s in states),
        S_X_same_choice_states=sum(s["choices"]["S"]["candidate_id"] == s["choices"]["X"]["candidate_id"] for s in states),
        equal_parent_mean_S_minus_G=statistics.mean(p["S_minus_G"] for p in parents),
        equal_parent_mean_S_minus_X=statistics.mean(p["S_minus_X"] for p in parents),
        boundary="Offline selections using already-recorded fixed scores; no new fitted models, fresh physical branches, efficacy gate or causal test of return overhead.")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("states", "parents", "checked_input_sha256")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.source, args.output)
