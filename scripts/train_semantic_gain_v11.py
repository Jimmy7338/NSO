#!/usr/bin/env python3
"""Auditable grouped-CV training for the V11 semantic observation-gain head."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nso.semantic_gain_v11 import TinyMLP, TinyMLPConfig, candidate_features


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False) + "\n")


def choose(scores, candidates):
    return min(range(len(candidates)), key=lambda i: (
        -float(scores[i]), int(candidates[i]["cost"]), int(candidates[i]["candidate_id"])))


def centered(histories, key):
    values = []
    for row in histories:
        y = row[key]
        values.append(y - y.mean())
    return np.concatenate(values)


def fit_ensemble(histories, feature, config, seeds):
    x = np.concatenate([h[feature] for h in histories])
    y = centered(histories, "utility")
    return [TinyMLP(config).fit(x, y, seed) for seed in seeds]


def predict_ensemble(models, x):
    members = np.asarray([model.predict(x) for model in models])
    return members.mean(axis=0), members.std(axis=0), members


def information_value(pair):
    if len(pair) != 2:
        raise ValueError("each context requires exactly two semantic arrangements")
    ids = [tuple(c["candidate_id"] for c in h["candidates"]) for h in pair]
    if ids[0] != ids[1]:
        raise ValueError("paired arrangements must share candidate IDs")
    conditional = float(np.mean([h["utility"].max() for h in pair]))
    without_semantics = float(np.max(np.mean([h["utility"] for h in pair], axis=0)))
    return conditional - without_semantics


def read_histories(data, protocol):
    histories = []
    input_hashes = {}
    required = ("fixture.json", "candidates.json", "predictions.json",
                "candidate_audit.json", "outcomes.json", "structural_audit.json")
    for context in protocol["data"]["contexts"]:
        for arrangement in protocol["data"]["arrangements"]:
            folder = data / context / arrangement
            for name in required:
                path = folder / name
                input_hashes[str(path.relative_to(data))] = sha256(path)
            candidates = json.loads((folder / "candidates.json").read_text())
            prediction = json.loads((folder / "predictions.json").read_text())
            audit = json.loads((folder / "candidate_audit.json").read_text())
            outcomes = json.loads((folder / "outcomes.json").read_text())
            if prediction["truth_or_future_outcomes_used"] or audit["truth_used_for_selection"]:
                raise ValueError("future truth entered V11 decision inputs")
            if len(candidates) != len(prediction["audit"]) or len(candidates) != len(outcomes):
                raise ValueError("candidate, prediction and outcome rows do not align")
            outcome_by_id = {int(r["candidate_id"]): r for r in outcomes}
            prediction_by_id = {int(r["candidate_id"]): r for r in prediction["audit"]}
            if len(outcome_by_id) != len(outcomes) or len(prediction_by_id) != len(candidates):
                raise ValueError("duplicate candidate IDs")
            assets = audit["measured_assets"]
            swapped_assets = [{**asset, "class_vote": -float(asset["class_vote"])}
                              for asset in assets]
            semantic, geometry, swapped = [], [], []
            ordered_outcomes = []
            for candidate in candidates:
                candidate_id = int(candidate["candidate_id"])
                outcome = outcome_by_id[candidate_id]
                pred = prediction_by_id[candidate_id]
                sx, gx, _ = candidate_features(pred, candidate, assets)
                xx, swapped_geometry, _ = candidate_features(pred, candidate, swapped_assets)
                np.testing.assert_array_equal(gx, swapped_geometry)
                semantic.append(sx); geometry.append(gx); swapped.append(xx)
                if (outcome["failure"] is not None or outcome["collision_count"] != 0
                        or not outcome["returned_to_anchor"] or outcome["paid_actions"] <= 0):
                    raise ValueError("V11 development pool contains failed or invalid branch")
                ordered_outcomes.append(outcome)
            utility = np.asarray([(r["after"]["area_times_f1_05cm"]
                - r["before"]["area_times_f1_05cm"]) / r["paid_actions"]
                for r in ordered_outcomes], dtype=float)
            secondary = np.asarray([r["new_area_m2"] / r["paid_actions"]
                                    for r in ordered_outcomes], dtype=float)
            histories.append(dict(history_id=f"{context}/{arrangement}", context=context,
                arrangement=arrangement, candidates=candidates, outcomes=ordered_outcomes,
                utility=utility, secondary=secondary,
                semantic=np.asarray(semantic), geometry=np.asarray(geometry),
                swapped=np.asarray(swapped), fixed_scores={name: np.asarray(
                    prediction["rate_scores"][name], dtype=float) for name in ("G", "S", "X")},
                fixed_choices={name: int(prediction["rate_selected_candidate_ids"][name])
                               for name in ("G", "S", "X")},
                feature_audit=dict(measured_assets=assets,
                    semantic_feature_sha256=hashlib.sha256(np.asarray(semantic).tobytes()).hexdigest(),
                    geometry_feature_sha256=hashlib.sha256(np.asarray(geometry).tobytes()).hexdigest(),
                    swapped_feature_sha256=hashlib.sha256(np.asarray(swapped).tobytes()).hexdigest())))
    return histories, input_hashes


def summarize(records, method):
    rows = [r["methods"][method] for r in records]
    return dict(histories=len(rows), selected_utility_mean=float(np.mean(
        [r["selected_utility"] for r in rows])), mean_regret=float(np.mean(
        [r["regret"] for r in rows])), exact_oracle_choices=sum(r["exact_oracle"] for r in rows),
        centered_mae=(float(np.mean([r["centered_mae"] for r in rows]))
                      if "centered_mae" in rows[0] else None),
        spearman_mean=(float(np.mean([r["spearman"] for r in rows]))
                       if "spearman" in rows[0] else None))


def main(data, output):
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT / "configs/virtual3d/semantic_gain_v11_development_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    source_paths = [Path(__file__).resolve(), protocol_path, ROOT / "nso/semantic_gain_v11.py"]
    verification = json.loads((data / "verification.json").read_text())
    if not (verification["status"] == "passed_full" and verification["passed_full"]
            and verification["branches_checked"] == verification["branches_total"] == 44
            and verification["collisions"] == verification["failed_branches"] == 0
            and verification["raw_hashes_rechecked_after_replay"]):
        raise ValueError("V9.1 physical source is not fully verified")
    artifact_manifest = json.loads((data / "artifact_hashes.json").read_text())
    if sha256(data / "artifact_hashes.json") != verification["artifact_manifest_sha256"]:
        raise ValueError("V9.1 artifact manifest is not bound to its independent verification")
    # Hash and freeze paths before parsing any outcome values.
    expected_paths = [Path(context) / arrangement / name
        for context in protocol["data"]["contexts"]
        for arrangement in protocol["data"]["arrangements"]
        for name in ("fixture.json", "candidates.json", "predictions.json",
                     "candidate_audit.json", "outcomes.json", "structural_audit.json")]
    frozen_inputs = {}
    for relative in expected_paths:
        name = str(relative)
        actual = sha256(data / relative)
        if artifact_manifest.get(name) != actual:
            raise ValueError(f"V9.1 artifact mismatch: {name}")
        frozen_inputs[name] = actual
    manifest = dict(schema_version="semantic_gain_v11_training/1", status="running",
        created_utc=datetime.now(timezone.utc).isoformat(), protocol=protocol,
        protocol_sha256=sha256(protocol_path), source_sha256={str(p.relative_to(ROOT)): sha256(p)
            for p in source_paths}, input_run=str(data),
        input_artifact_manifest_sha256=sha256(data / "artifact_hashes.json"),
        input_files_sha256=frozen_inputs, outcome_files_opened_before_manifest=0,
        independent_confirmation=False)
    write(output / "manifest.json", manifest)

    histories, loaded_hashes = read_histories(data, protocol)
    if loaded_hashes != frozen_inputs:
        raise AssertionError("loaded input inventory differs from frozen inventory")
    by_context = {context: [h for h in histories if h["context"] == context]
                  for context in protocol["data"]["contexts"]}
    information = {context: information_value(pair) for context, pair in by_context.items()}
    information_gate = (sum(value > protocol["pretraining_gate"]["minimum_value"]
                            for value in information.values())
                        >= protocol["pretraining_gate"]["required_nonzero_contexts"])
    if not information_gate:
        raise ValueError("paired semantic information-value gate failed")

    config = TinyMLPConfig(**{key: protocol["network"][key] for key in (
        "input_dim", "hidden_dim", "epochs", "learning_rate", "l2")})
    seeds = protocol["network"]["ensemble_seeds"]
    fold_reports = []
    for held in protocol["data"]["contexts"]:
        train = [h for h in histories if h["context"] != held]
        validation = by_context[held]
        geometry_models = fit_ensemble(train, "geometry", config, seeds)
        semantic_models = fit_ensemble(train, "semantic", config, seeds)
        validation_rows = []
        for history in validation:
            geometry_score, geometry_std, geometry_members = predict_ensemble(
                geometry_models, history["geometry"])
            semantic_score, semantic_std, semantic_members = predict_ensemble(
                semantic_models, history["semantic"])
            swapped_score, swapped_std, swapped_members = predict_ensemble(
                semantic_models, history["swapped"])
            utility = history["utility"]
            centered_utility = utility - utility.mean()
            oracle_index = choose(utility, history["candidates"])
            methods = {}
            model_values = (("geometry_mlp", geometry_score, geometry_std, geometry_members),
                            ("semantic_mlp", semantic_score, semantic_std, semantic_members),
                            ("semantic_swapped", swapped_score, swapped_std, swapped_members))
            for name, score, uncertainty, members in model_values:
                index = choose(score, history["candidates"])
                rho = spearmanr(score, centered_utility).statistic
                methods[name] = dict(selected_index=index,
                    selected_candidate_id=int(history["candidates"][index]["candidate_id"]),
                    selected_utility=float(utility[index]), regret=float(utility.max() - utility[index]),
                    exact_oracle=index == oracle_index, centered_mae=float(np.mean(
                        np.abs(score - centered_utility))), spearman=float(0.0 if np.isnan(rho) else rho),
                    scores=score, ensemble_std=uncertainty, member_scores=members)
            for name, fixed_name in (("fixed_geometry_G", "G"), ("fixed_semantic_S", "S"),
                                     ("fixed_swapped_X", "X")):
                index = next(i for i, c in enumerate(history["candidates"])
                             if int(c["candidate_id"]) == history["fixed_choices"][fixed_name])
                methods[name] = dict(selected_index=index,
                    selected_candidate_id=int(history["candidates"][index]["candidate_id"]),
                    selected_utility=float(utility[index]), regret=float(utility.max() - utility[index]),
                    exact_oracle=index == oracle_index, scores=history["fixed_scores"][fixed_name])
            methods["low_confidence_fallback"] = dict(methods["geometry_mlp"],
                rule="semantic confidence below threshold uses geometry MLP")
            methods["oracle"] = dict(selected_index=oracle_index,
                selected_candidate_id=int(history["candidates"][oracle_index]["candidate_id"]),
                selected_utility=float(utility[oracle_index]), regret=0., exact_oracle=True,
                scores=utility)
            validation_rows.append(dict(history_id=history["history_id"], context=held,
                arrangement=history["arrangement"], candidate_ids=[int(c["candidate_id"])
                    for c in history["candidates"]], costs=[int(c["cost"]) for c in history["candidates"]],
                primary_utility=utility, secondary_area_per_action=history["secondary"],
                oracle_index=oracle_index, methods=methods, feature_audit=history["feature_audit"]))
        fold_reports.append(dict(held_context=held,
            training_histories=[h["history_id"] for h in train],
            validation_histories=[h["history_id"] for h in validation],
            models={"geometry": [m.to_dict() for m in geometry_models],
                    "semantic": [m.to_dict() for m in semantic_models]},
            validation=validation_rows))

    records = [row for fold in fold_reports for row in fold["validation"]]
    methods = ("fixed_geometry_G", "fixed_semantic_S", "fixed_swapped_X", "geometry_mlp",
               "semantic_mlp", "semantic_swapped", "low_confidence_fallback", "oracle")
    summaries = {method: summarize(records, method) for method in methods}
    geometry_choices = [r["methods"]["geometry_mlp"]["selected_candidate_id"] for r in records]
    semantic_choices = [r["methods"]["semantic_mlp"]["selected_candidate_id"] for r in records]
    calibration_rows = []
    errors, uncertainties = [], []
    for row in records:
        method = row["methods"]["semantic_mlp"]
        centered_y = np.asarray(row["primary_utility"]) - np.mean(row["primary_utility"])
        errors.extend(np.abs(np.asarray(method["scores"]) - centered_y))
        uncertainties.extend(method["ensemble_std"])
    errors, uncertainties = np.asarray(errors), np.asarray(uncertainties)
    for indices in np.array_split(np.argsort(uncertainties), 3):
        calibration_rows.append(dict(count=len(indices), mean_ensemble_std=float(
            uncertainties[indices].mean()), mean_absolute_error=float(errors[indices].mean())))
    gates = dict(
        information_value_nonzero=information_gate,
        semantic_mean_regret_below_geometry=(summaries["semantic_mlp"]["mean_regret"]
                                             < summaries["geometry_mlp"]["mean_regret"]),
        semantic_selected_utility_above_fixed_semantic=(
            summaries["semantic_mlp"]["selected_utility_mean"]
            > summaries["fixed_semantic_S"]["selected_utility_mean"]),
        semantic_changes_at_least_one_action_from_geometry=any(
            a != b for a, b in zip(semantic_choices, geometry_choices)),
        swapped_labels_worse_than_correct_semantics=(
            summaries["semantic_swapped"]["selected_utility_mean"]
            < summaries["semantic_mlp"]["selected_utility_mean"]),
        low_confidence_falls_back_exactly_to_geometry=all(
            r["methods"]["low_confidence_fallback"]["selected_candidate_id"]
            == r["methods"]["geometry_mlp"]["selected_candidate_id"] for r in records),
        all_training_losses_decreased=all(model["diagnostics"]["loss_decreased"]
            for fold in fold_reports for family in fold["models"].values() for model in family))
    development_passed = all(gates.values())
    write(output / "folds.json", fold_reports)
    result = dict(schema_version="semantic_gain_v11_development_result/1",
        status="passed" if development_passed else "failed", development_only=True,
        input_histories=len(histories), physical_candidate_branches=sum(len(h["candidates"])
                                                                        for h in histories),
        information_value_by_context=information, summaries=summaries,
        semantic_choice_changes_from_geometry=sum(a != b for a, b in zip(
            semantic_choices, geometry_choices)), uncertainty_calibration=calibration_rows,
        gates=gates, development_passed=development_passed)
    write(output / "summary.json", result)

    if development_passed:
        full_geometry = fit_ensemble(histories, "geometry", config, seeds)
        full_semantic = fit_ensemble(histories, "semantic", config, seeds)
        frozen_model = dict(schema_version="semantic_gain_v11_frozen_development_model/1",
            status="frozen_after_grouped_development_gates", independent_confirmation=False,
            architecture="12x4x1", parameters_per_member=full_semantic[0].parameter_count,
            ensemble_seeds=seeds, feature_contract=protocol["features"], target=protocol["target"],
            geometry_members=[m.to_dict() for m in full_geometry],
            semantic_members=[m.to_dict() for m in full_semantic],
            protocol_sha256=manifest["protocol_sha256"], input_files_sha256=frozen_inputs,
            allowed_use="STGHP development integration; never independent-confirmation retraining")
        write(output / "frozen_model.json", frozen_model)
    manifest["status"] = "complete"
    manifest["development_passed"] = development_passed
    write(output / "manifest.json", manifest)
    for path in source_paths:
        if sha256(path) != manifest["source_sha256"][str(path.relative_to(ROOT))]:
            raise RuntimeError("source changed during V11 training")
    for name, digest in frozen_inputs.items():
        if sha256(data / name) != digest:
            raise RuntimeError("input changed during V11 training")
    write(output / "artifact_hashes.json", {str(p.relative_to(output)): sha256(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    print(json.dumps(json_value(result), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.data.resolve(), args.output.resolve())
