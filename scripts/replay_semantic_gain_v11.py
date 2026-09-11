#!/usr/bin/env python3
"""Independent numerical replay of V11 features, Adam fits and selections.

This verifier intentionally does not import the V11 model or training runner.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False) + "\n")


def independent_features(prediction, candidate, assets):
    cost = float(candidate["cost"])
    aperture = np.asarray(prediction["aperture_factors"], dtype=float)
    generic = np.asarray(prediction["generic_area_proxies"], dtype=float)
    votes = np.asarray([float(a["class_vote"]) for a in assets])
    confidence = np.asarray([abs(float(a["class_vote"])) * min(1., 20. *
        float(a["marked_points"]) / max(1, int(a["support_points"]))) for a in assets])
    support = np.asarray([float(a["support_m2_proxy"]) for a in assets])
    target = (float(aperture.mean()) if candidate.get("asset_index") is None
              else float(aperture[int(candidate["asset_index"])]))
    signed = votes * confidence
    signed_target = (float(np.mean(signed * aperture)) if candidate.get("asset_index") is None
                     else float(signed[int(candidate["asset_index"])] * target))
    common = np.asarray([float(prediction["common_proxy_per_action"]),
        float(prediction["potential_proxies_before_cost"]["G"]) / cost,
        cost / 48., 1. / cost, aperture.sum(), aperture.max(),
        abs(aperture[0] - aperture[1]), target])
    semantic = np.asarray([signed_target, np.sum(signed * aperture * generic) / cost,
        np.sum(signed * aperture * support) / cost,
        np.sum(signed * aperture**2) / max(float(aperture.sum()), 1e-8)])
    geometry = np.asarray([common[4]**2, common[5]**2, common[6]**2, common[7]**2])
    return np.r_[common, semantic], np.r_[common, geometry]


def independent_fit(x, y, config, seed):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mean = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-9] = 1.
    y_scale = max(float(y.std()), 1e-9)
    z, target = (x - mean) / scale, y / y_scale
    rng = np.random.default_rng(int(seed))
    w1 = rng.normal(0., .2, (config["input_dim"], config["hidden_dim"]))
    b1 = np.zeros(config["hidden_dim"])
    w2 = rng.normal(0., .2, (config["hidden_dim"], 1))
    b2 = np.zeros(1)
    params = [w1, b1, w2, b2]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]

    def loss():
        hidden = np.tanh(z @ w1 + b1)
        residual = (hidden @ w2 + b2)[:, 0] - target
        return float(np.mean(residual**2) + config["l2"] *
                     (np.sum(w1**2) + np.sum(w2**2)))

    curve = [loss()]
    for step in range(1, config["epochs"] + 1):
        hidden = np.tanh(z @ w1 + b1)
        prediction = (hidden @ w2 + b2)[:, 0]
        output_gradient = 2. * (prediction - target) / len(target)
        gradients = [None] * 4
        gradients[2] = hidden.T @ output_gradient[:, None] + 2. * config["l2"] * w2
        gradients[3] = np.asarray([output_gradient.sum()])
        hidden_gradient = output_gradient[:, None] @ w2.T * (1. - hidden**2)
        gradients[0] = z.T @ hidden_gradient + 2. * config["l2"] * w1
        gradients[1] = hidden_gradient.sum(axis=0)
        for i, (parameter, gradient) in enumerate(zip(params, gradients)):
            m[i] = .9 * m[i] + .1 * gradient
            v[i] = .999 * v[i] + .001 * gradient**2
            parameter -= config["learning_rate"] * (m[i] / (1. - .9**step)) / (
                np.sqrt(v[i] / (1. - .999**step)) + 1e-8)
        curve.append(loss())
    return dict(x_mean=mean, x_scale=scale, y_scale=y_scale, parameters=params,
                loss_curve=np.asarray(curve))


def predict(model, x):
    w1, b1, w2, b2 = model["parameters"]
    z = (np.asarray(x) - model["x_mean"]) / model["x_scale"]
    return (np.tanh(z @ w1 + b1) @ w2 + b2)[:, 0] * model["y_scale"]


def assert_close(actual, expected, label, checks, atol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=label)
    checks.append(label)


def load_histories(data, protocol, checks):
    histories = {}
    for context in protocol["data"]["contexts"]:
        for arrangement in protocol["data"]["arrangements"]:
            folder = data / context / arrangement
            candidates = json.loads((folder / "candidates.json").read_text())
            predictions = json.loads((folder / "predictions.json").read_text())
            audit = json.loads((folder / "candidate_audit.json").read_text())
            outcomes = {int(r["candidate_id"]): r for r in json.loads(
                (folder / "outcomes.json").read_text())}
            pred = {int(r["candidate_id"]): r for r in predictions["audit"]}
            semantic, geometry, swapped = [], [], []
            for candidate in candidates:
                i = int(candidate["candidate_id"])
                sx, gx = independent_features(pred[i], candidate, audit["measured_assets"])
                xx, xg = independent_features(pred[i], candidate,
                    [{**a, "class_vote": -float(a["class_vote"])}
                     for a in audit["measured_assets"]])
                assert_close(gx, xg, f"{context}/{arrangement}/{i}/geometry_label_invariant", checks)
                semantic.append(sx); geometry.append(gx); swapped.append(xx)
            ordered = [outcomes[int(c["candidate_id"])] for c in candidates]
            utility = np.asarray([(r["after"]["area_times_f1_05cm"]
                - r["before"]["area_times_f1_05cm"]) / r["paid_actions"] for r in ordered])
            histories[f"{context}/{arrangement}"] = dict(context=context,
                candidates=candidates, semantic=np.asarray(semantic), geometry=np.asarray(geometry),
                swapped=np.asarray(swapped), utility=utility)
    return histories


def replay_member(independent, recorded, label, checks):
    assert_close(independent["x_mean"], recorded["x_mean"], label + "/x_mean", checks)
    assert_close(independent["x_scale"], recorded["x_scale"], label + "/x_scale", checks)
    assert_close(independent["y_scale"], recorded["y_scale"], label + "/y_scale", checks)
    for i, (actual, expected) in enumerate(zip(independent["parameters"], recorded["parameters"])):
        assert_close(actual, expected, f"{label}/parameter_{i}", checks)
    assert_close(independent["loss_curve"], recorded["diagnostics"]["loss_curve"],
                 label + "/loss_curve", checks)


def main(run, data, output):
    output.mkdir(parents=True, exist_ok=False)
    hashes = json.loads((run / "artifact_hashes.json").read_text())
    for name, expected in hashes.items():
        if sha256(run / name) != expected:
            raise ValueError(f"training artifact changed: {name}")
    manifest = json.loads((run / "manifest.json").read_text())
    for name, expected in manifest["input_files_sha256"].items():
        if sha256(data / name) != expected:
            raise ValueError(f"training input changed: {name}")
    protocol = manifest["protocol"]
    folds = json.loads((run / "folds.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    checks = ["training_artifact_hashes", "input_file_hashes"]
    histories = load_histories(data, protocol, checks)
    config = {key: protocol["network"][key] for key in (
        "input_dim", "hidden_dim", "epochs", "learning_rate", "l2")}
    seeds = protocol["network"]["ensemble_seeds"]
    replayed_choices = {}
    for fold in folds:
        held = fold["held_context"]
        train = [histories[name] for name in fold["training_histories"]]
        x_by_family = {family: np.concatenate([h[family] for h in train])
                       for family in ("geometry", "semantic")}
        y = np.concatenate([h["utility"] - h["utility"].mean() for h in train])
        fitted = {}
        for family in ("geometry", "semantic"):
            fitted[family] = []
            for seed, recorded in zip(seeds, fold["models"][family]):
                model = independent_fit(x_by_family[family], y, config, seed)
                replay_member(model, recorded, f"{held}/{family}/seed_{seed}", checks)
                fitted[family].append(model)
        recorded_validation = {r["history_id"]: r for r in fold["validation"]}
        for name in fold["validation_histories"]:
            history = histories[name]; recorded = recorded_validation[name]
            for method, family, feature in (("geometry_mlp", "geometry", "geometry"),
                                            ("semantic_mlp", "semantic", "semantic"),
                                            ("semantic_swapped", "semantic", "swapped")):
                members = np.asarray([predict(model, history[feature]) for model in fitted[family]])
                mean, std = members.mean(axis=0), members.std(axis=0)
                row = recorded["methods"][method]
                assert_close(members, row["member_scores"], name + "/" + method + "/members", checks)
                assert_close(mean, row["scores"], name + "/" + method + "/mean", checks)
                assert_close(std, row["ensemble_std"], name + "/" + method + "/std", checks)
                index = min(range(len(mean)), key=lambda i: (-float(mean[i]),
                    int(history["candidates"][i]["cost"]), int(history["candidates"][i]["candidate_id"])))
                if index != row["selected_index"]:
                    raise AssertionError(name + "/" + method + "/selection")
                checks.append(name + "/" + method + "/selection")
                replayed_choices[name + "/" + method] = index
    if len(replayed_choices) != 24:
        raise AssertionError("expected 8 histories x 3 learned interventions")

    frozen = json.loads((run / "frozen_model.json").read_text())
    all_histories = [histories[f"{c}/{a}"] for c in protocol["data"]["contexts"]
                     for a in protocol["data"]["arrangements"]]
    full_y = np.concatenate([h["utility"] - h["utility"].mean() for h in all_histories])
    for family, key in (("geometry", "geometry_members"), ("semantic", "semantic_members")):
        full_x = np.concatenate([h[family] for h in all_histories])
        for seed, recorded in zip(seeds, frozen[key]):
            model = independent_fit(full_x, full_y, config, seed)
            replay_member(model, recorded, f"full/{family}/seed_{seed}", checks)
    if not summary["development_passed"] or not all(summary["gates"].values()):
        raise AssertionError("archived development gates did not pass")
    checks += ["development_gates", "frozen_full_development_model"]
    report = dict(schema_version="semantic_gain_v11_independent_numerical_replay/1",
        status="passed", verifier_imported_training_or_model_module=False,
        artifact_hashes_verified=len(hashes), input_files_verified=len(
            manifest["input_files_sha256"]), numerical_checks=len(checks),
        folds_retrained=4, ensemble_members_retrained=30,
        validation_histories=8, learned_intervention_choices_replayed=24,
        development_only=True, independent_scene_confirmation=False,
        checks=checks)
    write(output / "verification.json", report)
    (output / "REPORT.md").write_text(
        "# V11 independent numerical replay\n\n"
        "All four grouped folds and both full-development ensembles were independently retrained "
        "without importing the V11 model or training runner. Feature vectors, 30 member fits, "
        "loss curves, parameters, predictions, ensemble uncertainty and 24 learned choices match "
        "the archived run. This verifies numerical reproducibility, not independent-scene efficacy.\n")
    print(json.dumps({"status": "passed", "checks": len(checks),
                      "members": 30, "choices": 24}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.data.resolve(), args.output.resolve())
