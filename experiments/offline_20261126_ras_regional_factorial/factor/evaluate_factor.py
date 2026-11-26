"""Measure all four factor combinations with the archived surface evaluator."""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import sys

from factor_study import BASE, FREEZE, ROOT, RUNS


sys.path.insert(0, str(BASE / "evaluation"))
import evaluate as published
from evaluate_recovery import recovered_guards


CONTRASTS = (("D_at_G0", "D1G0", "D0G0"), ("D_at_G1", "D1G1", "D0G1"),
             ("G_at_D0", "D0G1", "D0G0"), ("G_at_D1", "D1G1", "D1G0"))


def check_factor_branch(job, directory, manifest):
    published.require((manifest["global_mode"], manifest["local_mode"]) == ("G", "S"),
                      f"Unexpected selector modes: {directory}")
    repairs = 0
    for event in published.read(directory / "regional_choices.json"):
        ranking = event["ranking"]
        repair = ranking.get("attribution_path_repair")
        expected = job.get("repair") if ranking.get("ranking") else None
        published.require(bool(repair) == bool(expected), f"Wrong score branch: {directory}/{event['paid_before']}")
        if repair:
            published.require(repair["score_repair"] == expected and repair["actual_paid"] == event["paid_before"]
                              and repair["ranking"] == ranking["ranking"],
                              f"Unmatched factor score evidence: {directory}")
            repairs += 1
    return repairs


def read_matrix(freeze_path, phase_path):
    freeze = published.read(freeze_path)
    digest = published.sha(freeze_path)
    phase = published.read(phase_path)
    published.require(phase["freeze_sha256"] == digest, f"Worker receipts do not match {freeze_path}")
    jobs = {(row["layout"], row["arm"]): row for row in freeze["jobs"]}
    receipts = {(row["layout"], row["arm"]): row for row in phase["episodes"]}
    published.require(jobs.keys() == receipts.keys() and len(jobs) == len(phase["episodes"]),
                      f"Worker schedule differs from completed receipts: {phase_path}")
    return freeze, digest, jobs, receipts


def effect(left, right):
    return left - right if left is not None and right is not None else None


def factor_effects(layouts, episodes, checkpoints):
    compare = []
    values = {(r["layout"], r["arm"]): r for r in checkpoints
              if r["checkpoint"] == "paid_96" and r["status"] == "captured"}
    for layout in layouts:
        arms = {a: values.get((layout, a), {}) for a in ("D0G0", "D1G0", "D0G1", "D1G1")}
        for contrast, left, right in CONTRASTS:
            nw, old = episodes[layout, left], episodes[layout, right]
            difference = {"layout": layout, "contrast": contrast, "left_arm": left, "right_arm": right,
                          "left_A95": nw.get("A95_paid"), "right_A95": old.get("A95_paid"),
                          "delta_A95_steps": effect(nw.get("A95_paid"), old.get("A95_paid"))}
            for name in ("C_free", "F1_scene_05cm", "J"):
                delta = effect(arms[left].get(name), arms[right].get(name))
                difference[f"delta_{name}_96_pp"] = delta * 100 if delta is not None else None
            compare.append(difference)
        js = [arms[name].get("J") for name in ("D1G1", "D1G0", "D0G1", "D0G0")]
        compare.append({"layout": layout, "contrast": "D_by_G",
                        "delta_J_96_pp": 100 * (js[0] - js[1] - js[2] + js[3])
                        if None not in js else None})
    return compare


def evaluate(output):
    spec = published.read(ROOT / "PROTOCOL.json")
    old = read_matrix(BASE / "FORMAL_FREEZE.json", BASE / "execution/formal/phase_receipt.json")
    new = read_matrix(FREEZE, RUNS / "phase_receipt.json")
    scene_root = Path(new[0]["scene_root"])
    published.require(old[0]["scene_root"] == new[0]["scene_root"], "Alternate scenes appeared in factorial study")
    native = published.import_file("factor_native_measurement", published.ORIGINAL / "evaluate.py")
    ev = published.import_file("factor_weighted_measurement", native.WEIGHTED_METRIC)
    ev.SCENES = scene_root
    world = published.import_file("factor_frozen_truth", scene_root / "scene.py")
    metric = ev.files.load_metric(published.read(ev.files.SCENES / "freeze.json"),
                                  published.read(ev.files.OLD_EVAL / "freeze.json"))
    published.guards = recovered_guards
    published.check_policy_trace = check_factor_branch
    output.mkdir(parents=True, exist_ok=False)
    tables, episodes, bindings = defaultdict(list), {}, {}
    cold_by_layout = defaultdict(list)
    for layout in spec["layouts"]:
        for arm, flags in spec["arms"].items():
            original_arm = flags.get("archive_arm")
            root = BASE if original_arm else ROOT
            frozen, digest, jobs, receipts = old if original_arm else new
            key = (layout, original_arm or arm)
            worker, job = receipts[key], jobs[key]
            directory, transport = published.checked_worker(root, worker, digest, frozen["frozen_unix_s"])
            episode, parts, binding = native.measure_episode(ev, directory, job, worker, frozen, digest,
                                                               scene_root, world, metric)
            episode.update(transport)
            if episode["status"] == "completed_execution":
                proof = published.check_success(ev, job, directory, episode, parts, binding, scene_root)
                cold_by_layout[layout].append(proof["cold"])
            else:
                proof = published.check_failure(job, episode, parts, directory, scene_root)
                binding = None
            episodes[layout, arm] = episode | {"arm": arm, "source_arm": key[1],
                                                "source_freeze_sha256": digest,
                                                "reused": bool(original_arm), "D": int(flags["target"]),
                                                "G": int(flags["outbound"])}
            bindings[layout, arm] = binding
            tables["measurement_bindings"].append({"layout": layout, "arm": arm, **proof})
            for name, rows in parts.items():
                tables[name].extend(row | {"arm": arm} for row in rows)
            print(layout, arm, episode["status"], episode.get("A95_paid"), flush=True)
    for layout, cold in cold_by_layout.items():
        published.require(all(row == cold[0] for row in cold), f"Paired cold input mismatch: {layout}")
    tables["episodes"] = list(episodes.values())
    tables["factor_effects"] = factor_effects(spec["layouts"], episodes, tables["checkpoints"])
    tables["first_divergence"] = [native.pair_divergence(ev, layout, pair, bindings)
                                 for layout in spec["layouts"] for pair in combinations(spec["arms"], 2)]
    for name, rows in tables.items():
        published.write_rows(output / f"{name}.csv", rows, leading=("layout", "arm"))
    summary = {"independent_units": len(spec["layouts"]), "registered_pairs": len(episodes),
               "completed_new": sum(e["status"] == "completed_execution" and not e["reused"] for e in episodes.values()),
               "completed_reused": sum(e["status"] == "completed_execution" and e["reused"] for e in episodes.values()),
               "failed": sum(e["status"] != "completed_execution" for e in episodes.values()),
               "measurements": len(tables["checkpoints"]), "effects": tables["factor_effects"],
               "actor_freeze": new[1], "archived_actor_freeze": old[1],
               "source_measurement": published.sha(Path(published.__file__)),
               "results_sha256": {path.name: published.sha(path) for path in sorted(output.glob("*.csv"))}}
    published.write_json(output / "summary.json", summary)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=ROOT / "results/formal")
    evaluate(cli.parse_args().out.resolve())
