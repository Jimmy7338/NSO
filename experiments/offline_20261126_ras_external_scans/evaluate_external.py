"""Remeasure the external scenes with the frozen evaluator and group by source floor."""

import argparse
import csv
from pathlib import Path
import statistics
import sys

import numpy as np

from study_external import ARMS, BASE, FREEZE, ROOT


sys.path.insert(0, str(BASE / "evaluation"))
import evaluate as published
from evaluate_recovery import recovered_guards


def stored_contract(native, root, known_smoke):
    spec = published.read(root / "FORMAL_PROTOCOL.json")
    frozen = published.read(FREEZE)
    digest = published.sha(FREEZE)
    path = root / "execution/formal/phase_receipt.json"
    phase = published.read(path)
    jobs = {(x["layout"], x["arm"]): x for x in frozen["jobs"]}
    receipts = {(x["layout"], x["arm"]): x for x in phase["episodes"]}
    expected = {(row["layout"], arm) for row in spec["starts"] for arm in ARMS}
    published.require(not known_smoke and phase["freeze_sha256"] == digest and
                      jobs.keys() == receipts.keys() == expected and
                      published.sha(root / "FORMAL_PROTOCOL.json") == frozen["protocol_sha256"],
                      "External measurement differs from the sealed paired starts.")
    for name in ("actor_files", "evaluator_files", "simulator_only_files"):
        for file, sha in frozen[name].items():
            published.require(published.sha(file) == sha, f"Frozen file changed: {file}")
    return frozen, spec, digest, jobs, receipts, path


def measured_room_labels(world, layout, shape, resolution):
    source = ROOT / "scenes/evaluator_references" / layout / "evaluator_truth.npz"
    with np.load(source, allow_pickle=False) as truth:
        if "room_labels" in truth:
            labels = truth["room_labels"]
            if labels.shape != tuple(shape):
                raise ValueError(f"External room label grid has wrong origin or shape: {layout}")
            return labels
    # Motion accounting requires a label for each grid cell. Source metadata
    # supplies any independently identified rooms, not this constant label.
    return np.full(tuple(shape), "whole_floor_motion_only", dtype="<U31")


def scene_only_reference(ev, layout, initial_receipt):
    base = ROOT / "scenes/evaluator_references" / layout
    initial = published.read(ROOT / "scenes/prepared" / layout / "initial_state.json")
    receipt = published.read(base / "RECEIPT.json")
    published.require(initial["initial_packet_sha256"] == initial_receipt["initial_packet_sha256"]
                      and initial["anchor"] == initial_receipt["anchor"], f"Inconsistent start: {layout}")
    with np.load(base / "evaluator_truth.npz", allow_pickle=False) as truth:
        reachable = truth["reachable"]
    with np.load(base / "fixed_reference.npz", allow_pickle=False) as fixed:
        points = fixed["all"]
    counts = receipt["metadata"]["all"]
    published.require(list(reachable.shape) == initial["shape"] and int(reachable.sum()) == receipt["reachable_cells"]
                      and points.shape == (counts["visible_samples"], 3) and len(points) > 0,
                      f"Frozen reference size differs: {layout}")
    empty = np.empty((0, 3), dtype=np.float32)
    return {"initial": initial, "reachable": reachable, "source_mesh": ev.files.read_mesh(base / "source_mesh.ply"),
            "samples": {"scene": points, "shelf": empty, "cabinet": empty},
            "weights": {"scene": counts["area_per_sample_m2"], "shelf": 0.0, "cabinet": 0.0},
            "source_reference_receipt_sha256": published.sha(base / "RECEIPT.json")}


def prepare(output):
    imported, write = published.import_file, published.write_json

    def load(name, path):
        module = imported(name, path)
        if name == "old_regional_floor_evaluation":
            module.room_map = measured_room_labels
        if name == "old_regional_weighted_for_new_floors":
            module.load_reference = lambda layout, initial: scene_only_reference(module, layout, initial)
            augment = module.engine.augment_measurement

            def suppress_unavailable_object_metrics(*args):
                result = augment(*args)
                return {key: value for key, value in result.items()
                        if not key.startswith(("shelf_", "cabinet_", "object_"))}

            module.engine.augment_measurement = suppress_unavailable_object_metrics
        return module

    def write_external_summary(path, obj):
        if path.name == "summary.json":
            obj.update(schema="nso_external_floor_paired_starts_v1", scope="independent_external_source_floors",
                       independent_unit="source floor with pre-registered paired starting positions")
        write(path, obj)

    published.import_file = load
    published.write_json = write_external_summary
    published.guards = recovered_guards
    published.input_contract = stored_contract
    published.evaluate(ROOT, output, False)


def mean(values):
    samples = [v for v in values if v is not None]
    return statistics.mean(samples) if samples else None


def aggregate(output):
    spec = published.read(ROOT / "FORMAL_PROTOCOL.json")
    with (output / "episodes.csv").open() as stream:
        episodes = {(row["layout"], row["arm"]): row for row in csv.DictReader(stream)}
    with (output / "checkpoints.csv").open() as stream:
        checkpoints = {(row["layout"], row["arm"], row["checkpoint"]): row for row in csv.DictReader(stream)}
    starts = []
    metrics = ("C_free", "F1_scene_05cm", "J")
    for start in spec["starts"]:
        layout = start["layout"]
        for control in ("region_S", "F_strict"):
            row = {"layout": layout, "source_floor": start["source_floor"], "start_index": start["start_index"],
                   "control": control}
            for name, arm in (("revised", "combined_S"), ("control", control)):
                episode = episodes[layout, arm]
                middle = checkpoints.get((layout, arm, "paid_96"), {})
                available = {metric: float(middle[metric]) if middle.get(metric) else None for metric in metrics}
                row[f"{name}_status"] = episode["status"]
                row[f"{name}_A95"] = int(episode["A95_paid"]) if episode.get("A95_paid") else None
                row[f"{name}_C_terminal"] = float(episode["C_free"]) if episode.get("C_free") else None
                row[f"{name}_returned"] = episode.get("returned")
                for metric, value in available.items():
                    row[f"{name}_{metric}_96"] = value
            a, b = row["revised_A95"], row["control_A95"]
            row["A95_delta"] = a - b if a is not None and b is not None else None
            for metric in metrics:
                a, b = row[f"revised_{metric}_96"], row[f"control_{metric}_96"]
                row[f"delta_{metric}_96_pp"] = 100 * (a-b) if a is not None and b is not None else None
            starts.append(row)
    published.write_rows(output / "external_start_pairs.csv", starts)
    groups = {}
    for row in starts:
        groups.setdefault((row["source_floor"], row["control"]), []).append(row)
    floor_means = []
    for (floor, control), rows in groups.items():
        floor_means.append({"source_floor": floor, "control": control, "registered_starts": len(rows),
                            "paired_J96_starts": sum(r["delta_J_96_pp"] is not None for r in rows),
                            "paired_A95_starts": sum(r["A95_delta"] is not None for r in rows),
                            "revised_A95_reached": sum(r["revised_A95"] is not None for r in rows),
                            "control_A95_reached": sum(r["control_A95"] is not None for r in rows),
                            "revised_J96_wins": sum((r["delta_J_96_pp"] or 0) > 0 for r in rows),
                            "revised_J96_ties": sum(r["delta_J_96_pp"] == 0 for r in rows),
                            "revised_J96_losses": sum((r["delta_J_96_pp"] or 0) < 0 for r in rows),
                            **{f"mean_{key}": mean(r[key] for r in rows) for key in
                               ("A95_delta", *(f"delta_{m}_96_pp" for m in metrics))}})
    published.write_rows(output / "external_source_floor_means.csv", floor_means)
    equivalent = {}
    for control in ("region_S", "F_strict"):
        floors = [row for row in floor_means if row["control"] == control]
        equivalent[control] = {"registered_source_floors": len(floors),
                               "J96_available_source_floors": sum(r["mean_delta_J_96_pp"] is not None for r in floors),
                               "A95_available_source_floors": sum(r["mean_A95_delta"] is not None for r in floors),
                               **{f"floor_equal_{field}": mean(r[field] for r in floors)
                                  for field in ("mean_A95_delta", *(f"mean_delta_{m}_96_pp" for m in metrics))}}
    published.write_json(output / "external_source_summary.json",
                         {"source_floor_count": len({r["source_floor"] for r in starts}),
                          "preregistered_start_count": len(spec["starts"]),
                          "source_floor_equal_means": equivalent,
                          "checkpoint_pairs_sha256": published.sha(output / "external_start_pairs.csv")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/formal")
    args = parser.parse_args()
    prepare(args.out.resolve())
    aggregate(args.out.resolve())
