"""Measure the active-observation formal runs on the unchanged V2 references."""

from argparse import ArgumentParser
import csv
import importlib.util
from pathlib import Path
import sys

from study import (BASE, BASELINE_CELLS, ELIGIBLE_LAYOUT, FREEZE, MASK_CELLS, GATE_VALIDATION,
                   PREVIOUS, ROOT, SCHEMA, V1, V2, supervisor)


EVALUATOR_SOURCE = ROOT.parent / "ras_20261126/heterogeneous"


def read_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archived_evaluator():
    """Bind the legacy unqualified evaluator import to its archived implementation."""
    saved = {name: sys.modules.pop(name, None) for name in ("evaluate", "evaluate_recovery")}
    sys.path.insert(0, str(EVALUATOR_SOURCE))
    try:
        sys.modules["evaluate"] = read_module("evaluate", BASE / "evaluation/evaluate.py")
        return read_module("archived_external_evaluator", EVALUATOR_SOURCE / "evaluate_external.py")
    finally:
        sys.path.remove(str(EVALUATOR_SOURCE))
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def bind_v2_references(external):
    """Keep the archived measurement implementation and its exact V2 assets."""
    shared = supervisor()
    frozen = shared.read(FREEZE)
    v2 = shared.read(V2 / "FORMAL_FREEZE.json")
    if (frozen["schema"] != SCHEMA or frozen["phase"] != "formal"
            or frozen["previous_online_sweep_freeze_sha256"] != shared.sha(PREVIOUS / "FORMAL_FREEZE.json")
            or any(frozen[key] != v2[key] for key in ("scene_root", "simulator_only_files", "evaluator_files", "jobs"))):
        raise ValueError("Formal evaluation requires the exact frozen V2 scenes and 12 policy jobs")
    shared.verify(frozen["qualification_files"])
    if str(GATE_VALIDATION) not in frozen["qualification_files"] or not shared.read(GATE_VALIDATION)["eligible_for_formal"]:
        raise ValueError("The independent gate does not qualify the formal results")
    external.ROOT, external.FREEZE = ROOT, FREEZE

    def v2_data(function):
        def call(*args):
            external.ROOT = V2
            try:
                return function(*args)
            finally:
                external.ROOT = ROOT
        return call

    external.measured_room_labels = v2_data(external.measured_room_labels)
    external.scene_only_reference = v2_data(external.scene_only_reference)


def table(path, *fields):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {tuple(row[k] for k in fields): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate measurement keys: {path}")
    return result


def reachable_mask(np, study, layout):
    path = study / "scenes/evaluator_references" / layout / "evaluator_truth.npz"
    with np.load(path, allow_pickle=False) as archive:
        return archive["reachable"]


def archived_measurements(result, published):
    audit_path = result / "measurement_audit.json"
    audit = published.read(audit_path)
    for filename in ("episodes.csv", "checkpoints.csv", "coverage_curve.csv"):
        published.require(published.sha(result / filename) == audit["results_sha256"][filename],
                          f"Measurement table differs from audit: {result / filename}")
    return (table(result / "episodes.csv", "layout", "arm"),
            table(result / "checkpoints.csv", "layout", "arm", "checkpoint"),
            table(result / "coverage_curve.csv", "layout", "arm", "paid"),
            {(item["layout"], item["arm"]): item for item in audit["recorded_arm_receipts"]})


def count_at_clocks(np, published, masks, requested, episode, audit, curve):
    if episode["status"] != "completed_execution":
        return {}
    path = Path(episode["coverage_trace_path"])
    published.require(str(path) == audit["coverage_trace_path"] and
                      published.sha(path) == audit["coverage_trace_sha256"],
                      f"Observed occupancy differs from independent audit: {path}")
    expected = np.arange(int(episode["actual_paid"]) + 1)
    with np.load(path, allow_pickle=False) as archive:
        ticks, belief = archive["paid"], archive["belief"]
    published.require(np.array_equal(ticks, expected) and belief.shape == (len(ticks), *masks[0].shape),
                      f"Misaligned paid clocks or reachable-mask coordinate frame: {path}")
    counts = {}
    key = (episode["layout"], episode["arm"])
    for paid in requested:
        if paid >= len(ticks):
            continue
        old, current = (int((belief[paid][mask] == 0).sum()) for mask in masks)
        recorded = curve[*key, str(paid)]
        published.require(current == int(recorded["correct_free_cells"]) and
                          int(masks[1].sum()) == int(recorded["reachable_cells"]),
                          f"Independent coverage recount disagrees: {path}, paid {paid}")
        counts[paid] = old
    return counts


def checkpoint_F1(checkpoints, key, paid):
    for name in (f"paid_{paid}" if paid else "start", "terminal"):
        row = checkpoints.get((*key, name), {})
        if row.get("eligible_for_paired_comparison") == "True" and int(row["actual_paid"]) == paid:
            return float(row["F1_scene_05cm"])
    return None


def delta_pp(left, right):
    return None if left is None or right is None else 100*(right-left)


def original_mask_sensitivity(output, external, root=ROOT, original=PREVIOUS, mask_source=V1, primary=V2):
    """Compare V3 and active runs at matching paid clocks on the common V1 mask."""
    old_output = original / "results/formal"
    pub, np = external.published, external.np
    before = archived_measurements(old_output, pub)
    after = archived_measurements(output, pub)
    protocol = pub.read(root / "FORMAL_PROTOCOL.json")
    original_protocol = pub.read(original / "FORMAL_PROTOCOL.json")
    for field in ("starts", "arms", "budget", "capture_paid"):
        pub.require(protocol[field] == original_protocol[field],
                    f"Original-mask sensitivity requires a common {field} registration")
    scheduled = protocol["capture_paid"]
    rows = []
    plateau = {}
    for start in protocol["starts"]:
        layout = start["layout"]
        old_mask, new_mask = (reachable_mask(np, scenario, layout) for scenario in (mask_source, primary))
        pub.require(old_mask.shape == new_mask.shape and bool(old_mask.any()) and
                    not np.any(old_mask & ~new_mask), f"Old mask is not a subset for {layout}")
        old_cells, new_cells = int(old_mask.sum()), int(new_mask.sum())
        for arm in protocol["arms"]:
            key = (layout, arm)
            episodes = [data[0][key] for data in (before, after)]
            terminal = [int(row["actual_paid"]) if row["status"] == "completed_execution" else None for row in episodes]
            last_common = (min(int(row["actual_paid"]) for row in episodes)
                           if all(row["status"] == "completed_execution" for row in episodes) else None)
            requested = sorted(set(scheduled) | set(range(last_common + 1 if last_common is not None else 0))
                               | {paid for paid in terminal if paid is not None})
            series = []
            for data in (before, after):
                episode, checkpoints, curve, audit = data
                values = count_at_clocks(np, pub, (old_mask, new_mask), requested, episode[key], audit[key], curve)
                values_at_paid = {}
                for paid, numerator in values.items():
                    coverage = numerator / old_cells
                    f1 = checkpoint_F1(checkpoints, key, paid)
                    values_at_paid[paid] = (numerator, coverage, f1, coverage*f1 if f1 is not None else None)
                series.append(values_at_paid)
            for paid in requested:
                old, new = (values.get(paid, (None,)*4) for values in series)
                row = {"source_floor": start["source_floor"], "start_index": start["start_index"],
                       "layout": layout, "arm": arm, "paid": paid,
                       "scheduled_paid_checkpoint": paid in scheduled, "last_common_paid": paid == last_common,
                       "original_reachable_cells": old_cells, "repaired_reachable_cells": new_cells,
                       "original_actual_paid": episodes[0]["actual_paid"],
                       "repaired_actual_paid": episodes[1]["actual_paid"],
                       "original_status": episodes[0]["status"], "repaired_status": episodes[1]["status"],
                       "coverage_pair_available": old[1] is not None and new[1] is not None}
                for prefix, measurements in (("original_run", old), ("repaired_run", new)):
                    for name, value in zip(("correct_free_original_mask", "C_original_mask", "F1_scene_05cm",
                                            "J_original_mask"), measurements, strict=True):
                        row[f"{prefix}_{name}"] = value
                row["delta_C_original_mask_pp"] = delta_pp(old[1], new[1])
                row["delta_J_original_mask_pp"] = delta_pp(old[3], new[3])
                rows.append(row)
            if layout == ELIGIBLE_LAYOUT:
                pub.require(old_cells == MASK_CELLS, "HM3D start 1 needs the archived 1,640-cell V1 mask")
                for version, paid, values in zip(("v3", "active_observation"), terminal, series, strict=True):
                    value = values.get(paid, (None,)*4)
                    at_budget = values.get(protocol["budget"], (None,)*4)
                    plateau.setdefault(arm, {})[version] = {
                        "actual_terminal_paid": paid, "terminal_correct_free_cells_on_v1": value[0],
                        "terminal_coverage_on_v1": value[1], "paid_192_coverage_on_v1": at_budget[1],
                        "terminal_exceeds_v3_regional_baseline": value[0] > BASELINE_CELLS if value[0] is not None else None,
                    }
    for arm in ("region_S", "combined_S"):
        pub.require(plateau[arm]["v3"]["terminal_correct_free_cells_on_v1"] == BASELINE_CELLS,
                    "Historical coverage must reproduce the archived 522/1,640 V3 plateau")
    csv_path = output / "original_mask_matched_paid.csv"
    pub.write_rows(csv_path, rows)
    pub.write_json(output / "original_mask_matched_paid_receipt.json", {
        "schema": "nso_active_sweep_common_v1_mask_sensitivity_v1",
        "comparison": "V3 and active-observation formal runs compared by policy and exact paid clock on the V1 reachable mask."
                      " C is correctly observed free cells divided by reachable cells in the V1 mask."
                      " J multiplies this C by audited same-clock F1@5cm, with terminal mesh permitted only at the exact requested clock."
                      " No carry forward. The last shared clock is pair-dependent and is not a fixed-budget comparison.",
        "original_measurement_audit_sha256": pub.sha(old_output / "measurement_audit.json"),
        "repaired_measurement_audit_sha256": pub.sha(output / "measurement_audit.json"),
        "hm3d_start1_previous_regional_plateau_on_v1_mask": {"reachable_cells": MASK_CELLS,
                                                              "correct_free_cells": BASELINE_CELLS,
                                                              "coverage": BASELINE_CELLS / MASK_CELLS, "arms": plateau},
        "scheduled_paid_clocks": scheduled,
        "by_scheduled_paid_clock": {str(paid): {"coverage_pairs": sum(r["coverage_pair_available"] for r in rows if r["paid"] == paid),
                                               "joint_pairs": sum(r["delta_J_original_mask_pp"] is not None for r in rows if r["paid"] == paid)}
                                    for paid in scheduled},
        "pair_last_common_clocks": [{k: r[k] for k in ("layout", "arm", "paid", "delta_C_original_mask_pp",
                                                       "delta_J_original_mask_pp")}
                                    for r in rows if r["last_common_paid"]],
        "csv_sha256": pub.sha(csv_path),
    })
    return rows


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=ROOT / "results/formal")
    args = cli.parse_args()
    output = args.out.resolve()
    external = archived_evaluator()
    bind_v2_references(external)
    external.prepare(output)
    external.aggregate(output)
    original_mask_sensitivity(output, external)
    report = read_module("archived_external_report", EVALUATOR_SOURCE / "report_external.py")
    report.ROOT, report.RESULTS = ROOT, output
    report.main()


if __name__ == "__main__":
    main()
