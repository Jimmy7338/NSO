"""Bind active-sensing receipts to native motion and recount the fixed mask."""

import csv
from pathlib import Path

import numpy as np

from study import ELIGIBLE_LAYOUT, MASK_FILE, PREVIOUS, ROOT, supervisor


def main():
    io = supervisor("formal")
    receipt = io.read(io.RUNS / "phase_receipt.json")
    prior = io.read(PREVIOUS / "execution/formal/phase_receipt.json")
    previous = {(row["layout"], row["arm"]): row for row in prior["episodes"]}
    assert receipt["freeze_sha256"] == io.sha(io.FREEZE) and len(receipt["episodes"]) == 12
    with np.load(MASK_FILE, allow_pickle=False) as data:
        mask = data["reachable"] & ~data["occupancy"]
    assert mask.sum() == 1640
    trace_items = []
    for episode in receipt["episodes"]:
        assert episode["exit_code"] == 0 and episode["result_sha256"] == io.sha(episode["result_path"])
        path = Path(episode["run_path"])
        key = episode["layout"], episode["arm"]
        before = Path(previous[key]["run_path"])
        manifest = io.read(path / "manifest.json")
        for item in ("steps.json", "audit.json", "observed_belief_trace.npz"):
            assert io.sha(path / item) == manifest["files_sha256"][item]
        steps, former = io.read(path / "steps.json"), io.read(before / "steps.json")
        fields = ("action", "position", "heading", "collision", "scan_ranges_sha256", "depth_values_sha256")
        same = len(steps) == len(former) and all(
            all(a[field] == b[field] for field in fields) for a, b in zip(steps, former, strict=True)
        )
        turn = {"layout": key[0], "arm": key[1], "old_paid": len(former), "new_paid": len(steps),
                "same_action_pose_and_sensor_values": same, "collisions": manifest["collisions"],
                "returned": manifest["returned"], "native_steps_sha256": io.sha(path / "steps.json")}
        audit = io.read(path / "audit.json")
        crossings = [row for row in audit if row["event"] == "active_observed_edge_executed"]
        if key[0] != ELIGIBLE_LAYOUT:
            assert not any(row["event"].startswith("active_observation") for row in audit) and not crossings
        else:
            def count(run):
                with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as archive:
                    return np.count_nonzero(archive["belief"][:, mask] == 0, axis=1)

            old, new = count(before), count(path)
            assert len(old) == len(former)+1 and len(new) == len(steps)+1
            turn.update(old_common_correct=int(old[-1]), new_common_correct=int(new[-1]),
                        new_common_peak=int(new.max()), common_reachable_cells=int(mask.sum()),
                        measured_acquisitions=[r for r in audit if r["event"] == "active_observation_acquired"],
                        actual_crossings=crossings)
            for edge in crossings:
                s, e = edge["scan_paid"], edge["edge_paid"]
                assert 0 < s < e <= len(steps)
                assert steps[s-1]["packet_sha256"] == edge["scan_packet_sha256"]
                assert steps[e-1]["packet_sha256"] == edge["edge_packet_sha256"]
                assert steps[e-2]["position"] == edge["from_cell"] and steps[e-1]["position"] == edge["to_cell"]
                assert steps[e-1]["action"] == "forward" and not steps[e-1]["collision"]
                assert edge["complete_sweep"] and edge["target_sweep_certificate"]["allowed"]
                edge["common_correct_at_crossing"] = int(new[e])
        trace_items.append(turn)
    with (ROOT / "results/formal/episodes.csv").open() as stream:
        measured = list(csv.DictReader(stream))
    assert len(measured) == 12 and all(row["status"] == "completed_execution" for row in measured)
    io.write(ROOT / "FORMAL_OBSERVATION_AUDIT.json",
             {"schema": "nso_formal_active_observation_execution_v1", "formal_runs": len(trace_items),
              "common_reference_sha256": io.sha(MASK_FILE),
              "actual_crossings": sum(len(row.get("actual_crossings", ())) for row in trace_items),
              "unchanged_other_starts": sum(row["same_action_pose_and_sensor_values"] for row in trace_items
                                           if row["layout"] != ELIGIBLE_LAYOUT),
              "reached_A95": sum(bool(row["A95_paid"]) for row in measured),
              "source_receipt_sha256": io.sha(io.RUNS / "phase_receipt.json"), "rows": trace_items})


if __name__ == "__main__":
    main()
