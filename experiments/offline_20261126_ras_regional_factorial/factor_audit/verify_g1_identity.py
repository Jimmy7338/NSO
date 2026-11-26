"""Compare the actual outbound-reward executions and application of doorway credit."""

from hashlib import file_digest
import json
from math import isclose
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD = ROOT.parents[1] / "region_independent_floors_20261123"
FACTOR = ROOT.parent / "factor"
RAW_FIELDS = ("step", "action", "position", "heading", "packet_sha256", "packet_chain_sha256",
              "depth_values_sha256", "scan_ranges_sha256", "rgb_values_sha256", "selected_target_pose",
              "route_action_index_zero_based", "phase_before_action", "guard_reason", "reserved_return_cost")


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def indexed_runs(path):
    phase = load(path)
    return {(row["layout"], row["arm"]): row for row in phase["episodes"]}


def run_path(worker):
    path = Path(worker["run_path"])
    assert worker["exit_code"] == 0
    assert sha(path / "manifest.json") == worker["result_sha256"]
    manifest = load(path / "manifest.json")
    for name in ("steps.json", "regional_choices.json"):
        assert sha(path / name) == manifest["files_sha256"][name]
    return path


def compare(left, right):
    steps = [load(run / "steps.json") for run in (left, right)]
    assert len(steps[0]) == len(steps[1]) == 192
    for a, b in zip(*steps, strict=True):
        assert all(a[k] == b[k] for k in RAW_FIELDS), (left, a["step"])
    a, b = [load(run / "regional_choices.json") for run in (left, right)]
    score_changes, rank_changes, changed_fields = 0, 0, 0
    largest = 0.0
    for lower, upper in zip(a, b, strict=True):
        assert all(lower[k] == upper[k] for k in ("paid_before", "selection_call_id", "actual_target_pose"))
        x, y = [[tuple(row["pose"]) for row in src["ranking"]["ranking"]] for src in (lower, upper)]
        assert set(x) == set(y)
        rank_changes += x != y
        rows = [{tuple(row["pose"]): row for row in src["ranking"]["ranking"]} for src in (lower, upper)]
        for pose, row in rows[0].items():
            match = rows[1][pose]
            delta = match["score_m2"] - row["score_m2"]
            score_changes += not isclose(delta, 0, abs_tol=1e-10, rel_tol=0)
            changed_fields += row["credited_region_id"] != match["credited_region_id"]
            largest = max(largest, abs(delta))
    same = []
    for label in ("start", "paid_48", "paid_96", "paid_144", "paid_192", "terminal"):
        raw = [load(run / label / "checkpoint.json")["files_sha256"] for run in (left, right)]
        records = {name:sha(left / label / name) for name in ("belief.npz", "native_mesh.ply")}
        assert all(raw[0][name] == raw[1][name] == digest == sha(right / label / name)
                   for name, digest in records.items())
        same.append({"checkpoint": label, **records})
    return {"same_action_sensor_and_dispatch_rows": len(steps[0]),
            "same_actual_global_choices": len(a), "differing_candidate_credit_rows": changed_fields,
            "differing_candidate_score_rows": score_changes, "global_calls_with_rank_reordering": rank_changes,
            "largest_absolute_candidate_score_delta_m2": largest, "identical_belief_and_mesh_checkpoints": same}


def main():
    original = indexed_runs(OLD / "execution/formal/phase_receipt.json")
    newer = indexed_runs(FACTOR / "execution/formal/phase_receipt.json")
    cases = []
    for layout in load(FACTOR / "PROTOCOL.json")["layouts"]:
        cases.append({"floor": layout, **compare(run_path(newer[layout, "D0G1"]), run_path(original[layout, "combined_S"]))})
    result = {"scope": "Four literal D0G1 / D1G1 executions. Raw command, measured observation, selected target, "
              "checkpoint belief and measured surface are compared at all archived clocks. Candidate score "
              "terms may change even with identical executed decisions.", "floor_count":len(cases),
              "factor_first_divergence_sha256":sha(FACTOR / "results/formal/first_divergence.csv"),
              "cases": cases}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
