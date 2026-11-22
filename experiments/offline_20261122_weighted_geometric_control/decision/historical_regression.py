#!/usr/bin/env python3
"""Recompute the two registered historical scalar-weight cases from saved states."""

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path


ANALYSES = Path(__file__).resolve().parents[2]
Q1_RELATIVE = Path("rapid_evidence_20261108/semantic")
WEST_RELATIVE = Path("reproduction_20261104/deployed_policy_matrix/shelf_west_96_S_choices.json")
WEST_AUDIT_RELATIVE = Path("scalar_equivalence_20261104/audit.json")


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(path, expected):
    result = sha256(path)
    if result != expected:
        raise ValueError(f"Historical archive changed: {path}")
    return result


class Interval:
    def __init__(self):
        self.lower = (Fraction(0), True, "nonnegative weight")
        self.upper = None
        self.impossible = []

    def add(self, constant, slope, strict, witness):
        if slope == 0:
            if constant < 0 or (strict and constant == 0):
                self.impossible.append(witness)
            return
        boundary = -constant / slope
        bound = (boundary, not strict, witness)
        if slope > 0:
            if (boundary, strict) > (self.lower[0], not self.lower[1]):
                self.lower = bound
        elif self.upper is None or (boundary, not strict) < self.upper[:2]:
            self.upper = bound

    def as_dict(self):
        lo, li, lw = self.lower
        hi, ui, uw = self.upper if self.upper is not None else (None, False, None)
        feasible = not self.impossible and (hi is None or lo < hi or (lo == hi and li and ui))
        return {
            "nonnegative_weight_exists": feasible,
            "lower_fraction": str(lo), "lower": float(lo), "lower_inclusive": li, "lower_witness": lw,
            "upper_fraction": None if hi is None else str(hi),
            "upper": None if hi is None else float(hi), "upper_inclusive": ui, "upper_witness": uw,
            "constant_score_conflicts": self.impossible,
        }


def winner(scores, candidates):
    best = min(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]["cost"],
                                                      candidates[i]["candidate_id"], i), default=None)
    return best if best is not None and scores[best] > 0 else None


def verify_call(call, mode, interval=None):
    data = call["outputs"]
    candidates, audits, scores = data["candidates"], data["score_audit"], data["scores"]
    assert len(candidates) == len(audits) == len(scores["S"]) == len(scores["G"])
    terms = []
    for i, (candidate, features) in enumerate(zip(candidates, audits, strict=True)):
        assert candidate["candidate_id"] == features["candidate_id"]
        base, prior = (features["common_total_proxy"], features["prior_potentials"]["G"])
        assert math.isclose(base + prior, scores["G"][i], abs_tol=1e-9, rel_tol=1e-12)
        terms.append((Fraction.from_float(float(base)), Fraction.from_float(float(prior))))
    actual = winner(scores[mode], candidates)
    assert (candidates[actual]["candidate_id"] if actual is not None else None) == (
        data.get("selected") or {}).get("candidate_id")
    selected = winner(scores["S"], candidates)
    paid = 96 - call["inputs"]["remaining_budget"]
    own = Interval()
    def bound(a, b, strict, rival):
        label = f"paid {paid}, {rival}"
        own.add(a, b, strict, label)
        if interval is not None:
            interval.add(a, b, strict, label)
    if selected is None:
        for rival, (a, b) in zip(candidates, terms, strict=True):
            bound(-a, -b, False, f"candidate {rival['candidate_id']} must be nonpositive")
    else:
        preferred = candidates[selected]
        a, b = terms[selected]
        bound(a, b, True, f"selected candidate {preferred['candidate_id']} must be positive")
        for j, (candidate, (c, d)) in enumerate(zip(candidates, terms, strict=True)):
            if j != selected:
                strict = (preferred["cost"], preferred["candidate_id"], selected) > (
                    candidate["cost"], candidate["candidate_id"], j)
                bound(a - c, b - d, strict, f"candidate {candidate['candidate_id']}")
    probe = winner([float(a) + 7.0 * float(b) for a, b in terms], candidates)
    return {
        "mode": mode, "paid_actions": paid, "candidate_count": len(candidates),
        "selected_S_candidate": None if selected is None else candidates[selected]["candidate_id"],
        "selected_S_pose": None if selected is None else candidates[selected]["pose"],
        "interval": own.as_dict(), "interior_probe_lambda_7_matches_S": probe == selected,
    }


def run(analyses):
    q1 = analyses / Q1_RELATIVE
    west = analyses / WEST_RELATIVE
    q1_manifest = q1 / "manifest.json"
    west_audit = analyses / WEST_AUDIT_RELATIVE
    records = read(west_audit)
    sources = {str(q1_manifest): sha256(q1_manifest), str(west_audit): sha256(west_audit)}
    q1_files = {file["path"]: file["sha256"] for file in read(q1_manifest)["files"]}
    for relative in ("comparison.json", "execution/protocol.json"):
        sources[str(q1 / relative)] = validate(q1 / relative, q1_files[relative])
    sources[str(west)] = validate(west, records["input_hashes"][str(ANALYSES / WEST_RELATIVE)])
    for name, expected in records["selection_source_hashes"].items():
        path = analyses / Path(name).relative_to(ANALYSES)
        sources[str(path)] = validate(path, expected)
    assert read(q1 / "execution/protocol.json")["budget"] == 96
    q1_common = Interval()
    q1_calls = []
    for mode in ("G", "S", "X"):
        relative = f"execution/Q1_shelf_west_96_{mode}_choices.json"
        path = q1 / relative
        sources[str(path)] = validate(path, q1_files[relative])
        q1_calls.extend(verify_call(call, mode, q1_common if mode == "S" else None) for call in read(path))
    actual_s = [row for row in q1_calls if row["mode"] == "S"]
    q1_result = q1_common.as_dict()
    expected_q1 = read(q1 / "comparison.json")["actual_S_common_uniform_generic_weight_interval"]
    assert len(q1_calls) == 25 and len(actual_s) == 13
    assert all(row["interval"]["nonnegative_weight_exists"] for row in q1_calls)
    assert q1_result["nonnegative_weight_exists"] and all(row["interior_probe_lambda_7_matches_S"] for row in actual_s)
    for field in ("lower_fraction", "lower_inclusive", "upper_fraction", "upper_inclusive"):
        assert q1_result[field] == expected_q1[field]
    west_result = verify_call(read(west)[2], "S")
    assert west_result["paid_actions"] == 41 and west_result["selected_S_candidate"] == 6
    assert not west_result["interval"]["nonnegative_weight_exists"]
    prior = next(row for row in records["states"] if row["file"] == west.name
                 and row["family"] == "deployed_policy_matrix" and row["call_1based"] == 3)
    for field in ("lower_fraction", "lower_inclusive", "upper_fraction", "upper_inclusive"):
        assert west_result["interval"][field] == prior["interval"][field]
    return {
        "status": "verified_two_historical_regressions_from_saved_states",
        "scope": "Historical global score C + lambda A_neutral. No simulation or fitting for current experiment.",
        "tie_rule": "Positive winner by score then smaller cost then candidate ID. Strict constraints use exact saved binary fractions.",
        "floating_boundary_caveat": "Rounded runtime addition can differ at a mathematical boundary; the interior lambda=7 probe is separate.",
        "Q1": {"actual_S_states": len(actual_s), "tested_state_total": len(q1_calls),
               "all_states_individually_feasible": True, "actual_S_common_interval": q1_result,
               "all_actual_S_states_match_at_lambda_7": True, "states": q1_calls},
        "west96_paid41": west_result | {"planning_call_1based": 3, "matrix": "deployed_policy_matrix"},
        "source_files_sha256": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyses", type=Path, default=ANALYSES)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("historical_regression.json"))
    args = parser.parse_args()
    result = run(args.analyses.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["Q1"]["actual_S_common_interval"], result["west96_paid41"]["interval"])


if __name__ == "__main__":
    main()
