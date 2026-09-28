#!/usr/bin/env python3
"""Finite numerical checks for the semantic coverage theory document.

These checks certify only the stated mathematical examples. They neither run
the planner nor establish NSO performance, SLAM accuracy, or risk calibration.
"""
import argparse
import json
import math
import random
from pathlib import Path


def check_submodular():
    rng = random.Random(4102)
    views, patches, shapes = 6, 9, 3
    probabilities = [0.2, 0.3, 0.5]
    areas = [[rng.uniform(0.01, 1.0) for _ in range(patches)] for _ in range(shapes)]
    quality = [[[rng.random() for _ in range(views)] for _ in range(patches)]
               for _ in range(shapes)]
    values = {}
    for mask in range(1 << views):
        selected = [v for v in range(views) if mask & (1 << v)]
        values[mask] = sum(probabilities[h] * areas[h][j]
                           * max((quality[h][j][v] for v in selected), default=0.0)
                           for h in range(shapes) for j in range(patches))
    checks = 0
    minimum_slack = math.inf
    for a in values:
        for b in values:
            if a & ~b:
                continue
            assert values[b] + 1e-12 >= values[a]
            for v in range(views):
                if b & (1 << v):
                    continue
                slack = (values[a | (1 << v)] - values[a]
                         - values[b | (1 << v)] + values[b])
                assert slack >= -1e-12
                minimum_slack = min(minimum_slack, slack)
                checks += 1
    assert values[0] == 0
    return dict(views=views, patches_per_shape=patches, shapes=shapes,
                diminishing_return_checks=checks, minimum_slack=minimum_slack)


def check_information_value():
    high, low, cost = 8.0, 2.0, 0.2
    cases = []
    for accuracy in (0.5, 0.8, 1.0):
        blind = (high + low) / 2
        informed = accuracy * high + (1 - accuracy) * low - cost
        formula = (accuracy - 0.5) * (high - low) - cost
        assert math.isclose(informed - blind, formula, abs_tol=1e-12)
        cases.append(dict(accuracy=accuracy, expected_increment=formula))
    assert cases[0]["expected_increment"] < 0 < cases[1]["expected_increment"]
    return cases


def check_counterexamples():
    marginal_before = 0.2 * (0.7 - 0.5)
    marginal_after = 0.3 * (0.7 - 0.5)
    assert marginal_after > marginal_before
    prior_gain = (0 + 2) / 2
    posterior_gain = 2.0
    assert posterior_gain > prior_gain
    # Mutually exclusive failure events, each with probability 0.1.
    actual_union = 0.2
    independence_formula = 1 - (1 - 0.1) ** 2
    assert independence_formula < actual_union
    return dict(product_not_submodular=dict(before=marginal_before, after=marginal_after),
                semantic_revelation_not_adaptive_submodular=dict(before=prior_gain,
                                                                 after=posterior_gain),
                dependent_path_risks=dict(actual_union=actual_union,
                                          independence_formula=independence_formula,
                                          union_bound=0.2))


def check_delay_bounds():
    rng = random.Random(918)
    horizon = 40
    checks, worst_slack = 0, math.inf
    for _ in range(100):
        # C_0(0)=0; measurements are made at integer times 1,...,B.
        coverage = [0.0] + sorted(rng.random() for _ in range(horizon))
        for delay in range(horizon + 1):
            shifted = [coverage[max(0, t - delay)] for t in range(1, horizon + 1)]
            loss = sum(coverage[t] - shifted[t - 1]
                       for t in range(1, horizon + 1)) / horizon
            slack = delay / horizon - loss
            assert slack >= -1e-12
            checks += 1
            worst_slack = min(worst_slack, slack)
    baseline = [0.2] * (horizon + 1)
    baseline[-1] = 0.9
    terminal_loss = baseline[-1] - baseline[-2]
    assert terminal_loss > 1 / horizon
    # A detour returns to the identical route state and preserves all route cells.
    reserve = ["start", "hall", "door", "room", "end"]
    detour = ["start", "object", "start"]
    available = (len(reserve) - 1) + (len(detour) - 1)
    concatenated = detour + reserve[1:]
    assert len(concatenated) - 1 == available
    assert set(reserve) <= set(concatenated)
    return dict(curves=100, delay_checks=checks, minimum_bound_slack=worst_slack,
                terminal_counterexample=dict(one_step_fraction=1 / horizon,
                                             terminal_coverage_loss=terminal_loss),
                reserve_route_cells=len(set(reserve)), exact_total_actions=available)


def check_evidence_ledger():
    rng = random.Random(337)
    areas = [rng.random() for _ in range(20)]
    evidence = [0.0] * len(areas)
    total = 0.0
    for _ in range(50):
        measured = [rng.random() for _ in areas]
        newer = [max(a, b) for a, b in zip(evidence, measured)]
        gain = sum(w * (b - a) for w, a, b in zip(areas, evidence, newer))
        assert gain >= 0
        total += gain
        evidence = newer
        duplicate = [max(a, b) for a, b in zip(evidence, measured)]
        assert duplicate == evidence
    telescoped = sum(w * e for w, e in zip(areas, evidence))
    assert math.isclose(total, telescoped, abs_tol=1e-12)
    return dict(patches=20, updates=50, duplicate_credit=0.0,
                accumulated=total, telescoped=telescoped)


def check_one_sided_variance_bound():
    checks = 0
    for probability in (0.01, 0.1, 0.3, 0.5, 0.8):
        # D equals 1 with given probability, otherwise 0; moments are exact.
        mean, variance = probability, probability * (1 - probability)
        for margin in (0.01, 0.1, 0.5, 1.0, 3.0):
            threshold = mean + margin
            actual = probability if threshold <= 1 else 0.0
            bound = variance / (variance + margin * margin)
            assert actual <= bound + 1e-12
            checks += 1
    return dict(exact_moment_distribution_checks=checks,
                note="Does not substitute MC-dropout sample variance for real conditional moments.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(scope="finite mathematical examples only; no planner or system efficacy claim",
                  information_value=check_information_value(),
                  frozen_surface_submodularity=check_submodular(),
                  counterexamples=check_counterexamples(),
                  delay_and_reserve=check_delay_bounds(),
                  measured_evidence_ledger=check_evidence_ledger(),
                  one_sided_variance=check_one_sided_variance_bound(),
                  passed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
