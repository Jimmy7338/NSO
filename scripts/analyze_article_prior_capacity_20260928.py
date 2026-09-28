#!/usr/bin/env python3
"""Analytical shared-prior bounds and synthetic arithmetic verification.

This is not a navigation or reconstruction experiment. It reads the fixed
public category table and exercises only the finite belief implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.semantic_reliability import SemanticReliabilityBelief


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(public_spec, output):
    spec = json.loads(public_spec.read_text())
    priors = spec['structure_prior']['probability_by_category']
    geometry = np.full(4, .25)
    rng = np.random.default_rng(9282026)
    records = []
    for label, values in sorted(priors.items()):
        q = np.asarray(values, dtype=float)
        ratio = q / geometry
        rho_lo, rho_hi = float(ratio.min()/(1+ratio.min())), float(ratio.max()/(1+ratio.max()))
        tv = float(.5*np.abs(q-geometry).sum())
        lower_delta, upper_delta = (.5-rho_lo)*tv, (rho_hi-.5)*tv
        shared = SemanticReliabilityBelief(geometry_prior=geometry,
            class_structure_priors=priors, share_across_instances=True)
        ordinary = SemanticReliabilityBelief(geometry_prior=geometry,
            class_structure_priors=priors, share_across_instances=False)
        for belief in (shared, ordinary):
            belief.register('target', label); belief.register('peer', label)
        # Finite evidence conforms to the runtime cumulative difference scale;
        # extreme likelihoods approach, but do not assert attainment of, limits.
        evidence_rows = [np.where(np.arange(4)==h, 0., -24.) for h in range(4)]
        evidence_rows += list(rng.uniform(-24., 0., size=(1000, 4)))
        max_tv = 0.; checks = 0
        for evidence in evidence_rows:
            shared.replace_log_evidence('peer', evidence)
            ordinary.replace_log_evidence('peer', evidence)
            left, right = shared.posterior('target'), ordinary.posterior('target')
            rho = left['rho']
            measured = float(.5*np.abs(np.array(left['active_structure_prior'])-
                                     np.array(right['active_structure_prior'])).sum())
            assert rho_lo-1e-12 <= rho <= rho_hi+1e-12
            assert abs(measured-abs(rho-.5)*tv) < 1e-12
            assert right['rho'] == .5
            assert measured <= max(lower_delta, upper_delta)+1e-12
            max_tv = max(max_tv, measured); checks += 4
        records.append(dict(category=label, nominal_prior=q.tolist(), geometry_prior=geometry.tolist(),
            mixture_prior_B=((q+geometry)/2).tolist(), peer_count=1,
            peer_evidence_ratio_interval=[float(ratio.min()), float(ratio.max())],
            rho_interval=[rho_lo, rho_hi], TV_nominal_vs_geometry=tv,
            maximum_active_prior_TV_S_vs_B=max(lower_delta, upper_delta),
            maximum_TV_when_peer_supports_semantic_model=upper_delta,
            synthetic_evidence_vectors=len(evidence_rows), arithmetic_checks=checks,
            largest_synthetic_active_prior_TV=max_tv))
    # General linear utility continuity bound: for fixed u, |u.(p-q)| <=
    # oscillation(u)*TV(p,q). Does not assume calibrated probabilities.
    max_error = 0.
    for _ in range(1000):
        p, q = rng.dirichlet(np.ones(4), size=2)
        u = rng.uniform(-5., 5., size=4)
        change = abs(float(u@(p-q)))
        bound = float(np.ptp(u)*.5*np.abs(p-q).sum())
        max_error = max(max_error, change-bound)
        assert change <= bound+1e-12
    result = dict(schema='article.shared_prior_capacity.v1', categories=records,
        utility_bound_checks=1000, maximum_positive_bound_violation=max_error,
        total_arithmetic_checks=sum(r['arithmetic_checks'] for r in records)+1000,
        interpretation='Algebraic bounds and deterministic synthetic arithmetic checks, not measured robot performance.',
        limits=['One eligible same-class peer, equal initial model weights and unchanged public priors.',
                'Bounds concern the transferred active prior, not the posterior after target evidence.',
                'Current association/category qualification may prevent any peer transfer.',
                'Candidate-gain vectors and costs must be held fixed for the utility perturbation bound.'],
        new_worlds=0, new_sensor_queries=0, new_policy_runs=0, new_surface_evaluations=0,
        sources={str(path.relative_to(ROOT)):dict(bytes=path.stat().st_size,sha256=sha(path))
                 for path in (public_spec,ROOT/'nso/semantic_reliability.py',Path(__file__).resolve())})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps(dict(output=str(output), checks=result['total_arithmetic_checks'], categories=records), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-spec', type=Path, default=ROOT/'audit_results/article_stage_20260928/scene_assets_v1/ART1_AISLE_DEV/public_planner_spec.json')
    parser.add_argument('--output', type=Path, default=ROOT/'audit_results/article_stage_20260928/diagnostics/shared_prior_capacity.json')
    args = parser.parse_args()
    analyze(args.public_spec.resolve(), args.output.resolve())
