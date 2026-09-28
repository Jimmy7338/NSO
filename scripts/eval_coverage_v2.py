#!/usr/bin/env python3
"""Audited CPU feasibility/ablation runs; all previous frozen files stay intact."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.grid_exploration import GridConfig, GridExplorationEnv
from env.grid_layouts import generate_layout
from env.grid_semantics import synthetic_semantics
from env.targeted_scenes import generate_scene
from nso.frontier_policy import FrontierPolicy
from nso.grid_mechanisms import MechanismPolicy
from nso.coverage_planner_v2 import CostCorrectedPolicy, CoveragePlannerV2
from nso.navigable_frontier_v2 import NavigableFrontierPolicy, ProjectedCoveragePolicy
from nso.route_coverage_v2 import RouteCoveragePolicy
from utils.cpu_protocol import file_hash, geometry_hash, digest_json, runtime_versions

METHODS = ('nearest', 'geometry', 'combined', 'combined_no_timeout',
           'geometry_cost', 'combined_cost', 'safe_single', 'safe_bundle', 'path_bundle',
           'metric_gain', 'metric_cost', 'projected_single', 'projected_bundle', 'projected_path',
           'route32', 'route64')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def make_policy(method, ec, candidates):
    if method in ('route32', 'route64'):
        return RouteCoveragePolicy(ec, candidates, horizon=int(method[5:]))
    if method in ('metric_gain', 'metric_cost'):
        return NavigableFrontierPolicy(ec, candidates, exact_cost=method == 'metric_cost')
    if method.startswith('projected_'):
        return ProjectedCoveragePolicy(ec, candidates, bundle=method != 'projected_single',
                                       path_gain=method == 'projected_path')
    if method == 'nearest':
        return FrontierPolicy(ec, 'nearest_frontier', candidates)
    if method in ('safe_single', 'safe_bundle', 'path_bundle'):
        return CoveragePlannerV2(ec, candidates, bundle=method != 'safe_single',
                                 path_gain=method == 'path_bundle')
    cls = CostCorrectedPolicy if method.endswith('_cost') else MechanismPolicy
    variant = 'gain_semantic_structure' if method.startswith('combined') else 'gain_control'
    return cls(ec, variant, candidates,
               dict(goal_budget=ec.max_steps if method == 'combined_no_timeout' else 32))


def make_scene(entry, config):
    ec = GridConfig(**config['environments'][entry['suite']])
    if entry['suite'] == 'targeted':
        scene = generate_scene(entry['family'], entry['size'], entry['seed'], ec.resolution_m)
        return ec, scene.occupancy, scene.semantic_fields['aligned'], dict(
            start=scene.entrances[entry['entrance']], heading=1 if entry['entrance'] == 0 else 3)
    world = generate_layout(entry['family'], entry['size'], entry['seed'])
    return ec, world, synthetic_semantics(world.shape, entry['seed']+10000), dict(seed=entry['start_seed'])


def episode(ec, world, field, reset, method, candidates, output):
    output.mkdir()
    env = GridExplorationEnv(world, ec, field)
    obs = env.reset(**reset)
    policy = make_policy(method, ec, candidates)
    rows, poses, masks, durations = [], [], [], []
    def record(action, duration=0.):
        rows.append(dict(env.evaluation_metrics(), action=action, decision_ms=duration))
        poses.append((*obs.position, obs.heading))
        masks.append(np.packbits(obs.visible.ravel()))
    record('reset')
    started = time.perf_counter()
    while True:
        t = time.perf_counter()
        decision = policy.act(obs)
        duration = 1000*(time.perf_counter()-t)
        durations.append(duration)
        obs, done = env.step(decision.action)
        if hasattr(policy, 'observe_outcome'):
            policy.observe_outcome(obs, done)
        record(decision.action, duration)
        if done:
            break
    elapsed = time.perf_counter()-started
    with (output/'steps.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for name in ('selection_audit', 'attempts'):
        with (output/(name+'.jsonl')).open('w') as f:
            for row in getattr(policy, name, []):
                f.write(json.dumps(row, allow_nan=False)+'\n')
    assets = env.evaluation_assets()
    packed = np.stack(masks)
    np.savez_compressed(output/'observations.npz', **assets, visible_packed=packed,
                        poses=poses, final_belief=obs.belief)
    # Independent reconstruction from archived sensor masks, not reward/scalars.
    known = np.zeros_like(world, bool)
    reconstructed = []
    for mask in packed:
        known |= np.unpackbits(mask)[:world.size].reshape(world.shape).astype(bool)
        reconstructed.append(float(np.count_nonzero(known & assets['reachable']) / assets['reachable'].sum()))
    if not np.allclose(reconstructed, [r['coverage_ratio'] for r in rows], rtol=0, atol=1e-12):
        raise AssertionError('coverage reconstruction mismatch')
    curve = np.interp(np.arange(ec.max_steps+1), np.arange(len(rows)), reconstructed)
    result = dict(rows[-1], method=method, coverage_auc=float(np.trapz(curve)/ec.max_steps),
                  wall_time_s=elapsed, decision_ms_p95=float(np.percentile(durations, 95)),
                  decision_ms_max=float(max(durations)), decision_ms_mean=float(np.mean(durations)),
                  map_hash=geometry_hash(world), initial_pose=poses[0],
                  mask_audit_passed=True, artifact_dir=output.name,
                  turns=sum(r['action'] in ('left', 'right') for r in rows))
    result.update(policy.diagnostics() if hasattr(policy, 'diagnostics') else dict(goal_changes=policy.goal_count))
    write_json(output/'episode.json', result)
    return result


def summarize(output, records, config):
    summary = dict(means={}, contrasts={}, gate={})
    for suite in sorted({r['suite'] for r in records}):
        means = {}
        for method in config['methods']:
            subset = [r for r in records if r['suite'] == suite and r['method'] == method]
            means[method] = {key: float(np.mean([r[key] for r in subset])) for key in
                ('coverage_auc', 'coverage_ratio', 'revisit_ratio', 'goal_changes', 'turns', 'decision_ms_mean')}
            means[method]['episodes'] = len(subset)
            means[method]['collisions'] = sum(r['collisions'] for r in subset)
        summary['means'][suite] = means
        contrast = {}
        for method in config['methods']:
            if method == 'geometry':
                continue
            pairs = []
            for entry in config['maps']:
                if entry['suite'] != suite:
                    continue
                found = {r['method']:r for r in records if r['map_id'] == entry['id']}
                pairs.append(dict(map_id=entry['id'], seed=entry['seed'],
                    auc=found[method]['coverage_auc']-found['geometry']['coverage_auc'],
                    final=found[method]['coverage_ratio']-found['geometry']['coverage_ratio']))
            # Same-seed family variants / paired starts are not independent.
            blocks = [np.mean([p['auc'] for p in pairs if p['seed'] == seed])
                      for seed in sorted({p['seed'] for p in pairs})]
            rng = np.random.default_rng(20260910)
            bootstrap = rng.choice(blocks, (10000, len(blocks)), replace=True).mean(axis=1)
            contrast[method] = dict(auc_difference=float(np.mean([p['auc'] for p in pairs])),
                relative_auc=means[method]['coverage_auc']/means['geometry']['coverage_auc']-1,
                final_difference=float(np.mean([p['final'] for p in pairs])),
                seed_block_ci95=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                independent_seed_blocks=len(blocks), wins=sum(p['auc']>0 for p in pairs), pairs=pairs)
        summary['contrasts'][suite] = contrast
    gate = config.get('gate')
    if gate:
        chosen = gate['method']
        checks = {}
        for suite in summary['means']:
            value = summary['contrasts'][suite][chosen]
            means = summary['means'][suite]
            checks[suite] = dict(relative_auc=value['relative_auc'] >= gate['min_relative_auc'],
                ci_positive=value['seed_block_ci95'][0] > 0,
                final_coverage=value['final_difference'] >= -gate['max_final_drop'],
                strongest_baseline=means[chosen]['coverage_auc'] >= max(means[b]['coverage_auc'] for b in gate['baselines']))
        checks['safety'] = all(r['collisions'] == 0 for r in records if r['method'] == chosen)
        checks['latency'] = all(r['decision_ms_p95'] < gate['max_p95_ms'] for r in records if r['method'] == chosen)
        summary['gate'] = dict(checks=checks, passed=all(
            all(v.values()) if isinstance(v, dict) else v for v in checks.values()))
    write_json(output/'summary.json', summary)
    return summary


def run(config, output):
    if not config['methods'] or set(config['methods']) - set(METHODS) or 'geometry' not in config['methods']:
        raise ValueError('invalid methods')
    if len(set(config['methods'])) != len(config['methods']) or len({e['id'] for e in config['maps']}) != len(config['maps']):
        raise ValueError('duplicate method or episode map id')
    prepared = [(entry, make_scene(entry, config)) for entry in config['maps']]
    seen = set(config.get('excluded_geometry_hashes', []))
    # Multiple starts may share geometry, but distinct map seeds may not.
    generated = {}
    for entry, (_, world, _, _) in prepared:
        value = geometry_hash(world)
        key = (entry['suite'], entry['family'], entry['size'], entry['seed'])
        if value in seen or (value in generated and generated[value] != key):
            raise ValueError('previous or duplicate geometry: '+entry['id'])
        generated[value] = key
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    names = ['scripts/eval_coverage_v2.py', 'nso/coverage_planner_v2.py', 'nso/navigable_frontier_v2.py',
             'nso/route_coverage_v2.py', 'nso/grid_mechanisms.py',
             'nso/grid_topology.py', 'nso/frontier_policy.py', 'env/grid_exploration.py',
             'env/grid_layouts.py', 'env/grid_semantics.py', 'env/targeted_scenes.py',
             'utils/grid_geometry.py', 'utils/paper_eval.py', 'utils/cpu_protocol.py']
    signatures = {name:file_hash(ROOT/name) for name in names}
    metadata = dict(status='running', created_at=datetime.now(timezone.utc).isoformat(),
        files_sha256=signatures, config_sha256=digest_json(config), runtime=runtime_versions(),
        expected_episodes=len(prepared)*len(config['methods']), scope=config['scope'],
        limitations=['synthetic occupancy and visible salience; perfect odometry',
                     'not original neural NSO; not TARE/FALCON reproduction',
                     'seed-block bootstrap is exploratory for small samples'])
    write_json(output/'config.json', config); write_json(output/'run_metadata.json', metadata)
    with zipfile.ZipFile(output/'sources.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(ROOT/name, name)
    records = []
    for entry, (ec, world, field, reset) in prepared:
        for method in config['methods']:
            result = episode(ec, world, field, reset, method, config['max_candidates'],
                             output/(entry['id']+'_'+method))
            result.update(suite=entry['suite'], family=entry['family'], size=entry['size'],
                          map_seed=entry['seed'], map_id=entry['id'])
            write_json(output/result['artifact_dir']/'episode.json', result)
            records.append(result)
            with (output/'episodes.jsonl').open('a') as f:
                f.write(json.dumps(result, allow_nan=False)+'\n')
            print(entry['id'], method, 'AUC', round(result['coverage_auc'],4),
                  'final', round(result['coverage_ratio'],4), flush=True)
    if signatures != {name:file_hash(ROOT/name) for name in names}:
        raise AssertionError('source changed during experiment')
    summary = summarize(output, records, config)
    metadata.update(status='complete', frozen_sources_verified=True, all_masks_audited=True,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(output/'run_metadata.json', metadata)
    return summary


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(json.loads(args.config.read_text()), args.output)
