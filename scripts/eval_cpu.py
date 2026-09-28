#!/usr/bin/env python3
"""Run deterministic CPU grid exploration baselines and archive every episode."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import subprocess
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from env.grid_exploration import GridConfig, GridExplorationEnv
from env.grid_layouts import generate_layout, LAYOUTS
from nso.frontier_policy import FrontierPolicy, METHODS
from nso.grid_mechanisms import MechanismPolicy, MECHANISM_METHODS
from env.grid_semantics import synthetic_semantics
from nso.cpu_reachability import CandidateRPN
from utils.eval_protocol import code_fingerprint
from utils.cpu_protocol import verify_protocol

SCHEMA = 'cpu_grid_v1'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('schema') != SCHEMA:
        raise ValueError('expected cpu_grid_v1 config')
    GridConfig(**config['environment'])
    if not config['maps'] or not config['start_seeds'] or not config['methods']:
        raise ValueError('empty experiment matrix')
    if any(m not in METHODS + MECHANISM_METHODS for m in config['methods']) or len(set(config['methods'])) != len(config['methods']):
        raise ValueError('unknown/duplicate method')
    identities = [(m['layout'], m['seed']) for m in config['maps']]
    if len(set(identities)) != len(identities) or any(k not in LAYOUTS for k, _ in identities):
        raise ValueError('unknown/duplicate map')
    if len(set(config['start_seeds'])) != len(config['start_seeds']):
        raise ValueError('duplicate start seeds')
    if any(m in MECHANISM_METHODS for m in config['methods']) and config.get('semantics', {}).get('source') != 'synthetic_visible_salience_v1':
        raise ValueError('mechanism experiments must declare synthetic semantics source')
    return config


def run_episode(world, env_config, method, start_seed, output, identity, max_candidates=24,
                semantic_field=None, options=None, predictor=None):
    output.mkdir(parents=True)
    env = GridExplorationEnv(world, env_config, semantic_field)
    obs = env.reset(seed=start_seed)
    policy = (MechanismPolicy(env_config, method, max_candidates, options, predictor)
              if method in MECHANISM_METHODS else FrontierPolicy(env_config, method, max_candidates))
    initial_pose = (*obs.position, obs.heading)
    assets = env.evaluation_assets()
    metrics = env.evaluation_metrics()
    rows = [dict(metrics, action='reset', row=obs.position[0], col=obs.position[1], heading=obs.heading,
                 goal_row=None, goal_col=None, goal_heading=None, new_goal=False,
                 predicted_gain_cells=0, estimated_distance_cells=0, goal_score=0., decision_ms=0.)]
    masks = [np.packbits(obs.visible.ravel())]
    durations = []
    started = time.perf_counter()
    while True:
        decision_start = time.perf_counter()
        # Do not pass env, metrics, world or evaluation assets into the policy.
        decision = policy.act(obs)
        duration_ms = (time.perf_counter() - decision_start) * 1000
        durations.append(duration_ms)
        obs, done = env.step(decision.action)
        if isinstance(policy, MechanismPolicy):
            policy.observe_outcome(obs, done)
        metrics = env.evaluation_metrics()
        rows.append(dict(metrics, action=decision.action, row=obs.position[0], col=obs.position[1],
                         heading=obs.heading, goal_row=decision.goal[0] if decision.goal else None,
                         goal_col=decision.goal[1] if decision.goal else None,
                         goal_heading=decision.goal_heading, new_goal=decision.new_goal,
                         predicted_gain_cells=decision.predicted_gain_cells,
                         estimated_distance_cells=decision.distance_cells, goal_score=decision.score,
                         decision_ms=duration_ms))
        masks.append(np.packbits(obs.visible.ravel()))
        if done:
            break
    elapsed = time.perf_counter() - started
    if isinstance(policy, MechanismPolicy):
        for filename, records in [('candidates.jsonl', policy.selection_audit), ('goal_attempts.jsonl', policy.attempts)]:
            with (output / filename).open('w') as stream:
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
        write_json(output / 'topology.json', dict(nodes=policy.topology.nodes, edges=policy.topology.edges))
        assets['final_semantic_belief'] = policy.semantic
        if policy.topology.labels is not None:
            assets['final_region_labels'] = policy.topology.labels
    if semantic_field is not None:
        assets['synthetic_semantic_ground_truth'] = semantic_field
    with (output / 'steps.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output / 'observations.npz', visible_packed=np.stack(masks),
                        map_shape=np.array(world.shape), final_belief=obs.belief,
                        poses=np.array([(r['row'], r['col'], r['heading']) for r in rows]),
                        **assets)
    coverage = [r['coverage_ratio'] for r in rows]
    # STOP does not remove the remaining fixed action budget from AUC.
    curve = np.interp(np.arange(env_config.max_steps + 1),
                      np.arange(len(coverage)), coverage)
    result = dict(identity, **metrics, schema=SCHEMA, status='complete',
                  method=method, start_seed=start_seed, initial_pose=list(initial_pose),
                  goal_count=policy.goal_count, forward_attempts=sum(r['action'] == 'forward' for r in rows),
                  coverage_auc=float(np.trapz(curve) / env_config.max_steps),
                  steps_to_80=next((i for i, c in enumerate(coverage) if c >= .8), None),
                  steps_to_90=next((i for i, c in enumerate(coverage) if c >= .9), None),
                  episode_wall_time_s=elapsed,
                  decision_ms_median=float(np.median(durations)),
                  decision_ms_p95=float(np.percentile(durations, 95)),
                  artifact_dir=str(output.name),
                  ground_truth_sha256=hashlib.sha256(world.tobytes()).hexdigest())
    result.update(policy.diagnostics() if isinstance(policy, MechanismPolicy) else
                  dict(semantic_updates=0, topo_updates=0, rpn_calls=0, goal_changes=policy.goal_count))
    if semantic_field is not None:
        result['semantic_sha256'] = hashlib.sha256(semantic_field.tobytes()).hexdigest()
    if predictor is not None:
        predictor.verify_frozen()
        result['rpn_sha256'] = predictor.sha256
        result['rpn_frozen_verified'] = True
    write_json(output / 'episode.json', result)
    return result


def run(config, output):
    protocol = verify_protocol(config, ROOT) if config.get('frozen_protocol') else None
    predictor = None
    if any(m.endswith('_rpn') for m in config['methods']):
        model_path = Path(config['rpn_checkpoint'])
        predictor = CandidateRPN(model_path if model_path.is_absolute() else ROOT / model_path)
        if predictor.goal_budget != config.get('mechanisms', {}).get('goal_budget', 32):
            raise ValueError('RPN/controller goal budget mismatch')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    if protocol is not None:
        write_json(output / 'frozen_protocol.json', protocol)
        with zipfile.ZipFile(output / 'frozen_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in protocol['files_sha256']:
                archive.write(ROOT / name, name)
    (output / 'requirements.lock.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    metadata = {'schema': SCHEMA, 'status': 'running', 'config_sha256': digest,
                'created_at': datetime.now(timezone.utc).isoformat(), 'code': code_fingerprint(ROOT),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'python': platform.python_version(),
                'expected_episodes': len(config['maps']) * len(config['start_seeds']) * len(config['methods']),
                'policy_input': 'observed occupancy, visibility, odometry and collision only',
                'coverage_reference': 'start_connected_robot_center_grid',
                'neural_training': False,
                'limitations': ['synthetic 2D geometry', 'perfect odometry',
                                'no RGB-D, semantics, learned SLAM or PPO',
                                'optimistic unknown-space gain heuristic, not measured mutual information']}
    if config.get('semantics'):
        metadata['policy_input'] += '; currently visible synthetic salience (remembered by policy)'
        metadata['limitations'][2] = 'synthetic salience only; no RGB-D, open-vocabulary perception, learned SLAM or PPO'
        metadata['mechanisms'] = config.get('mechanisms', {})
    if predictor is not None:
        metadata['rpn'] = dict(sha256=predictor.sha256, task='bounded_controller_success',
                               architecture='candidate_mlp_5_16_1', frozen=True)
        (output / 'rpn_frozen.npz').write_bytes(predictor.path.read_bytes())
    if protocol is not None:
        metadata['frozen_protocol_verified_at_start'] = True
    write_json(output / 'run_metadata.json', metadata)
    env_config = GridConfig(**config['environment'])
    failures = 0
    try:
        for entry in config['maps']:
            world = generate_layout(entry['layout'], config['map_size'], entry['seed'], config['door_width'])
            map_id = f"{entry['layout']}_seed{entry['seed']}"
            semantic = (synthetic_semantics(world.shape, entry['seed'] + config['semantics'].get('seed_offset', 10000))
                        if config.get('semantics') else None)
            for seed in config['start_seeds']:
                for method in config['methods']:
                    name = f'{map_id}_start{seed}_{method}'
                    identity = {'map_id': map_id, 'layout': entry['layout'], 'map_seed': entry['seed'],
                                'episode_key': name}
                    try:
                        result = run_episode(world, env_config, method, seed, output / name, identity,
                                             config.get('max_candidates', 24), semantic,
                                             config.get('mechanisms'), predictor if method.endswith('_rpn') else None)
                        print(f"{name}: coverage={result['coverage_ratio']:.3f} steps={result['steps']} "
                              f"time={result['episode_wall_time_s']:.2f}s", flush=True)
                    except Exception as exc:
                        failures += 1
                        result = dict(identity, method=method, start_seed=seed, schema=SCHEMA,
                                      status='failed', error=str(exc))
                        (output / f'{name}.error.txt').write_text(traceback.format_exc())
                        print(f'{name}: FAILED: {exc}', file=sys.stderr, flush=True)
                    with (output / 'episodes.jsonl').open('a') as stream:
                        stream.write(json.dumps(result, allow_nan=False) + '\n')
        if protocol is not None:
            verify_protocol(config, ROOT)
            metadata['frozen_protocol_verified_at_end'] = True
        metadata['status'] = 'complete' if failures == 0 else 'completed_with_failures'
    finally:
        metadata['failed_episodes'] = failures
        metadata['peak_process_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        write_json(output / 'run_metadata.json', metadata)
    from scripts.analyze_cpu_results import analyze
    analyze(output)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/cpu/pilot.json')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sys.exit(1 if run(load_config(args.config), args.output) else 0)


if __name__ == '__main__':
    main()
