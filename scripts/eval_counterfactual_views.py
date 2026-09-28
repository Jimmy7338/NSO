#!/usr/bin/env python3
"""Paid-action, fixed-history view-utility experiment; development only.

Simulator truth is confined to physical execution and post-score evaluation.
Every scorer shares one candidate pool and each physical branch is run once.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
import open3d as o3d
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.virtual3d_inspection_v4 import InspectionConfigV4, InspectionWorldV4
from env.virtual3d import camera_pose
from env.grid_exploration import GridConfig
from nso.quality_coverage3d import QualityCoveragePolicy
from nso.semantic_completion_v3 import SemanticHistoryMapperV3, ObjectCompletionModel
from nso.route_coverage_v2 import orientation_graph
from utils.cpu_protocol import file_hash, digest_json
from utils.grid_geometry import DIRECTIONS
from utils.reconstruction_metrics import ReconstructionEvaluator
from utils.rgbd_contract import RGBDFrame, PlanarScan


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save_mesh(path, mesh):
    np.savez_compressed(path, vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles))


def grid_config(c):
    return GridConfig(resolution_m=c.resolution_m, robot_radius_m=c.robot_radius_m,
                      sensor_range_m=c.max_depth_m, sensor_fov_deg=360, max_steps=c.max_steps)


def advance_states(initial, actions):
    """Known-map path simulation only; real world.step later checks execution."""
    states = [tuple(map(int, initial))]
    for action in actions:
        r, col, heading = states[-1]
        if action == 'forward':
            dr, dc = DIRECTIONS[heading]; r += dr; col += dc
        elif action in ('left', 'right'):
            heading = (heading + (1 if action == 'right' else -1)) % 4
        else:
            raise ValueError('only paid movement/turn actions are allowed')
        states.append((int(r), int(col), int(heading)))
    return states


def states_to_actions(states):
    actions = []
    for a, b in zip(states, states[1:]):
        if a[:2] == b[:2]:
            delta = (b[2] - a[2]) % 4
            if delta not in (1, 3): raise ValueError('invalid turn in orientation route')
            actions.append('right' if delta == 1 else 'left')
        else:
            dr, dc = DIRECTIONS[a[2]]
            if b != (a[0]+dr, a[1]+dc, a[2]): raise ValueError('invalid forward orientation route')
            actions.append('forward')
    return actions


def path_states(previous, start, target, cells, reverse=False):
    chain = [int(target)]
    while chain[-1] != start:
        nxt = int(previous[chain[-1]])
        if nxt < 0 or len(chain) > len(previous): raise ValueError('unreachable route')
        chain.append(nxt)
    if not reverse: chain.reverse()
    return [(int(cells[state//4, 0]), int(cells[state//4, 1]), state % 4) for state in chain]


def quantile_take(rows, count):
    if len(rows) <= count: return rows[:]
    return [rows[i] for i in np.linspace(0, len(rows)-1, count, dtype=int)]


def candidate_routes(mapper, obs, max_actions=48, max_candidates=12):
    """Geometric pool only, with actual directed outgoing and return costs."""
    c = mapper.config; policy = QualityCoveragePolicy(grid_config(c), c, 'coverage')
    policy.set_surface_evidence(mapper.evidence())
    safe = policy._safe(obs)
    if not safe[obs.position]: return [], {'reason': 'initial_pose_not_in_known_safe_map'}
    graph, cells, ids = orientation_graph(safe)
    start = int(ids[obs.position])*4 + obs.heading
    outward, prev = dijkstra(graph, directed=True, indices=start, return_predecessors=True)
    backward, backprev = dijkstra(graph.T.tocsr(), directed=True, indices=start, return_predecessors=True)
    minimum = outward.reshape(-1, 4).min(axis=1)
    distances = np.full(safe.shape, -1.)
    valid = np.isfinite(minimum); distances[tuple(cells[valid].T)] = minimum[valid]
    model = ObjectCompletionModel(mapper, False)
    anchor = None
    if policy._select(obs, safe): anchor = (*policy.goal, int(policy.goal_heading))
    frontier = [(*cell, h) for cell in policy._candidates(obs, safe, distances) for h in range(4)]
    objects = []
    for cell in model.candidates(safe, distances, limit=64):
        xy = np.array([(cell[1]+.5)*c.resolution_m, (safe.shape[0]-cell[0]-.5)*c.resolution_m])
        center = min(model.objects, key=lambda obj: np.linalg.norm(obj['center'][:2]-xy))['center'][:2]
        delta = center - xy
        heading = int(np.argmax(np.array([[0, 1], [1, 0], [0, -1], [-1, 0]]) @ delta))
        objects.append((*cell, heading))
    uniform = [(int(r), int(col), h) for r, col in cells[valid]
               if r % 4 == 0 and col % 4 == 0 for h in range(4)]
    pools = {}; rejected = []
    for group, poses in [('frontier', frontier), ('object', objects), ('uniform', uniform)]:
        rows = []
        for r, col, heading in sorted(set(poses)):
            state = int(ids[r, col])*4 + heading
            cost = outward[state] + backward[state]
            if not np.isfinite(cost) or outward[state] < 1 or cost > max_actions:
                rejected.append({'group': group, 'pose': [r, col, heading], 'reason': 'unreachable_or_outside_paid_round_trip_budget'})
                continue
            rows.append({'pose': (r, col, heading), 'state': state, 'cost': int(cost), 'group': group})
        pools[group] = sorted(rows, key=lambda row: (row['cost'], row['pose']))
    picked = []
    anchor_row = next((row for row in pools['frontier'] if row['pose'] == anchor), None)
    if anchor_row is not None: picked.append(anchor_row)
    picked += quantile_take([row for row in pools['frontier'] if row is not anchor_row], 4-len(picked))
    picked += quantile_take(pools['object'], 4) + quantile_take(pools['uniform'], 4)
    chosen = []; seen = set()
    for row in picked:
        if row['pose'] not in seen: chosen.append(row); seen.add(row['pose'])
    for row in sorted(sum(pools.values(), []), key=lambda row: (row['cost'], row['pose'], row['group'])):
        if len(chosen) >= max_candidates: break
        if row['pose'] not in seen: chosen.append(row); seen.add(row['pose'])
    routes = []
    for i, row in enumerate(chosen[:max_candidates]):
        out = path_states(prev, start, row['state'], cells)
        back = path_states(backprev, start, row['state'], cells, reverse=True)
        states = out + back[1:]; actions = states_to_actions(states)
        assert len(actions) == row['cost'] and states[0] == states[-1]
        assert advance_states(states[0], actions) == states
        routes.append({'candidate_id': i, 'group': row['group'], 'pose': row['pose'], 'states': states,
                       'actions': actions, 'cost': len(actions), 'arrival_action': len(out)-1,
                       'is_coverage_anchor': row['pose'] == anchor})
    return routes, {'safe_hash': hashlib.sha256(safe.tobytes()).hexdigest(),
                    'pool_counts': {k: len(v) for k, v in pools.items()}, 'rejected': rejected,
                    'coverage_anchor': anchor, 'anchor_within_budget': anchor_row is not None,
                    'candidate_geometry_objects': len(model.objects)}


def update_collision(mapper, position, heading, collision):
    if collision:
        dr, dc = DIRECTIONS[heading]; r, col = position[0]+dr, position[1]+dc
        if 0 <= r < mapper.shape[0] and 0 <= col < mapper.shape[1]: mapper.belief[r, col] = 1


def remap(frames, scans, records, c, shape, condition='aligned'):
    mapper = SemanticHistoryMapperV3(shape, c, c.truncation_m)
    for frame, scan, row in zip(frames, scans, records):
        if condition != 'aligned':
            labels = np.zeros_like(frame.semantic)
            if condition == 'shuffled':
                labels = frame.semantic.copy(); labels[frame.semantic == 2] = 3; labels[frame.semantic == 3] = 2
            frame = replace(frame, semantic=labels)
        mapper.update(frame, scan)
        update_collision(mapper, row['position'], row['heading'], row['collision'])
    return mapper


def generate_prefix(world, checkpoints, out):
    c = world.config; mapper = SemanticHistoryMapperV3(world.shape, c, c.truncation_m)
    policy = QualityCoveragePolicy(grid_config(c), c, 'coverage')
    out.mkdir(); (out/'frames').mkdir(); (out/'scans').mkdir()
    frames = []; scans = []; rows = []; action = 'reset'; done = collision = False
    frame = world.sense()
    while True:
        scan = world.scan(); mapper.update(frame, scan)
        update_collision(mapper, world.position, world.heading, collision)
        obs = mapper.observation(world.position, world.heading, world.step_count, collision)
        if world.step_count: policy.observe_outcome(obs, done)
        row = {'step': world.step_count, 'position': list(world.position), 'heading': world.heading,
               'collision': collision, 'collisions': world.collisions, 'moves': world.moves,
               'action': action, 'done': done}
        frame.save(out/'frames'/f'{world.step_count:04d}.npz'); scan.save(out/'scans'/f'{world.step_count:04d}.npz')
        frames.append(frame); scans.append(scan); rows.append(row)
        if done or world.step_count >= max(checkpoints): break
        policy.set_surface_evidence(mapper.evidence())
        decision = policy.act(obs); action = decision.action
        frame, collision, done = world.step(action)
    write_json(out/'records.json', rows)
    write_json(out/'source_hashes.json', {str(p.relative_to(out)): file_hash(p) for p in out.rglob('*.npz')})
    return frames, scans, rows


def state_restore(world, row, expected_frame, expected_scan):
    # Fork only a previously executed prefix state; no endpoint teleportation.
    world.position = tuple(row['position']); world.heading = int(row['heading'])
    world.step_count = int(row['step']); world.collisions = int(row['collisions']); world.moves = int(row['moves'])
    actual = world.sense(); scan = world.scan()
    for key in RGBDFrame.__dataclass_fields__: np.testing.assert_array_equal(getattr(actual, key), getattr(expected_frame, key))
    for key in PlanarScan.__dataclass_fields__: np.testing.assert_array_equal(getattr(scan, key), getattr(expected_scan, key))


def coverage(mapper, reachable):
    return float(np.count_nonzero((mapper.belief != -1) & reachable)/reachable.sum())


def fixture_task(config, entry, output):
    from nso.counterfactual_view_scoring import score_routes, CounterfactualScoreConfig
    from utils.counterfactual_surface_visibility import reference_visible, surface_increment
    c = InspectionConfigV4(**(config['environment'] | entry.get('environment', {})))
    if c.pose_noise_m != 0: raise ValueError('this protocol requires explicit zero pose-input error')
    world = InspectionWorldV4(c, seed=entry['seed'])
    key = f"{entry.get('group','main')}_{entry['seed']}_{digest_json(asdict(c))[:12]}"
    out = output/key; out.mkdir(); write_json(out/'fixture.json', {'entry': entry, 'environment': asdict(c)})
    checkpoints = entry.get('checkpoints', config['checkpoints'])
    frames, scans, records = generate_prefix(world, checkpoints, out/'prefix')
    prepared = []
    for index in checkpoints:
        folder = out/f'history_{index:04d}'; folder.mkdir()
        if index >= len(frames) or records[index]['done']:
            write_json(folder/'status.json', {'status': 'unavailable_prefix', 'requested_step': index, 'last_step': records[-1]['step']})
            continue
        past_f, past_s, past_r = frames[:index+1], scans[:index+1], records[:index+1]
        mappers = {condition: remap(past_f, past_s, past_r, c, world.shape, condition)
                   for condition in ('aligned', 'shuffled', 'absent')}
        row = past_r[-1]; obs = mappers['aligned'].observation(tuple(row['position']), row['heading'], index, row['collision'])
        routes, audit = candidate_routes(mappers['aligned'], obs, config['branch_actions'], config['max_candidates'])
        write_json(folder/'candidates.json', routes); write_json(folder/'candidate_audit.json', audit)
        predictions = score_routes(mappers, routes, CounterfactualScoreConfig(grid_config(c), c))
        write_json(folder/'predictions.json', predictions)
        save_mesh(folder/'prefix_mesh.npz', mappers['aligned'].mesh())
        np.savez_compressed(folder/'prefix_map.npz', belief=mappers['aligned'].belief)
        prepared.append((index, folder, routes, predictions))
        del mappers
    # All candidate scores are persisted before evaluator truth is queried.
    sealed = {str(p.relative_to(out)): file_hash(p) for p in out.glob('history_*/*.json')}
    write_json(out/'pre_outcome_seal.json', sealed)
    evaluator = ReconstructionEvaluator(world, config['reference_samples'])
    point_weight = world.mesh.get_surface_area()/config['reference_samples']
    np.savez_compressed(out/'reference.npz', points=evaluator.reference, classes=evaluator.classes,
                        weights=np.full(len(evaluator.reference), point_weight), reachable=world.reachable,
                        vertices=np.asarray(world.mesh.vertices), triangles=np.asarray(world.mesh.triangles))
    summaries = []
    for index, folder, routes, predictions in prepared:
        past_f, past_s, past_r = frames[:index+1], scans[:index+1], records[:index+1]
        initial = remap(past_f, past_s, past_r, c, world.shape)
        c0 = coverage(initial, world.reachable)
        before = evaluator.evaluate(initial.mesh(), c0, config['thresholds_m'])
        seen = np.zeros(len(evaluator.reference), bool)
        for frame, row in zip(past_f, past_r):
            pose = camera_pose(tuple(row['position']), row['heading'], c, world.shape[0])
            seen |= reference_visible(evaluator.reference, frame, evaluator.truth, pose, c.max_depth_m)
        outcomes = []
        for route in routes:
            if shutil.disk_usage(output).free < 100*2**20: raise RuntimeError('disk reserve reached; preserve partial evidence')
            branch_dir = folder/f"candidate_{route['candidate_id']:03d}"; branch_dir.mkdir()
            (branch_dir/'frames').mkdir(); (branch_dir/'scans').mkdir()
            branch = remap(past_f, past_s, past_r, c, world.shape)
            state_restore(world, past_r[-1], past_f[-1], past_s[-1])
            masks = {'outbound': np.zeros_like(seen), 'endpoint': np.zeros_like(seen), 'return': np.zeros_like(seen)}
            branch_rows = []; checkpoints_metrics = [{'action_index': 0, 'coverage_2d': c0, **before}]
            failure = None
            for j, action in enumerate(route['actions'], 1):
                frame, collision, done = world.step(action); scan = world.scan()
                frame.save(branch_dir/'frames'/f'{j:04d}.npz'); scan.save(branch_dir/'scans'/f'{j:04d}.npz')
                branch.update(frame, scan); update_collision(branch, world.position, world.heading, collision)
                actual = (*world.position, world.heading)
                expected = tuple(route['states'][j])
                stage = 'outbound' if j < route['arrival_action'] else ('endpoint' if j == route['arrival_action'] else 'return')
                pose = camera_pose(world.position, world.heading, c, world.shape[0])
                masks[stage] |= reference_visible(evaluator.reference, frame, evaluator.truth, pose, c.max_depth_m)
                now_c = coverage(branch, world.reachable)
                branch_rows.append({'action_index': j, 'absolute_step': world.step_count, 'action': action,
                                    'position': list(world.position), 'heading': world.heading, 'collision': collision,
                                    'coverage_2d': now_c, 'stage': stage})
                if collision or actual != expected: failure = 'collision_or_execution_mismatch'
                if done and j < len(route['actions']): failure = 'world_budget_exhausted'
                if j == route['arrival_action'] or j == len(route['actions']) or failure:
                    measured = {'action_index': j, 'coverage_2d': now_c,
                                **evaluator.evaluate(branch.mesh(), now_c, config['thresholds_m'])}
                    checkpoints_metrics.append(measured)
                    save_mesh(branch_dir/('arrival_mesh.npz' if j == route['arrival_action'] else 'final_mesh.npz'), branch.mesh())
                if failure: break
            if not branch_rows: raise AssertionError('zero-cost branch was admitted')
            if not (branch_dir/'final_mesh.npz').exists(): save_mesh(branch_dir/'final_mesh.npz', branch.mesh())
            if failure is None: assert (*world.position, world.heading) == tuple(route['states'][0])
            future = np.logical_or.reduce(list(masks.values())); cumulative = seen.copy(); parts = {}
            for stage, mask in masks.items():
                parts[stage] = surface_increment(cumulative, mask, point_weight); cumulative |= mask
            new_area = surface_increment(seen, future, point_weight)
            np.testing.assert_allclose(sum(parts.values()), new_area, atol=1e-10, rtol=0)
            final = checkpoints_metrics[-1]; paid = len(branch_rows)
            result = {'candidate_id': route['candidate_id'], 'paid_actions': paid, 'planned_actions': route['cost'],
                      'arrival_actions': route['arrival_action'], 'failure': failure,
                      'new_area_m2': new_area, 'area_per_action': new_area/paid, 'stage_new_area_m2': parts,
                      'f1_gain_05cm': final['f1_05cm']-before['f1_05cm'],
                      'f1_gain_per_action': (final['f1_05cm']-before['f1_05cm'])/paid,
                      'coverage_gain_m2': (final['coverage_2d']-c0)*world.reachable.sum()*c.resolution_m**2,
                      'before': before, 'after': final}
            for threshold in config['thresholds_m']:
                tag = f'{round(threshold*100):02d}cm'
                result[f'branch_joint_auc_{tag}'] = float(np.trapz(np.interp(np.arange(config['branch_actions']+1),
                    [x['action_index'] for x in checkpoints_metrics], [x[f'joint_{tag}'] for x in checkpoints_metrics]))/config['branch_actions'])
            np.savez_compressed(branch_dir/'visibility.npz', prefix=seen, **masks, union=future)
            write_json(branch_dir/'actions.json', branch_rows); write_json(branch_dir/'metrics.json', checkpoints_metrics)
            write_json(branch_dir/'outcome.json', result)
            outcomes.append(result); del branch
        write_json(folder/'outcomes.json', outcomes)
        summary = {'fixture': key, 'history_step': index, 'seed': entry['seed'], 'group': entry.get('group','main'),
                   'depth_sigma_m': c.depth_sigma_m, 'candidate_count': len(routes), 'branch_failures': sum(x['failure'] is not None for x in outcomes),
                   'marker_pixels_history': sum(int(np.count_nonzero(f.semantic)) for f in past_f),
                   'status': 'complete' if routes else 'no_feasible_candidates', 'predictions': str((folder/'predictions.json').relative_to(output)),
                   'outcomes': str((folder/'outcomes.json').relative_to(output))}
        write_json(folder/'summary.json', summary); summaries.append(summary)
        print('completed', key, 'step', index, 'branches', len(routes), flush=True)
        del initial
    assert all(file_hash(out/name) == sha for name, sha in sealed.items()), 'pre-outcome scoring changed'
    write_json(out/'summary.json', summaries)
    return summaries


def summarize(output, records):
    rows = []
    for history in records:
        predictions = json.loads((output/history['predictions']).read_text())
        outcomes = json.loads((output/history['outcomes']).read_text())
        by_id = {x['candidate_id']: x for x in outcomes}
        if not by_id: continue
        candidates = [p for p in predictions['candidates'] if p['candidate_id'] in by_id]
        for scorer in ('G', 'O', 'S', 'X', 'M', 'N'):
            chosen = next(p for p in candidates if p['candidate_id'] == predictions['selected'][scorer])
            result = by_id[chosen['candidate_id']]
            pred = np.array([p['scores'][scorer]['score'] for p in candidates])
            actual = np.array([by_id[p['candidate_id']]['area_per_action'] for p in candidates])
            correlation = spearmanr(pred, actual).statistic if np.ptp(pred)>0 and np.ptp(actual)>0 else np.nan
            pairs = [(i,j) for i in range(len(pred)) for j in range(i+1,len(pred)) if actual[i] != actual[j]]
            ranking = float(np.mean([1 if (pred[i]-pred[j])*(actual[i]-actual[j])>0 else .5 if pred[i]==pred[j] else 0 for i,j in pairs])) if pairs else None
            rows.append({**{k: history[k] for k in ('fixture','history_step','seed','group','depth_sigma_m')},
                         'scorer': scorer, 'chosen': chosen['candidate_id'],
                         **{k: result[k] for k in ('new_area_m2','area_per_action','f1_gain_05cm','f1_gain_per_action','coverage_gain_m2','failure')},
                         'spearman_area_rate': float(correlation) if np.isfinite(correlation) else None,
                         'pairwise_area_rate_accuracy': ranking,
                         'area_rate_regret': float(actual.max()-result['area_per_action'])})
    write_json(output/'choices.json', rows)
    averages = {scorer: {key: float(np.mean([r[key] for r in rows if r['scorer']==scorer]))
                         for key in ('new_area_m2','area_per_action','f1_gain_05cm','f1_gain_per_action','coverage_gain_m2','area_rate_regret')}
                for scorer in ('G','O','S','X','M','N')} if rows else {}
    write_json(output/'summary.json', {'histories': records, 'scorer_averages': averages,
                'scope': 'development common-history paid-action utility, not whole-system advantage',
                'independent_confirmation': False, 'calibration_fitted': False})


def run(config, output):
    reserved = set(range(301,309)) | set(range(401,413))
    if any(entry['seed'] in reserved for entry in config['fixtures']): raise ValueError('reserved confirmation seed')
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    names = sorted({str(p.relative_to(ROOT)) for folder in ('env','nso','utils') for p in (ROOT/folder).glob('*.py')} |
                   {'scripts/eval_counterfactual_views.py', 'docs/research/COUNTERFACTUAL_VIEW_UTILITY_PROTOCOL.md', 'requirements-3d.lock.txt'})
    hashes = {name: file_hash(ROOT/name) for name in names}
    metadata = {'status': 'running', 'config_sha256': digest_json(config), 'source_sha256': hashes,
                'scope': config['scope'], 'started_unix': time.time(), 'replayed': False}
    write_json(output/'metadata.json', metadata); write_json(output/'config.json', config)
    with zipfile.ZipFile(output/'sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in names: archive.write(ROOT/name, name)
    records = []
    try:
        with ProcessPoolExecutor(max_workers=config.get('workers',2)) as pool:
            futures = [pool.submit(fixture_task, config, entry, output) for entry in config['fixtures']]
            for future in as_completed(futures): records.extend(future.result())
        assert all(file_hash(ROOT/name)==sha for name,sha in hashes.items()), 'sources changed during run'
        summarize(output, records)
        unavailable = [json.loads(p.read_text()) | {'path': str(p.relative_to(output))}
                       for p in output.glob('*/history_*/status.json')]
        metadata.update(status='complete', histories=len(records), branches=sum(r['candidate_count'] for r in records),
                        unavailable_histories=unavailable,
                        elapsed_s=time.time()-metadata['started_unix'])
    except Exception as exc:
        metadata.update(status='failed', error=repr(exc)); write_json(output/'metadata.json', metadata)
        raise
    write_json(output/'metadata.json', metadata)
    write_json(output/'artifact_hashes.json', {str(p.relative_to(output)): file_hash(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(json.loads(args.config.read_text()), args.output)
