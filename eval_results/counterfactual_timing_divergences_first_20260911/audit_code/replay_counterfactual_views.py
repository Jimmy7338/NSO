#!/usr/bin/env python3
"""Independently replay a sealed development counterfactual run.

Only verification.json is written to the run. All simulator, mapper, evaluator
and visibility dependencies are loaded from its verified sources.zip in spawned
workers, never from today's checkout. Recorded actions are executed verbatim;
no policy is called and truth never repairs an action or selects a candidate.

--max-branches is a smoke check and always reports partial, even if its limit
exceeds the number of branches. A successful replay certifies reproducibility
and accounting, not semantic efficacy, calibration, or independent confirmation.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
import hashlib
import importlib
import json
import multiprocessing
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import tempfile
import time
import traceback
import zipfile

import numpy as np

SCORERS = ('G', 'O', 'S', 'X', 'M', 'N')
STAGES = ('outbound', 'endpoint', 'return')
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value, compact=False):
    options = {'sort_keys': True, 'allow_nan': False}
    if compact:
        options['separators'] = (',', ':')
    return hashlib.sha256(json.dumps(value, **options).encode()).hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f'nonfinite JSON value: {value}')
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def relative_file(root, name):
    require(isinstance(name, str) and name, 'empty/non-string relative path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in name,
            f'unsafe artifact path: {name}')
    result = root.joinpath(*path.parts)
    require(result.resolve().is_relative_to(root.resolve()), f'path escaped root: {name}')
    require(not result.is_symlink(), f'symlink artifact: {name}')
    return result


def compare(actual, expected, context, tolerance=1e-9):
    """Recursive structural check; numerical tolerance never ignores missing keys."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f'{context}: keys differ')
        for key in expected:
            compare(actual[key], expected[key], f'{context}/{key}', tolerance)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f'{context}: length differs')
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f'{context}/{i}', tolerance)
    elif isinstance(expected, (bool, str)) or expected is None:
        require(type(actual) is type(expected) and actual == expected, f'{context}: {actual!r} != {expected!r}')
    elif isinstance(expected, (int, np.integer)):
        require(isinstance(actual, (int, np.integer)) and not isinstance(actual, bool) and actual == expected,
                f'{context}: integer differs')
    else:
        require(isinstance(actual, (int, float, np.number)) and not isinstance(actual, bool)
                and np.isfinite(actual) and np.isfinite(expected)
                and abs(actual - expected) <= tolerance,
                f'{context}: {actual!r} != {expected!r}')


def compare_array(actual, expected, context):
    left, right = np.asarray(actual), np.asarray(expected)
    require(left.shape == right.shape and left.dtype == right.dtype, f'{context}: array shape/dtype differs')
    np.testing.assert_array_equal(left, right, err_msg=context)


def compare_npz(path, expected):
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(expected), f'{path}: NPZ keys differ')
        for key, value in expected.items():
            compare_array(saved[key], value, f'{path}/{key}')


def mesh_arrays(mesh):
    return {'vertices': np.asarray(mesh.vertices), 'triangles': np.asarray(mesh.triangles)}


def check_manifest(run, manifest):
    require(isinstance(manifest, dict) and manifest, 'missing artifact manifest')
    require({'metadata.json', 'config.json', 'sources.zip'} <= manifest.keys(), 'incomplete artifact manifest')
    actual = {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file()
              and p != run / 'artifact_hashes.json' and p != run / 'verification.json'
              and (str(p.relative_to(run)) in manifest or not p.is_relative_to(run / 'analysis'))}
    require(actual == set(manifest), 'artifact inventory differs from sealed manifest')
    for name, expected in manifest.items():
        path = relative_file(run, name)
        require(path.is_file() and file_hash(path) == expected, f'artifact hash mismatch: {name}')


def extract_sources(run, destination, hashes):
    """Validate every byte before extracting; reject duplicate/unsafe entries."""
    require(isinstance(hashes, dict) and hashes, 'missing source hashes')
    with zipfile.ZipFile(run / 'sources.zip') as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(hashes), 'source archive inventory differs')
        for name in names:
            target = relative_file(destination, name)
            payload = archive.read(name)
            require(hashlib.sha256(payload).hexdigest() == hashes[name], f'archived source hash mismatch: {name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)


def validate_route(route, initial, safe, budget):
    states = np.asarray(route['states'])
    require(states.ndim == 2 and states.shape[1] == 3 and len(states) >= 3,
            'route must contain a paid closed tour')
    require(np.issubdtype(states.dtype, np.integer), 'noninteger route state')
    require(route['cost'] == len(route['actions']) == len(states) - 1
            and 0 < route['cost'] <= budget, 'route budget/cost mismatch')
    require(isinstance(route['arrival_action'], int)
            and 0 < route['arrival_action'] < route['cost'], 'invalid arrival action')
    compare(states[0].tolist(), list(initial), 'route initial')
    compare(states[-1].tolist(), list(initial), 'route return')
    compare(states[route['arrival_action']].tolist(), route['pose'], 'route endpoint')
    for r, col, h in states:
        require(0 <= r < safe.shape[0] and 0 <= col < safe.shape[1] and 0 <= h < 4,
                'route outside map/headings')
        require(safe[r, col] or (r, col) == tuple(initial[:2]), 'route leaves common known-safe map')
    for before, after, action in zip(states, states[1:], route['actions']):
        r, col, h = map(int, before)
        if action == 'forward':
            dr, dc = DIRECTIONS[h]
            successor = [r + dr, col + dc, h]
        elif action in ('left', 'right'):
            successor = [r, col, (h + (1 if action == 'right' else -1)) % 4]
        else:
            raise ValueError('unpaid or unknown branch action')
        compare(after.tolist(), successor, 'route successor')


def validate_predictions(predictions, routes):
    compare(predictions['scorers'], list(SCORERS), 'scorer inventory')
    rows = predictions['candidates']
    compare([x['candidate_id'] for x in rows], [x['candidate_id'] for x in routes], 'shared scorer candidate order')
    prepared = []
    for row, route in zip(rows, routes):
        compare(row['cost'], route['cost'], 'scorer cost')
        compare(row['states_sha256'], json_hash(route['states'], compact=True), 'scorer path hash')
        require(set(row['scores']) == set(SCORERS), 'scorer keys differ')
        for scorer in SCORERS:
            score = row['scores'][scorer]
            require(np.isfinite(score['score']) and np.isfinite(score['final_score']), 'nonfinite prediction')
            compare(score['score'], score['final_score'], 'score/final_score', tolerance=0)
        compare(row['scores']['M'], row['scores']['G'], 'missing-label geometry invariant', tolerance=0)
        prepared.append({key: route[key] for key in ('candidate_id', 'states', 'cost')})
    for scorer in SCORERS:
        order = [row['candidate_id'] for _, row in sorted(enumerate(rows), key=lambda pair:
                 (-pair[1]['scores'][scorer]['final_score'], pair[1]['cost'], pair[0]))]
        compare(predictions['rankings'][scorer], order, f'{scorer} sealed ranking')
        compare(predictions['selected'][scorer], order[0] if order else None, f'{scorer} sealed selection')
    require(predictions['invariants']['validated'] is True, 'scoring invariants were not validated')
    compare(predictions['invariants']['candidate_routes_sha256'], json_hash(prepared, compact=True),
            'common candidate pool hash')


def area_accounting(prefix, masks, weights):
    """Independent weighted union ledger: no renormalization after visibility filtering."""
    prefix = np.asarray(prefix)
    weights = np.asarray(weights)
    require(prefix.ndim == 1 and prefix.dtype == bool and weights.shape == prefix.shape,
            'invalid reference ledger shape')
    require(np.isfinite(weights).all() and (weights >= 0).all(), 'invalid physical area weights')
    cumulative = prefix.copy()
    union = np.zeros_like(prefix)
    parts = {}
    for stage in STAGES:
        mask = np.asarray(masks[stage])
        require(mask.dtype == bool and mask.shape == prefix.shape, 'invalid stage visibility mask')
        parts[stage] = float(weights[mask & ~cumulative].sum())
        cumulative |= mask
        union |= mask
    total = float(weights[union & ~prefix].sum())
    require(abs(sum(parts.values()) - total) <= 1e-10, 'repeated surface gain')
    return union, parts, total


def joint_auc(metrics, budget, tag):
    indices = [row['action_index'] for row in metrics]
    require(budget > 0 and indices[0] == 0 and all(a < b for a, b in zip(indices, indices[1:]))
            and indices[-1] <= budget, 'invalid joint AUC action checkpoints')
    return float(np.trapz(np.interp(np.arange(budget + 1), indices,
                 [row[f'joint_{tag}'] for row in metrics])) / budget)


def worker_init(frozen_root):
    global InspectionConfigV4, InspectionWorldV4, SemanticHistoryMapperV3
    global ReconstructionEvaluator, RGBDFrame, PlanarScan, camera_pose, reference_visible, inflated_obstacles
    sys.path.insert(0, frozen_root)
    module_names = ('env.virtual3d_inspection_v4', 'env.virtual3d', 'nso.semantic_completion_v3',
                    'utils.reconstruction_metrics', 'utils.rgbd_contract',
                    'utils.counterfactual_surface_visibility', 'utils.grid_geometry')
    modules = {name: importlib.import_module(name) for name in module_names}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('env', 'nso', 'utils') and getattr(module, '__file__', None):
            require(Path(module.__file__).resolve().is_relative_to(Path(frozen_root).resolve()),
                    f'live-checkout module imported: {name}')
    InspectionConfigV4 = modules['env.virtual3d_inspection_v4'].InspectionConfigV4
    InspectionWorldV4 = modules['env.virtual3d_inspection_v4'].InspectionWorldV4
    SemanticHistoryMapperV3 = modules['nso.semantic_completion_v3'].SemanticHistoryMapperV3
    ReconstructionEvaluator = modules['utils.reconstruction_metrics'].ReconstructionEvaluator
    RGBDFrame, PlanarScan = modules['utils.rgbd_contract'].RGBDFrame, modules['utils.rgbd_contract'].PlanarScan
    camera_pose = modules['env.virtual3d'].camera_pose
    reference_visible = modules['utils.counterfactual_surface_visibility'].reference_visible
    inflated_obstacles = modules['utils.grid_geometry'].inflated_obstacles


def collision_update(mapper, row):
    if row['collision']:
        dr, dc = DIRECTIONS[row['heading']]
        r, col = row['position'][0] + dr, row['position'][1] + dc
        if 0 <= r < mapper.shape[0] and 0 <= col < mapper.shape[1]:
            mapper.belief[r, col] = 1


def remap_prefix(frames, scans, records, config, shape):
    require(len(frames) == len(scans) == len(records), 'truncated prefix pairing')
    mapper = SemanticHistoryMapperV3(shape, config, config.truncation_m)
    for frame, scan, record in zip(frames, scans, records):
        mapper.update(frame, scan)
        collision_update(mapper, record)
    return mapper


def compare_sensor(actual, expected, context):
    for field in fields(expected):
        compare_array(getattr(actual, field.name), getattr(expected, field.name), f'{context}/{field.name}')


def occupancy_coverage(mapper, reachable):
    return float(np.count_nonzero((mapper.belief != -1) & reachable) / reachable.sum())


def inventory_npz(folder, indices):
    require({p.name for p in folder.iterdir()} == {f'{i:04d}.npz' for i in indices},
            f'{folder}: missing or extra raw frames')


def replay_branch(folder, route, world, evaluator, frames, scans, records, config, weights, seen, before):
    c = world.config
    mapper = remap_prefix(frames, scans, records, c, world.shape)
    last = records[-1]
    world.position, world.heading = tuple(last['position']), last['heading']
    world.step_count, world.collisions, world.moves = last['step'], last['collisions'], last['moves']
    compare_sensor(world.sense(), frames[-1], 'branch restored RGB-D')
    compare_sensor(world.scan(), scans[-1], 'branch restored scan')
    rows = read_json(folder / 'actions.json')
    require(0 < len(rows) <= len(route['actions']), 'invalid paid branch length')
    for name in ('frames', 'scans'):
        inventory_npz(folder / name, range(1, len(rows) + 1))
    c0 = occupancy_coverage(mapper, world.reachable)
    metrics = [{'action_index': 0, 'coverage_2d': c0, **before}]
    masks = {stage: np.zeros_like(seen) for stage in STAGES}
    failure = None
    arrival_saved = False
    for j, row in enumerate(rows, 1):
        require(failure is None, 'branch continued after failure')
        action = route['actions'][j - 1]
        generated, collision, done = world.step(action)
        generated_scan = world.scan()
        frame = RGBDFrame.load(folder / 'frames' / f'{j:04d}.npz')
        scan = PlanarScan.load(folder / 'scans' / f'{j:04d}.npz')
        compare_sensor(generated, frame, f'branch action {j} RGB-D')
        compare_sensor(generated_scan, scan, f'branch action {j} scan')
        mapper.update(frame, scan)
        collision_update(mapper, {'position': world.position, 'heading': world.heading, 'collision': collision})
        actual = (*world.position, world.heading)
        stage = 'outbound' if j < route['arrival_action'] else 'endpoint' if j == route['arrival_action'] else 'return'
        now_c = occupancy_coverage(mapper, world.reachable)
        compare(row, {'action_index': j, 'absolute_step': world.step_count, 'action': action,
                      'position': list(world.position), 'heading': world.heading, 'collision': collision,
                      'coverage_2d': now_c, 'stage': stage}, 'branch action record', tolerance=1e-12)
        physical = camera_pose(world.position, world.heading, c, world.shape[0])
        masks[stage] |= reference_visible(evaluator.reference, frame, evaluator.truth, physical, c.max_depth_m)
        if collision or actual != tuple(route['states'][j]):
            failure = 'collision_or_execution_mismatch'
        if done and j < len(route['actions']):
            failure = 'world_budget_exhausted'
        if j == route['arrival_action'] or j == len(route['actions']) or failure:
            mesh = mapper.mesh()
            metrics.append({'action_index': j, 'coverage_2d': now_c,
                            **evaluator.evaluate(mesh, now_c, config['thresholds_m'])})
            mesh_name = 'arrival_mesh.npz' if j == route['arrival_action'] else 'final_mesh.npz'
            compare_npz(folder / mesh_name, mesh_arrays(mesh))
            arrival_saved |= j == route['arrival_action']
    require(len(rows) == route['cost'] or failure is not None, 'unexplained truncated branch')
    if failure is None:
        compare(list((*world.position, world.heading)), route['states'][0], 'successful branch return')
    require((folder / 'arrival_mesh.npz').exists() == arrival_saved, 'unexpected arrival mesh')
    compare_npz(folder / 'final_mesh.npz', mesh_arrays(mapper.mesh()))
    compare(read_json(folder / 'metrics.json'), metrics, 'branch metrics')
    union, parts, area = area_accounting(seen, masks, weights)
    compare_npz(folder / 'visibility.npz', {'prefix': seen, **masks, 'union': union})
    final, paid = metrics[-1], len(rows)
    expected = {'candidate_id': route['candidate_id'], 'paid_actions': paid, 'planned_actions': route['cost'],
                'arrival_actions': route['arrival_action'], 'failure': failure, 'new_area_m2': area,
                'area_per_action': area / paid, 'stage_new_area_m2': parts,
                'f1_gain_05cm': final['f1_05cm'] - before['f1_05cm'],
                'f1_gain_per_action': (final['f1_05cm'] - before['f1_05cm']) / paid,
                'coverage_gain_m2': (final['coverage_2d'] - c0) * world.reachable.sum() * c.resolution_m ** 2,
                'before': before, 'after': final}
    for threshold in config['thresholds_m']:
        tag = f'{round(threshold * 100):02d}cm'
        expected[f'branch_joint_auc_{tag}'] = joint_auc(metrics, config['branch_actions'], tag)
    compare(read_json(folder / 'outcome.json'), expected, 'branch outcome')
    return {'path': str(folder), 'paid_actions': paid, 'failure': failure, 'new_area_m2': area,
            'status': 'passed'}


def verify_fixture(run_name, fixture_name, config, selected):
    run, folder = Path(run_name), Path(run_name) / fixture_name
    description = read_json(folder / 'fixture.json')
    entry = description['entry']
    c = InspectionConfigV4(**(config['environment'] | entry.get('environment', {})))
    compare(description['environment'], asdict(c), 'fixture environment', tolerance=0)
    require(c.pose_noise_m == 0, 'protocol requires explicit zero pose noise')
    world = InspectionWorldV4(c, seed=entry['seed'])
    require(folder.name == f"{entry.get('group', 'main')}_{entry['seed']}_{json_hash(asdict(c))[:12]}",
            'fixture key/environment mismatch')
    evaluator = ReconstructionEvaluator(world, config['reference_samples'])
    weights = np.full(len(evaluator.reference), world.mesh.get_surface_area() / config['reference_samples'])
    reference_hash = file_hash(folder / 'reference.npz')
    compare_npz(folder / 'reference.npz', {'points': evaluator.reference, 'classes': evaluator.classes,
                'weights': weights, 'reachable': world.reachable, **mesh_arrays(world.mesh)})
    checkpoints = entry.get('checkpoints', config['checkpoints'])
    require(len(checkpoints) == len(set(checkpoints)), 'duplicate history checkpoint')
    require({p.name for p in folder.glob('history_*')} == {f'history_{i:04d}' for i in checkpoints},
            'history inventory differs')
    seal = read_json(folder / 'pre_outcome_seal.json')
    expected_sealed = set()
    for i in checkpoints:
        history = folder / f'history_{i:04d}'
        names = ('status.json',) if (history / 'status.json').exists() else ('candidates.json', 'candidate_audit.json', 'predictions.json')
        expected_sealed.update(str((history / name).relative_to(folder)) for name in names)
    require(set(seal) == expected_sealed, 'pre-outcome seal inventory differs')
    for name, sha in seal.items():
        require(file_hash(relative_file(folder, name)) == sha, f'pre-outcome score seal mismatch: {name}')
    prefix = folder / 'prefix'
    records = read_json(prefix / 'records.json')
    require(records, 'empty prefix')
    source_hashes = read_json(prefix / 'source_hashes.json')
    expected_raw = {f'{kind}/{i:04d}.npz' for kind in ('frames', 'scans') for i in range(len(records))}
    require(set(source_hashes) == expected_raw, 'prefix source hash inventory differs')
    for name, sha in source_hashes.items():
        require(file_hash(relative_file(prefix, name)) == sha, 'prefix raw source hash differs')
    for kind in ('frames', 'scans'):
        inventory_npz(prefix / kind, range(len(records)))
    frames, scans = [], []
    mapper = SemanticHistoryMapperV3(world.shape, c, c.truncation_m)
    seen = np.zeros(len(evaluator.reference), bool)
    seen_at = {}
    for i, row in enumerate(records):
        require(i == 0 or not records[i - 1]['done'], 'prefix continues after stop/budget')
        if i == 0:
            generated, collision, done, action = world.sense(), False, False, 'reset'
        else:
            action = row['action']
            require(action in ('forward', 'left', 'right', 'stop'), 'illegal prefix action')
            generated, collision, done = world.step(action)
        frame = RGBDFrame.load(prefix / 'frames' / f'{i:04d}.npz')
        scan = PlanarScan.load(prefix / 'scans' / f'{i:04d}.npz')
        compare_sensor(generated, frame, f'prefix {i} RGB-D')
        compare_sensor(world.scan(), scan, f'prefix {i} scan')
        compare(row, {'step': i, 'position': list(world.position), 'heading': world.heading,
                      'collision': collision, 'collisions': world.collisions, 'moves': world.moves,
                      'action': action, 'done': done}, 'prefix record')
        frames.append(frame); scans.append(scan)
        mapper.update(frame, scan); collision_update(mapper, row)
        physical = camera_pose(world.position, world.heading, c, world.shape[0])
        seen |= reference_visible(evaluator.reference, frame, evaluator.truth, physical, c.max_depth_m)
        if i in checkpoints and not done:
            history = folder / f'history_{i:04d}'
            compare_npz(history / 'prefix_mesh.npz', mesh_arrays(mapper.mesh()))
            compare_npz(history / 'prefix_map.npz', {'belief': mapper.belief})
            seen_at[i] = seen.copy()
    require(records[-1]['done'] or records[-1]['step'] == max(checkpoints), 'unexplained prefix truncation')
    del mapper
    summaries, unavailable, checked, branch_count = [], [], [], 0
    for index in checkpoints:
        history = folder / f'history_{index:04d}'
        if index >= len(records) or records[index]['done']:
            expected = {'status': 'unavailable_prefix', 'requested_step': index, 'last_step': records[-1]['step']}
            compare(read_json(history / 'status.json'), expected, 'unavailable history')
            require({p.name for p in history.iterdir()} == {'status.json'}, 'unavailable history contains outcomes')
            unavailable.append(expected | {'path': str((history / 'status.json').relative_to(run))})
            continue
        past = (frames[:index + 1], scans[:index + 1], records[:index + 1])
        initial = remap_prefix(*past, c, world.shape)
        compare_npz(history / 'prefix_mesh.npz', mesh_arrays(initial.mesh()))
        compare_npz(history / 'prefix_map.npz', {'belief': initial.belief})
        c0 = occupancy_coverage(initial, world.reachable)
        before = evaluator.evaluate(initial.mesh(), c0, config['thresholds_m'])
        routes = read_json(history / 'candidates.json')
        compare([r['candidate_id'] for r in routes], list(range(len(routes))), 'candidate identifiers')
        require(len(routes) <= config['max_candidates'], 'candidate pool exceeds budget')
        require(len({tuple(r['pose']) for r in routes}) == len(routes), 'duplicate endpoint candidate')
        safe = ~inflated_obstacles(initial.belief != 0, c.robot_radius_m / c.resolution_m)
        start = (*records[index]['position'], records[index]['heading'])
        safe[start[:2]] = True
        audit = read_json(history / 'candidate_audit.json')
        if 'safe_hash' in audit:
            compare(audit['safe_hash'], hashlib.sha256(safe.tobytes()).hexdigest(), 'candidate common-safe map')
        require({p.name for p in history.glob('candidate_[0-9]*')} == {f"candidate_{r['candidate_id']:03d}" for r in routes},
                'branch inventory differs from candidate pool')
        predictions = read_json(history / 'predictions.json')
        validate_predictions(predictions, routes)
        outcomes = read_json(history / 'outcomes.json')
        compare([o['candidate_id'] for o in outcomes], [r['candidate_id'] for r in routes], 'outcome candidate order')
        for route, outcome in zip(routes, outcomes):
            validate_route(route, start, safe, config['branch_actions'])
            branch = history / f"candidate_{route['candidate_id']:03d}"
            compare(read_json(branch / 'outcome.json'), outcome, 'history/branch outcome copy', tolerance=0)
            branch_count += 1
            relative = str(branch.relative_to(run))
            if relative in selected:
                try:
                    result = replay_branch(branch, route, world, evaluator, *past, config, weights, seen_at[index], before)
                except Exception as exc:
                    raise ValueError(f'{relative}: {exc}') from exc
                result['path'] = relative
                checked.append(result)
        expected_summary = {'fixture': folder.name, 'history_step': index, 'seed': entry['seed'],
            'group': entry.get('group', 'main'), 'depth_sigma_m': c.depth_sigma_m, 'candidate_count': len(routes),
            'branch_failures': sum(o['failure'] is not None for o in outcomes),
            'marker_pixels_history': sum(int(np.count_nonzero(f.semantic)) for f in past[0]),
            'status': 'complete' if routes else 'no_feasible_candidates',
            'predictions': str((history / 'predictions.json').relative_to(run)),
            'outcomes': str((history / 'outcomes.json').relative_to(run))}
        compare(read_json(history / 'summary.json'), expected_summary, 'history summary')
        summaries.append(expected_summary)
        print(f'verified {folder.name} history {index}: {sum(str(p.relative_to(run)) in selected for p in history.glob("candidate_[0-9]*"))}/{len(routes)} branches', flush=True)
        del initial
    compare(read_json(folder / 'summary.json'), summaries, 'fixture summary')
    require(file_hash(folder / 'reference.npz') == reference_hash, 'reference changed during replay')
    for name, sha in seal.items():
        require(file_hash(folder / name) == sha, 'pre-outcome score changed during replay')
    import open3d
    import scipy
    return {'fixture': folder.name, 'reference_sha256': reference_hash, 'prefix_frames': len(frames),
            'histories': summaries, 'unavailable_histories': unavailable, 'branch_count': branch_count,
            'checked_branches': checked, 'dependencies': {'numpy': np.__version__, 'open3d': open3d.__version__, 'scipy': scipy.__version__}}


def verify_aggregate(run, summaries):
    from scipy.stats import spearmanr
    summary = read_json(run / 'summary.json')
    expected_by_history = {(x['fixture'], x['history_step']): x for x in summaries}
    require(len(summary['histories']) == len(expected_by_history), 'duplicate or missing aggregate history')
    for history in summary['histories']:
        compare(history, expected_by_history.pop((history['fixture'], history['history_step'])), 'aggregate history')
    require(not expected_by_history, 'aggregate omitted history')
    choices = []
    for history in summary['histories']:
        predictions = read_json(relative_file(run, history['predictions']))
        outcomes = read_json(relative_file(run, history['outcomes']))
        by_id = {row['candidate_id']: row for row in outcomes}
        candidates = predictions['candidates']
        if not candidates:
            continue
        for scorer in SCORERS:
            chosen_id = predictions['selected'][scorer]
            actual = np.array([by_id[p['candidate_id']]['area_per_action'] for p in candidates])
            pred = np.array([p['scores'][scorer]['score'] for p in candidates])
            correlation = float(spearmanr(pred, actual).statistic) if np.ptp(pred) > 0 and np.ptp(actual) > 0 else None
            pairs = [(i, j) for i in range(len(pred)) for j in range(i + 1, len(pred)) if actual[i] != actual[j]]
            accuracy = float(np.mean([1 if (pred[i] - pred[j]) * (actual[i] - actual[j]) > 0
                                     else .5 if pred[i] == pred[j] else 0 for i, j in pairs])) if pairs else None
            result = by_id[chosen_id]
            choices.append({**{key: history[key] for key in ('fixture', 'history_step', 'seed', 'group', 'depth_sigma_m')},
                'scorer': scorer, 'chosen': chosen_id,
                **{key: result[key] for key in ('new_area_m2', 'area_per_action', 'f1_gain_05cm', 'f1_gain_per_action', 'coverage_gain_m2', 'failure')},
                'spearman_area_rate': correlation, 'pairwise_area_rate_accuracy': accuracy,
                'area_rate_regret': float(actual.max() - result['area_per_action'])})
    compare(read_json(run / 'choices.json'), choices, 'aggregate choices')
    averages = {scorer: {key: float(np.mean([r[key] for r in choices if r['scorer'] == scorer]))
        for key in ('new_area_m2', 'area_per_action', 'f1_gain_05cm', 'f1_gain_per_action', 'coverage_gain_m2', 'area_rate_regret')}
        for scorer in SCORERS} if choices else {}
    compare(summary, {'histories': summary['histories'], 'scorer_averages': averages,
        'scope': 'development common-history paid-action utility, not whole-system advantage',
        'independent_confirmation': False, 'calibration_fitted': False}, 'aggregate summary')


def completion_status(limit, checked, total):
    require(checked <= total, 'more replayed branches than recorded')
    full = limit is None and checked == total
    return {'status': 'passed_full' if full else 'partial', 'passed_full': full,
            'partial': not full, 'branches_checked': checked, 'branches_total': total}


def verify(run, workers=2, max_branches=None):
    run = Path(run).resolve()
    require(run.is_dir(), 'run directory does not exist')
    require(workers > 0 and (max_branches is None or max_branches >= 0), 'invalid replay limits')
    started = time.time()
    own_sha = file_hash(__file__)
    report = {'schema_version': 1, 'status': 'failed', 'passed_full': False, 'partial': max_branches is not None,
              'run': str(run), 'verifier_sha256': own_sha, 'max_branches': max_branches, 'workers': workers,
              'started_unix': started, 'python': platform.python_version(),
              'scope': 'archived-source raw-sensor replay and accounting; not efficacy or independent confirmation',
              'policy_regenerated': False, 'raw_artifacts_modified': False}
    try:
        manifest_path = run / 'artifact_hashes.json'
        manifest_sha = file_hash(manifest_path)
        manifest = read_json(manifest_path)
        check_manifest(run, manifest)
        metadata, config = read_json(run / 'metadata.json'), read_json(run / 'config.json')
        require(metadata['status'] == 'complete', 'run is not complete')
        compare(metadata['config_sha256'], json_hash(config), 'config hash')
        compare(metadata['scope'], config['scope'], 'metadata scope')
        require(metadata['replayed'] is False, 'input metadata was changed to claim replay')
        reserved = set(range(301, 309)) | set(range(401, 413))
        require(all(entry['seed'] not in reserved for entry in config['fixtures']), 'reserved confirmation seed')
        fixtures = sorted(p.parent for p in run.glob('*/fixture.json'))
        require(len(fixtures) == len(config['fixtures']), 'fixture count differs')
        compare(sorted(json_hash(read_json(p / 'fixture.json')['entry']) for p in fixtures),
                sorted(json_hash(entry) for entry in config['fixtures']), 'fixture entry multiset')
        branches = sorted(str(p.relative_to(run)) for p in run.glob('*/history_*/candidate_[0-9]*') if p.is_dir())
        selected = set(branches if max_branches is None else branches[:max_branches])
        results = []
        with tempfile.TemporaryDirectory(prefix='counterfactual-replay-') as temp:
            frozen = Path(temp) / 'sources'; frozen.mkdir()
            extract_sources(run, frozen, metadata['source_sha256'])
            # Preserve the exact independent verifier beside, outside the archived dependency tree.
            shutil.copyfile(__file__, Path(temp) / 'replay_counterfactual_views.py')
            context = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=worker_init,
                                     initargs=(str(frozen),)) as pool:
                futures = {pool.submit(verify_fixture, str(run), fixture.name, config, selected): fixture.name for fixture in fixtures}
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        raise ValueError(f"fixture {futures[future]}: {exc}") from exc
        results.sort(key=lambda row: row['fixture'])
        histories = [history for result in results for history in result['histories']]
        unavailable = [history for result in results for history in result['unavailable_histories']]
        checked = [branch for result in results for branch in result['checked_branches']]
        total = sum(result['branch_count'] for result in results)
        require(total == len(branches), 'global branch inventory differs')
        require({row['path'] for row in checked} == selected, 'requested replay branch missing/duplicated')
        compare(metadata['histories'], len(histories), 'metadata history count')
        compare(metadata['branches'], total, 'metadata branch count')
        compare(sorted(metadata['unavailable_histories'], key=lambda row: row['path']),
                sorted(unavailable, key=lambda row: row['path']), 'unavailable history metadata')
        verify_aggregate(run, histories)
        check_manifest(run, manifest)
        require(file_hash(manifest_path) == manifest_sha, 'artifact manifest changed during verification')
        require(file_hash(__file__) == own_sha, 'verifier source changed during verification')
        report.update(completion_status(max_branches, len(checked), total), fixtures=results,
                      artifact_manifest_sha256=manifest_sha, source_archive_sha256=file_hash(run / 'sources.zip'),
                      source_sha256=metadata['source_sha256'], prefix_histories_checked=len(histories),
                      unavailable_histories_checked=len(unavailable), raw_hashes_rechecked_after_replay=True,
                      unsealed_posthoc_analysis_files=sorted(str(p.relative_to(run)) for p in (run / 'analysis').rglob('*')
                          if p.is_file() and str(p.relative_to(run)) not in manifest),
                      aggregate_validation='recomputed from recorded outcomes; physical outcome replay limited to checked branches')
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    report['elapsed_s'] = time.time() - started
    # The existing artifact manifest deliberately excludes this independent report.
    with tempfile.NamedTemporaryFile(mode='w', prefix='.verification-', suffix='.json', dir=run, delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        temporary = Path(stream.name)
    temporary.replace(run / 'verification.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-branches', type=int)
    args = parser.parse_args()
    result = verify(args.run, args.workers, args.max_branches)
    print(json.dumps({key: result[key] for key in ('status', 'passed_full', 'elapsed_s')}
                     | {'verification': str(args.run.resolve() / 'verification.json')}, ensure_ascii=False), flush=True)
    if result['status'] == 'failed':
        print(result['error'], file=sys.stderr)
        sys.exit(1)
