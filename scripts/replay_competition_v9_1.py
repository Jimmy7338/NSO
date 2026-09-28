#!/usr/bin/env python3
"""Independently replay the complete sealed V9.1 44-branch prospective run.

The archived physical world and TSDF/history mapper are reused. This file owns
the execution state machine, EDT footprint gate, orientation BFS, camera-pose
construction, surface sampling/reference filtering, visibility, P/R/F1, coverage,
and deduplicated area/AUC accounting. No production guard, candidate generator,
planner, scorer, evaluator, visibility helper or runner function is called.

Only verification.json is added to the run; any previous verification is retained
outside it. A replay is reproducibility/accounting evidence, not independent
confirmation of efficacy, physical safety under map error, or full ANS testing.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import deque
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
import numpy as np
from scipy.ndimage import distance_transform_edt

STAGES = ('outbound', 'endpoint', 'return')
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(root, name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and name and not p.is_absolute()
            and '..' not in p.parts and '\\' not in name, 'unsafe relative artifact path')
    target = root.joinpath(*p.parts)
    require(target.resolve().is_relative_to(root.resolve()) and not target.is_symlink(), 'escaped/symlink artifact')
    return target


def compare(actual, expected, where, tolerance=1e-9):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), where + ': keys')
        for key in expected:
            compare(actual[key], expected[key], where + '/' + key, tolerance)
    elif isinstance(expected, (tuple, list)):
        require(isinstance(actual, (tuple, list)) and len(actual) == len(expected), where + ': length')
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{where}/{i}', tolerance)
    elif expected is None or isinstance(expected, (str, bool)):
        require(type(actual) is type(expected) and actual == expected, f'{where}: {actual!r} != {expected!r}')
    elif isinstance(expected, (int, np.integer)):
        require(isinstance(actual, (int, np.integer)) and not isinstance(actual, bool) and actual == expected,
                f'{where}: integer {actual!r} != {expected!r}')
    else:
        require(isinstance(actual, (int, float, np.number)) and not isinstance(actual, bool)
                and np.isfinite(actual) and np.isfinite(expected) and abs(actual - expected) <= tolerance,
                f'{where}: number {actual!r} != {expected!r}')


def arrays(actual, expected, where):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.dtype == b.dtype, where + ': shape/dtype')
    np.testing.assert_array_equal(a, b, err_msg=where)


def npz(path, expected):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == set(expected), str(path) + ': NPZ inventory')
        for key in data.files:
            arrays(data[key], expected[key], str(path) + '/' + key)


def mesh_arrays(mesh):
    return {'vertices': np.asarray(mesh.vertices), 'triangles': np.asarray(mesh.triangles)}


def successor(state, action):
    r, c, h = state
    require(h in range(4), 'invalid heading')
    if action == 'forward':
        dr, dc = DIRECTIONS[h]
        return r + dr, c + dc, h
    require(action in ('left', 'right'), 'unpaid or unknown motion')
    return r, c, (h + (1 if action == 'right' else -1)) % 4


def inside(safe, cell):
    return 0 <= cell[0] < safe.shape[0] and 0 <= cell[1] < safe.shape[1]


def safe_grid(belief, config):
    require(belief.ndim == 2 and np.isin(belief, [-1, 0, 1]).all(), 'invalid belief encoding')
    distance = distance_transform_edt(np.pad(belief == 0, 1, constant_values=False))[1:-1, 1:-1]
    return distance > config.robot_radius_m / config.resolution_m + np.sqrt(2) / 2


def distances_to_anchor(safe, anchor):
    """Reverse unweighted BFS; rotation and translation each cost one action."""
    distances = {}
    if not inside(safe, anchor[:2]) or not safe[anchor[:2]]:
        return distances
    distances[anchor] = 0
    queue = deque([anchor])
    while queue:
        state = queue.popleft()
        r, c, h = state
        dr, dc = DIRECTIONS[h]
        predecessors = ((r, c, (h - 1) % 4), (r, c, (h + 1) % 4), (r - dr, c - dc, h))
        for previous in predecessors:
            if previous not in distances and inside(safe, previous[:2]) and safe[previous[:2]]:
                distances[previous] = distances[state] + 1
                queue.append(previous)
    return distances


def plan_spec(safe, state, anchor, budget, distances):
    require(budget >= 0, 'negative remaining budget')
    if not inside(safe, state[:2]) or not safe[state[:2]]:
        return False, 'current_footprint_not_known_safe', 0
    if not inside(safe, anchor[:2]) or not safe[anchor[:2]]:
        return False, 'anchor_footprint_not_known_safe', 0
    if state not in distances:
        return False, 'anchor_disconnected_in_latest_map', 0
    cost = distances[state]
    if cost > budget:
        return False, 'known_return_exceeds_remaining_budget', cost
    return True, 'known_safe_return_within_budget', cost


def validate_plan(record, safe, state, anchor, budget, distances):
    available, reason, cost = plan_spec(safe, state, anchor, budget, distances)
    compare({k: record[k] for k in ('available', 'reason', 'paid_cost')},
            {'available': available, 'reason': reason, 'paid_cost': cost}, 'independent return BFS')
    actions = record['actions']
    require(isinstance(actions, list), 'return plan actions are not a list')
    if available:
        require(len(actions) == cost, 'return action cost/heading restoration mismatch')
        for action in actions:
            state = successor(state, action)
            require(inside(safe, state[:2]) and safe[state[:2]], 'return plan crosses unknown/obstacle footprint')
        require(state == anchor, 'return does not restore anchor position and heading')
    else:
        require(actions == [], 'unavailable plan carries executable actions')
    return available, reason, cost


def assessment_spec(safe, state, action, remaining):
    target = successor(state, action)[:2]
    reason = ('no_action_budget' if remaining < 1 else
              'current_footprint_not_known_safe' if not inside(safe, state[:2]) or not safe[state[:2]] else
              'next_footprint_not_known_safe' if not inside(safe, target) or not safe[target] else
              'latest_observed_grid_allows_action')
    return {'allowed': reason == 'latest_observed_grid_allows_action', 'reason': reason, 'target': list(target)}


def camera_matrix(state, config, shape):
    r, c, h = state
    forward = np.array(((0., 1., 0.), (1., 0., 0.), (0., -1., 0.), (-1., 0., 0.))[h])
    right = np.array((forward[1], -forward[0], 0.))
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = right, (0., 0., -1.), forward
    pose[:3, 3] = ((c + .5) * config.resolution_m,
                      (shape[0] - r - .5) * config.resolution_m, config.camera_height_m)
    return pose


def ray_scene(mesh):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def sample_surface(mesh, count, seed):
    """Independent area-CDF triangle sampling with the frozen RNG specification."""
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return np.empty((0, 3)), np.empty(0, int)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    areas = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    if areas.sum() <= 0:
        return np.empty((0, 3)), np.empty(0, int)
    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(areas), count, p=areas / areas.sum())
    u, v = np.sqrt(rng.random(count)), rng.random(count)
    points = ((1 - u)[:, None] * a[indexes] + (u * (1 - v))[:, None] * b[indexes]
              + (u * v)[:, None] * c[indexes])
    return points, indexes


def make_reference(world, count):
    """Own deterministic reachable-lattice observability filter, seed 2026."""
    import open3d as o3d
    points, triangle_ids = sample_surface(world.mesh, count, 2026)
    scene = ray_scene(world.mesh)
    visible = np.zeros(len(points), bool)
    c = world.config
    vertical = c.height_px / c.width_px * np.tan(np.deg2rad(c.fov_deg / 2))
    for r, col in np.argwhere(world.reachable):
        if r % 4 or col % 4:
            continue
        origin = np.array(((col + .5) * c.resolution_m, (world.shape[0] - r - .5) * c.resolution_m,
                           c.camera_height_m))
        delta = points - origin
        horizontal = np.linalg.norm(delta[:, :2], axis=1)
        ids = np.flatnonzero((~visible) & (horizontal > .15) & (horizontal < c.max_depth_m)
                             & (np.abs(delta[:, 2]) <= horizontal * vertical))
        if len(ids):
            rays = np.column_stack((np.tile(origin, (len(ids), 1)), delta[ids])).astype(np.float32)
            hits = scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
            visible[ids[np.abs(hits - 1.) < 1e-4]] = True
    return points[visible].astype(np.float32), world.triangle_classes[triangle_ids[visible]], scene


def observed_reference(points, frame, truth, pose, max_depth):
    """Independent continuous first-hit test plus actual nearest-pixel validity."""
    import open3d as o3d
    cloud = np.asarray(points, float)
    local = (cloud - pose[:3, 3]) @ pose[:3, :3]
    z = local[:, 2]
    mask = (z > .15) & (z <= max_depth)
    ids = np.flatnonzero(mask)
    result = np.zeros(len(cloud), bool)
    if not len(ids):
        return result
    projected = local[ids] @ frame.intrinsic.T
    u, v = projected[:, 0] / z[ids], projected[:, 1] / z[ids]
    h, w = frame.depth_m.shape
    valid = (u >= -.5) & (u < w - .5) & (v >= -.5) & (v < h - .5)
    ids, u, v = ids[valid], u[valid], v[valid]
    sampled = frame.depth_m[np.floor(v + .5).astype(int), np.floor(u + .5).astype(int)]
    valid = np.isfinite(sampled) & (sampled > .15) & (sampled <= max_depth)
    ids = ids[valid]
    if len(ids):
        rays = np.column_stack((np.tile(pose[:3, 3], (len(ids), 1)), cloud[ids] - pose[:3, 3])).astype(np.float32)
        hits = truth.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
        result[ids] = np.isfinite(hits) & (np.abs(hits - 1.) < 1e-4)
    return result


def metric_snapshot(mapper, world, reference, truth, thresholds):
    """Independent metric algebra/sample generation; Open3D distance engine reused."""
    import open3d as o3d
    mesh = mapper.mesh()
    cov = float(np.count_nonzero((mapper.belief != -1) & world.reachable) / world.reachable.sum())
    predicted, _ = sample_surface(mesh, 12000, 812)
    result = {'surface_samples': len(predicted), 'reference_samples': len(reference['points'])}
    reconstruction = ray_scene(mesh) if len(mesh.triangles) else None
    if len(predicted):
        accuracy = truth.compute_distance(o3d.core.Tensor(predicted.astype(np.float32)), nthreads=1).numpy()
        completeness = reconstruction.compute_distance(o3d.core.Tensor(reference['points']), nthreads=1).numpy()
        result.update(surface_error_mean_m=float(accuracy.mean()), surface_error_p95_m=float(np.percentile(accuracy, 95)))
    else:
        accuracy, completeness = np.empty(0), np.full(len(reference['points']), np.inf)
        result.update(surface_error_mean_m=None, surface_error_p95_m=None)
    for threshold in thresholds:
        tag = f'{round(threshold * 100):02d}cm'
        p = float(np.mean(accuracy <= threshold)) if len(accuracy) else 0.
        r = float(np.mean(completeness <= threshold))
        f1 = 2 * p * r / (p + r) if p + r else 0.
        result.update({f'precision_{tag}': p, f'recall_{tag}': r, f'f1_{tag}': f1, f'joint_{tag}': cov * f1})
        for label, name in ((2, 'storage_shelf'), (3, 'closed_equipment_cabinet')):
            subset = reference['classes'] == label
            result[f'{name}_recall_{tag}'] = float(np.mean(completeness[subset] <= threshold)) if subset.any() else None
    area = cov * int(world.reachable.sum()) * world.config.resolution_m ** 2
    result.update(coverage_2d=cov, covered_area_m2=area)
    for threshold in thresholds:
        tag = f'{round(threshold * 100):02d}cm'
        result[f'area_times_f1_{tag}'] = area * result[f'f1_{tag}']
    old = reference['points'][reference['prefix_seen']]
    if len(old) and reconstruction is not None:
        distance = reconstruction.compute_distance(o3d.core.Tensor(old.astype(np.float32)), nthreads=1).numpy()
        result.update(fixed_old_support_error_mean_m=float(distance.mean()),
                      fixed_old_support_error_p95_m=float(np.percentile(distance, 95)))
    else:
        result.update(fixed_old_support_error_mean_m=None, fixed_old_support_error_p95_m=None)
    return mesh, result


def worker(args):
    sys.path.insert(0, str(args.snapshot))
    from env.virtual3d_competition_v9 import CompetitionWorldV9, CompetitionContextV9, CompetitionConfigV9
    from nso.semantic_completion_v3 import SemanticHistoryMapperV3
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    seal = read(args.run / 'pre_execution_seal.json')
    protocol = seal['protocol']
    compare(protocol, read(args.snapshot / 'configs/virtual3d/competition_v9_1_validation_protocol.json'), 'archived protocol equals execution seal')
    require(protocol['parent_order'] == ['Q0', 'Q1', 'Q2', 'Q3'] and protocol['arrangement_order'] == ['shelf_west', 'shelf_east'], 'finite history inventory changed')
    require(protocol['schema_version'] == 'competition_v9_1_validation_protocol/1', 'V9 protocol required')
    budget = protocol['branch_actions']
    thresholds = protocol['metrics']['thresholds_m']
    require(budget == 48 and thresholds == [.02, .05] and protocol['prefix_paid_actions'] == 150, 'frozen budget/metric contract')
    require(protocol['reference']['samples_requested'] == 32000 and seal['physical_branch_limit'] == 44, 'reference/branch limits')
    summary = read(args.run / 'summary.json')
    all_outcomes, reports = [], []
    prefix_sensor_comparisons = branch_sensor_comparisons = 0
    def sensors(frame, scan, prefix, index):
        for value, expected in ((frame, RGBDFrame.load(prefix / 'frames' / f'{index:04d}.npz')),
                                (scan, PlanarScan.load(prefix / 'scans' / f'{index:04d}.npz'))):
            for field in fields(type(value)):
                arrays(getattr(value, field.name), getattr(expected, field.name), f'{prefix}/{index}/{field.name}')
    def assimilate(mapper, frame, scan, world, collision):
        mapper.update(frame, scan)
        if collision:
            cell = successor((*world.position, world.heading), 'forward')[:2]
            if inside(mapper.belief, cell):
                mapper.belief[cell] = 1
    for context_id in protocol['parent_order']:
        paired_routes = []
        for arrangement in protocol['arrangement_order']:
            folder = args.run / context_id / arrangement
            fixture = read(folder / 'fixture.json')
            context = CompetitionContextV9(**fixture['context'])
            config = CompetitionConfigV9(**fixture['config'])
            require(context.context_id == context_id and fixture['arrangement'] == arrangement, 'fixture identity')
            routes = read(folder / 'candidates.json')
            ids = protocol['candidate_ids_by_parent'][context_id]
            require([r['candidate_id'] for r in routes] == ids, 'missing/extra shared candidate IDs')
            compare([r['group'] for r in routes], protocol['candidate_roles_by_parent'][context_id], 'fixed roles')
            paired_routes.append(routes)
            prediction = read(folder / 'predictions.json')
            require(set(prediction['scores']) == {'N', 'G', 'O', 'S', 'X', 'M'}, 'method inventory')
            for method, scores in prediction['scores'].items():
                require(len(scores) == len(ids) and np.isfinite(scores).all(), 'invalid sealed score vector')
                selected = max(ids, key=lambda i: (scores[i], -routes[i]['cost'], -i))
                require(prediction['selected_candidate_ids'][method] == selected, 'choice differs from sealed shared scores')
            compare(prediction['scores']['M'], prediction['scores']['G'], 'missing-label fallback', 0.)
            require(set(prediction['rate_scores']) == set(prediction['scores']), 'rate method inventory')
            for method, scores in prediction['rate_scores'].items():
                require(len(scores) == len(ids) and np.isfinite(scores).all(), 'invalid rate score vector')
                selected = max(ids, key=lambda i: (scores[i], -routes[i]['cost'], -i))
                require(prediction['rate_selected_candidate_ids'][method] == selected, 'rate tie rule changed')
                compare(prediction['scores'][method], [v * r['cost'] for v, r in zip(scores, routes)],
                        'prospectively frozen total=rate*cost', 0.)
            compare(prediction['rate_scores']['M'], prediction['rate_scores']['G'], 'rate missing fallback', 0.)
            with np.load(folder / 'reference.npz', allow_pickle=False) as saved:
                reference = {k: saved[k].copy() for k in saved.files}
            prefix = folder / 'prefix'
            records = read(prefix / 'records.json')
            require(len(records) == 151, 'prefix frame inventory')
            expected_names = {f'{i:04d}.npz' for i in range(151)}
            for modality in ('frames', 'scans'):
                require({p.name for p in (prefix / modality).iterdir()} == expected_names, 'prefix file inventory')
            branch_outcomes = []
            truth = None
            for route in routes:
                world = CompetitionWorldV9(context, arrangement, config)
                mapper = SemanticHistoryMapperV3(world.shape, config, config.truncation_m)
                if truth is None:
                    points, classes, truth = make_reference(world, 32000)
                    arrays(points, reference['points'], 'independent GT reference sample/filter')
                    arrays(classes, reference['classes'], 'independent GT reference classes')
                    arrays(world.reachable, reference['reachable'], 'GT reachable')
                    for key, value in mesh_arrays(world.mesh).items():
                        arrays(value, reference[key], 'GT exterior mesh/' + key)
                    arrays(np.full(len(points), float(world.mesh.get_surface_area()) / 32000), reference['weights'], 'unrenormalized physical surface weights')
                    for name, predicate in (('west_slot', points[:, 0] < config.width_m / 2),
                                             ('east_slot', points[:, 0] >= config.width_m / 2)):
                        arrays((classes != 1) & predicate, reference[name], name)
                seen = np.zeros(len(reference['points']), bool)
                for j, row in enumerate(records):
                    if j:
                        require(row['action'] == world.prefix_actions[j - 1], 'prefix action contract')
                        frame, collision, done = world.step(row['action'])
                    else:
                        require(row['action'] == 'reset', 'initial prefix record')
                        frame, collision, done = world.sense(), False, False
                    scan = world.scan()
                    sensors(frame, scan, prefix, j)
                    prefix_sensor_comparisons += 1
                    compare(row, {'step': world.step_count, 'position': list(world.position), 'heading': world.heading,
                                  'collision': bool(collision), 'collisions': world.collisions, 'moves': world.moves,
                                  'action': row['action'], 'done': bool(done)}, 'prefix record')
                    require(not collision and not done, 'failed paid prefix')
                    pose = camera_matrix((*world.position, world.heading), config, world.shape)
                    arrays(pose, frame.world_from_camera, 'zero-noise physical camera pose')
                    if route['candidate_id'] == 0:
                        seen |= observed_reference(reference['points'], frame, truth, pose, config.max_depth_m)
                    assimilate(mapper, frame, scan, world, collision)
                if route['candidate_id'] == 0:
                    arrays(seen, reference['prefix_seen'], 'actual full-prefix visibility')
                npz(folder / 'prefix_map.npz', {'belief': mapper.belief, 'camera_seen': mapper.camera_seen})
                anchor = (*world.position, world.heading)
                initial_safe = safe_grid(mapper.belief, config)
                require(tuple(route['states'][0]) == tuple(route['states'][-1]) == anchor, 'closed paid route anchor')
                require(route['cost'] == len(route['actions']) == len(route['states']) - 1
                        and 0 < route['cost'] <= budget, 'route cost')
                require(0 < route['arrival_action'] < route['cost'] and
                        route['states'][route['arrival_action']] == route['pose'], 'declared route endpoint')
                for a, action, b in zip(route['states'], route['actions'], route['states'][1:]):
                    require(successor(tuple(a), action) == tuple(b), 'unpaid route state change')
                for state in route['states']:
                    require(inside(initial_safe, state[:2]) and initial_safe[tuple(state[:2])], 'original route outside complete knownsafe footprint')
                branch = folder / f"candidate_{route['candidate_id']:03d}"
                decisions = [read_line for read_line in (json.loads(x) for x in (branch / 'decisions.jsonl').read_text().splitlines())]
                actions = read(branch / 'actions.json')
                compare([json.loads(x) for x in (branch / 'actions.jsonl').read_text().splitlines()], actions, 'action stream')
                metric_records = read(branch / 'metrics.json')
                mesh, before = metric_snapshot(mapper, world, reference, truth, thresholds)
                npz(folder / 'executed_prefix_mesh.npz', mesh_arrays(mesh))
                npz(branch / 'prefix_mesh.npz', mesh_arrays(mesh))
                metrics = [{'action_index': 0, **before}]
                paid = cursor = event_index = collisions = 0
                mode, terminal, failure, arrival = 'route', None, None, None
                masks = {stage: np.zeros(len(reference['points']), bool) for stage in STAGES}
                denials = []
                def take(kind, **extra):
                    nonlocal event_index
                    require(event_index < len(decisions), 'missing decision ' + kind)
                    record = decisions[event_index]
                    base = {'decision_id': event_index, 'kind': kind, 'mode': mode,
                            'paid_actions_before': paid, 'absolute_step': world.step_count,
                            'position': list(world.position), 'heading': world.heading,
                            'remaining_actions': budget - paid, **extra}
                    require(base.keys() <= record.keys(), 'missing decision fields')
                    compare({key: record[key] for key in base}, base, 'decision order/state/' + kind)
                    event_index += 1
                    return record
                while terminal is None:
                    state = (*world.position, world.heading)
                    remaining = budget - paid
                    safe = safe_grid(mapper.belief, config)
                    distances = distances_to_anchor(safe, anchor)
                    now = take('current_return_feasibility')
                    available, reason, cost = validate_plan(now, safe, state, anchor, remaining, distances)
                    if not available:
                        terminal = failure = 'return_unavailable:' + reason
                        break
                    if mode == 'route' and cursor == len(route['actions']):
                        terminal = 'original_route_completed' if state == anchor else 'route_ended_away_from_anchor'
                        break
                    if mode == 'return' and state == anchor:
                        terminal = 'returned_after_route_denial'
                        break
                    if remaining == 0:
                        terminal = 'budget_exhausted'
                        failure = terminal if state != anchor else None
                        break
                    action = route['actions'][cursor] if mode == 'route' else now['actions'][0]
                    assessment = take('action_assessment', action=action, route_cursor=cursor)
                    expected = assessment_spec(safe, state, action, remaining)
                    compare({k: assessment[k] for k in expected}, expected, 'independent current/target footprint gate')
                    denied, reason = not expected['allowed'], expected['reason']
                    if not denied:
                        after_plan = take('successor_return_feasibility', action=action)
                        available, after_reason, _ = validate_plan(after_plan, safe, successor(state, action), anchor,
                                                                  remaining - 1, distances)
                        denied, reason = not available, 'successor_return:' + after_reason
                    if denied:
                        denials.append({'paid_actions_before': paid, 'mode': mode, 'action': action, 'reason': reason})
                        if mode == 'route':
                            mode = 'return'
                            take('switch_to_return', reason=reason)
                            continue
                        terminal = failure = 'return_action_denied:' + reason
                        break
                    require(paid < len(actions), 'unrecorded executed action')
                    frame, collision, done = world.step(action)
                    scan = world.scan()
                    paid += 1
                    sensors(frame, scan, branch, paid)
                    branch_sensor_comparisons += 1
                    assimilate(mapper, frame, scan, world, collision)
                    actual = (*world.position, world.heading)
                    target_now = arrival is None and actual == tuple(route['pose'])
                    if target_now:
                        arrival = paid
                    stage = 'endpoint' if target_now else 'return' if mode == 'return' or arrival is not None else 'outbound'
                    expected_row = {'action_index': paid, 'absolute_step': world.step_count, 'action': action,
                                    'position': list(world.position), 'heading': world.heading, 'collision': bool(collision),
                                    'done': bool(done), 'mode': mode, 'stage': stage,
                                    'expected_state': list(successor(state, action))}
                    compare(actions[paid - 1], expected_row, 'paid action/physical state')
                    pose = camera_matrix(actual, config, world.shape)
                    arrays(pose, frame.world_from_camera, 'branch physical pose')
                    masks[stage] |= observed_reference(reference['points'], frame, truth, pose, config.max_depth_m)
                    if mode == 'route':
                        cursor += 1
                    if target_now:
                        mesh, values = metric_snapshot(mapper, world, reference, truth, thresholds)
                        metrics.append({'action_index': paid, **values})
                        npz(branch / 'arrival_mesh.npz', mesh_arrays(mesh))
                    collisions += int(collision)
                    if collision or actual != successor(state, action):
                        terminal = failure = 'collision_or_physical_state_mismatch'
                    elif mode == 'route' and actual != tuple(route['states'][cursor]):
                        terminal = failure = 'route_state_mismatch'
                    elif done:
                        terminal = failure = 'world_budget_exhausted'
                take('terminal_stop', reason=terminal, paid_sensor_capture_on_stop=False)
                require(event_index == len(decisions) and paid == len(actions) and paid <= budget,
                        'extra decisions/actions after stop or budget exceeded')
                for modality in ('frames', 'scans'):
                    require({p.name for p in (branch / modality).iterdir()} == {f'{i:04d}.npz' for i in range(1, paid + 1)}, 'unpaid or missing sensor captures')
                require((branch / 'arrival_mesh.npz').exists() == (arrival is not None), 'false arrival mesh')
                mesh, after = metric_snapshot(mapper, world, reference, truth, thresholds)
                npz(branch / 'final_mesh.npz', mesh_arrays(mesh))
                npz(branch / 'final_map.npz', {'belief': mapper.belief, 'camera_seen': mapper.camera_seen})
                if metrics[-1]['action_index'] == paid:
                    metrics[-1] = {'action_index': paid, **after}
                else:
                    metrics.append({'action_index': paid, **after})
                compare(metric_records, metrics, 'all metric checkpoints')
                final = (*world.position, world.heading)
                safe = safe_grid(mapper.belief, config)
                available, _, _ = plan_spec(safe, final, anchor, budget - paid, distances_to_anchor(safe, anchor))
                returned = final == anchor and available
                if not returned and failure is None:
                    failure = 'stopped_away_from_safe_anchor'
                union = np.zeros(len(reference['points']), bool)
                cumulative = reference['prefix_seen'].copy()
                weights, increments = reference['weights'], {}
                for stage in STAGES:
                    increments[stage] = float(weights[masks[stage] & ~cumulative].sum())
                    union |= masks[stage]
                    cumulative |= masks[stage]
                new = union & ~reference['prefix_seen']
                area = float(weights[new].sum())
                require(abs(sum(increments.values()) - area) <= 1e-10, 'surface double counting')
                npz(branch / 'visibility.npz', {'prefix': reference['prefix_seen'], **masks, 'union': union})
                slots = [reference['west_slot'], reference['east_slot']]
                remaining_area = [float(weights[s & ~reference['prefix_seen']].sum()) for s in slots]
                slot_area = [float(weights[s & new].sum()) for s in slots]
                outcome = {'context': context_id, 'arrangement': arrangement, 'candidate_id': route['candidate_id'],
                           'group': route['group'], 'planned_actions': route['cost'], 'paid_actions': paid,
                           'prefix_paid_actions': 150, 'task_paid_actions': 150 + paid, 'failure': failure,
                           'terminal_reason': terminal, 'collision_count': collisions,
                           'original_target_reached': arrival is not None, 'actual_arrival_action': arrival,
                           'full_original_route_completed': terminal == 'original_route_completed',
                           'returned_to_anchor': bool(returned), 'new_area_m2': area,
                           'area_per_action': area / paid if paid else None, 'before': before, 'after': after,
                           'f1_gain_05cm': after['f1_05cm'] - before['f1_05cm'],
                           'f1_gain_per_action': (after['f1_05cm'] - before['f1_05cm']) / paid if paid else None,
                           'coverage_gain_m2': after['covered_area_m2'] - before['covered_area_m2'],
                           'slot_new_area_m2': slot_area, 'slot_remaining_area_m2': remaining_area,
                           'slot_new_fractions': [a / b if b else None for a, b in zip(slot_area, remaining_area)],
                           'background_new_area_m2': float(weights[new & (reference['classes'] == 1)].sum()),
                           'stage_new_area_m2': increments, 'zero_paid_action_rate_is_undefined': paid == 0}
                for threshold in thresholds:
                    tag = f'{round(threshold * 100):02d}cm'
                    xs = [m['action_index'] for m in metrics]
                    require(xs[0] == 0 and all(a < b for a, b in zip(xs, xs[1:])) and xs[-1] <= budget, 'AUC checkpoint accounting')
                    trace = np.interp(np.arange(budget + 1), xs, [m[f'joint_{tag}'] for m in metrics])
                    outcome[f'branch_joint_auc_{tag}'] = float(np.trapz(trace) / budget)
                compare(read(branch / 'outcome.json'), outcome, 'all branch outcome fields')
                branch_outcomes.append(outcome)
                all_outcomes.append(outcome)
                reports.append({'context': context_id, 'arrangement': arrangement, 'candidate_id': route['candidate_id'],
                                'paid_actions': paid, 'decisions': len(decisions), 'terminal_reason': terminal,
                                'original_target_reached': arrival is not None, 'returned_to_anchor': bool(returned),
                                'failure': failure, 'collisions': collisions, 'denials': denials,
                                'final_footprint_known_safe': bool(safe[final[:2]]),
                                'independent_BFS_EDT_decisions': True, 'raw_fields_and_mesh_arrays_exact': True,
                                'independent_reference_visibility_metrics_area_auc_match': True})
                print('independent replay', context_id, arrangement, route['candidate_id'], 'paid', paid,
                      'terminal', terminal, flush=True)
                del mapper, world, mesh
            compare(read(folder / 'outcomes.json'), branch_outcomes, 'history outcome inventory')
        compare(paired_routes[0], paired_routes[1], 'paired shared candidates', 0.)
    compare(summary, {'outcomes': all_outcomes, 'physical_branches': 44,
                      'paid_branch_actions': sum(r['paid_actions'] for r in reports),
                      'shared_prefix_actions_per_history': 150, 'unique_prefix_histories': 8}, 'complete shared physical outcome set')
    meta = read(args.run / 'metadata.json')
    require(meta['new_physical_branches'] == 44 and meta['paid_actions'] == sum(r['paid_actions'] for r in reports)
            and meta['failures'] == sum(r['failure'] is not None for r in reports), 'metadata outcome count')
    # Imports happen through archived modules, but no production assessment or
    # evaluator functions were called. Check the direct loaded shared classes.
    for module_name in ('env.virtual3d_competition_v9', 'nso.semantic_completion_v3', 'utils.rgbd_contract'):
        require(Path(sys.modules[module_name].__file__).resolve().is_relative_to(args.snapshot), 'checkout import escaped frozen snapshot')
    write(args.result, {'branches': reports, 'physical_branches': 44, 'branches_checked': 44, 'branches_total': 44,
                        'paid_actions': sum(r['paid_actions'] for r in reports),
                        'unique_prefix_histories': 8, 'unique_prefix_paid_actions': 1200,
                        'unique_prefix_rgbd_scan_pairs': 1208, 'prefix_rgbd_scan_pair_comparisons': prefix_sensor_comparisons,
                        'branch_rgbd_scan_pair_comparisons': branch_sensor_comparisons,
                        'returned_branches': sum(r['returned_to_anchor'] for r in reports),
                        'collisions': sum(r['collisions'] for r in reports),
                        'failed_branches': sum(r['failure'] is not None for r in reports)})


def main(args):
    started = time.monotonic()
    own_hash = sha(Path(__file__))
    report = {'status': 'failed', 'passed_full': False, 'partial': False, 'max_branches': None,
              'run': str(args.run), 'scope': 'all 44 shared prospectively declared V9.1 branches; independent raw replay and accounting only'}
    try:
        manifest = read(args.run / 'artifact_hashes.json')
        require({'metadata.json', 'sources.zip', 'pre_execution_seal.json', 'summary.json'} <= manifest.keys(), 'incomplete run manifest')
        manifest_hash = sha(args.run / 'artifact_hashes.json')
        source_hash = sha(args.run / 'sources.zip')
        report.update(artifact_manifest_sha256=manifest_hash, source_archive_sha256=source_hash)
        def check():
            require(sha(args.run / 'artifact_hashes.json') == manifest_hash, 'manifest changed during verification')
            actual = {str(p.relative_to(args.run)) for p in args.run.rglob('*') if p.is_file()
                      and p not in (args.run / 'artifact_hashes.json', args.run / 'verification.json')
                      and not p.is_relative_to(args.run / 'analysis')}
            require(actual == set(manifest), 'run file inventory differs from frozen manifest')
            for name, expected in manifest.items():
                require(sha(relative(args.run, name)) == expected, 'changed input: ' + name)
        check()
        meta = read(args.run / 'metadata.json')
        require(meta['status'] == 'complete_execution_pending_independent_replay', 'run has not completed all execution')
        seal = read(args.run / 'pre_execution_seal.json')
        require(seal['source_sha256'] == meta['source_sha256'], 'run/source seal disagreement')
        for name, expected in seal['decision_assets'].items():
            require(name in manifest and manifest[name] == expected, 'decision asset seal differs')
        prepared = Path(meta['prepared']).resolve()
        require(sha(prepared / 'artifact_hashes.json') == seal['prepared_manifest_sha256'], 'prepared binding differs')
        prepared_manifest = read(prepared / 'artifact_hashes.json')
        for name, expected in seal['decision_assets'].items():
            require(prepared_manifest.get(name) == expected and sha(relative(prepared, name)) == expected, 'copied prepared asset differs')
        report.update(prepared_manifest_sha256=seal['prepared_manifest_sha256'],
                      pre_execution_seal_sha256=sha(args.run / 'pre_execution_seal.json'),
                      structural_review_sha256=seal['structural_review_sha256'])
        with tempfile.TemporaryDirectory(prefix='competition-independent-replay-') as temporary:
            temp = Path(temporary)
            snapshot = temp / 'sources'
            snapshot.mkdir()
            with zipfile.ZipFile(args.run / 'sources.zip') as archive:
                names = archive.namelist()
                require(len(names) == len(set(names)) and set(names) == set(seal['source_sha256']), 'source archive inventory')
                for name in names:
                    target = relative(snapshot, name)
                    data = archive.read(name)
                    require(hashlib.sha256(data).hexdigest() == seal['source_sha256'][name], 'source archive byte hash: ' + name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
            result = temp / 'worker_result.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--run', str(args.run),
                       '--snapshot', str(snapshot), '--result', str(result)]
            completed = subprocess.run(command, cwd=temp,
                                       env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(snapshot)})
            require(completed.returncode == 0, 'isolated independent worker failed')
            report.update(read(result))
        check()
        require(sha(Path(__file__)) == own_hash, 'verifier changed during replay')
        report.update(status='passed_full', passed_full=True, raw_hashes_rechecked_after_replay=True,
                      production_guard_planner_scorer_runner_called=False,
                      metric_evaluator_or_visibility_helper_called=False,
                      independent_confirmation=False, raw_artifacts_modified=False,
                      reused_parts=['archived physical sensor world', 'archived RGB-D/scan contract',
                                    'archived SemanticHistoryMapperV3/TSDF', 'Open3D ray/distance engine',
                                    'NumPy RNG and SciPy EDT'],
                      independent_parts=['execution decision state machine', 'known-free footprint gate',
                                         'orientation and remaining-budget BFS', 'physical camera matrix',
                                         'reference triangle sampling and reachable-lattice filter',
                                         'actual-frame reference visibility', 'P/R/F1 and fixed-old-support error',
                                         'coverage, slots, unique area ledger and sparse 48-action AUC'])
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    report.update(elapsed_s=time.monotonic() - started, verifier_sha256=own_hash)
    target = args.run / 'verification.json'
    if target.exists():
        root = Path(__file__).resolve().parents[1]
        retained = root / 'audit_results' / 'competition_v9_1_replay_attempts' / sha(target)
        retained.mkdir(parents=True, exist_ok=True)
        (retained / 'verification.json').write_bytes(target.read_bytes())
    write(target, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'branches'}, ensure_ascii=False), flush=True)
    return 0 if report['passed_full'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--result', type=Path)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments)
    else:
        raise SystemExit(main(arguments))
