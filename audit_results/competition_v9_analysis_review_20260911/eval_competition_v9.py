#!/usr/bin/env python3
"""Execute the complete sealed 48-branch prospective finite-domain validation.

All methods share every physical branch. The latest observed-map guard is
checked before each paid action, including a known return within the remainder.
Truth is confined to sensing and the evaluator; it never repairs a route.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
import open3d as o3d
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from env.virtual3d_competition_v9 import create_competition_world
from env.virtual3d import camera_pose
from nso.execution_guard_v8 import ObservedExecutionGuard
from scripts.eval_counterfactual_views import remap, state_restore, update_collision, coverage, save_mesh, write_json
from utils.rgbd_contract import RGBDFrame, PlanarScan
from utils.cpu_protocol import file_hash
from utils.grid_geometry import DIRECTIONS
from utils.reconstruction_metrics import ReconstructionEvaluator, ray_scene
from utils.counterfactual_surface_visibility import reference_visible, surface_increment


def read(path): return json.loads(Path(path).read_text())


def successor(state, action):
    row, col, heading = state
    if action == 'forward':
        dr, dc = DIRECTIONS[heading]
        return (int(row + dr), int(col + dc), heading)
    if action not in ('left', 'right'):
        raise ValueError('only actual movement and rotation actions are permitted')
    return (row, col, (heading + (1 if action == 'right' else -1)) % 4)


def restore_evaluator(world, reference):
    """Keep exactly the frozen point sample, weights, classes and full truth."""
    evaluator = ReconstructionEvaluator.__new__(ReconstructionEvaluator)
    evaluator.config = world.config
    evaluator.truth = ray_scene(world.mesh)
    np.testing.assert_array_equal(reference['vertices'], np.asarray(world.mesh.vertices))
    np.testing.assert_array_equal(reference['triangles'], np.asarray(world.mesh.triangles))
    evaluator.reference = reference['points']; evaluator.classes = reference['classes']
    return evaluator


def snapshot_metrics(mapper, world, evaluator, reference, thresholds):
    mesh = mapper.mesh(); cov = coverage(mapper, world.reachable)
    values = evaluator.evaluate(mesh, cov, thresholds)
    # The reused evaluator's legacy simple/complex names do not describe these
    # new assets. Rename class diagnostics explicitly; primary metrics unchanged.
    for threshold in thresholds:
        tag = f'{round(threshold * 100):02d}cm'
        values[f'storage_shelf_recall_{tag}'] = values.pop(f'simple_recall_{tag}')
        values[f'closed_equipment_cabinet_recall_{tag}'] = values.pop(f'complex_recall_{tag}')
    covered_area = cov * int(world.reachable.sum()) * world.config.resolution_m ** 2
    values.update(coverage_2d=cov, covered_area_m2=covered_area)
    for threshold in thresholds:
        tag = f'{round(threshold * 100):02d}cm'
        values[f'area_times_f1_{tag}'] = covered_area * values[f'f1_{tag}']
    old = evaluator.reference[reference['prefix_seen']]
    if len(old) and len(mesh.triangles):
        distances = ray_scene(mesh).compute_distance(o3d.core.Tensor(old.astype(np.float32)), nthreads=1).numpy()
        values['fixed_old_support_error_mean_m'] = float(distances.mean())
        values['fixed_old_support_error_p95_m'] = float(np.percentile(distances, 95))
    else:
        values['fixed_old_support_error_mean_m'] = None
        values['fixed_old_support_error_p95_m'] = None
    return mesh, values


def execute(world, route, frames, scans, records, evaluator, reference, folder, protocol):
    folder.mkdir(); (folder / 'frames').mkdir(); (folder / 'scans').mkdir()
    config = world.config; budget = protocol['branch_actions']
    state_restore(world, records[-1], frames[-1], scans[-1])
    mapper = remap(frames, scans, records, config, world.shape)
    guard = ObservedExecutionGuard(config.resolution_m, config.robot_radius_m)
    anchor = (*world.position, world.heading)
    if tuple(route['states'][0]) != anchor or tuple(route['states'][-1]) != anchor:
        raise ValueError('sealed route does not restore the actual prefix pose and heading')
    for a, action, b in zip(route['states'], route['actions'], route['states'][1:]):
        if successor(tuple(a), action) != tuple(b):
            raise ValueError('unpaid jump in sealed route')
    if len(route['actions']) != route['cost'] or not 1 <= route['cost'] <= budget:
        raise ValueError('invalid sealed cost')
    mesh, before = snapshot_metrics(mapper, world, evaluator, reference, protocol['metrics']['thresholds_m'])
    common_prefix_mesh = folder.parent / 'executed_prefix_mesh.npz'
    if common_prefix_mesh.exists():
        saved = np.load(common_prefix_mesh)
        np.testing.assert_array_equal(saved['vertices'], np.asarray(mesh.vertices))
        np.testing.assert_array_equal(saved['triangles'], np.asarray(mesh.triangles))
    else:
        save_mesh(common_prefix_mesh, mesh)
    os.link(common_prefix_mesh, folder / 'prefix_mesh.npz')
    metrics = [{'action_index': 0, **before}]
    prefix_seen = reference['prefix_seen']; weights = reference['weights']
    masks = {name: np.zeros_like(prefix_seen) for name in ('outbound', 'endpoint', 'return')}
    actions = []; decisions = []; mode = 'route'; cursor = 0; terminal = None; arrival = None
    collision_count = 0; failure = None

    def decision(kind, **detail):
        row = {'decision_id': len(decisions), 'kind': kind, 'mode': mode,
               'paid_actions_before': len(actions), 'absolute_step': world.step_count,
               'position': list(world.position), 'heading': world.heading,
               'remaining_actions': budget - len(actions), **detail}
        decisions.append(row)
        with (folder / 'decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    while terminal is None:
        current = (*world.position, world.heading); remaining = budget - len(actions)
        return_now = guard.return_plan(mapper.belief, world.position, world.heading, anchor, remaining)
        decision('current_return_feasibility', **asdict(return_now))
        if not return_now.available:
            terminal = 'return_unavailable:' + return_now.reason; failure = terminal; break
        if mode == 'route' and cursor == len(route['actions']):
            terminal = 'original_route_completed' if current == anchor else 'route_ended_away_from_anchor'
            break
        if mode == 'return' and current == anchor:
            terminal = 'returned_after_route_denial'; break
        if remaining == 0:
            terminal = 'budget_exhausted'; failure = terminal if current != anchor else None; break
        action = route['actions'][cursor] if mode == 'route' else return_now.actions[0]
        assessment = guard.assess(mapper.belief, world.position, world.heading, action, remaining)
        decision('action_assessment', action=action, route_cursor=cursor, **asdict(assessment))
        denied = not assessment.allowed
        reason = assessment.reason
        if not denied:
            after_state = successor(current, action)
            after_plan = guard.return_plan(mapper.belief, after_state[:2], after_state[2], anchor, remaining - 1)
            decision('successor_return_feasibility', action=action, **asdict(after_plan))
            denied = not after_plan.available; reason = 'successor_return:' + after_plan.reason
        if denied:
            if mode == 'route':
                mode = 'return'; decision('switch_to_return', reason=reason); continue
            terminal = 'return_action_denied:' + reason; failure = terminal; break
        if shutil.disk_usage(folder).free < protocol['storage']['reserve_bytes']:
            raise RuntimeError('disk reserve reached; retain all partial artifacts and do not skip any branch')
        frame, collision, done = world.step(action); scan = world.scan()
        paid = len(actions) + 1
        frame.save(folder / 'frames' / f'{paid:04d}.npz'); scan.save(folder / 'scans' / f'{paid:04d}.npz')
        mapper.update(frame, scan); update_collision(mapper, world.position, world.heading, collision)
        actual = (*world.position, world.heading)
        target_now = arrival is None and actual == tuple(route['pose'])
        if target_now:
            arrival = paid
        stage = 'endpoint' if target_now else 'return' if mode == 'return' or arrival is not None else 'outbound'
        pose = camera_pose(world.position, world.heading, config, world.shape[0])
        masks[stage] |= reference_visible(evaluator.reference, frame, evaluator.truth, pose, config.max_depth_m)
        row = {'action_index': paid, 'absolute_step': world.step_count, 'action': action,
               'position': list(world.position), 'heading': world.heading, 'collision': bool(collision),
               'done': bool(done), 'mode': mode, 'stage': stage, 'expected_state': list(successor(current, action))}
        actions.append(row)
        with (folder / 'actions.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        if mode == 'route':
            cursor += 1
        if target_now:
            mesh, values = snapshot_metrics(mapper, world, evaluator, reference, protocol['metrics']['thresholds_m'])
            metrics.append({'action_index': paid, **values}); save_mesh(folder / 'arrival_mesh.npz', mesh)
        collision_count += int(collision)
        if collision or actual != successor(current, action):
            terminal = 'collision_or_physical_state_mismatch'; failure = terminal
        elif mode == 'route' and actual != tuple(route['states'][cursor]):
            terminal = 'route_state_mismatch'; failure = terminal
        elif done:
            terminal = 'world_budget_exhausted'; failure = terminal
    decision('terminal_stop', reason=terminal, paid_sensor_capture_on_stop=False)
    mesh, after = snapshot_metrics(mapper, world, evaluator, reference, protocol['metrics']['thresholds_m'])
    save_mesh(folder / 'final_mesh.npz', mesh)
    paid = len(actions)
    if metrics[-1]['action_index'] == paid:
        metrics[-1] = {'action_index': paid, **after}
    else:
        metrics.append({'action_index': paid, **after})
    returned = (*world.position, world.heading) == anchor and guard.return_plan(
        mapper.belief, world.position, world.heading, anchor, budget - paid).available
    if not returned and failure is None:
        failure = 'stopped_away_from_safe_anchor'
    future = np.logical_or.reduce(list(masks.values())); new = future & ~prefix_seen
    area = surface_increment(prefix_seen, future, weights)
    slots = [reference['west_slot'], reference['east_slot']]
    slot_remaining = [float(weights[s & ~prefix_seen].sum()) for s in slots]
    slot_new = [float(weights[s & new].sum()) for s in slots]
    cumulative = prefix_seen.copy(); increments = {}
    for stage, mask in masks.items():
        increments[stage] = surface_increment(cumulative, mask, weights); cumulative |= mask
    np.testing.assert_allclose(sum(increments.values()), area, rtol=0, atol=1e-10)
    result = {'context': world.context.context_id, 'arrangement': world.arrangement,
              'candidate_id': route['candidate_id'], 'group': route['group'], 'planned_actions': route['cost'],
              'paid_actions': paid, 'prefix_paid_actions': 150, 'task_paid_actions': 150 + paid,
              'failure': failure, 'terminal_reason': terminal, 'collision_count': collision_count,
              'original_target_reached': arrival is not None, 'actual_arrival_action': arrival,
              'full_original_route_completed': terminal == 'original_route_completed',
              'returned_to_anchor': bool(returned), 'new_area_m2': area,
              'area_per_action': area / paid if paid else None,
              'before': before, 'after': after, 'f1_gain_05cm': after['f1_05cm'] - before['f1_05cm'],
              'f1_gain_per_action': (after['f1_05cm'] - before['f1_05cm']) / paid if paid else None,
              'coverage_gain_m2': after['covered_area_m2'] - before['covered_area_m2'],
              'slot_new_area_m2': slot_new, 'slot_remaining_area_m2': slot_remaining,
              'slot_new_fractions': [v / r if r else None for v, r in zip(slot_new, slot_remaining)],
              'background_new_area_m2': float(weights[new & (reference['classes'] == 1)].sum()),
              'stage_new_area_m2': increments, 'zero_paid_action_rate_is_undefined': paid == 0}
    for threshold in protocol['metrics']['thresholds_m']:
        tag = f'{round(threshold * 100):02d}cm'
        trace = np.interp(np.arange(budget + 1), [m['action_index'] for m in metrics], [m[f'joint_{tag}'] for m in metrics])
        result[f'branch_joint_auc_{tag}'] = float(np.trapz(trace) / budget)
    np.savez_compressed(folder / 'visibility.npz', prefix=prefix_seen, **masks, union=future)
    np.savez_compressed(folder / 'final_map.npz', belief=mapper.belief, camera_seen=mapper.camera_seen)
    write_json(folder / 'actions.json', actions); write_json(folder / 'metrics.json', metrics)
    write_json(folder / 'outcome.json', result)
    return result


def run(prepared, review_path, output):
    if output.exists(): raise FileExistsError(output)
    metadata = read(prepared / 'metadata.json'); inputs = read(prepared / 'artifact_hashes.json')
    if metadata['status'] != 'complete_structural_pass' or not read(prepared / 'structure_summary.json')['passed']:
        raise ValueError('all eight structural histories must pass before any branch')
    if not all(file_hash(prepared / name) == digest for name, digest in inputs.items()):
        raise ValueError('sealed preparation changed')
    review = read(review_path)
    if not review['release_allowed'] or review['prepared_manifest_sha256'] != file_hash(prepared / 'artifact_hashes.json'):
        raise ValueError('independent structure review has not released this exact preparation')
    for name, digest in metadata['source_sha256'].items():
        if file_hash(ROOT / name) != digest: raise ValueError(f'prepared source changed: {name}')
    protocol = read(ROOT / metadata['protocol_path'])
    if (protocol['schema_version'] != 'competition_v9_validation_protocol/1'
            or protocol['parent_order'] != ['Q0', 'Q1', 'Q2', 'Q3']
            or protocol['physical_branch_limit'] != 48):
        raise ValueError('only the complete fixed V9 batch is permitted')
    if shutil.disk_usage(ROOT).free < protocol['storage']['reserve_bytes']:
        raise RuntimeError('disk reserve reached before execution')
    guard_reviews = [ROOT / f'eval_results/execution_guard_v8_{suffix}_20260911/verification.json'
                     for suffix in ('failures', 'success_impacts')]
    if not all(read(p)['passed_full'] for p in guard_reviews):
        raise ValueError('complete guard side-effect evidence required')
    output.mkdir(parents=True)
    source_names = sorted(set(metadata['source_sha256']) | {'scripts/eval_competition_v9.py', 'scripts/replay_competition_v9.py', 'scripts/analyze_competition_v9.py', 'scripts/analyze_competition_v8_1.py'})
    hashes = {name: file_hash(ROOT / name) for name in source_names}
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in source_names: archive.write(ROOT / name, name)
    for context in protocol['parent_order']:
        for arrangement in protocol['arrangement_order']:
            source = prepared / context / arrangement; target = output / context / arrangement
            for p in source.rglob('*'):
                if p.is_file():
                    destination = target / p.relative_to(source); destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(p, destination)
    seal = {str(p.relative_to(output)): file_hash(p) for context in protocol['parent_order']
            for p in (output / context).rglob('*') if p.is_file()}
    write_json(output / 'pre_execution_seal.json', {'decision_assets': seal, 'source_sha256': hashes,
               'prepared_manifest_sha256': file_hash(prepared / 'artifact_hashes.json'),
               'structural_review_sha256': file_hash(review_path),
               'guard_verification_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in guard_reviews},
               'protocol': protocol, 'physical_branch_limit': 48,
               'input_pose_is_exact': True, 'complete_ANS_or_mainstream_superiority_claim': False})
    meta = {'status': 'running', 'prepared': str(prepared.resolve()), 'source_sha256': hashes,
            'scope': 'all 48 prospectively fixed shared validation branches; explicit prior; no training or calibration',
            'independent_replay': 'pending', 'new_physical_branches': 0}
    write_json(output / 'metadata.json', meta); started = time.time(); outcomes = []
    try:
        for context in protocol['parent_order']:
            for arrangement in protocol['arrangement_order']:
                folder = output / context / arrangement; prefix = folder / 'prefix'
                world = create_competition_world(context, arrangement)
                reference = dict(np.load(folder / 'reference.npz'))
                evaluator = restore_evaluator(world, reference)
                records = read(prefix / 'records.json')
                frames = [RGBDFrame.load(prefix / 'frames' / f'{i:04d}.npz') for i in range(len(records))]
                scans = [PlanarScan.load(prefix / 'scans' / f'{i:04d}.npz') for i in range(len(records))]
                routes = read(folder / 'candidates.json')
                if [r['candidate_id'] for r in routes] != list(range(6)):
                    raise ValueError('exactly all six shared candidates must be executed')
                for route in routes:
                    result = execute(world, route, frames, scans, records, evaluator, reference,
                                     folder / f"candidate_{route['candidate_id']:03d}", protocol)
                    outcomes.append(result)
                    print('executed', context, arrangement, route['candidate_id'], 'paid', result['paid_actions'],
                          'failure', result['failure'], 'new_area', result['new_area_m2'], flush=True)
                write_json(folder / 'outcomes.json', [r for r in outcomes if r['context'] == context and r['arrangement'] == arrangement])
        if len(outcomes) != 48: raise RuntimeError('finite branch set is incomplete')
        if not all(file_hash(output / name) == digest for name, digest in seal.items()):
            raise RuntimeError('sealed inputs changed')
        if not all(file_hash(ROOT / name) == digest for name, digest in hashes.items()):
            raise RuntimeError('source changed during execution')
        write_json(output / 'summary.json', {'outcomes': outcomes, 'physical_branches': 48,
                   'paid_branch_actions': sum(r['paid_actions'] for r in outcomes),
                   'shared_prefix_actions_per_history': 150, 'unique_prefix_histories': 8})
        meta.update(status='complete_execution_pending_independent_replay', new_physical_branches=48,
                    paid_actions=sum(r['paid_actions'] for r in outcomes),
                    failures=sum(r['failure'] is not None for r in outcomes))
    except Exception as exc:
        meta.update(status='failed_execution', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        meta['elapsed_s'] = time.time() - started
        write_json(output / 'metadata.json', meta)
        write_json(output / 'artifact_hashes.json', {str(p.relative_to(output)): file_hash(p)
                    for p in output.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--structural-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.prepared, args.structural_review, args.output)
