#!/usr/bin/env python3
"""Execute all sealed paired V7 routes, recording actual RGB-D and metrics."""
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
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from env.virtual3d import camera_pose
from env.virtual3d_response_v7 import ResponseContextV7, ResponseConfigV7, ResponseWorldV7, RESPONSE_FAMILIES
from scripts.eval_counterfactual_views import (remap, state_restore, update_collision,
                                              coverage, save_mesh, write_json)
from utils.cpu_protocol import file_hash
from utils.rgbd_contract import RGBDFrame, PlanarScan
from utils.reconstruction_metrics import ReconstructionEvaluator
from utils.counterfactual_surface_visibility import reference_visible, surface_increment


def read(path): return json.loads(path.read_text())


def execute_branch(world, evaluator, route, frames, scans, records, before, seen, weight, folder, config):
    folder.mkdir(); (folder / 'frames').mkdir(); (folder / 'scans').mkdir()
    c = world.config
    mapper = remap(frames, scans, records, c, world.shape)
    state_restore(world, records[-1], frames[-1], scans[-1])
    c0 = coverage(mapper, world.reachable)
    metrics = [{'action_index': 0, 'coverage_2d': c0, **before}]
    masks = {name: np.zeros_like(seen) for name in ('outbound', 'endpoint', 'return')}
    actions = []; failure = None
    for j, action in enumerate(route['actions'], 1):
        if shutil.disk_usage(folder).free < config['storage']['reserve_bytes']:
            raise RuntimeError('disk reserve reached; keep the incomplete branch and do not skip candidates')
        frame, collision, done = world.step(action); scan = world.scan()
        frame.save(folder / 'frames' / f'{j:04d}.npz'); scan.save(folder / 'scans' / f'{j:04d}.npz')
        mapper.update(frame, scan); update_collision(mapper, world.position, world.heading, collision)
        actual = (*world.position, world.heading)
        stage = 'outbound' if j < route['arrival_action'] else 'endpoint' if j == route['arrival_action'] else 'return'
        pose = camera_pose(world.position, world.heading, c, world.shape[0])
        masks[stage] |= reference_visible(evaluator.reference, frame, evaluator.truth, pose, c.max_depth_m)
        now_c = coverage(mapper, world.reachable)
        actions.append({'action_index': j, 'absolute_step': world.step_count, 'action': action,
            'position': list(world.position), 'heading': world.heading, 'collision': collision,
            'coverage_2d': now_c, 'stage': stage})
        if collision or actual != tuple(route['states'][j]): failure = 'collision_or_execution_mismatch'
        if done and j < len(route['actions']): failure = 'world_budget_exhausted'
        if j == route['arrival_action'] or j == len(route['actions']) or failure:
            mesh = mapper.mesh()
            metrics.append({'action_index': j, 'coverage_2d': now_c,
                            **evaluator.evaluate(mesh, now_c, config['thresholds_m'])})
            save_mesh(folder / ('arrival_mesh.npz' if j == route['arrival_action'] else 'final_mesh.npz'), mesh)
        if failure: break
    if not actions: raise ValueError('zero-cost executed branch')
    if not (folder / 'final_mesh.npz').exists(): save_mesh(folder / 'final_mesh.npz', mapper.mesh())
    if failure is None: assert (*world.position, world.heading) == tuple(route['states'][0])
    future = np.logical_or.reduce(list(masks.values())); cumulative = seen.copy(); parts = {}
    for stage, mask in masks.items():
        parts[stage] = surface_increment(cumulative, mask, weight); cumulative |= mask
    area = surface_increment(seen, future, weight)
    np.testing.assert_allclose(sum(parts.values()), area, atol=1e-10, rtol=0)
    final = metrics[-1]; paid = len(actions)
    result = {'candidate_id': route['candidate_id'], 'paid_actions': paid, 'planned_actions': route['cost'],
        'arrival_actions': route['arrival_action'], 'failure': failure, 'new_area_m2': area,
        'area_per_action': area / paid, 'stage_new_area_m2': parts,
        'f1_gain_05cm': final['f1_05cm'] - before['f1_05cm'],
        'f1_gain_per_action': (final['f1_05cm'] - before['f1_05cm']) / paid,
        'coverage_gain_m2': (final['coverage_2d'] - c0) * world.reachable.sum() * c.resolution_m**2,
        'before': before, 'after': final}
    for threshold in config['thresholds_m']:
        tag = f'{round(threshold * 100):02d}cm'
        trace = np.interp(np.arange(config['branch_actions'] + 1),
                          [r['action_index'] for r in metrics], [r[f'joint_{tag}'] for r in metrics])
        result[f'branch_joint_auc_{tag}'] = float(np.trapz(trace) / config['branch_actions'])
    np.savez_compressed(folder / 'visibility.npz', prefix=seen, **masks, union=future)
    write_json(folder / 'actions.json', actions); write_json(folder / 'metrics.json', metrics)
    write_json(folder / 'outcome.json', result)
    return result


def run(prepared, output):
    metadata = read(prepared / 'metadata.json')
    assert metadata['status'] == 'complete' and metadata['candidate_structure_gate_passed']
    inputs = read(prepared / 'artifact_hashes.json')
    assert all(file_hash(prepared / name) == sha for name, sha in inputs.items())
    if output.exists(): raise FileExistsError(output)
    config_path = ROOT / 'configs/virtual3d/response_v7_pipeline.json'
    config = read(config_path)
    assert config['prefix_actions'] == metadata['context']['prefix_step']
    assert metadata['context']['context_id'] in config['train_contexts'] + config['calibration_contexts']
    if metadata['context']['role'] == 'calibration':
        assert metadata['bank_sha256'] is not None
    for name, sha in metadata['source_sha256'].items():
        assert file_hash(ROOT / name) == sha, f'prepared dependency changed: {name}'
    output.mkdir(parents=True); started = time.time()
    names = sorted(set(metadata['source_sha256']) | {'scripts/eval_response_v7.py', str(config_path.relative_to(ROOT))})
    hashes = {name: file_hash(ROOT / name) for name in names}
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names: archive.write(ROOT / name, name)
    write_json(output / 'config.json', config)
    meta = {'status': 'running', 'source_sha256': hashes, 'prepared': str(prepared.resolve()),
        'prepared_manifest_sha256': file_hash(prepared / 'artifact_hashes.json'),
        'context': metadata['context'], 'scope': 'paired development response acquisition, not system efficacy',
        'candidate_limit': 6, 'replayed': False, 'storage': 'all raw observations and meshes retained pending replay'}
    write_json(output / 'metadata.json', meta)
    # Copy only the already sealed input artifacts before any evaluator is constructed.
    for family in RESPONSE_FAMILIES:
        folder = output / family; folder.mkdir()
        for p in (prepared / family).rglob('*'):
            if p.is_file():
                target = folder / p.relative_to(prepared / family); target.parent.mkdir(parents=True, exist_ok=True)
                os.link(p, target)
    seal = {str(p.relative_to(output)): file_hash(p) for family in RESPONSE_FAMILIES
            for p in (output / family).rglob('*') if p.is_file()}
    write_json(output / 'pre_outcome_seal.json', seal)
    all_outcomes = []
    try:
        for family in RESPONSE_FAMILIES:
            folder = output / family; fixture = read(folder / 'fixture.json')
            context_data = dict(fixture['context']); context_data['offset_xy_m'] = tuple(context_data['offset_xy_m'])
            world = ResponseWorldV7(ResponseContextV7(**context_data), family, ResponseConfigV7(**fixture['config']))
            c = world.config; prefix = folder / 'prefix'; records = read(prefix / 'records.json')
            frames = [RGBDFrame.load(prefix / 'frames' / f'{i:04d}.npz') for i in range(len(records))]
            scans = [PlanarScan.load(prefix / 'scans' / f'{i:04d}.npz') for i in range(len(records))]
            routes = read(folder / 'candidates.json')
            assert len(routes) == config['candidate_limit']
            evaluator = ReconstructionEvaluator(world, config['reference_samples'])
            weight = world.mesh.get_surface_area() / config['reference_samples']
            np.savez_compressed(folder / 'reference.npz', points=evaluator.reference, classes=evaluator.classes,
                weights=np.full(len(evaluator.reference), weight), reachable=world.reachable,
                vertices=np.asarray(world.mesh.vertices), triangles=np.asarray(world.mesh.triangles))
            mapper = remap(frames, scans, records, c, world.shape)
            c0 = coverage(mapper, world.reachable); mesh = mapper.mesh()
            before = evaluator.evaluate(mesh, c0, config['thresholds_m'])
            save_mesh(folder / 'prefix_mesh.npz', mesh)
            np.savez_compressed(folder / 'prefix_map.npz', belief=mapper.belief)
            del mapper
            seen = np.zeros(len(evaluator.reference), bool)
            for frame, row in zip(frames, records):
                seen |= reference_visible(evaluator.reference, frame, evaluator.truth,
                    camera_pose(tuple(row['position']), row['heading'], c, world.shape[0]), c.max_depth_m)
            outcomes = []
            for route in routes:
                result = execute_branch(world, evaluator, route, frames, scans, records, before, seen, weight,
                                        folder / f"candidate_{route['candidate_id']:03d}", config)
                outcomes.append(result)
                print('executed', context_data['context_id'], family, route['candidate_id'],
                      'actions', result['paid_actions'], 'failure', result['failure'], flush=True)
            write_json(folder / 'outcomes.json', outcomes)
            all_outcomes.extend({'family': family, **row} for row in outcomes)
        assert all(file_hash(output / name) == sha for name, sha in seal.items())
        assert all(file_hash(prepared / name) == sha for name, sha in inputs.items())
        assert all(file_hash(ROOT / name) == sha for name, sha in hashes.items()), 'source changed during execution'
        meta.update(status='complete', physical_branches=len(all_outcomes),
                    paid_actions=sum(r['paid_actions'] for r in all_outcomes),
                    failures=sum(r['failure'] is not None for r in all_outcomes), elapsed_s=time.time()-started)
        write_json(output / 'summary.json', {'context': context_data['context_id'], 'outcomes': all_outcomes,
                    'scope': 'all predeclared candidates, no fitted V7 superiority claim'})
    except Exception as exc:
        meta.update(status='failed_execution', error=f'{type(exc).__name__}: {exc}', elapsed_s=time.time()-started)
        raise
    finally:
        write_json(output / 'metadata.json', meta)
        write_json(output / 'artifact_hashes.json', {str(p.relative_to(output)): file_hash(p)
            for p in output.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.prepared, args.output)
