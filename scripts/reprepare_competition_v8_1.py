#!/usr/bin/env python3
"""Repair candidate coordinate consistency on the same sealed V8 raw prefixes.

No sensor is recaptured; no physical future branch, reward, or world constructor
is called. All original failed construction records remain untouched.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from env.virtual3d_competition_v8 import CompetitionConfigV8
from nso.competition_candidates_v8_1 import candidate_routes
from nso.competition_prior_v8 import score_routes
from scripts.prepare_competition_v8 import check_pair
from scripts.prepare_response_v7 import array_hash
from scripts.eval_counterfactual_views import remap, write_json
from utils.rgbd_contract import RGBDFrame, PlanarScan
from utils.cpu_protocol import file_hash


def read(path):
    return json.loads(Path(path).read_text())


def run(base, output):
    if output.exists():
        raise FileExistsError(output)
    inputs = read(base / 'artifact_hashes.json')
    if not all(file_hash(base / name) == digest for name, digest in inputs.items()):
        raise ValueError('original sealed V8 preparation has changed')
    original = read(base / 'metadata.json')
    if original['status'] != 'halted_structural_failure' or original['new_candidate_branches'] != 0:
        raise ValueError('this revision requires the unchanged, halted pre-outcome V8 construction')
    protocol_path = ROOT / 'configs/virtual3d/competition_v8_1_protocol.json'
    protocol = read(protocol_path)
    revision = protocol['revision_provenance']
    for name, digest in revision['original_bindings'].items():
        if file_hash(base / name) != digest:
            raise ValueError(f'base differs from the explicitly bound failed construction: {name}')
    for key in ('base_protocol', 'immutable_scorer', 'immutable_scoring_geometry_module'):
        if file_hash(ROOT / revision[key]) != revision[key + '_sha256']:
            raise ValueError(f'unchanged contract dependency differs: {key}')
    old_summary = read(base / 'structure_summary.json')
    output.mkdir(parents=True)
    names = sorted({str(p.relative_to(ROOT)) for folder in ('env', 'nso', 'utils')
                    for p in (ROOT / folder).glob('*.py')} | {
        'scripts/reprepare_competition_v8_1.py', 'scripts/prepare_competition_v8.py',
        'scripts/prepare_response_v7.py', 'scripts/eval_counterfactual_views.py',
        'configs/virtual3d/competition_v8_contexts.json',
        'configs/virtual3d/competition_v8_protocol.json',
        'configs/virtual3d/competition_v8_1_protocol.json', 'requirements-3d.lock.txt'})
    hashes = {name: file_hash(ROOT / name) for name in names}
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, name)
    metadata = {'status': 'running', 'source_sha256': hashes, 'base_preparation': str(base.resolve()),
                'base_manifest_sha256': file_hash(base / 'artifact_hashes.json'),
                'new_sensor_captures': 0, 'new_candidate_branches': 0,
                'scope': 'V8.1 geometric coordinate consistency correction; identical measured prefixes',
                'protocol_path': str(protocol_path.relative_to(ROOT))}
    write_json(output / 'metadata.json', metadata)
    started = time.time(); summaries = []; structures = []; pairs = {}
    try:
        for context in protocol['parent_order']:
            pair = []
            for arrangement in protocol['arrangement_order']:
                source = base / context / arrangement; folder = output / context / arrangement
                folder.mkdir(parents=True)
                for name in ('fixture.json', 'prefix_map.npz', 'reference.npz'):
                    os.link(source / name, folder / name)
                for p in (source / 'prefix').rglob('*'):
                    if p.is_file():
                        target = folder / p.relative_to(source); target.parent.mkdir(parents=True, exist_ok=True)
                        os.link(p, target)
                fixture = read(folder / 'fixture.json'); config = CompetitionConfigV8(**fixture['config'])
                records = read(folder / 'prefix/records.json')
                frames = [RGBDFrame.load(folder / 'prefix/frames' / f'{i:04d}.npz') for i in range(len(records))]
                scans = [PlanarScan.load(folder / 'prefix/scans' / f'{i:04d}.npz') for i in range(len(records))]
                prefix_poses = [f.world_from_camera for f in frames]
                shape = np.load(folder / 'prefix_map.npz')['belief'].shape
                mapper = remap(frames, scans, records, config, shape)
                last = records[-1]; obs = mapper.observation(tuple(last['position']), last['heading'], last['step'], False)
                routes, audit = candidate_routes(mapper, obs)
                prediction = score_routes(mapper, routes, prefix_poses)
                write_json(folder / 'candidates.json', routes); write_json(folder / 'candidate_audit.json', audit)
                write_json(folder / 'predictions.json', prediction)
                for mode, expected in (('shuffled', 'X'), ('absent', 'G')):
                    changed = remap(frames, scans, records, config, shape, mode)
                    changed_routes, _ = candidate_routes(changed, obs)
                    if routes != changed_routes:
                        raise ValueError('label intervention changed candidate pool')
                    actual = score_routes(changed, routes, prefix_poses)
                    np.testing.assert_array_equal(actual['scores']['S'], prediction['scores'][expected])
                    del changed
                q = mapper.quality_evidence(max_points=10**9); mesh = mapper.mesh()
                geom = array_hash([('belief', mapper.belief), ('camera_seen', mapper.camera_seen),
                                   ('vertices', np.asarray(mesh.vertices)), ('triangles', np.asarray(mesh.triangles)),
                                   ('normals', np.asarray(mesh.vertex_normals))]
                                  + ([] if q is None else [(k, q[k]) for k in sorted(q) if k != 'label']))
                old = next(r for r in old_summary['prefix_summaries'] if r['context'] == context and r['arrangement'] == arrangement)
                if geom != old['geometry_sha256']:
                    raise ValueError('original observed nonlabel geometry changed')
                pair.append({'frames': frames, 'scans': scans, 'records': records, 'routes': routes,
                             'geometry_hash': geom, 'predictions': prediction})
                summary = {**old, 'candidate_count': len(routes), 'candidate_status': audit['status'],
                           'roles': [r['group'] for r in routes], 'costs': [r['cost'] for r in routes],
                           'choices': prediction['selected_candidate_ids']}
                summaries.append(summary)
                structural = dict(read(source / 'structural_audit.json'))
                checks = dict(structural['checks'])
                checks['candidate_roles'] = summary['roles'] == protocol['pre_outcome_structural_gates']['candidate_roles_in_order']
                checks['two_markers'] = len(audit['measured_assets']) == 2 and all(a['marked_points'] > 0 for a in audit['measured_assets'])
                structural.update(checks=checks, passed=all(checks.values()),
                                  original_structure_sha256=file_hash(source / 'structural_audit.json'),
                                  original_reference_sha256=file_hash(source / 'reference.npz'),
                                  actual_prefix_and_reference_reused=True)
                structures.append(structural); write_json(folder / 'structural_audit.json', structural)
                print('reprepared', summary, flush=True)
                del mapper, mesh
            pairs[context] = check_pair(*pair)
            pairs[context]['union_area_equal'] = old_summary['pairs'][context]['union_area_equal']
            del pair
        passed = all(s['passed'] for s in structures) and all(p['union_area_equal'] for p in pairs.values())
        write_json(output / 'pre_evaluator_choices_seal.json', {str(p.relative_to(output)): file_hash(p)
                    for context in protocol['parent_order'] for p in (output / context).rglob('*') if p.is_file()})
        write_json(output / 'structure_summary.json', {'passed': passed, 'histories': structures, 'pairs': pairs,
                    'prefix_summaries': summaries, 'new_candidate_branches': 0, 'new_sensor_captures': 0})
        if not all(file_hash(ROOT / name) == digest for name, digest in hashes.items()):
            raise RuntimeError('source changed during preparation')
        if not all(file_hash(base / name) == digest for name, digest in inputs.items()):
            raise RuntimeError('original inputs changed')
        metadata.update(status='complete_structural_pass' if passed else 'halted_structural_failure',
                        structural_gate_passed=passed, elapsed_s=time.time() - started)
    except Exception as exc:
        metadata.update(status='failed_preparation', error=f'{type(exc).__name__}: {exc}', elapsed_s=time.time() - started)
        raise
    finally:
        write_json(output / 'metadata.json', metadata)
        write_json(output / 'artifact_hashes.json', {str(p.relative_to(output)): file_hash(p)
                    for p in output.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.base, args.output)
