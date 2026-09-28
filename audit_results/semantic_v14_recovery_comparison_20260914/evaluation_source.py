#!/usr/bin/env python3
"""Evaluator-only paired reconstruction comparison for the V14 recovery pilot."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from env.virtual3d_inspection_v4 import InspectionConfigV4, InspectionWorldV4
from nso.decision_replay_v13 import load_packet
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from scripts.collect_semantic_gain_v13_history import sha, write
from utils.counterfactual_surface_visibility import reference_visible
from utils.reconstruction_metrics import ReconstructionEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    m = json.loads((args.source / 'manifest.json').read_text())
    if m['status'] != 'complete':
        raise ValueError('complete recovery execution and physical replay required')
    protocol = m['protocol']
    sources = {'baseline': Path(protocol['baseline']), 'recovery': args.source}
    for source in sources.values():
        sm = json.loads((source / 'manifest.json').read_text())
        for name, expected in sm['source_sha256'].items():
            assert sha(Path(name)) == expected, name
        for name, expected in json.loads((source / 'artifact_hashes.json').read_text()).items():
            assert sha(source / name) == expected, name
    scene = json.loads(Path(m['history_protocol']['scene_protocol']).read_text())
    declared = next(c for c in scene['contexts'] if c['id'] == protocol['context'])
    settings = {**scene['shared_conditions'], **{k:v for k,v in declared.items() if k not in ('id','seed')}}
    world = InspectionWorldV4(InspectionConfigV4(**settings), seed=declared['seed'], semantic_condition='aligned')
    evaluator = ReconstructionEvaluator(world, count=protocol['reference_sample_count'], seed=protocol['reference_seed'])
    weight = world.mesh.get_surface_area() / protocol['reference_sample_count']
    np.savez_compressed(args.output / 'reference.npz', points=evaluator.reference, classes=evaluator.classes,
                        sample_area_weight=weight)
    results = {}
    for name, source in sources.items():
        mapper = ObservedRuntimeMapperV10(world.shape, world.config)
        files = sorted((source / 'packets').glob('*.npz'))
        history = json.loads((source / 'history_summary.json').read_text())
        assert len(files) == history['paid_actions'] + 1
        visible = np.zeros(len(evaluator.reference), bool)
        curve, initial_visible = [], None
        for i, path in enumerate(files):
            p = load_packet(path)
            assert p.action_id == i
            mapper.update(p.frame, p.scan)
            visible |= reference_visible(evaluator.reference, p.frame, evaluator.truth,
                                          p.frame.world_from_camera, world.config.max_depth_m)
            if i == 0:
                initial_visible = visible.copy()
            if i % protocol['metrics_every_paid_actions'] == 0 or i == len(files) - 1:
                coverage = float(np.count_nonzero((mapper.belief != -1) & world.reachable) / world.reachable.sum())
                measured = evaluator.evaluate(mapper.mesh(), coverage, thresholds=protocol['thresholds_m'])
                measured.update(action_id=i, coverage_2d=coverage,
                    covered_area_m2=coverage * world.reachable.sum() * world.config.resolution_m ** 2,
                    new_visible_surface_m2=float(np.count_nonzero(visible & ~initial_visible)) * weight)
                curve.append(measured)
        mesh = mapper.mesh()
        np.savez_compressed(args.output / f'{name}_final_mesh.npz', vertices=np.asarray(mesh.vertices),
                            triangles=np.asarray(mesh.triangles), vertex_colors=np.asarray(mesh.vertex_colors))
        final = curve[-1]
        actions = json.loads((source / 'actions.json').read_text())
        if curve[-1]['action_id'] < protocol['total_budget']:
            curve.append({**final, 'action_id':protocol['total_budget'], 'held_after_stop':True})
        results[name] = dict(final=final, curve=curve, history=history,
            path_distance_m=sum(a['action']=='forward' for a in actions) * world.config.resolution_m,
            action_time_s=len(actions) * world.config.action_duration_s,
            normalized_joint_auc=float(np.trapz([c['joint_05cm'] for c in curve],
                                               [c['action_id'] for c in curve]) / protocol['total_budget']))
        print(name, final['coverage_2d'], final['f1_05cm'], final['joint_05cm'], flush=True)
    differences = {key:results['recovery']['final'][key] - results['baseline']['final'][key]
                   for key in ('coverage_2d','precision_05cm','recall_05cm','f1_05cm','joint_05cm','new_visible_surface_m2')}
    result = dict(status='complete', protocol=protocol, results=results, differences=differences,
                  primary_deadline_joint_improved=differences['joint_05cm'] > 0,
                  semantic_efficacy_proven=False, independent_confirmation=False,
                  scope='Common execution recovery on one existing development layout; no semantic policy evaluated.')
    write(args.output / 'result.json', result)
    write(args.output / 'manifest.json', dict(status='complete',
        input_inventories={str(p):sha(p/'artifact_hashes.json') for p in sources.values()},
        evaluation_source_sha256=sha(Path(__file__)), frozen_metric_protocol_sha256=sha(Path('configs/virtual3d/semantic_v14_recovery_comparison.json'))))
    (args.output / 'evaluation_source.py').write_bytes(Path(__file__).read_bytes())
    write(args.output / 'artifact_hashes.json', {str(p.relative_to(args.output)):sha(p)
        for p in sorted(args.output.rglob('*')) if p.is_file()})


if __name__ == '__main__':
    main()
