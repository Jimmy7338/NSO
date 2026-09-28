#!/usr/bin/env python3
"""Supplement a verified trajectory after a diagnosed positive-face rejection.

No trajectory, sensor, policy or TSDF is rerun. The original failed review and
raw mesh stay immutable. Every positive face participates in the same metric.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.development_sensor_v41 import runtime_counts_v41
from nso.complete_surface_evaluation import evaluate_complete_surface
from nso.episode_driver_v43 import canonical_bytes,file_sha256
from nso.offline_evaluation_v44 import load_reference_v44,measure_navigation_coverage_v44
from nso.saved_replay_v44 import _arrays
from nso.semantic_experiment import inspect_experiment,read_json,reference_spec


def evaluate(episode_root,episode_pin,original_review,review_pin,output):
    episode_root,original_review,output = map(lambda x:Path(x).resolve(),
                                             (episode_root,original_review,output))
    if output.exists() or output.is_relative_to(episode_root):
        raise ValueError('new supplemental output outside immutable episode required')
    if file_sha256(original_review)!=review_pin:
        raise ValueError('original review external pin mismatch')
    previous=read_json(original_review)
    episode=inspect_experiment(episode_root,episode_pin)
    if (not episode['current_sources_match'] or previous['episode_manifest_sha256']!=episode_pin
            or previous['status']!='experiment_review_failed'
            or previous.get('error',{}).get('message')!='nonfinite or degenerate triangle rejected'
            or previous.get('replay',{}).get('status')!='verified'
            or previous['replay']['frames_verified']!=episode['result']['acquired_and_saved_packets']
            or previous['source_sha256']!=episode['manifest']['source_sha256']):
        raise ValueError('bound independent replay and diagnosed original rejection required')
    sources=dict(episode['manifest']['source_sha256'])
    for name in ('nso/complete_surface_evaluation.py','scripts/evaluate_semantic_integration_endpoint.py'):
        sources[name]=file_sha256(ROOT/name)
    output.mkdir(parents=True,exist_ok=False)
    for name,digest in sources.items():
        path=output/'numerical_source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        content=(ROOT/name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('numerical source changed before archival')
        with path.open('xb') as stream:stream.write(content)
    before=runtime_counts_v41();began=time.monotonic()
    result=dict(schema='semantic.endpoint.numeric_supplement.v1',status='semantic_endpoint_evaluation_failed',
        episode_root=str(episode_root),episode_manifest_sha256=episode_pin,
        run_id=episode['started']['run_id'],original_review_path=str(original_review),
        original_review_sha256=review_pin,numerical_source_sha256=sources,
        numerical_source_root=str(output/'numerical_source'),numerical_evaluation_recomputed=True,
        original_replay_rerun=False,automatic_retry=False)
    with (output/'started.json').open('xb') as stream:stream.write(canonical_bytes(result))
    try:
        ref=reference_spec(episode['protocol'],episode['slot']['asset_id'])
        reference,domain,record=load_reference_v44(ROOT/ref['root'],manifest_sha256=ref['manifest_sha256'],asset_id=episode['slot']['asset_id'])
        mapper=read_json(episode_root/'prediction/mapper.json');descriptor=record['coverage']
        for key in ('shape','resolution_m','origin_xy_m','grid_convention'):
            if mapper[key]!=descriptor[key]:raise ValueError('evaluation/mapping coordinates differ')
        occupancy=_arrays(episode_root/'prediction/occupancy.npz',('belief','observed'))
        coverage=measure_navigation_coverage_v44(occupancy['belief'],domain,descriptor)
        mesh=_arrays(episode_root/'prediction/mesh.npz',('vertices','triangles','vertex_colors'))
        triangles=mesh['vertices'][mesh['triangles']]
        areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
        diagnosis=dict(prediction_triangles=len(areas),finite_vertices=bool(np.isfinite(mesh['vertices']).all()),
            zero_area_faces=int(np.count_nonzero(areas==0)),positive_faces_below_old_floor=int(np.count_nonzero((areas>0)&(areas<=5e-13))),
            minimum_area_m2=float(areas.min()),all_faces_preserved=True,
            mesh_sha256=file_sha256(episode_root/'prediction/mesh.npz'),old_floor_twice_area_m2=1e-12)
        if not diagnosis['finite_vertices'] or diagnosis['zero_area_faces'] or not diagnosis['positive_faces_below_old_floor']:
            raise ValueError('actual mesh does not match diagnosed strictly-positive small-face case')
        result['diagnosis']=diagnosis
        with (output/'diagnosis.json').open('xb') as stream:stream.write(canonical_bytes(diagnosis))
        metrics=evaluate_complete_surface(reference,mesh['vertices'],mesh['triangles'],
            C_map=coverage['C_nav'],**episode['protocol']['evaluation'])
        metrics.pop('C_map');metrics.pop('J')
        metrics.update(C_nav=coverage['C_nav'],J_nav=coverage['C_nav']*metrics['Q'])
        result['evaluation']=dict(metrics=metrics,coverage=coverage,reference_manifest_sha256=ref['manifest_sha256'],
            task_success=episode['result']['status']=='controller_stop' and episode['result']['sensor_status']['returned_xy_and_yaw'] is True,
            original_episode_status=episode['result']['status'],all_task_instances_in_macro_denominator=True,
            prediction_roi_cropped=False,new_worlds=0,new_sensor_packets=0,new_tsdf_integrations=0)
        if any(file_sha256(ROOT/name)!=digest for name,digest in sources.items()):
            raise ValueError('numeric source changed during evaluation')
        if file_sha256(original_review)!=review_pin or not inspect_experiment(episode_root,episode_pin)['current_sources_match']:
            raise ValueError('input or executed sources changed during evaluation')
        result['status']='semantic_endpoint_evaluation_completed'
    except Exception as exc:
        result['error']=dict(type=type(exc).__name__,message=str(exc))
    after=runtime_counts_v41()
    result.update(elapsed_s=time.monotonic()-began,runtime=dict(before=before,after=after,no_new_world_or_sensor_action=before==after))
    if before!=after:result['status']='semantic_endpoint_evaluation_failed'
    with (output/'receipt.json').open('xb') as stream:stream.write(canonical_bytes(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('episode','expected-manifest-sha256','original-review','original-review-sha256','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    result=evaluate(args.episode,args.expected_manifest_sha256,args.original_review,args.original_review_sha256,args.output)
    print(json.dumps({k:result[k] for k in ('status','run_id','elapsed_s')},sort_keys=True))
    return 0 if result['status']=='semantic_endpoint_evaluation_completed' else 2


if __name__=='__main__':sys.exit(main())
