#!/usr/bin/env python3
"""Supplemental endpoint evaluation after the original pilot sample-cap failure.

Only the technical allocation ceiling changes: every original sample remains,
with the same spacing, RNG seed, threshold, complete mesh and frozen reference.
The original pilot sources, failed review and successful replay stay immutable.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from env.development_sensor_v41 import runtime_counts_v41
from nso.episode_driver_v43 import canonical_bytes, file_sha256
from nso.mechanism_development import inspect_pilot
from nso.offline_evaluation_v44 import load_reference_v44, measure_navigation_coverage_v44
from nso.saved_replay_v44 import _arrays
from nso.surface_evaluation_v40 import evaluate_surface_v40

MAX_SAMPLES = 1_000_000
SPACING_M = .3
SEED = 4002
THRESHOLD_M = .05


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content)


def evaluate(episode, prior_review, *, expected_manifest_sha256, prior_review_sha256, output):
    episode, prior_review, output = (Path(p).resolve() for p in (episode, prior_review, output))
    if output.is_relative_to(episode) or output == prior_review or output.exists():
        raise ValueError('new output outside original episode/review is required')
    if file_sha256(prior_review) != prior_review_sha256:
        raise ValueError('prior review differs from external pin')
    protocol, slot, started, result, manifest = inspect_pilot(episode, expected_manifest_sha256)
    prior = read_json(prior_review)
    replay = prior.get('replay', {})
    if (prior.get('episode_manifest_sha256') != expected_manifest_sha256
            or prior.get('run_id') != started['run_id']
            or prior.get('source_sha256') != manifest['source_sha256']
            or replay.get('status') != 'verified' or replay.get('prediction_verified') is not True
            or replay.get('frames_verified') != result['acquired_and_saved_packets']
            or prior.get('runtime', {}).get('no_new_world_or_sensor_action') is not True):
        raise ValueError('original review must prove complete saved-frame policy/map replay of this episode')
    if prior.get('status') != 'pilot_review_failed' or prior.get('error') != {
            'type': 'ValueError', 'message': 'declared surface sample limit exceeded'}:
        raise ValueError('supplement is restricted to the diagnosed technical sample-cap failure')
    source = Path(__file__).resolve(); source_bytes = source.read_bytes(); source_sha = file_sha256(source)
    start_path = output.with_suffix('.started.json')
    source_path = output.with_suffix('.source.py')
    receipt = dict(schema='mechanism.pilot.supplemental_evaluation.v1', status='evaluation_started',
        run_id=started['run_id'], mode=slot['mode'], episode_root=str(episode),
        episode_manifest_sha256=expected_manifest_sha256,
        original_review_path=str(prior_review), original_review_sha256=prior_review_sha256,
        original_review_status=prior['status'], original_failure=prior['error'],
        original_replay=replay, original_episode_status=result['status'],
        original_executed_paid_actions=result['executed_paid_actions'],
        supplemental_source_sha256=source_sha, supplemental_source_copy=str(source_path),
        capacity_amendment=dict(original_max_samples=50000, common_max_samples=MAX_SAMPLES,
            reason='one sample minimum per original TSDF triangle exceeded technical allocation ceiling',
            changes_sample_positions_or_weights=False, subsampling=False, mesh_decimation=False,
            sample_spacing_m=SPACING_M, seed=SEED, threshold_m=THRESHOLD_M,
            applies_equally_to=['pilot_A_G', 'pilot_A_S']),
        new_worlds=0, new_sensor_packets=0, new_tsdf_integrations=0,
        prediction_reintegrated=False, original_sources_or_outputs_modified=False,
        semantic_performance_claim=False, primary_experiment_started=False)
    write_new(source_path, source_bytes)
    write_new(start_path, canonical_bytes(receipt))
    began=time.monotonic(); runtime_before=runtime_counts_v41()
    try:
        reference, domain, record = load_reference_v44(ROOT/protocol['reference_root'],
            manifest_sha256=protocol['reference_manifest_sha256'], asset_id=slot['asset_id'])
        mapper=read_json(episode/'prediction/mapper.json'); descriptor=record['coverage']
        for key in ('shape','resolution_m','origin_xy_m','grid_convention'):
            if mapper[key] != descriptor[key]:
                raise ValueError('fixed coverage coordinates differ from prediction')
        occupancy=_arrays(episode/'prediction/occupancy.npz', ('belief','observed'))
        coverage=measure_navigation_coverage_v44(occupancy['belief'],domain,descriptor)
        mesh=_arrays(episode/'prediction/mesh.npz', ('vertices','triangles','vertex_colors'))
        xyz=mesh['vertices'][mesh['triangles']]
        areas=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)/2
        receipt['geometry_capacity_diagnostic']=dict(vertices=len(mesh['vertices']),
            submitted_triangles=len(mesh['triangles']), total_area_m2=float(areas.sum()),
            before_duplicate_removal_required_samples=int(np.maximum(1,np.ceil(areas/SPACING_M**2)).sum()))
        metrics=evaluate_surface_v40(reference, mesh['vertices'],mesh['triangles'],C_map=coverage['C_nav'],
            threshold_m=THRESHOLD_M,sample_spacing_m=SPACING_M,seed=SEED,max_samples=MAX_SAMPLES)
        metrics.pop('C_map');metrics.pop('J')
        metrics.update(C_nav=coverage['C_nav'],J_nav=coverage['C_nav']*metrics['Q'])
        receipt.update(status='pilot_endpoint_scored',metrics=metrics,coverage=coverage,
            reference_manifest_sha256=protocol['reference_manifest_sha256'],
            task_success=result['status']=='controller_stop' and result['sensor_status'].get('returned_xy_and_yaw') is True,
            all_task_instances_in_macro_denominator=True,prediction_roi_cropped=False,
            scope='cost and observation-chain pilot; existing V43 greedy controls, not new-mechanism validation')
        inspect_pilot(episode,expected_manifest_sha256)
        if file_sha256(prior_review)!=prior_review_sha256 or file_sha256(source)!=source_sha:
            raise ValueError('supplemental input/source changed during evaluation')
    except Exception as exc:
        receipt.update(status='supplemental_evaluation_failed',error=dict(type=type(exc).__name__,message=str(exc)))
    runtime_after=runtime_counts_v41()
    receipt['runtime']=dict(before=runtime_before,after=runtime_after,
        no_new_world_or_sensor_action=runtime_before==runtime_after)
    if runtime_before!=runtime_after:
        receipt['status']='supplemental_evaluation_failed'
        receipt['runtime_error']='unexpected live sensor activity'
    receipt['elapsed_s']=time.monotonic()-began
    write_new(output,canonical_bytes(receipt))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',required=True,type=Path)
    parser.add_argument('--prior-review',required=True,type=Path)
    parser.add_argument('--expected-manifest-sha256',required=True)
    parser.add_argument('--prior-review-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=evaluate(args.episode,args.prior_review,expected_manifest_sha256=args.expected_manifest_sha256,
        prior_review_sha256=args.prior_review_sha256,output=args.output)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result['status']=='pilot_endpoint_scored' else 2


if __name__=='__main__':
    sys.exit(main())
