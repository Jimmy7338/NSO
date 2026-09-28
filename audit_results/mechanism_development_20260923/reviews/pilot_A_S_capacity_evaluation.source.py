#!/usr/bin/env python3
"""Reuse a completed pilot score only after proving all numerical inputs identical.

This is explicitly not a second independent numerical evaluation. Both live
trajectories and their independent saved-frame policy/TSDF replays remain real.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.development_sensor_v41 import runtime_counts_v41
from nso.episode_driver_v43 import canonical_bytes, file_sha256
from nso.mechanism_development import inspect_pilot
from nso.offline_evaluation_v44 import load_reference_v44


def read(path):
    return json.loads(Path(path).read_text())


def replay_receipt(path, expected_sha, episode_sha, started, result, manifest):
    if file_sha256(path) != expected_sha:
        raise ValueError('original review differs from pin')
    review=read(path); replay=review.get('replay', {})
    if (review.get('episode_manifest_sha256') != episode_sha or review.get('run_id') != started['run_id']
            or review.get('source_sha256') != manifest['source_sha256']
            or replay.get('status') != 'verified' or replay.get('prediction_verified') is not True
            or replay.get('frames_verified') != result['acquired_and_saved_packets']
            or review.get('runtime', {}).get('no_new_world_or_sensor_action') is not True):
        raise ValueError('independent complete saved-frame replay must bind each episode')
    return review


def reuse(*, source_evaluation, source_evaluation_sha256, episode, expected_manifest_sha256,
          prior_review, prior_review_sha256, output):
    began=time.monotonic(); before=runtime_counts_v41()
    source_evaluation, episode, prior_review, output=map(lambda p:Path(p).resolve(),
        (source_evaluation,episode,prior_review,output))
    if output.exists() or output==prior_review or output.is_relative_to(episode):
        raise ValueError('new supplemental output outside original episode required')
    if file_sha256(source_evaluation)!=source_evaluation_sha256:
        raise ValueError('completed source evaluation differs from external pin')
    source=read(source_evaluation)
    if (source.get('schema')!='mechanism.pilot.supplemental_evaluation.v1'
            or source.get('status')!='pilot_endpoint_scored'
            or source.get('runtime',{}).get('no_new_world_or_sensor_action') is not True):
        raise ValueError('a completed actual numerical supplemental evaluation is required')
    numerical_source=ROOT/'scripts/evaluate_mechanism_pilot.py'
    if (file_sha256(numerical_source)!=source['supplemental_source_sha256']
            or file_sha256(source['supplemental_source_copy'])!=source['supplemental_source_sha256']):
        raise ValueError('numerical evaluator differs from its executed source snapshot')
    source_episode=Path(source['episode_root'])
    sp,ss,sstart,sresult,sm=inspect_pilot(source_episode,source['episode_manifest_sha256'])
    tp,ts,tstart,tresult,tm=inspect_pilot(episode,expected_manifest_sha256)
    original_source_review=replay_receipt(Path(source['original_review_path']),source['original_review_sha256'],
        source['episode_manifest_sha256'],sstart,sresult,sm)
    target_review=replay_receipt(prior_review,prior_review_sha256,expected_manifest_sha256,tstart,tresult,tm)
    if target_review.get('status')!='pilot_review_failed' or target_review.get('error')!={
            'type':'ValueError','message':'declared surface sample limit exceeded'}:
        raise ValueError('only the already diagnosed numerical capacity failure may use this supplement')
    if (ss['asset_id']!=ts['asset_id'] or sp!=tp
            or source['reference_manifest_sha256']!=tp['reference_manifest_sha256']):
        raise ValueError('scene, reference or frozen evaluation protocol differs')
    equal_files={}
    for name in ('prediction/mesh.npz','prediction/occupancy.npz','public_workspace.json','public_spec.json'):
        left,right=file_sha256(source_episode/name),file_sha256(episode/name)
        if left!=right:
            raise ValueError('numerical-input bytes differ: '+name)
        equal_files[name]=dict(source_sha256=left,target_sha256=right,byte_identical=True)
    source_mapper=read(source_episode/'prediction/mapper.json'); target_mapper=read(episode/'prediction/mapper.json')
    fields=('shape','resolution_m','origin_xy_m','grid_convention','frames','backend_poisoned','occupancy_sha256')
    if any(source_mapper[key]!=target_mapper[key] for key in fields):
        raise ValueError('mapper coordinates or completeness differs')
    if sm['source_sha256']!=tm['source_sha256']:
        raise ValueError('executed original source closure differs')
    cap=source['capacity_amendment']; metrics=source['metrics']
    if (cap['common_max_samples']!=1000000 or cap['sample_spacing_m']!=.3 or cap['seed']!=4002
            or cap['threshold_m']!=.05 or cap['subsampling'] is not False or cap['mesh_decimation'] is not False
            or metrics['prediction_sample_spacing_m']!=.3 or metrics['prediction_seed']!=4002
            or metrics['threshold_m']!=.05):
        raise ValueError('numerical settings are not the common fixed capacity amendment')
    load_reference_v44(ROOT/tp['reference_root'],manifest_sha256=tp['reference_manifest_sha256'],asset_id=ts['asset_id'])
    receipt={key:deepcopy(source[key]) for key in ('capacity_amendment','metrics','coverage',
        'geometry_capacity_diagnostic','reference_manifest_sha256','all_task_instances_in_macro_denominator',
        'prediction_roi_cropped','scope')}
    receipt.update(schema='mechanism.pilot.identical_input_evaluation_reuse.v1',status='pilot_endpoint_scored',
        run_id=tstart['run_id'],mode=ts['mode'],episode_root=str(episode),
        episode_manifest_sha256=expected_manifest_sha256,
        original_review_path=str(prior_review),original_review_sha256=prior_review_sha256,
        original_review_status=target_review['status'],original_failure=target_review['error'],
        original_replay=target_review['replay'],original_episode_status=tresult['status'],
        original_executed_paid_actions=tresult['executed_paid_actions'],
        task_success=tresult['status']=='controller_stop' and tresult['sensor_status'].get('returned_xy_and_yaw') is True,
        numerical_evaluation_performed=False,
        evaluation_execution='reused exact-identical inputs; NOT independently recomputed',
        reused_evaluation=dict(path=str(source_evaluation),sha256=source_evaluation_sha256,
            run_id=sstart['run_id'],episode_manifest_sha256=source['episode_manifest_sha256'],
            actual_numerical_evaluation_elapsed_s=source['elapsed_s']),
        equivalence_proof=dict(byte_identical_files=equal_files,
            identical_mapper_evaluation_fields={key:source_mapper[key] for key in fields},
            identical_original_source_closure=True,identical_frozen_protocol=True,
            same_scene_asset_id=ts['asset_id'],same_reference_manifest_sha256=tp['reference_manifest_sha256'],
            same_sample_spacing_m=.3,same_seed=4002,same_threshold_m=.05,same_max_samples=1000000,
            no_mode_or_class_input_to_evaluator=True,both_complete_saved_frame_replays_verified=True),
        new_worlds=0,new_sensor_packets=0,new_tsdf_integrations=0,prediction_reintegrated=False,
        primary_experiment_started=False,semantic_performance_claim=False,original_sources_or_outputs_modified=False)
    # Revalidate the immutable episodes after every source/reference check.
    inspect_pilot(source_episode,source['episode_manifest_sha256']);inspect_pilot(episode,expected_manifest_sha256)
    if file_sha256(source_evaluation)!=source_evaluation_sha256 or file_sha256(prior_review)!=prior_review_sha256:
        raise ValueError('pinned evidence changed during reuse proof')
    after=runtime_counts_v41()
    if before!=after:
        raise ValueError('unexpected live sensor activity during saved-input proof')
    receipt['runtime']=dict(before=before,after=after,no_new_world_or_sensor_action=True)
    receipt['elapsed_s']=time.monotonic()-began
    receipt['supplemental_source_sha256']=file_sha256(__file__)
    source_copy=output.with_suffix('.source.py')
    receipt['supplemental_source_copy']=str(source_copy)
    output.parent.mkdir(parents=True,exist_ok=True)
    with source_copy.open('xb') as stream:stream.write(Path(__file__).read_bytes())
    with output.open('xb') as stream:stream.write(canonical_bytes(receipt))
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-evaluation',required=True,type=Path)
    p.add_argument('--source-evaluation-sha256',required=True)
    p.add_argument('--episode',required=True,type=Path)
    p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--prior-review',required=True,type=Path)
    p.add_argument('--prior-review-sha256',required=True)
    p.add_argument('--output',required=True,type=Path)
    result=reuse(**vars(p.parse_args()))
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    sys.exit(main())
