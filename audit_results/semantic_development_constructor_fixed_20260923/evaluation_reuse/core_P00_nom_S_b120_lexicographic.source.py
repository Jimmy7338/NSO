#!/usr/bin/env python3
"""Reuse a complete new-scene score after exact input and replay verification.

This standalone receipt does not replace either original review. It calls only
archive inspection and reference loading, never policy replay, TSDF integration,
World construction or the numerical surface evaluator.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.development_sensor_v41 import runtime_counts_v41
from nso.episode_driver_v43 import canonical_bytes,file_sha256
from nso.offline_evaluation_v44 import _array_sha
from nso.semantic_experiment import EVALUATION,_repo_path,read_json
from nso.semantic_scene_evaluation import PREDICTION_FILES,load_semantic_scene_reference
from nso.semantic_scene_experiment import ARTIFACT_SCHEMA,SCHEMA,inspect_experiment,source_names


EXPECTED_EXECUTED_SOURCE_COUNT=58
MAPPER_FIELDS=('shape','resolution_m','origin_xy_m','grid_convention','frames',
    'backend_poisoned','occupancy_sha256','tsdf_integration_count','voxel_m','sdf_trunc_m','near_m','far_m','backend')
ARRAY_FILES={'prediction/mesh.npz':('vertices','triangles','vertex_colors'),
    'prediction/occupancy.npz':('belief','observed')}
EVALUATOR_SOURCES=('nso/semantic_scene_experiment.py','nso/semantic_experiment_replay.py',
    'nso/semantic_scene_evaluation.py','nso/complete_surface_evaluation.py',
    'nso/offline_evaluation_v44.py','nso/surface_evaluation_v40.py','nso/saved_replay_v44.py')
INPUT_NAMES=('asset_manifest.json','navigation_manifest.json','reference_index.json','reference_manifest.json')


def _pin(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('explicit external SHA256 pin required')


def _checked_episode(root,pin):
    _pin(pin)
    episode=inspect_experiment(root,pin)
    manifest=episode['manifest'];protocol=episode['protocol'];sources=manifest['source_sha256']
    if (episode.get('current_sources_match') is not True or episode.get('current_source_differences')
            or manifest.get('schema')!=ARTIFACT_SCHEMA or protocol.get('schema')!=SCHEMA
            or protocol.get('status')!='frozen' or protocol.get('evaluation')!=EVALUATION
            or len(sources)!=EXPECTED_EXECUTED_SOURCE_COUNT or set(sources)!=set(source_names())
            or any(file_sha256(ROOT/name)!=digest for name,digest in sources.items())):
        raise ValueError('frozen new-scene episode and exactly matching executed 58-source closure required')
    expected=dict(zip(INPUT_NAMES,(protocol['asset_manifest_sha256'],protocol['navigation_manifest_sha256'],
        protocol['reference_index_sha256'],episode['reference']['manifest_sha256'])))
    if manifest.get('input_sha256')!=expected:
        raise ValueError('episode asset/navigation/reference input pins disagree')
    return episode


def _checked_review(path,pin,episode,*,complete_numerical):
    _pin(pin)
    if path.is_symlink() or file_sha256(path)!=pin:
        raise ValueError('original review differs from external pin')
    review=read_json(path);runtime=review.get('runtime',{});replay=review.get('replay',{})
    statuses=('experiment_reviewed',) if complete_numerical else ('experiment_reviewed','experiment_replay_verified')
    if (review.get('schema')!='semantic.scene_experiment.review.v1' or review.get('status') not in statuses
            or review.get('error') is not None or review.get('run_id')!=episode['started']['run_id']
            or review.get('phase_id')!=episode['protocol']['phase_id']
            or review.get('episode_manifest_sha256')!=episode['manifest_sha256']
            or review.get('source_sha256')!=episode['manifest']['source_sha256']
            or review.get('input_sha256')!=episode['manifest']['input_sha256']
            or review.get('current_sources_match') is not True or review.get('current_source_differences')
            or runtime.get('no_new_world_or_sensor_action') is not True
            or 'before' not in runtime or runtime['before']!=runtime.get('after')):
        raise ValueError('original successful new-scene review must bind episode, executed sources and all input pins')
    mapper=read_json(episode['root']/'prediction/mapper.json');rr=replay.get('runtime',{})
    if (replay.get('status')!='verified'
            or replay.get('verification_kind')!='same_driver_saved_packet_policy_and_TSDF_replay'
            or replay.get('prediction_verified') is not True or replay.get('occupancy_exact') is not True
            or replay.get('mesh_order_invariant_tolerance_m')!=1e-9
            or replay.get('frames_verified')!=episode['result']['acquired_and_saved_packets']
            or replay.get('terminal_status_verified')!=episode['result']['status']
            or replay.get('replayed_tsdf_integrations')!=mapper.get('tsdf_integration_count')
            or replay.get('counterfactual_trajectory') is not False
            or any(replay.get(key)!=0 for key in ('new_worlds','new_sensor_packets','physical_actions'))
            or 'before' not in rr or rr['before']!=rr.get('after')):
        raise ValueError('independent complete saved-policy and TSDF replay required for each episode')
    if complete_numerical and (not isinstance(review.get('evaluation'),dict)
            or review.get('numerical_evaluation_recomputed') is False):
        raise ValueError('source must contain an original complete numerical review, not a reuse chain')
    return review


def _equal_arrays(source,target):
    proof={}
    for name,keys in ARRAY_FILES.items():
        with np.load(source/name,allow_pickle=False) as left,np.load(target/name,allow_pickle=False) as right:
            if set(left.files)!=set(keys) or set(right.files)!=set(keys):
                raise ValueError('complete original prediction array inventory required: '+name)
            arrays={}
            for key in keys:
                a,b=left[key],right[key]
                if (a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b)
                        or np.ascontiguousarray(a).tobytes()!=np.ascontiguousarray(b).tobytes()):
                    raise ValueError('numerical input array differs exactly: '+name+'/'+key)
                arrays[key]=dict(dtype=a.dtype.str,shape=list(a.shape),array_sha256=_array_sha(a),
                    values_dtype_shape_and_bits_identical=True)
            proof[name]=dict(source_file_sha256=file_sha256(source/name),target_file_sha256=file_sha256(target/name),
                zip_bytes_identical=file_sha256(source/name)==file_sha256(target/name),arrays=arrays)
    return proof


def _load_reference(episode):
    protocol=episode['protocol'];reference=episode['reference']
    return load_semantic_scene_reference(_repo_path(reference['root']),
        manifest_sha256=reference['manifest_sha256'],asset_id=episode['slot']['asset_id'],
        asset_root=_repo_path(protocol['asset_root']),asset_manifest_sha256=protocol['asset_manifest_sha256'])


def _prediction_pins(episode):
    return {name:episode['manifest']['files']['prediction/'+name]['sha256'] for name in PREDICTION_FILES}


def _task_success(episode):
    return episode['result']['status']=='controller_stop' and episode['result']['sensor_status'].get('returned_xy_and_yaw') is True


def reuse(*,source_episode,source_manifest_sha256,source_review,source_review_sha256,
        target_episode,target_manifest_sha256,target_review,target_review_sha256,output):
    began=time.monotonic();before=runtime_counts_v41()
    source_episode,target_episode,source_review,target_review,output=map(lambda value:Path(value).resolve(),
        (source_episode,target_episode,source_review,target_review,output))
    source_copy=output.with_suffix('.source.py');script_sha=file_sha256(__file__)
    if (output.exists() or source_copy.exists() or source_episode==target_episode
            or any(output.is_relative_to(path) or source_copy.is_relative_to(path)
                   for path in (source_episode,target_episode))):
        raise ValueError('distinct episodes and new supplemental output outside their immutable trees required')
    source=_checked_episode(source_episode,source_manifest_sha256)
    target=_checked_episode(target_episode,target_manifest_sha256)
    if source['started']['run_id']==target['started']['run_id']:
        raise ValueError('distinct source and target runs required')
    sr=_checked_review(source_review,source_review_sha256,source,complete_numerical=True)
    tr=_checked_review(target_review,target_review_sha256,target,complete_numerical=False)
    source_slot={key:value for key,value in source['slot'].items() if key!='method'}
    target_slot={key:value for key,value in target['slot'].items() if key!='method'}
    if (source['protocol']!=target['protocol'] or source_slot!=target_slot
            or source['reference']!=target['reference']
            or source['manifest']['source_sha256']!=target['manifest']['source_sha256']
            or source['manifest']['input_sha256']!=target['manifest']['input_sha256']):
        raise ValueError('frozen protocol, source closure, paired slot or asset/navigation/reference pins differ')
    reference=source['reference'];ref_root=_repo_path(reference['root'])
    if output.is_relative_to(ref_root) or source_copy.is_relative_to(ref_root):
        raise ValueError('reuse output must be outside immutable reference')
    surface,domain,record=_load_reference(source)
    public={}
    for name in ('public_workspace.json','public_spec.json','public_graph.json'):
        left,right=file_sha256(source_episode/name),file_sha256(target_episode/name)
        if left!=right:raise ValueError('shared pinned public input differs: '+name)
        public[name]=dict(source_sha256=left,target_sha256=right,byte_identical=True)
    arrays=_equal_arrays(source_episode,target_episode)
    sm,tm=(read_json(root/'prediction/mapper.json') for root in (source_episode,target_episode))
    if (any(key not in sm or key not in tm or sm[key]!=tm[key] for key in MAPPER_FIELDS)
            or sm['backend_poisoned'] is not False
            or sm['frames']!=source['result']['acquired_and_saved_packets']
            or tm['frames']!=target['result']['acquired_and_saved_packets']
            or any(sm[key]!=record['coverage'][key] for key in ('shape','resolution_m','origin_xy_m','grid_convention'))
            or arrays['prediction/occupancy.npz']['arrays']['belief']['array_sha256']!=sm['occupancy_sha256']):
        raise ValueError('mapper coordinate, history, settings or occupancy binding differs')
    evaluation=sr['evaluation'];metrics=evaluation.get('metrics',{});coverage=evaluation.get('coverage',{})
    evaluator_sources={name:source['manifest']['source_sha256'][name] for name in EVALUATOR_SOURCES}
    if (evaluation.get('schema')!='semantic_scene.complete_saved_prediction_evaluation.v1'
            or evaluation.get('episode_manifest_sha256')!=source_manifest_sha256
            or evaluation.get('asset_id')!=source['slot']['asset_id']
            or evaluation.get('asset_manifest_sha256')!=source['protocol']['asset_manifest_sha256']
            or evaluation.get('reference_manifest_sha256')!=reference['manifest_sha256']
            or evaluation.get('input_prediction_sha256')!=_prediction_pins(source)
            or evaluation.get('evaluator_source_sha256')!=evaluator_sources['nso/complete_surface_evaluation.py']
            or evaluation.get('task_success')!=_task_success(source)
            or evaluation.get('original_episode_status')!=source['result']['status']
            or evaluation.get('all_task_instances_in_macro_denominator') is not True
            or evaluation.get('prediction_roi_cropped') is not False
            or evaluation.get('prediction_reintegrated') is not False
            or evaluation.get('independent_replay_performed_here') is not False
            or any(evaluation.get(key)!=0 for key in ('new_worlds','new_sensor_packets','new_tsdf_integrations'))
            or metrics.get('reference_fingerprint')!=surface.fingerprint
            or metrics.get('prediction_mesh_validation')!='strict_positive_area_without_absolute_area_floor'
            or coverage.get('denominator')!=record['coverage']
            or metrics.get('prediction_sample_spacing_m')!=EVALUATION['sample_spacing_m']
            or metrics.get('prediction_seed')!=EVALUATION['seed'] or metrics.get('threshold_m')!=EVALUATION['threshold_m']
            or metrics.get('C_nav')!=coverage.get('C_nav')
            or metrics.get('J_nav')!=metrics.get('C_nav',0)*metrics.get('Q',0)):
        raise ValueError('original complete numerical evaluation does not bind shared inputs and positive-face implementation')
    copied=deepcopy(evaluation)
    copied.update(episode_manifest_sha256=target_manifest_sha256,input_prediction_sha256=_prediction_pins(target),
        task_success=_task_success(target),original_episode_status=target['result']['status'],
        numerical_evaluation_recomputed=False,independent_replay_performed_here=False)
    result=dict(schema='semantic.scene_endpoint_evaluation_reused.v1',status='experiment_endpoint_evaluation_reused',
        phase_id=target['protocol']['phase_id'],run_id=target['started']['run_id'],method=target['slot']['method'],
        asset_id=target['slot']['asset_id'],episode_root=str(target_episode),episode_manifest_sha256=target_manifest_sha256,
        source_episode_root=str(source_episode),source_episode_manifest_sha256=source_manifest_sha256,
        source_run_id=source['started']['run_id'],source_review_path=str(source_review),source_review_sha256=source_review_sha256,
        target_review_path=str(target_review),target_review_sha256=target_review_sha256,
        source_original_review_status=sr['status'],target_original_review_status=tr['status'],
        source_original_review_elapsed_s=sr['elapsed_s'],source_replay=deepcopy(sr['replay']),target_replay=deepcopy(tr['replay']),
        source_sha256=deepcopy(target['manifest']['source_sha256']),input_sha256=deepcopy(target['manifest']['input_sha256']),
        evaluation=copied,numerical_evaluation_recomputed=False,independent_replay_performed_here=False,
        evaluation_execution='Exact numerical-input reuse from one original complete review; no target numerical reevaluation.',
        rebound_target_provenance_fields=['episode_manifest_sha256','input_prediction_sha256','task_success','original_episode_status'],
        equivalence_proof=dict(exact_original_arrays=arrays,byte_identical_public_files=public,
            identical_mapper_evaluation_fields={key:sm[key] for key in MAPPER_FIELDS},
            identical_complete_source_closure=True,executed_source_count=len(source['manifest']['source_sha256']),
            evaluator_source_sha256=evaluator_sources,identical_frozen_protocol=True,
            identical_nonmethod_slot=source_slot,evaluation_configuration=deepcopy(EVALUATION),
            asset_id=source['slot']['asset_id'],reference_spec=reference,input_sha256=deepcopy(source['manifest']['input_sha256']),
            complete_source_and_target_saved_policy_TSDF_replays_verified=True),
        new_worlds=0,new_sensor_packets=0,new_tsdf_integrations=0,physical_actions=0,
        primary_experiment_started=target['protocol'].get('primary_experiment_started',False),
        formal_performance_evidence=False,semantic_performance_claim=False,automatic_retry=False,
        original_sources_or_outputs_modified=False,supplemental_source_sha256=script_sha,supplemental_source_copy=str(source_copy))
    # The inspected original sources include the positive-area evaluator. Both
    # full inventories and current pinned inputs are checked again before save.
    final_source=_checked_episode(source_episode,source_manifest_sha256)
    final_target=_checked_episode(target_episode,target_manifest_sha256)
    _checked_review(source_review,source_review_sha256,final_source,complete_numerical=True)
    _checked_review(target_review,target_review_sha256,final_target,complete_numerical=False)
    _load_reference(final_source)
    if (file_sha256(__file__)!=script_sha
            or any(file_sha256(ROOT/name)!=value for name,value in evaluator_sources.items())):
        raise ValueError('reuse proof or numerical source changed during verification')
    after=runtime_counts_v41()
    if before!=after:raise ValueError('unexpected live sensor activity during saved-input proof')
    result.update(runtime=dict(before=before,after=after,no_new_world_or_sensor_action=True),elapsed_s=time.monotonic()-began)
    output.parent.mkdir(parents=True,exist_ok=True)
    with source_copy.open('xb') as stream:stream.write(Path(__file__).read_bytes())
    with output.open('xb') as stream:stream.write(canonical_bytes(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for role in ('source','target'):
        parser.add_argument('--'+role+'-episode',required=True,type=Path)
        parser.add_argument('--'+role+'-manifest-sha256',required=True)
        parser.add_argument('--'+role+'-review',required=True,type=Path)
        parser.add_argument('--'+role+'-review-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    print(canonical_bytes(reuse(**vars(parser.parse_args()))).decode(),end='')


if __name__=='__main__':
    main()
