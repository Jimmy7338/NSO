#!/usr/bin/env python3
"""Reuse an integration endpoint score only for exactly equal numerical inputs.

Both episodes must have their own completed policy/TSDF replay. The source
must additionally have an original complete numerical review or the explicitly
diagnosed positive-face supplemental evaluation. This script runs neither
replay nor numerical surface evaluation.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.development_sensor_v41 import runtime_counts_v41
from nso.episode_driver_v43 import canonical_bytes,file_sha256
from nso.offline_evaluation_v44 import _array_sha,load_reference_v44
from nso.semantic_experiment import EVALUATION,inspect_experiment,read_json,reference_spec


RUNS={'integration_A_G':'G','integration_A_B':'bayes_semantic','integration_A_S':'shared_semantic'}
PHASE='semantic_integration_20260923'
MAPPER_FIELDS=('shape','resolution_m','origin_xy_m','grid_convention','frames',
    'backend_poisoned','occupancy_sha256','tsdf_integration_count')
ARRAY_FILES={'prediction/mesh.npz':('vertices','triangles','vertex_colors'),
    'prediction/occupancy.npz':('belief','observed')}
EVALUATOR_SOURCES=('nso/semantic_experiment.py','nso/semantic_experiment_replay.py',
    'nso/offline_evaluation_v44.py','nso/surface_evaluation_v40.py','nso/saved_replay_v44.py')
SUPPLEMENTAL_SOURCES=('nso/complete_surface_evaluation.py','scripts/evaluate_semantic_integration_endpoint.py')
POSITIVE_FACE_FAILURE=dict(type='ValueError',message='nonfinite or degenerate triangle rejected')


def _checked_episode(root,pin):
    episode=inspect_experiment(root,pin)
    run=episode['started']['run_id'];protocol=episode['protocol'];slot=episode['slot']
    if (episode.get('current_sources_match') is not True or episode.get('current_source_differences')
            or run not in RUNS or protocol.get('phase_id')!=PHASE
            or set(protocol.get('slots',{}))!=set(RUNS)
            or slot['method']!=RUNS[run] or slot['asset_id']!='DEV_A_00'
            or protocol.get('primary_experiment_started') is not False
            or protocol.get('evaluation')!=EVALUATION):
        raise ValueError('matching executed sources and one of the three frozen integration slots required')
    return episode


def _checked_review(path,pin,episode,*,complete_numerical,diagnosed_positive_face_failure=False):
    path=Path(path)
    if path.is_symlink() or file_sha256(path)!=pin:
        raise ValueError('original review differs from external SHA256 pin')
    review=read_json(path);replay=review.get('replay',{});runtime=review.get('runtime',{})
    required_status=('experiment_reviewed',) if complete_numerical else (
        'experiment_reviewed','experiment_replay_verified')
    if diagnosed_positive_face_failure:
        required_status=('experiment_review_failed',)
    expected_error=POSITIVE_FACE_FAILURE if diagnosed_positive_face_failure else None
    if (review.get('schema')!='semantic.experiment.review.v1' or review.get('status') not in required_status
            or review.get('error')!=expected_error or review.get('run_id')!=episode['started']['run_id']
            or review.get('phase_id')!=PHASE or review.get('episode_manifest_sha256')!=episode['manifest_sha256']
            or review.get('source_sha256')!=episode['manifest']['source_sha256']
            or review.get('current_sources_match') is not True or review.get('current_source_differences')
            or runtime.get('no_new_world_or_sensor_action') is not True
            or 'before' not in runtime or runtime['before']!=runtime.get('after')):
        raise ValueError('original successful review must bind each inspected episode and unchanged source')
    mapper=read_json(episode['root']/'prediction/mapper.json')
    rr=replay.get('runtime',{})
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
        raise ValueError('source must contain original complete numerical evaluation, not a reuse chain')
    return review


def _checked_numeric_supplement(path,pin,episode,original_review,original_review_sha256):
    path=Path(path)
    if path.is_symlink() or file_sha256(path)!=pin:
        raise ValueError('numerical supplement differs from external pin')
    value=read_json(path);runtime=value.get('runtime',{})
    if (value.get('schema')!='semantic.endpoint.numeric_supplement.v1'
            or value.get('status')!='semantic_endpoint_evaluation_completed' or value.get('error') is not None
            or value.get('numerical_evaluation_recomputed') is not True
            or value.get('original_replay_rerun') is not False
            or value.get('episode_manifest_sha256')!=episode['manifest_sha256']
            or value.get('run_id')!=episode['started']['run_id']
            or Path(value.get('episode_root','')).resolve()!=episode['root']
            or Path(value.get('original_review_path','')).resolve()!=original_review
            or value.get('original_review_sha256')!=original_review_sha256
            or runtime.get('no_new_world_or_sensor_action') is not True
            or 'before' not in runtime or runtime['before']!=runtime.get('after')):
        raise ValueError('completed original positive-face supplement must bind source episode and retained failed review')
    sources=value.get('numerical_source_sha256',{})
    expected=dict(episode['manifest']['source_sha256'])
    expected.update({name:file_sha256(ROOT/name) for name in SUPPLEMENTAL_SOURCES})
    archive=Path(value['numerical_source_root'])
    if sources!=expected or archive.is_symlink():
        raise ValueError('supplemental numerical source closure differs from executed original plus explicit repair')
    for name,digest in sources.items():
        relative=Path(name);archived=archive/relative
        if (relative.is_absolute() or '..' in relative.parts or archived.is_symlink()
                or not archived.resolve().is_relative_to(archive.resolve())
                or not archived.is_file() or file_sha256(archived)!=digest
                or file_sha256(ROOT/name)!=digest):
            raise ValueError('archived or current supplemental numerical source differs: '+name)
    mesh=episode['root']/'prediction/mesh.npz'
    with np.load(mesh,allow_pickle=False) as data:
        xyz=data['vertices'][data['triangles']]
        areas=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)/2
        finite=bool(np.isfinite(data['vertices']).all())
    diagnosis=dict(prediction_triangles=len(areas),finite_vertices=finite,
        zero_area_faces=int(np.count_nonzero(areas==0)),
        positive_faces_below_old_floor=int(np.count_nonzero((areas>0)&(areas<=5e-13))),
        minimum_area_m2=float(areas.min()),all_faces_preserved=True,
        mesh_sha256=file_sha256(mesh),old_floor_twice_area_m2=1e-12)
    metrics=value.get('evaluation',{}).get('metrics',{})
    if (value.get('diagnosis')!=diagnosis or not finite or not np.isfinite(areas).all()
            or np.any(areas<=0) or not diagnosis['positive_faces_below_old_floor']
            or metrics.get('prediction_mesh_validation')!='strict_positive_area_without_absolute_area_floor'
            or metrics.get('positive_subthreshold_faces_preserved')!=diagnosis['positive_faces_below_old_floor']
            or metrics.get('submitted_prediction_triangles')!=len(areas)):
        raise ValueError('actual complete mesh or metric does not match retained positive-face diagnosis')
    return value


def _equal_arrays(source,target):
    proof={}
    for name,keys in ARRAY_FILES.items():
        with np.load(source/name,allow_pickle=False) as left,np.load(target/name,allow_pickle=False) as right:
            if set(left.files)!=set(keys) or set(right.files)!=set(keys):
                raise ValueError('complete original array inventory required: '+name)
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


def reuse(*,source_episode,source_manifest_sha256,source_review,source_review_sha256,
        target_episode,target_manifest_sha256,target_review,target_review_sha256,output,
        source_numerical_evaluation=None,source_numerical_evaluation_sha256=None):
    began=time.monotonic();before=runtime_counts_v41()
    source_episode,target_episode,source_review,target_review,output=map(lambda x:Path(x).resolve(),
        (source_episode,target_episode,source_review,target_review,output))
    source_copy=output.with_suffix('.source.py')
    if (output.exists() or source_copy.exists() or source_episode==target_episode
            or any(output.is_relative_to(path) or source_copy.is_relative_to(path)
                   for path in (source_episode,target_episode))):
        raise ValueError('distinct episodes and new receipt outside their immutable trees required')
    script_sha=file_sha256(__file__)
    source=_checked_episode(source_episode,source_manifest_sha256)
    target=_checked_episode(target_episode,target_manifest_sha256)
    if source['started']['run_id']==target['started']['run_id']:
        raise ValueError('source and target must be distinct integration runs')
    if (source_numerical_evaluation is None)!=(source_numerical_evaluation_sha256 is None):
        raise ValueError('supplemental numerical evaluation path and external pin must be supplied together')
    supplemented=source_numerical_evaluation is not None
    sr=_checked_review(source_review,source_review_sha256,source,complete_numerical=not supplemented,
        diagnosed_positive_face_failure=supplemented)
    tr=_checked_review(target_review,target_review_sha256,target,complete_numerical=False)
    numerical=None
    if supplemented:
        source_numerical_evaluation=Path(source_numerical_evaluation).resolve()
        numerical=_checked_numeric_supplement(source_numerical_evaluation,source_numerical_evaluation_sha256,
            source,source_review,source_review_sha256)
    if (source['protocol']!=target['protocol']
            or source['manifest']['source_sha256']!=target['manifest']['source_sha256']
            or source['slot']['asset_id']!=target['slot']['asset_id']):
        raise ValueError('asset, complete frozen protocol or executed source closure differs')
    evaluator_sources={name:source['manifest']['source_sha256'][name] for name in EVALUATOR_SOURCES}
    if any(file_sha256(ROOT/name)!=digest for name,digest in evaluator_sources.items()):
        raise ValueError('numerical evaluator source differs from original completed review')
    reference=reference_spec(source['protocol'],source['slot']['asset_id'])
    if reference!=reference_spec(target['protocol'],target['slot']['asset_id']):
        raise ValueError('fixed reference pin or path differs')
    ref_path=Path(reference['root'])
    if ref_path.is_absolute() or '..' in ref_path.parts:
        raise ValueError('repository-relative pinned reference required')
    ref_root=ROOT/ref_path
    if output.is_relative_to(ref_root) or source_copy.is_relative_to(ref_root):
        raise ValueError('reuse receipt must be outside immutable reference')
    surface,domain,record=load_reference_v44(ref_root,manifest_sha256=reference['manifest_sha256'],
        asset_id=source['slot']['asset_id'])
    equal_files={}
    for name in ('public_workspace.json','public_spec.json'):
        left,right=file_sha256(source_episode/name),file_sha256(target_episode/name)
        if left!=right:
            raise ValueError('shared numerical public input differs: '+name)
        equal_files[name]=dict(source_sha256=left,target_sha256=right,byte_identical=True)
    arrays=_equal_arrays(source_episode,target_episode)
    sm,tm=(read_json(root/'prediction/mapper.json') for root in (source_episode,target_episode))
    if (any(sm.get(key)!=tm.get(key) for key in MAPPER_FIELDS)
            or sm.get('backend_poisoned') is not False
            or sm.get('frames')!=source['result']['acquired_and_saved_packets']
            or tm.get('frames')!=target['result']['acquired_and_saved_packets']
            or any(sm.get(key)!=record['coverage'][key] for key in
                   ('shape','resolution_m','origin_xy_m','grid_convention'))
            or arrays['prediction/occupancy.npz']['arrays']['belief']['array_sha256']!=sm['occupancy_sha256']):
        raise ValueError('mapper coordinate, history, poison or occupancy binding differs')
    evaluation=(numerical if supplemented else sr)['evaluation']
    metrics=evaluation.get('metrics',{});coverage=evaluation.get('coverage',{})
    source_success=source['result']['status']=='controller_stop' and source['result']['sensor_status'].get('returned_xy_and_yaw') is True
    if (evaluation.get('reference_manifest_sha256')!=reference['manifest_sha256']
            or evaluation.get('task_success')!=source_success
            or evaluation.get('original_episode_status')!=source['result']['status']
            or evaluation.get('all_task_instances_in_macro_denominator') is not True
            or evaluation.get('prediction_roi_cropped') is not False
            or any(evaluation.get(key)!=0 for key in ('new_worlds','new_sensor_packets','new_tsdf_integrations'))
            or metrics.get('reference_fingerprint')!=surface.fingerprint
            or coverage.get('denominator')!=record['coverage']
            or metrics.get('prediction_sample_spacing_m')!=EVALUATION['sample_spacing_m']
            or metrics.get('prediction_seed')!=EVALUATION['seed']
            or metrics.get('threshold_m')!=EVALUATION['threshold_m']
            or metrics.get('C_nav')!=coverage.get('C_nav')
            or metrics.get('J_nav')!=metrics.get('C_nav',0)*metrics.get('Q',0)):
        raise ValueError('original numerical evaluation is not bound to the shared reference and fixed configuration')
    copied=deepcopy(evaluation)
    copied.update(task_success=target['result']['status']=='controller_stop'
        and target['result']['sensor_status'].get('returned_xy_and_yaw') is True,
        original_episode_status=target['result']['status'],numerical_evaluation_recomputed=False,
        independent_replay_performed_here=False)
    result=dict(schema='semantic.experiment.identical_input_evaluation_reuse.v1',
        status='experiment_endpoint_evaluation_reused',phase_id=PHASE,run_id=target['started']['run_id'],
        method=target['slot']['method'],episode_root=str(target_episode),episode_manifest_sha256=target_manifest_sha256,
        source_episode_root=str(source_episode),source_episode_manifest_sha256=source_manifest_sha256,
        source_run_id=source['started']['run_id'],source_review_path=str(source_review),source_review_sha256=source_review_sha256,
        target_review_path=str(target_review),target_review_sha256=target_review_sha256,
        source_original_review_status=sr['status'],target_original_review_status=tr['status'],
        source_original_review_error=deepcopy(sr.get('error')),
        source_original_review_elapsed_s=sr['elapsed_s'],
        source_numeric_supplement=None if not supplemented else dict(
            path=str(source_numerical_evaluation),sha256=source_numerical_evaluation_sha256,
            status=numerical['status'],elapsed_s=numerical['elapsed_s'],
            numerical_source_root=numerical['numerical_source_root'],
            numerical_source_sha256=deepcopy(numerical['numerical_source_sha256']),
            diagnosis=deepcopy(numerical['diagnosis']),numerical_evaluation_recomputed=True),
        source_replay=deepcopy(sr['replay']),target_replay=deepcopy(tr['replay']),evaluation=copied,
        numerical_evaluation_recomputed=False,independent_replay_performed_here=False,
        evaluation_execution='Exact numerical-input reuse from one completed original numerical evaluation; target surface scoring was not rerun.',
        equivalence_proof=dict(exact_original_arrays=arrays,byte_identical_public_files=equal_files,
            identical_mapper_evaluation_fields={key:sm[key] for key in MAPPER_FIELDS},
            identical_complete_source_closure=True,evaluator_source_sha256=evaluator_sources,
            identical_frozen_protocol=True,evaluation_configuration=deepcopy(EVALUATION),
            asset_id=source['slot']['asset_id'],reference_spec=reference,
            complete_source_and_target_saved_policy_TSDF_replays_verified=True),
        source_sha256=deepcopy(target['manifest']['source_sha256']),
        new_worlds=0,new_sensor_packets=0,new_tsdf_integrations=0,primary_experiment_started=False,
        formal_performance_evidence=False,semantic_performance_claim=False,automatic_retry=False,
        original_sources_or_outputs_modified=False,supplemental_source_sha256=script_sha,
        supplemental_source_copy=str(source_copy))
    # Recheck both full episode inventories, reviews, reference and evaluator
    # sources after all comparisons; no result is emitted from changing inputs.
    _checked_episode(source_episode,source_manifest_sha256)
    _checked_episode(target_episode,target_manifest_sha256)
    load_reference_v44(ref_root,manifest_sha256=reference['manifest_sha256'],asset_id=source['slot']['asset_id'])
    if supplemented:
        _checked_numeric_supplement(source_numerical_evaluation,source_numerical_evaluation_sha256,
            source,source_review,source_review_sha256)
    if (file_sha256(source_review)!=source_review_sha256 or file_sha256(target_review)!=target_review_sha256
            or file_sha256(__file__)!=script_sha
            or any(file_sha256(ROOT/name)!=digest for name,digest in evaluator_sources.items())):
        raise ValueError('pinned review or numerical source changed during reuse proof')
    after=runtime_counts_v41()
    if before!=after:
        raise ValueError('unexpected live sensor activity during numerical reuse proof')
    result['runtime']=dict(before=before,after=after,no_new_world_or_sensor_action=True)
    result['elapsed_s']=time.monotonic()-began
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
    parser.add_argument('--source-numerical-evaluation',type=Path)
    parser.add_argument('--source-numerical-evaluation-sha256')
    parser.add_argument('--output',required=True,type=Path)
    result=reuse(**vars(parser.parse_args()))
    print(canonical_bytes(result).decode(),end='')


if __name__=='__main__':
    main()
