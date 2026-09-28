#!/usr/bin/env python3
"""Compare explicitly pinned saved trajectories; no World, replay or scoring."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.development_sensor_v41 import runtime_counts_v41
from nso.episode_driver_v43 import ACTION_TO_SENSOR_V43,canonical_bytes,file_sha256
from nso.offline_evaluation_v44 import _array_sha
from scripts.inspect_semantic_hierarchy_execution import array_digest,compact_choice
from scripts.reuse_semantic_scene_endpoint_evaluation import _checked_episode
from scripts.summarize_semantic_integration import _records,digest
from scripts.summarize_semantic_scene_results import parse_scene_steps


def inspect(root,pin):
    episode=_checked_episode(root,pin);history=[]
    def observed():
        for index,step,receipt in _records(root,episode['result'],episode['manifest'],episode['protocol']):
            evidence=step['controller_evidence'];decision=step['decision'];account=step['accounting']
            selection=decision.get('global_selection') or {};observation=receipt['observation_sha256']
            if (any(row['observation_sha256']!=observation for row in (account,evidence,step['mapper']))
                    or decision['source_observation_sha256']!=observation
                    or account['paid_step']!=index or account['action']!=receipt['execution']['action']):
                raise ValueError('saved action/observation/update binding differs')
            if index and ACTION_TO_SENSOR_V43[history[-1]['next_action']]!=account['action']:
                raise ValueError('selected action is not next saved paid action')
            direct=sorted(selection.get('direct_options',[]),key=lambda row:-row['score'])
            diagnostics=sorted(selection.get('diagnostic_options',[]),key=lambda row:-row['score'])
            history.append(dict(paid_step=index,actual_action=account['action'],next_action=decision['action'],
                pose=receipt['execution']['pose_xyyaw_rad'],global_replanned=decision['global_replanned'],
                macro_id=decision['macro_id'],macro_target=decision['macro_target'],
                selected=compact_choice(selection.get('selected')),best_direct_score=None if not direct else direct[0]['score'],
                best_diagnostic_score=None if not diagnostics else diagnostics[0]['score'],
                direct_ranked=[compact_choice(row) for row in direct],
                diagnostic_ranked=[compact_choice(row) for row in diagnostics],
                belief=deepcopy(evidence['structure_belief']['instances']),
                geometry_feedback=deepcopy(evidence.get('geometry_feedback',[])),
                first_actual_reliable_planes=evidence.get('first_actual_reliable_planes',[]),
                tsdf_integration_count=step['mapper']['tsdf_integration_count'],
                rgbd_excluding_frame_id_sha256=array_digest(root/f'packets/{index:03d}_rgbd.npz'),
                step_sha256=file_sha256(root/f'steps/{index:03d}.json.gz')))
            yield index,step,receipt
    activation=parse_scene_steps(observed());result=episode['result']
    arrays={}
    for name in ('mesh.npz','occupancy.npz'):
        with np.load(root/'prediction'/name,allow_pickle=False) as saved:
            arrays[name]={key:dict(shape=list(saved[key].shape),dtype=saved[key].dtype.str,
                array_sha256=_array_sha(saved[key])) for key in saved.files}
    compact=dict(root=str(root),manifest_sha256=pin,method=episode['slot']['method'],
        status=result['status'],paid_actions=result['executed_paid_actions'],saved_packets=result['acquired_and_saved_packets'],
        returned_xy_and_yaw=result['sensor_status'].get('returned_xy_and_yaw'),collisions=result['collisions'],
        action_sequence_sha256=digest([row['next_action'] for row in history]),
        pose_sequence_sha256=digest([row['pose'] for row in history]),prediction_arrays=arrays,
        mesh_container_sha256=file_sha256(root/'prediction/mesh.npz'),
        counts=activation['counts'],first_events=activation['first_events'],
        diagnostic_evi_max=activation['diagnostic_evi_max'],
        selected_diagnostic_evi_max=activation['selected_diagnostic_evi_max'],
        feedback_application=activation['feedback_application'],acquisition=activation['measurement_acquisition'],
        observed_instances=activation['observed_instances'])
    return episode,compact,history


def identity(row):
    return None if row is None else tuple(row.get(key) for key in ('kind','target','instance_id'))


def first_difference(left,right,key):
    return next((index for index,(a,b) in enumerate(zip(left,right)) if key(a)!=key(b)),None)


def compact_witness(row):
    if row is None:return None
    result=deepcopy(row)
    for key in ('direct_ranked','diagnostic_ranked'):
        result[key+'_count']=len(result[key]);result[key]=result[key][:3]
    return result


def compare(left,right,ls,rs):
    action_steps=[i for i,(a,b) in enumerate(zip(left,right)) if a['next_action']!=b['next_action']]
    first=action_steps[0] if action_steps else None
    prefix=min(len(left),len(right)) if first is None else first+1
    goal_steps=[i for i,(a,b) in enumerate(zip(left,right)) if identity(a['selected'])!=identity(b['selected'])]
    ranks=[i for i,(a,b) in enumerate(zip(left,right)) if
        [[identity(row) for row in a[key]] for key in ('direct_ranked','diagnostic_ranked')]
        !=[[identity(row) for row in b[key]] for key in ('direct_ranked','diagnostic_ranked')]]
    score_steps=[i for i,(a,b) in enumerate(zip(left,right)) if
        [a[key] for key in ('best_direct_score','best_diagnostic_score')]
        !=[b[key] for key in ('best_direct_score','best_diagnostic_score')]]
    posterior=first_difference(left,right,lambda row:[(v['instance_id'],v['structure_probabilities']) for v in row['belief']])
    rho=first_difference(left,right,lambda row:[(v['instance_id'],v['rho'],v['peer_instance_ids']) for v in row['belief']])
    focus=first if first is not None else (ranks[0] if ranks else (score_steps[0] if score_steps else posterior))
    return dict(left=ls['method'],right=rs['method'],
        action_sequences_identical=not action_steps and len(left)==len(right),
        first_different_decision_paid_step=first,action_difference_paid_steps=action_steps,
        first_selected_goal_difference_paid_step=goal_steps[0] if goal_steps else None,
        first_candidate_rank_difference_paid_step=ranks[0] if ranks else None,
        candidate_rank_difference_paid_steps=ranks,first_best_score_difference_paid_step=score_steps[0] if score_steps else None,
        best_score_difference_paid_steps=score_steps,first_posterior_difference_paid_step=posterior,
        first_rho_or_peer_difference_paid_step=rho,
        common_rgbd_prefix_through_first_different_decision=all(a['rgbd_excluding_frame_id_sha256']==b['rgbd_excluding_frame_id_sha256']
            for a,b in zip(left[:prefix],right[:prefix])),
        prediction_array_bits_dtype_and_shape_identical=ls['prediction_arrays']==rs['prediction_arrays'],
        mesh_array_bits_dtype_and_shape_identical=ls['prediction_arrays']['mesh.npz']==rs['prediction_arrays']['mesh.npz'],
        mesh_container_bytes_identical=ls['mesh_container_sha256']==rs['mesh_container_sha256'],
        diagnostic_focus_paid_step=focus,
        focus_reason='first action divergence, else first ranking, best-score or posterior difference',
        focus_window=[] if focus is None else [dict(left=compact_witness(a),right=compact_witness(b))
            for a,b in zip(left[max(0,focus-1):focus+3],right[max(0,focus-1):focus+3])])


def analyze(entries,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('new diagnostic output required')
    before=runtime_counts_v41();episodes=[];summaries=[];histories=[]
    for path,pin in entries:
        episode,compact,history=inspect(Path(path).resolve(),pin)
        episodes.append(episode);summaries.append(compact);histories.append(history)
    base=episodes[0]
    if any(e['protocol']!=base['protocol'] or {k:v for k,v in e['slot'].items() if k!='method'}
            !={k:v for k,v in base['slot'].items() if k!='method'} for e in episodes):
        raise ValueError('only identical declared nonmethod experimental conditions may be paired')
    pairs=[compare(histories[i],histories[j],summaries[i],summaries[j])
        for i in range(len(entries)) for j in range(i+1,len(entries))]
    for episode in episodes:_checked_episode(episode['root'],episode['manifest_sha256'])
    after=runtime_counts_v41()
    if before!=after:raise ValueError('saved-log diagnostic unexpectedly executed sensing')
    result=dict(schema='semantic.paired_saved_mechanism_diagnostic.v1',asset_id=base['slot']['asset_id'],
        protocol_sha256=base['manifest']['protocol_sha256'],episodes=summaries,pairs=pairs,
        source_sha256={name:file_sha256(ROOT/name) for name in
            ('scripts/inspect_semantic_paired_mechanism.py','scripts/inspect_semantic_hierarchy_execution.py',
             'scripts/summarize_semantic_scene_results.py','scripts/summarize_semantic_integration.py',
             'scripts/reuse_semantic_scene_endpoint_evaluation.py')},
        runtime=dict(before=before,after=after),new_worlds=0,new_policy_runs=0,
        independent_replay_performed=False,numerical_endpoint_evaluation_performed=False,
        semantic_advantage_claim=False,
        limitations=['A first divergence on identical measured inputs can locate a mechanism affecting a choice; it cannot establish terminal advantage.',
            'Different scores or beliefs with identical executed actions are not behavioral gains.',
            'A B/S comparison isolates shared-reliability behavior only under the same saved observation history before divergence.',
            'No ground-truth structure or future sensor information is read for this diagnosis.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(canonical_bytes(result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',nargs=2,action='append',required=True,metavar=('PATH','MANIFEST_SHA256'))
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=analyze(args.episode,args.output)
    print(canonical_bytes(dict(asset_id=result['asset_id'],pairs=[{k:v for k,v in row.items() if k!='focus_window'}
        for row in result['pairs']],episodes=[dict(method=row['method'],counts=row['counts']) for row in result['episodes']])).decode())
