#!/usr/bin/env python3
"""Read the first eleven sealed acquisition episodes; no replay or scoring."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.development_sensor_v41 import runtime_counts_v41
from nso.episode_driver_v43 import ACTION_TO_SENSOR_V43,canonical_bytes,file_sha256
from nso.semantic_scene_experiment import inspect_experiment
from scripts.summarize_semantic_integration import _records,digest
from scripts.summarize_semantic_scene_results import parse_scene_steps


PHASE=ROOT/'audit_results/semantic_development_acquisition_20260923'
RUNS=[f'core_{parent}_nom_{method}_b120_lexicographic'
    for parent,methods in (('P00','GFBSK'),('P01','GBS'),('P02','GBS')) for method in methods]


def array_digest(path):
    with np.load(path,allow_pickle=False) as arrays:
        return digest({key:dict(dtype=arrays[key].dtype.str,shape=list(arrays[key].shape),
            bytes_sha256=hashlib.sha256(np.ascontiguousarray(arrays[key]).tobytes()).hexdigest())
            for key in sorted(arrays.files) if key!='frame_id'})


def compact_choice(selection):
    if not selection:return None
    names=('kind','target','instance_id','score','evi','total_cost','before_observation_score',
        'after_observation_score','future_information_used','cross_instance_future_used','branches')
    return {key:deepcopy(selection[key]) for key in names if key in selection}


def analyze(output):
    before=runtime_counts_v41();summaries={};histories={}
    for run_id in RUNS:
        root=PHASE/'episodes'/run_id;pin=file_sha256(root/'artifact_manifest.json')
        episode=inspect_experiment(root,pin)
        if not episode['current_sources_match']:raise ValueError('sealed execution source drift')
        result=episode['result'];history=[];refusal=Counter();prior_support={}
        def records():
            for index,step,receipt in _records(root,result,episode['manifest'],episode['protocol']):
                evidence=step['controller_evidence'];decision=step['decision'];account=step['accounting']
                observation=receipt['observation_sha256'];selection=decision.get('global_selection') or {}
                if (account['paid_step']!=index or receipt['execution']['paid_step']!=index
                        or any(owner['observation_sha256']!=observation for owner in (account,evidence,step['mapper']))
                        or decision['source_observation_sha256']!=observation
                        or account['action']!=receipt['execution']['action']
                        or account['action_cost']!=(0 if index==0 else 1)):
                    raise ValueError('saved action, acquisition, fusion and controller source binding differs')
                if index:
                    previous=history[-1];action=previous['next_action'];actual=result['actions'][index-1]
                    if (ACTION_TO_SENSOR_V43[action]!=account['action'] or actual['controller_action']!=action
                            or actual['paid_step']!=index or actual['observation_sha256']!=observation):
                        raise ValueError('selected primitive does not bind next actual paid packet')
                    if evidence['completed_macro_id'] is not None and (not previous['macro_observe_submitted']
                            or evidence['completed_macro_id']!=previous['macro_id'] or account['action']!='observe'):
                        raise ValueError('macro completion lacks preceding actual observe submission')
                routing=decision.get('routing') or {}
                if routing.get('reason')=='upstream_goal_with_paid_return_reserve':
                    if routing.get('target')!=decision['macro_target'] or len(routing['candidates'])!=1:
                        raise ValueError('local router did not receive exactly the committed global target')
                if routing.get('return_cost_after_action') is not None:
                    if 1+routing['return_cost_after_action']>episode['slot']['budget']-index:
                        raise ValueError('logged primitive violates full return allowance')
                instances={row['instance_id']:row for row in evidence['association']['instances']}
                associations={row['instance_id']:row for row in evidence['association']['accepted']}
                contexts=[]
                for row in evidence.get('geometry_feedback',[]):
                    key=row['instance_id'];instance=instances[key];association=associations[key]
                    support=len(instance['support_points_world_m'])
                    context=dict(instance_id=key,applied=row['applied'],saved_reason=row['reason'],
                        novel_support_voxels=association['novel_support_voxels'],no_new_support=association['no_new_support'],
                        geometry_feedback_eligible=association['geometry_feedback_eligible'],support_points=support,
                        support_points_before=prior_support.get(key,0),novel_support_frames=instance['novel_support_frames'])
                    if not row['applied']:
                        refusal['refused']+=1
                        refusal['no_new_support']+=int(context['no_new_support'])
                        refusal['support_already_full_before_current_frame']+=int(prior_support.get(key,0)==4096)
                    contexts.append(context)
                prior_support.update({key:len(row['support_points_world_m']) for key,row in instances.items()})
                best_direct=max((r['score'] for r in selection.get('direct_options',[])),default=None)
                best_diagnostic=max((r['score'] for r in selection.get('diagnostic_options',[])),default=None)
                history.append(dict(paid_step=index,pose=receipt['execution']['pose_xyyaw_rad'],
                    executed_sensor_action=account['action'],actual_action_cost=account['action_cost'],
                    next_action=decision['action'],macro_id=decision['macro_id'],macro_target=decision['macro_target'],
                    global_replanned=decision['global_replanned'],macro_observe_submitted=decision['macro_observe_submitted'],
                    completed_macro_id=evidence['completed_macro_id'],initialization_cancelled=decision.get('initialization_cancelled'),
                    first_actual_reliable_planes=evidence.get('first_actual_reliable_planes',[]),
                    selected=compact_choice(selection.get('selected')),routing=deepcopy(routing),
                    best_direct_score=best_direct,best_diagnostic_score=best_diagnostic,
                    diagnostic_minus_direct=None if best_direct is None or best_diagnostic is None else best_diagnostic-best_direct,
                    feedback=contexts,belief=deepcopy(evidence['structure_belief']['instances']),
                    tsdf_integrated=step['mapper']['tsdf_integrated'],tsdf_integration_count=step['mapper']['tsdf_integration_count'],
                    newly_known_cells=step['mapper']['newly_known_cells'],
                    observed_payload_excluding_frame_id_sha256=array_digest(root/f'packets/{index:03d}_rgbd.npz'),
                    step_sha256=file_sha256(root/f'steps/{index:03d}.json.gz'),
                    packet_receipt_sha256=file_sha256(root/f'packets/{index:03d}_receipt.json')))
                yield index,step,receipt
        parsed=parse_scene_steps(records());histories[run_id]=history
        summaries[run_id]=dict(episode_root=str(root),episode_manifest_sha256=pin,
            manifest_pin_origin='current sealed manifest snapshot; not a new independent review',
            protocol_sha256=episode['manifest']['protocol_sha256'],executed_source_sha256=episode['manifest']['source_sha256'],
            status=result['status'],paid_actions=result['executed_paid_actions'],saved_packets=result['acquired_and_saved_packets'],
            returned_xy_and_yaw=result['sensor_status'].get('returned_xy_and_yaw'),collisions=result['collisions'],
            counts=parsed['counts'],first_events=parsed['first_events'],
            acquisition=parsed['measurement_acquisition'],feedback_application=parsed['feedback_application'],
            feedback_refusal_context=dict(refusal),rho_ranges=[dict(instance_id=row['observed_instance_id'],
                minimum=row['rho_min'],maximum=row['rho_max']) for row in parsed['observed_instances']],
            selected_global_options=parsed['selected_global_options'],
            positive_diagnostic_margin_steps=[dict(paid_step=row['paid_step'],margin=row['diagnostic_minus_direct'])
                for row in history if row['diagnostic_minus_direct'] is not None and row['diagnostic_minus_direct']>1e-12],
            action_sequence_sha256=digest([row['controller_action'] for row in result['actions']]),
            pose_sequence_sha256=digest([row['pose'] for row in history]),
            mesh_container_sha256=file_sha256(root/'prediction/mesh.npz'),
            assertions=dict(all_primitive_to_paid_packet_links=True,all_observation_update_hashes_match=True,
                completed_macros_bind_actual_paid_observe=True,local_goal_is_single_committed_target=True,
                recorded_full_pose_return_reserve_preserved=True))
    pairs=[]
    for parent,arms in (('P00',('G','F','B','S','K')),('P01',('G','B','S')),('P02',('G','B','S'))):
        for left,right in ([('G',m) for m in arms if m!='G']+[('B','S')]):
            a,b=(f'core_{parent}_nom_{m}_b120_lexicographic' for m in (left,right));ha,hb=histories[a],histories[b]
            differences=[i for i,(ra,rb) in enumerate(zip(ha,hb)) if ra['next_action']!=rb['next_action']]
            first=differences[0] if differences else None
            prefix=len(ha) if first is None else first+1
            pairs.append(dict(left=a,right=b,identical_next_action_sequences=not differences and len(ha)==len(hb),
                first_different_decision_paid_step=first,next_action_different_steps=differences,
                identical_rgbd_payloads_through_first_different_decision=all(
                    ra['observed_payload_excluding_frame_id_sha256']==rb['observed_payload_excluding_frame_id_sha256']
                    for ra,rb in zip(ha[:prefix],hb[:prefix])),
                mesh_container_bytes_identical=summaries[a]['mesh_container_sha256']==summaries[b]['mesh_container_sha256'],
                first_different_decision=None if first is None else dict(left=ha[first],right=hb[first])))
    witness_run='core_P02_nom_S_b120_lexicographic';witness=histories[witness_run][69:75]
    after=runtime_counts_v41()
    if before!=after:raise ValueError('diagnostic unexpectedly performed sensor work')
    result=dict(schema='semantic.hierarchy_execution_evidence.v1',explicit_run_ids=RUNS,
        phase_id=PHASE.name,episodes=summaries,pairs=pairs,closed_loop_witness=dict(run_id=witness_run,paid_steps=witness),
        diagnostic_script_sha256=file_sha256(__file__),parser_source_sha256={name:file_sha256(ROOT/name) for name in
            ('scripts/summarize_semantic_integration.py','scripts/summarize_semantic_scene_results.py')},
        runtime=dict(before=before,after=after),new_worlds=0,new_policy_runs=0,policy_replay_performed=False,
        endpoint_scored=False,semantic_advantage_claim=False,
        limitations=['Saved-log consistency and causal attribution audit; not independent policy/TSDF replay.',
            'Shared reliability changing a posterior does not itself identify a contribution to actions or terminal quality.',
            'No endpoint quality or out-of-development generalization conclusion is drawn.'])
    output=Path(output)
    if output.exists():raise FileExistsError('retain previous diagnostic')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(canonical_bytes(result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    result=analyze(parser.parse_args().output)
    print(json.dumps(dict(episodes=len(result['episodes']),pairs=len(result['pairs']),new_worlds=0)))
