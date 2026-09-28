#!/usr/bin/env python3
"""Small saved-log P05 signal-path supplement; no sensing/replay/evaluation."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nso.episode_driver_v43 import canonical_bytes,file_sha256
from scripts import review_semantic_finite_queue as queue_proof
from scripts.reuse_semantic_scene_endpoint_evaluation import _checked_episode


def run(mechanism_diagnostic,sha256,output):
    mechanism_diagnostic=Path(mechanism_diagnostic).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('new saved-log supplemental output required')
    if file_sha256(mechanism_diagnostic)!=sha256:raise ValueError('paired diagnostic pin differs')
    paired=json.loads(mechanism_diagnostic.read_text());histories={};bindings={}
    for episode in paired['episodes']:
        root=Path(episode['root']);pin=episode['manifest_sha256'];_checked_episode(root,pin)
        history=[]
        for step in range(episode['saved_packets']):
            value=json.loads(gzip.decompress((root/'steps'/f'{step:03d}.json.gz').read_bytes()))
            evidence=value['controller_evidence'];selection=value['decision'].get('global_selection') or {}
            options=[]
            for row in selection.get('direct_options',[])+selection.get('diagnostic_options',[]):
                options.append({k:row[k] for k in ('kind','target','score','expected_gain','evi','instance_id',
                    'total_cost','branches','followups','before_observation_score','after_observation_score') if k in row})
            forecasts=[]
            for forecast in selection.get('forecasts',[]):
                views=forecast['candidates']
                forecasts.append(dict(instance_id=forecast['instance_id'],views=len(views),
                    fallback_views=sum(v['fallback'] for v in views),
                    fallback_reasons=sorted({v['fallback_reason'] for v in views if v['fallback']}),
                    maximum_expected_area_m2=max((v['expected_new_surface_area_m2'] for v in views),default=0)))
            history.append(dict(paid_step=step,remaining=selection.get('remaining'),
                belief=evidence['structure_belief']['instances'],geometry_feedback=evidence.get('geometry_feedback',[]),
                options=options,forecasts=forecasts,selected=selection.get('selected'),
                step_sha256=file_sha256(root/'steps'/f'{step:03d}.json.gz')))
        histories[episode['method']]=history;bindings[episode['method']]=dict(root=str(root),manifest_sha256=pin)
    b,s=histories['bayes_semantic'],histories['shared_semantic']
    def key(option):return json.dumps([option['kind'],option['target'],option.get('instance_id')],sort_keys=True)
    score_deltas=[];posterior_deltas=[];peer_frames=[]
    for left,right in zip(b,s):
        lm,rm=({key(row):row for row in value['options']} for value in (left,right))
        if set(lm)!=set(rm):raise ValueError('this supplementary zero-action case has differing candidate sets')
        maximum=max((abs(lm[k]['score']-rm[k]['score']) for k in lm),default=0.)
        if maximum:score_deltas.append(dict(paid_step=left['paid_step'],maximum_absolute_score_delta=maximum))
        lb,rb=({row['instance_id']:row for row in value['belief']} for value in (left,right))
        delta=max((float(np.max(np.abs(np.asarray(lb[k]['structure_probabilities'])-
            np.asarray(rb[k]['structure_probabilities'])))) for k in lb),default=0.)
        if delta:posterior_deltas.append(dict(paid_step=left['paid_step'],maximum_absolute_posterior_delta=delta))
        peer_frames.extend(dict(paid_step=right['paid_step'],**row) for row in right['belief']
            if row['rho'] is not None and abs(row['rho']-.5)>1e-12)
    selected_steps=(72,75,76,82,98)
    # Bind S's publication boundary, rather than inferring completion from a
    # file existing. This is the same record checked before adding S to the pair.
    qpath=ROOT/'audit_results/semantic_finite_queues_20260923/continuation_A_declaration.json'
    declaration=queue_proof.proof.read_json(qpath)
    run_id='core_P05_nom_S_b120_lexicographic'
    batch=next(row for row in declaration['batches'] if run_id in row['run_ids'])
    completion=queue_proof.completed_slot_event(batch,run_id,declaration['protocol_sha256'])
    if completion is None or completion['manifest_sha256']!=bindings['shared_semantic']['manifest_sha256']:
        raise ValueError('S lacks matching explicit completion receipt')
    recipient=next(row for row in next(e for e in paired['episodes'] if e['method']=='shared_semantic')['observed_instances']
        if row['observed_instance_id']=='instance_0001')
    diagnostic=[row for row in s[75]['options'] if row['kind']=='diagnose_then_observe']
    if (recipient['counts'].get('accepted_plane_fits',0)!=0
            or recipient['counts']['forecast_fallbacks']!=240 or recipient['counts']['view_forecasts']!=240
            or len(diagnostic)!=1 or diagnostic[0]['evi']!=0.
            or len(diagnostic[0]['branches'])!=4 or len({r['best_action'] for r in diagnostic[0]['branches']})!=1
            or any(history[i]['options'] for history in histories.values() for i in range(98,len(history)))
            or any(row['maximum_absolute_score_delta']>1e-12 for row in score_deltas)
            or any(not p['action_sequences_identical'] or not p['prediction_array_bits_dtype_and_shape_identical']
                or p['first_candidate_rank_difference_paid_step'] is not None for p in paired['pairs'])):
        raise ValueError('saved receipts do not support this specific P05 zero-behavior signal-path interpretation')
    result=dict(schema='semantic.P05_signal_path_diagnostic.v1',paired_diagnostic_path=str(mechanism_diagnostic),
        paired_diagnostic_sha256=sha256,episodes=bindings,S_completion_receipt=completion,
        source_sha256=file_sha256(__file__),
        exact_comparison_caveat='Paired diagnostic reports literal floating-point differences; these are not all substantive signal changes.',
        rounding_audit_absolute_tolerance=1e-12,tolerance_changes_no_planner_or_evaluation_rule=True,
        B_S_candidate_score_differences=score_deltas,
        B_S_maximum_candidate_score_absolute_difference=max((row['maximum_absolute_score_delta'] for row in score_deltas),default=0.),
        B_S_first_posterior_difference_above_rounding_tolerance=next((r for r in posterior_deltas
            if r['maximum_absolute_posterior_delta']>1e-12),None),
        B_S_first_exact_posterior_difference=posterior_deltas[0] if posterior_deltas else None,
        shared_peer_adjusted_instance_frames=len(peer_frames),
        shared_peer_adjusted_instance_ids=sorted({row['instance_id'] for row in peer_frames}),
        shared_receiver_rho_range=[min(row['rho'] for row in peer_frames),max(row['rho'] for row in peer_frames)],
        selected_paid_step_witnesses={method:[history[i] for i in selected_steps] for method,history in histories.items()},
        interpretation=dict(actual_behavior_difference=False,
            first_material_shared_transfer_paid_step=75,recipient_instance='instance_0001',source_instance='instance_0002',
            receiver_had_any_accepted_plane=False,receiver_all_240_view_forecasts_fell_back=True,
            sole_diagnostic_all_four_branches_same_best_action=True,sole_diagnostic_evi=0.,
            all_B_S_candidate_score_differences_are_roundoff=True,
            no_eligible_direct_or_diagnostic_options_from_paid_step_98=True,
            statement='Shared belief transfer occurred, but its recipient had no reliable plane and hence zero predicted surface utility; '
                'no scored ranking, selected target, paid action or prediction array changed.'),
        new_worlds=0,new_replays=0,new_numeric_evaluations=0,source_or_threshold_changes=False)
    for row in bindings.values():_checked_episode(Path(row['root']),row['manifest_sha256'])
    if file_sha256(mechanism_diagnostic)!=sha256:raise ValueError('paired diagnostic changed')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(canonical_bytes(result))
    with output.with_suffix('.source.py').open('xb') as stream:stream.write(Path(__file__).read_bytes())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism-diagnostic',required=True,type=Path)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    result=run(**vars(parser.parse_args()))
    print(canonical_bytes({key:value for key,value in result.items() if key!='selected_paid_step_witnesses'}).decode())
