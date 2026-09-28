#!/usr/bin/env python3
"""Read only the first completed AISLE B/G ground arms and their paired Off arms.

Describes actual targets, paid initialization/observations and recorded utility
components. No World, policy replay, new scoring candidate or quality measure.
"""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import analyze_article_ground_comparisons_20260928 as analysis


def explain(output):
    require=analysis.require;inputs=analysis.old.Inputs();stage=analysis.STAGE
    protocol_path=analysis.ROOT/'configs/virtual3d/article_ground_ablation_v2_20260928.json'
    protocol=inputs.json(protocol_path);protocol_sha=analysis.old.digest(inputs.read(protocol_path))
    pin=protocol['paired_baseline_protocol'];baseline=inputs.json(analysis.ROOT/pin['path'],pin=pin['sha256'])
    plan_path=stage/'common_evaluation_plan_v1.json';plan=inputs.json(plan_path);plan_sha=analysis.old.digest(inputs.read(plan_path))
    on_entries,on_phase=analysis.ledger_entries(inputs,protocol,protocol_sha)
    off_entries,off_phase=analysis.ledger_entries(inputs,baseline,pin['sha256'])
    snapshot=stage/'analysis_v1/ground_first2'
    require(analysis.sealed_folder(inputs,snapshot,{'article.ground_comparison_manifest.v1'}) is not None,'root released first-two analysis snapshot')
    witnesses=inputs.json(snapshot/'first_mechanism_witnesses.json')
    summaries={};goals=[];observes=[];early=[]
    for method in ('B','G'):
        on_id=f'ground_AISLE_{method}_b160_n92801';slot=protocol['slots'][on_id];off_id=slot['paired_baseline_run_id']
        for arm,run,entry in [('GroundOff',off_id,off_entries[off_id]),('GroundOn',on_id,on_entries[on_id])]:
            require(entry['status']!='reserved','completed inputs only')
            row=analysis.blank_row(arm,run,slot,entry)
            ready=(analysis.load_on(inputs,on_phase,stage/'episode_reviews_ground_v2',row,entry,protocol_sha) if arm=='GroundOn' else
                analysis.load_off(inputs,off_phase,stage/'episode_reviews_v1',analysis.ROOT/plan['output_root'],plan,plan_sha,row,entry,pin['sha256']))
            require(ready is not None and row['review_passed'] and row['quality_measurement_available'],'sealed complete and independently reviewed input')
            episode,manifest=ready
            def record(name):return json.loads(analysis.old.sealed_read(inputs,episode,manifest,name))
            result=record('result.json');graph=record('public_graph.json');final=record('controller_final.json')
            action_counts=Counter(a['sensor_action'] for a in result['actions']);positions=[];first={};plane_first={};feedback_first={}
            observed_first={};direct_pure_inspection=[];return_step=None
            for step in range(result['acquired_and_saved_packets']):
                log=json.loads(gzip.decompress(analysis.old.sealed_read(inputs,episode,manifest,f'steps/{step:03d}.json.gz')))
                d=log['decision'];e=log['controller_evidence'];packet=record(f'packets/{step:03d}_receipt.json')['execution']
                positions.append(packet['pose_xyyaw_rad'])
                for instance in e['association']['instances']:
                    key=instance['instance_id'];observed_first.setdefault(key,dict(first_observed_step=step))
                    if instance['semantic_conditioning_used']:observed_first[key].setdefault('first_qualified_class_step',step)
                for key in e['first_actual_reliable_planes']:plane_first.setdefault(key,step)
                for feedback in e['geometry_feedback']:
                    if feedback['applied']:feedback_first.setdefault(feedback['instance_id'],step)
                if d.get('reason')=='return' and return_step is None:return_step=step
                if packet['action']=='observe':
                    observes.append(dict(run_id=run,arm=arm,method=method,paid_step=step,
                        x_m=positions[-1][0],y_m=positions[-1][1],yaw_rad=positions[-1][2],
                        initialization_actions_spent=d['initialization_actions_spent']))
                if step<=13:
                    early.append(dict(run_id=run,arm=arm,method=method,paid_step=step,action=d['action'],
                        initialization_cancelled=d.get('initialization_cancelled'),initialization_actions_spent=d['initialization_actions_spent'],
                        instance_states=[{k:r[k] for k in ('instance_id','observed_class','association_uncertain','semantic_conditioning_used')}
                            for r in e['association']['instances']],first_reliable_planes=e['first_actual_reliable_planes']))
                selection=d.get('global_selection')
                if not d['global_replanned'] or not selection or not selection.get('selected'):continue
                selected=selection['selected'];target=selected['target'];view_id=target['node']+':'+str(target['heading'])
                beliefs={r['instance_id']:r for r in e['structure_belief']['instances']};contributors=[]
                for forecast in selection.get('forecasts',[]):
                    for candidate in forecast['candidates']:
                        if candidate['view_id']==view_id:
                            posterior=beliefs[forecast['instance_id']]['structure_probabilities']
                            gain=protocol['controller']['inspection_weight']*sum(p*g for p,g in zip(posterior,candidate['structure_new_surface_area_m2']))
                            contributors.append(dict(instance_id=forecast['instance_id'],inspection_gain=gain,
                                fallback_reason=candidate.get('fallback_reason'),repeated_view_excluded=candidate['repeated_view_excluded']))
                expected=selected.get('expected_gain');inspection=sum(r['inspection_gain'] for r in contributors)
                residual=None if expected is None or selected['kind']!='direct' else expected-inspection
                if residual is not None and abs(residual)<=1e-12 and inspection>1e-12:
                    direct_pure_inspection.append(dict(paid_step=step,view_id=view_id,expected_gain=expected,
                        contributors=[r for r in contributors if r['inspection_gain']>1e-12]))
                xy=graph['nodes'][target['node']]
                goals.append(dict(run_id=run,arm=arm,method=method,paid_step=step,kind=selected['kind'],
                    target_node=target['node'],x_m=xy[0],y_m=xy[1],heading=target['heading'],
                    total_cost=selected.get('total_cost'),score=selected.get('score'),expected_gain=expected,
                    initialization_instance_id=selected.get('instance_id'),initialization_attempt=selected.get('attempt_number'),
                    selected_inspection_components=contributors,remaining_discovery_component=residual))
            instances=final['observed_instances']['instances']
            canonical_metrics=(record('evaluation.json')['metrics'] if arm=='GroundOn' else
                inputs.json(analysis.ROOT/plan['output_root']/run/'measurement/result.json')['metrics'])
            summaries[run]=dict(arm=arm,method=method,original_status=row['online_status'],original_qualified=row['original_end_to_end_qualified'],
                metric_version=row['metric_version'],metrics={k:row[k] for k in analysis.METRICS},per_instance_metrics=canonical_metrics['per_instance'],
                actual_actions=dict(action_counts),path_length_m=row['path_length_m'],initialization_actions_spent=final['initialization_actions_spent'],
                return_first_decision_step=return_step,observed_instance_count=len(instances),
                observed_instances=[dict(instance_id=r['instance_id'],observed_class=r['observed_class'],anchor_world_m=r['anchor_world_m'],
                    first_steps=observed_first[r['instance_id']],first_reliable_plane=plane_first.get(r['instance_id']),
                    first_applied_feedback=feedback_first.get(r['instance_id'])) for r in instances],
                xy_bounds_m=[[min(p[i] for p in positions),max(p[i] for p in positions)] for i in (0,1)],
                selected_pure_inspection_goals=direct_pure_inspection,timings_s=result['timings'],elapsed_s=result['elapsed_s'])
    inputs.unchanged();require(not output.exists(),'exclusive new explanation output');output.mkdir(parents=True)
    def save(name,value):
        with (output/name).open('xb') as stream:stream.write(analysis.old.canonical(value))
    analysis.old.write_csv(output/'selected_goals.csv',goals,['run_id','paid_step'])
    analysis.old.write_csv(output/'paid_observations.csv',observes,['run_id','paid_step'])
    save('early_initialization_receipts.json',early)
    save('summary.json',dict(schema='article.ground_first_two_explanation.v1',runs=summaries,
        first_action_witnesses={key:value['first_action_witness'] for key,value in witnesses.items()
            if key in {f'ground_AISLE_{m}_b160_n92801__vs__dev_AISLE_{m}_b160_n92801' for m in ('B','G')}},
        new_worlds=0,new_tsdf_integrations=0,new_quality_evaluations=0,
        limits='First completed B/G only. Recorded utility decomposition, not new candidate scoring. After executed-action step6, trajectories differ and only descriptive downstream mechanisms are claimed. No S result or generalization conclusion.'))
    files={p.name:dict(bytes=p.stat().st_size,sha256=analysis.old.digest(p.read_bytes())) for p in output.iterdir()}
    save('manifest.json',dict(files=files,input_sha256=inputs.pins,source_sha256={
        str(Path(p).relative_to(analysis.ROOT)):analysis.old.digest(Path(p).read_bytes())
        for p in (__file__,analysis.__file__,analysis.old.__file__)},new_worlds=0,new_quality_evaluations=0))
    print(json.dumps(dict(output=str(output),runs=list(summaries),goals=len(goals),paid_observations=len(observes))))


if __name__=='__main__':
    explain(analysis.STAGE/'analysis_v1/ground_first2_explanation')
