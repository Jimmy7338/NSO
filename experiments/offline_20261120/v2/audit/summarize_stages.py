#!/usr/bin/env python3
"""Paired comparison of all v1 and pooling/applicability diagnostic variants."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
V1=ROOT.parent/'dual_template_geometry_20261120'
STATES={'shelf_left':(24,54,109),'shelf_right':(24,53,110)}
VARIANTS=('pooling','pooling_geometry','pooling_applicability','pooling_geometry_applicability')
BASE_SHA='f4bddf2cd7c8410f299d8c3dd81abcdeac6111c25e4c60f1ed15fc9c5fc2ec2e'
spec=importlib.util.spec_from_file_location('prior_stage_summary_helpers',V1/'audit/summarize_stages.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
sha,rho,avg,object_total,selected_payload=helper.sha,helper.rho,helper.avg,helper.object_total,helper.selected_payload


def summarize_state(layout,paid,baseline):
    attempt=ROOT/'replay'/f'{layout}_{paid}_attempt_001'
    audit=ROOT/'audit'/f'{layout}_{paid}'/'result.json'
    run=json.loads((attempt/'result.json').read_text())
    assert run['status']=='completed_instance_fixed_prefix_reused_branches'
    assert sha(attempt/'manifest.json')==run['manifest']['sha256']
    evaluation=json.loads(audit.read_text())
    assert evaluation['manifest_sha256']==sha(attempt/'manifest.json')
    source=json.loads((attempt/'fixed_candidates.json').read_text())
    assert source==json.loads((V1/'replay'/f'{layout}_{paid}_attempt_001'/'fixed_candidates.json').read_text())
    costs={int(x['candidate_id']):x['cost']for x in source}
    oldeval=json.loads((V1/'audit'/f'{layout}_{paid}'/'result.json').read_text())
    oldrows={r['candidate_id']:r for r in oldeval['stages']if r['variant']=='B2'}
    rows={v:{}for v in VARIANTS}
    for row in evaluation['stages']:
        assert row['layout']==layout and row['state_action']==paid
        assert row['actual_fixed_reference_increment']==oldrows[row['candidate_id']]['actual_fixed_reference_increment']
        assert row['execution_receipt']==oldrows[row['candidate_id']]['execution_receipt']
        rows[row['variant']][row['candidate_id']]=row
    state=copy.deepcopy(baseline)
    state.update(instance_evaluator_sha256=sha(audit),instance_execution_sha256=sha(attempt/'result.json'))
    for variant in VARIANTS:
        assert set(rows[variant])==set(costs)
        captured=json.loads((attempt/(variant+'_predictions.json')).read_text())
        predicted={int(x['candidate_id']):x for x in captured['routes']}
        assert set(predicted)==set(costs)
        ids=sorted(costs)
        actual=np.array([object_total(rows[variant][cid],'net_correct_surface_area_estimate_m2')for cid in ids])
        gross=np.array([object_total(rows[variant][cid],'gross_new_correct_surface_area_estimate_m2')for cid in ids])
        value={'models':{},'model_audit':{name:{
            'raw_component_count':captured['model_audits'][name]['raw_component_count'],
            'model_group_count_before_size_eligibility':captured['model_audits'][name]['model_group_count_before_size_eligibility'],
            'constructed_object_count':len(captured['model_audits'][name]['objects']),
            'skipped_pooled_groups':captured['model_audits'][name]['skipped_pooled_groups']}
            for name in ('G','S')}}
        for name in ('G','S'):
            matching=[rows[variant][cid]['prediction_groups'][name]['primary_predicted_point_match']for cid in ids]
            for cid,match in zip(ids,matching):
                assert np.isclose(match['total_weighted_template_area_m2'],predicted[cid]['template_area'][name],rtol=0,atol=1e-10)
            mass=sum(x['total_weighted_template_area_m2']for x in matching)
            support=sum(x['source_supported_weighted_template_area_m2']for x in matching)
            available=sum(x['source_supported_previously_unmatched_weight_m2']for x in matching)
            realized=sum(x['partition']['true_previously_unmatched_now_matched']['weighted_template_area_m2']for x in matching)
            template=[predicted[cid]['template_area'][name]for cid in ids]
            winner=min(ids,key=lambda cid:(-predicted[cid]['scores'][name],costs[cid],cid))
            twinner=min(ids,key=lambda cid:(-predicted[cid]['template_area'][name],costs[cid],cid))
            ep=Path(rows[variant][winner]['execution_receipt']['path'])
            assert sha(ep)==rows[variant][winner]['execution_receipt']['sha256']
            selection=selected_payload(rows[variant][winner],json.loads(ep.read_text()))
            value['models'][name]=dict(mean_route_predicted_weight_m2=mass/len(ids),
                mean_route_source_supported_weight_m2=support/len(ids),pooled_route_source_support_fraction=support/mass if mass else None,
                mean_positive_route_source_support_fraction=avg([x['source_supported_weight_fraction']for x in matching]),
                mean_route_realized_supported_weight_m2=realized/len(ids),pooled_route_realization_fraction=realized/available if available else None,
                template_rank_rho_vs_object_net=rho(template,actual),template_rank_rho_vs_object_gross=rho(template,gross),
                template_only_best_candidate=twinner,template_has_variation=bool(np.ptp(template)>0),selected=selection,
                object_net_best_available_m2=float(actual.max()),
                selected_object_net_gap_from_best_m2=float(actual.max()-selection['actual_object_net_new_m2']))
        value['G_S_selected_candidate_differs']=value['models']['G']['selected']['candidate_id']!=value['models']['S']['selected']['candidate_id']
        value['prediction_file_sha256']=sha(attempt/(variant+'_predictions.json'))
        state['variants'][variant]=value
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'audit/staged_summary.json')
    args=parser.parse_args()
    basepath=V1/'audit/staged_summary.json'
    assert sha(basepath)==BASE_SHA,'previous six-state results changed'
    base=json.loads(basepath.read_text())
    originals={(x['layout'],x['state_action']):x for x in base['states']}
    available=[];pending=[]
    for layout,states in STATES.items():
        for paid in states:
            if (ROOT/'audit'/f'{layout}_{paid}'/'result.json').exists():
                available.append(summarize_state(layout,paid,originals[(layout,paid)]))
            else:pending.append([layout,paid])
    variants=(*helper.VARIANTS,*VARIANTS)
    aggregate={};table=[]
    for variant in variants:
        aggregate[variant]={'G_S_different_winner_states':sum(s['variants'][variant]['G_S_selected_candidate_differs']for s in available),'models':{}}
        if variant in VARIANTS:
            aggregate[variant]['pooled_eligibility_audit']={name:{
                'total_skipped_groups':sum(len(s['variants'][variant]['model_audit'][name]['skipped_pooled_groups'])for s in available),
                'states_with_skipped_groups':sum(bool(s['variants'][variant]['model_audit'][name]['skipped_pooled_groups'])for s in available),
                'state_records':[dict(layout=s['layout'],state_action=s['state_action'],**s['variants'][variant]['model_audit'][name])for s in available]}
                for name in ('G','S')}
        for model in ('G','S'):
            rows=[s['variants'][variant]['models'][model]for s in available]
            fields=['mean_route_predicted_weight_m2','mean_route_source_supported_weight_m2','pooled_route_source_support_fraction',
                    'mean_positive_route_source_support_fraction','mean_route_realized_supported_weight_m2','pooled_route_realization_fraction',
                    'template_rank_rho_vs_object_net','template_rank_rho_vs_object_gross','selected_object_net_gap_from_best_m2']
            out={key:avg([r[key]for r in rows])for key in fields}
            out['rank_defined_states']=sum(r['template_rank_rho_vs_object_net']is not None for r in rows)
            out['mean_selected']={key:avg([r['selected'][key]for r in rows])for key in
                ['actual_object_gross_new_m2','actual_object_lost_m2','actual_object_net_new_m2','actual_global_gross_new_m2',
                 'actual_global_lost_m2','actual_global_net_new_m2','completed_actions','planned_actions','path_length_m',
                 'simulated_collisions','coverage_delta_percentage_points']}
            aggregate[variant]['models'][model]=out
            if rows:
                table.append(dict(variant=variant,model=model,predicted_weight_m2=out['mean_route_predicted_weight_m2'],
                    source_support_percent=100*out['pooled_route_source_support_fraction']if out['pooled_route_source_support_fraction']is not None else None,
                    source_supported_weight_m2=out['mean_route_source_supported_weight_m2'],
                    realized_supported_weight_m2=out['mean_route_realized_supported_weight_m2'],
                    template_object_net_rank_rho=out['template_rank_rho_vs_object_net'],
                    selected_object_net_m2=out['mean_selected']['actual_object_net_new_m2'],
                    selected_global_net_m2=out['mean_selected']['actual_global_net_new_m2'],
                    selected_actions=out['mean_selected']['completed_actions'],
                    selected_coverage_delta_pp=out['mean_selected']['coverage_delta_percentage_points'],
                    G_S_different_winner_states=aggregate[variant]['G_S_different_winner_states']))
    result={'status':'complete_instance_six_state_diagnostic_summary'if not pending else'partial_instance_development_summary',
        'completed_states':len(available),'declared_states':6,'pending_states':pending,'states':available,
        'equal_state_weight_aggregate':aggregate,'summarizer_sha256':sha(__file__),
        'prior_summary_sha256':BASE_SHA,'helper_sha256':sha(V1/'audit/summarize_stages.py'),
        'definitions':{**base['definitions'],'branch_reuse':'Exactly the original66 fixed-candidate branch meshes and execution receipts, not new outcomes. New variants only change measured-prefix predictions and score-derived choices.',
            'applicability':'Absolute free-evidence discount is a heuristic model-consistency mass, not a calibrated probability. Lower absolute predicted/supported mass may be intentional; examine realised selections and predicted support jointly.',
            'comparison':'Every prior v1 variant and new pooling factorial is retained on matched completed states; never compare a partial subset against all6priorstates.'}}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if table:
        with args.output.with_name('staged_aggregate.csv').open('w',newline='')as handle:
            writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps({'status':result['status'],'completed_states':len(available),'pending':pending}))


if __name__=='__main__':main()
