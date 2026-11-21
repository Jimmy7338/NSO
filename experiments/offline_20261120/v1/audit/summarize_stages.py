#!/usr/bin/env python3
"""Summarize all declared diagnostic states without selecting favourable cases."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
STATES = {'shelf_left': (24,54,109), 'shelf_right': (24,53,110)}
VARIANTS = ('B2','sampling','geometry','geometry_free')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rho(prediction, actual):
    x,y = np.asarray(prediction),np.asarray(actual)
    if len(x)<2 or np.all(x==x[0]) or np.all(y==y[0]):
        return None
    return float(np.corrcoef(rankdata(x),rankdata(y))[0,1])


def avg(values):
    valid=[x for x in values if x is not None]
    return float(np.mean(valid)) if valid else None


def object_total(row, quantity):
    return sum(row['actual_fixed_reference_increment'][name][quantity] for name in ('open31','closed118'))


def selected_payload(row, execution):
    inc=row['actual_fixed_reference_increment']
    return dict(candidate_id=row['candidate_id'],
        actual_object_gross_new_m2=object_total(row,'gross_new_correct_surface_area_estimate_m2'),
        actual_object_lost_m2=object_total(row,'lost_correct_surface_area_estimate_m2'),
        actual_object_net_new_m2=object_total(row,'net_correct_surface_area_estimate_m2'),
        actual_global_gross_new_m2=inc['all']['gross_new_correct_surface_area_estimate_m2'],
        actual_global_lost_m2=inc['all']['lost_correct_surface_area_estimate_m2'],
        actual_global_net_new_m2=inc['all']['net_correct_surface_area_estimate_m2'],
        completed_actions=execution['completed_actions'],planned_actions=execution['planned_actions'],
        path_length_m=execution['path_length_m'],endpoint_reached=execution['endpoint_reached'],
        simulated_collisions=execution['simulated_collisions'],execution_status=execution['status'],
        coverage_delta_percentage_points=100*(execution['final_coverage']['coverage_fraction']-execution['initial_coverage']['coverage_fraction']))


def summarize_state(layout,paid):
    attempt=ROOT/'replay'/f'{layout}_{paid}_attempt_001'
    audit=ROOT/'audit'/f'{layout}_{paid}'/'result.json'
    run=json.loads((attempt/'result.json').read_text())
    assert run['status']=='completed_staged_fixed_prefix_all_candidate_branches'
    assert sha(attempt/'manifest.json')==run['manifest']['sha256']
    evaluation=json.loads(audit.read_text())
    assert evaluation['manifest_sha256']==sha(attempt/'manifest.json')
    source=json.loads((attempt/'fixed_candidates.json').read_text())
    costs={int(x['candidate_id']):x['cost'] for x in source}
    rows={v:{}for v in VARIANTS}
    for row in evaluation['stages']:
        assert row['layout']==layout and row['state_action']==paid
        rows[row['variant']][row['candidate_id']]=row
    state=dict(layout=layout,state_action=paid,candidates=len(costs),variants={},
               evaluator_result_sha256=sha(audit),execution_result_sha256=sha(attempt/'result.json'))
    for variant in VARIANTS:
        assert set(rows[variant])==set(costs)
        captured=json.loads((attempt/(variant+'_predictions.json')).read_text())
        predicted={int(x['candidate_id']):x for x in captured['routes']}
        assert set(predicted)==set(costs)
        ids=sorted(costs)
        for cid in ids:
            assert rows[variant][cid]['actual_fixed_reference_increment']==rows['B2'][cid]['actual_fixed_reference_increment']
            for name in ('G','S'):
                measured_total=rows[variant][cid]['prediction_groups'][name]['primary_predicted_point_match']['total_weighted_template_area_m2']
                assert np.isclose(measured_total,predicted[cid]['template_area'][name],rtol=0,atol=1e-10)
        actual=np.array([object_total(rows[variant][cid],'net_correct_surface_area_estimate_m2')for cid in ids])
        gross=np.array([object_total(rows[variant][cid],'gross_new_correct_surface_area_estimate_m2')for cid in ids])
        value={'models':{}}
        for name in ('G','S'):
            matching=[rows[variant][cid]['prediction_groups'][name]['primary_predicted_point_match']for cid in ids]
            mass=sum(x['total_weighted_template_area_m2']for x in matching)
            support=sum(x['source_supported_weighted_template_area_m2']for x in matching)
            available=sum(x['source_supported_previously_unmatched_weight_m2']for x in matching)
            realized=sum(x['partition']['true_previously_unmatched_now_matched']['weighted_template_area_m2']for x in matching)
            template=[predicted[cid]['template_area'][name]for cid in ids]
            winner=min(ids,key=lambda cid:(-predicted[cid]['scores'][name],costs[cid],cid))
            twinner=min(ids,key=lambda cid:(-predicted[cid]['template_area'][name],costs[cid],cid))
            execution_path=Path(rows[variant][winner]['execution_receipt']['path'])
            assert sha(execution_path)==rows[variant][winner]['execution_receipt']['sha256']
            execution=json.loads(execution_path.read_text())
            selection=selected_payload(rows[variant][winner],execution)
            value['models'][name]=dict(mean_route_predicted_weight_m2=mass/len(ids),
                mean_route_source_supported_weight_m2=support/len(ids),
                pooled_route_source_support_fraction=support/mass if mass else None,
                mean_positive_route_source_support_fraction=avg([x['source_supported_weight_fraction']for x in matching]),
                mean_route_realized_supported_weight_m2=realized/len(ids),
                pooled_route_realization_fraction=realized/available if available else None,
                template_rank_rho_vs_object_net=rho(template,actual),
                template_rank_rho_vs_object_gross=rho(template,gross),
                template_only_best_candidate=twinner,template_has_variation=bool(np.ptp(template)>0),
                selected=selection,
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
    available=[];pending=[]
    for layout,states in STATES.items():
        for paid in states:
            if (ROOT/'audit'/f'{layout}_{paid}'/'result.json').exists():
                available.append(summarize_state(layout,paid))
            else:pending.append([layout,paid])
    aggregate={}
    for variant in VARIANTS:
        aggregate[variant]={'G_S_different_winner_states':sum(s['variants'][variant]['G_S_selected_candidate_differs']for s in available),'models':{}}
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
    result={'status':'complete_six_state_diagnostic_summary'if not pending else 'partial_development_summary',
        'completed_states':len(available),'declared_states':6,'pending_states':pending,'states':available,
        'equal_state_weight_aggregate':aggregate,'summarizer_sha256':sha(__file__),
        'definitions':{'sampling_units':'Six dependent acquired states from two permutations of one floorplan; neither66candidatebranches nor checkpoints are independent scenes.',
          'aggregation':'Arithmetic mean of state-level means/fractions; each completed state equalweight. A partial summary is not a stage-selection result.',
          'template_ranking':'Within each state, rank template-only predicted gain vs actual object gross/net newly reconstructed source area; Spearman undefined if constant.',
          'full_score':'Original common_total_proxy + template gain. Selected candidate actualglobal/object area, coverage and cost reported separately, not conflated into an invented measured utility.',
          'weights':'Weighted alternative-template quadrature integrals are not unique true surface area.',
          'positive_result':'G/S divergence alone is not evidence of benefit; all states/variants and actual deficits retained.',
          'selection_gap':'Descriptive gap from best observed object area among declared candidates; no evaluator-based choice was executed by the policy.'}}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'completed_states':len(available),'pending':pending,'output':str(args.output)}))


if __name__=='__main__':main()
