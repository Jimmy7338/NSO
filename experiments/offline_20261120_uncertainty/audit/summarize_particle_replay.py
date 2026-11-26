#!/usr/bin/env python3
"""Report every B8 fixed-replay arm, equally weighting six dependent states."""
import csv,importlib.util,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'particle_replay'
V1=ROOT.parent/'dual_template_geometry_20261120'
B5=ROOT.parent/'dual_template_objectfit_20261120';B6=ROOT.parent/'dual_template_parts_20261120'
STATES={'shelf_left':(24,54,109),'shelf_right':(24,53,110)}
ARMS={'G_fixed':'G','S_fixed':'S','G_uncertainty':'G','S_uncertainty':'S','S_mean_collapse':'S'}
spec=importlib.util.spec_from_file_location('B8_unchanged_summary_helpers',B6/'audit/summarize_parts.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
sha,avg,rho,object_total,selected_payload=h.sha,h.avg,h.rho,h.object_total,h.selected_payload
KEYS=('predicted_weight_m2','true_object_supported_weight_m2','realized_true_object_weight_m2','true_object_support_fraction')
SELECT=('actual_object_gross_new_m2','actual_object_lost_m2','actual_object_net_new_m2','actual_global_gross_new_m2','actual_global_lost_m2','actual_global_net_new_m2','completed_actions','path_length_m','coverage_delta_percentage_points')

def read(p):return json.loads(Path(p).read_text())
def resource(p):return dict(path=str(p),sha256=sha(p))

def state_summary(layout,paid):
    attempt=ROOT/'replay'/f'{layout}_{paid}_attempt_001';ep=OUT/f'{layout}_{paid}'/'result.json'
    run=read(attempt/'result.json');e=read(ep)
    assert run['status']=='completed_five_arm_fixed_prefix_reused_branches'
    assert e['status']=='completed_five_arm_particle_fixed_branch_evaluation'
    assert run['manifest']['sha256']==e['manifest_sha256']==sha(attempt/'manifest.json')
    routes=read(attempt/'fixed_candidates.json')
    assert routes==read(V1/'replay'/f'{layout}_{paid}_attempt_001/fixed_candidates.json')
    costs={int(r['candidate_id']):r['cost']for r in routes}
    original={r['candidate_id']:r for r in read(V1/'audit'/f'{layout}_{paid}/result.json')['stages']if r['variant']=='B2'}
    by={a:{}for a in ARMS}
    for r in e['stages']:
        assert r['layout']==layout and r['state_action']==paid
        cid=r['candidate_id'];assert r['actual_fixed_reference_increment']==original[cid]['actual_fixed_reference_increment']
        assert r['execution_receipt']==original[cid]['execution_receipt']
        by[r['arm']][cid]=r
    state=dict(layout=layout,state_action=paid,candidates=len(costs),evaluation=resource(ep),arms={})
    for arm,name in ARMS.items():
        rows=by[arm];assert set(rows)==set(costs)
        capture=read(attempt/(arm+'_predictions.json'));pred={int(r['candidate_id']):r for r in capture['routes']}
        assert set(pred)==set(rows);ids=sorted(rows)
        actual=[object_total(rows[c],'net_correct_surface_area_estimate_m2')for c in ids]
        gross=[object_total(rows[c],'gross_new_correct_surface_area_estimate_m2')for c in ids]
        primary=[rows[c]['prediction_groups'][name]['primary_predicted_point_match']for c in ids]
        attrib=[rows[c]['observed_object_attribution'][name]for c in ids]
        current=[h.sum_object_metrics(x['current']['all_observed_object_tracks'])for x in attrib]
        template=[pred[c]['template_area'][name]for c in ids]
        for expected,p,q in zip(template,primary,current):
            assert np.isclose(expected,p['total_weighted_template_area_m2'],atol=1e-10,rtol=0)
            assert np.isclose(expected,q['predicted_weight_m2'],atol=1e-10,rtol=0)
        winner=min(ids,key=lambda c:(-pred[c]['scores'][name],costs[c],c))
        ex=rows[winner]['execution_receipt'];assert sha(ex['path'])==ex['sha256']
        selection=selected_payload(rows[winner],read(ex['path']))
        selection['actual_objects']={n:rows[winner]['actual_fixed_reference_increment'][n]for n in ('open31','closed118')}
        selection['after_object_recall_5cm']={n:rows[winner]['after_native_TSDF_metrics']['target_surface_metrics'][n]['recall_05cm']for n in ('open31','closed118')}
        all_candidates=[]
        for i,c in enumerate(ids):
            ex=rows[c]['execution_receipt'];assert sha(ex['path'])==ex['sha256']
            all_candidates.append(dict(candidate_id=c,prediction=current[i],template_score=template[i],total_score=pred[c]['scores'][name],
                common_score=pred[c]['common_total_proxy'],actual=selected_payload(rows[c],read(ex['path']))))
        state['arms'][arm]=dict(mean_candidate={k:avg([x[k]for x in current])for k in KEYS},
            mean_per_object={n:{k:avg([x['per_object'][n][k]for x in current])for k in ('supported_weight_m2','realized_weight_m2')}for n in ('open31','closed118')},
            global_source_supported_weight_m2=avg([p['source_supported_weighted_template_area_m2']for p in primary]),
            template_rank_rho_vs_object_net=rho(template,actual),template_rank_rho_vs_object_gross=rho(template,gross),
            selected=selection,support_validation=attrib[0]['support_validation'],all_candidates=all_candidates)
    state['matched_comparisons']={}
    for a,b in [('G_uncertainty','G_fixed'),('S_uncertainty','S_fixed'),('S_uncertainty','S_mean_collapse'),('S_fixed','G_fixed'),('S_uncertainty','G_uncertainty')]:
        x,y=state['arms'][a],state['arms'][b]
        state['matched_comparisons'][a+'_minus_'+b]=dict(candidate_changed=x['selected']['candidate_id']!=y['selected']['candidate_id'],
            prediction={k:x['mean_candidate'][k]-y['mean_candidate'][k] if x['mean_candidate'][k]is not None and y['mean_candidate'][k]is not None else None for k in KEYS},
            rank_net_delta=x['template_rank_rho_vs_object_net']-y['template_rank_rho_vs_object_net'] if x['template_rank_rho_vs_object_net']is not None and y['template_rank_rho_vs_object_net']is not None else None,
            selected={k:x['selected'][k]-y['selected'][k]for k in SELECT})
    return state

def main():
    OUT.mkdir(exist_ok=True);states=[];pending=[]
    for layout,paid_states in STATES.items():
        for paid in paid_states:
            if (OUT/f'{layout}_{paid}/result.json').exists():states.append(state_summary(layout,paid))
            else:pending.append([layout,paid])
    aggregate={};table=[];comparisons={}
    for arm in ARMS:
        rows=[s['arms'][arm]for s in states]
        if not rows:continue
        a=dict(mean_candidate={k:avg([r['mean_candidate'][k]for r in rows])for k in KEYS},
            template_rank_rho_vs_object_net=avg([r['template_rank_rho_vs_object_net']for r in rows]),
            template_rank_rho_vs_object_gross=avg([r['template_rank_rho_vs_object_gross']for r in rows]),
            rank_defined_states=sum(r['template_rank_rho_vs_object_net']is not None for r in rows),
            mean_per_object={n:{k:avg([r['mean_per_object'][n][k]for r in rows])for k in ('supported_weight_m2','realized_weight_m2')}for n in ('open31','closed118')},
            selected_means={k:avg([r['selected'][k]for r in rows])for k in SELECT},
            selected_object_recall_5cm={n:avg([r['selected']['after_object_recall_5cm'][n]for r in rows])for n in ('open31','closed118')},
            states_with_unadmitted_tracks=sum(bool(r['support_validation']['unadmitted_tracks'])for r in rows))
        aggregate[arm]=a;table.append(dict(arm=arm,**a['mean_candidate'],rho_object_net=a['template_rank_rho_vs_object_net'],**a['selected_means']))
    if states:
        for key in states[0]['matched_comparisons']:
            rows=[s['matched_comparisons'][key]for s in states]
            comparisons[key]=dict(changed_winner_states=sum(r['candidate_changed']for r in rows),
                prediction={k:avg([r['prediction'][k]for r in rows])for k in KEYS},rank_net_delta=avg([r['rank_net_delta']for r in rows]),
                selected={k:avg([r['selected'][k]for r in rows])for k in SELECT},
                selected_object_net_positive_states=sum(r['selected']['actual_object_net_new_m2']>1e-12 for r in rows),
                selected_object_net_negative_states=sum(r['selected']['actual_object_net_new_m2']<-1e-12 for r in rows))
    result=dict(status='complete_six_state_five_arm_summary'if not pending else'partial_five_arm_summary',completed_states=len(states),pending=pending,
        states=states,equal_state_weight_aggregate=aggregate,matched_comparisons=comparisons,
        historical_references={n:dict(source=resource(p),aggregate=read(p)['equal_state_weight_aggregate'])for n,p in [('B5',B5/'audit/summary.json'),('B6',B6/'audit/summary.json')]},
        script=resource(Path(__file__)),definitions=['Mean within candidates of each state, then equal six-state means. No independent-trial count inflation.',
        'All 330 predictions reuse 66 original measured branches. No new closed-loop outcome.',
        'Absolute predicted source-supported and realized weights are alternate-geometry expectations, not unique physical surface area.',
        'Matched B8 fixed arm isolates extent-distribution modeling. Historical O and B6 have different geometry/likelihood and are not matched ablations.',
        'All five arms, negative actual effects, costs and fixed-budget reference recalls are retained; G/S divergence is not itself benefit.'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if table:
        with (OUT/'results.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print(json.dumps(dict(status=result['status'],completed_states=len(states),pending=pending)))
if __name__=='__main__':main()
