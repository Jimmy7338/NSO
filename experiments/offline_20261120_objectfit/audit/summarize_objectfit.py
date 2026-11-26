#!/usr/bin/env python3
"""Equal-state summary with positive-track baseline attribution and costs."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
V1=ROOT.parent/'dual_template_geometry_20261120'
V2=ROOT.parent/'dual_template_instance_20261120'
STATES={'shelf_left':(24,54,109),'shelf_right':(24,53,110)}
VARIANTS=('object_support','object_full_support','object_fit')
spec=importlib.util.spec_from_file_location('fixed_summary_helpers',V1/'audit/summarize_stages.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
sha,avg,rho,object_total,selected_payload=h.sha,h.avg,h.rho,h.object_total,h.selected_payload

def sum_object_metrics(targets):
    total=targets['open31']['total_weighted_template_area_m2']
    assert np.isclose(total,targets['closed118']['total_weighted_template_area_m2'],atol=1e-12,rtol=0)
    support=sum(v['source_supported_weighted_template_area_m2']for v in targets.values())
    realized=sum(v['partition']['true_previously_unmatched_now_matched']['weighted_template_area_m2']for v in targets.values())
    return dict(predicted_weight_m2=total,true_object_supported_weight_m2=support,
        realized_true_object_weight_m2=realized,true_object_support_fraction=support/total if total else None,
        per_object={name:dict(supported_weight_m2=v['source_supported_weighted_template_area_m2'],
            realized_weight_m2=v['partition']['true_previously_unmatched_now_matched']['weighted_template_area_m2'])for name,v in targets.items()})

def summarize_state(layout,paid):
    attempt=ROOT/'replay'/f'{layout}_{paid}_attempt_001';output=HERE/f'{layout}_{paid}'/'result.json'
    run=json.loads((attempt/'result.json').read_text());assert run['status']=='completed_objectfit_fixed_prefix_reused_branches'
    assert run['manifest']['sha256']==sha(attempt/'manifest.json')
    e=json.loads(output.read_text());assert e['status']=='completed_object_attributed_fixed_candidate_evaluation'
    assert e['manifest_sha256']==sha(attempt/'manifest.json')
    routes=json.loads((attempt/'fixed_candidates.json').read_text())
    assert routes==json.loads((V1/'replay'/f'{layout}_{paid}_attempt_001'/'fixed_candidates.json').read_text())
    costs={int(r['candidate_id']):r['cost']for r in routes}
    old=json.loads((V1/'audit'/f'{layout}_{paid}'/'result.json').read_text())
    oldrows={r['candidate_id']:r for r in old['stages']if r['variant']=='B2'}
    by_variant={v:{}for v in VARIANTS}
    for row in e['stages']:
        cid=row['candidate_id'];assert row['layout']==layout and row['state_action']==paid
        assert row['actual_fixed_reference_increment']==oldrows[cid]['actual_fixed_reference_increment']
        assert row['execution_receipt']==oldrows[cid]['execution_receipt']
        by_variant[row['variant']][cid]=row
    state=dict(layout=layout,state_action=paid,evaluation_sha256=sha(output),variants={})
    for variant in VARIANTS:
        rows=by_variant[variant];assert set(rows)==set(costs)
        capture=json.loads((attempt/(variant+'_predictions.json')).read_text())
        predictions={int(r['candidate_id']):r for r in capture['routes']}
        ids=sorted(rows);actual=np.array([object_total(rows[cid],'net_correct_surface_area_estimate_m2')for cid in ids])
        actual_gross=np.array([object_total(rows[cid],'gross_new_correct_surface_area_estimate_m2')for cid in ids])
        models={}
        for model in ('G','S'):
            primary=[rows[cid]['prediction_groups'][model]['primary_predicted_point_match']for cid in ids]
            attributed=[rows[cid]['observed_object_attribution'][model]for cid in ids]
            current=[sum_object_metrics(x['current']['all_observed_object_tracks'])for x in attributed]
            baseline=[sum_object_metrics(x['legacy_pooling_same_track_baseline']['all_observed_object_tracks'])for x in attributed]
            template=[predictions[cid]['template_area'][model]for cid in ids]
            for expected,p,c in zip(template,primary,current):
                assert np.isclose(expected,p['total_weighted_template_area_m2'],atol=1e-10,rtol=0)
                assert np.isclose(expected,c['predicted_weight_m2'],atol=1e-10,rtol=0)
            winner=min(ids,key=lambda cid:(-predictions[cid]['scores'][model],costs[cid],cid))
            ex=rows[winner]['execution_receipt'];assert sha(ex['path'])==ex['sha256']
            selection=selected_payload(rows[winner],json.loads(Path(ex['path']).read_text()))
            selection['actual_objects']={name:rows[winner]['actual_fixed_reference_increment'][name]for name in ('open31','closed118')}
            selection['after_object_recall_5cm']={name:rows[winner]['after_native_TSDF_metrics']['target_surface_metrics'][name]['recall_05cm']for name in ('open31','closed118')}
            keys=('predicted_weight_m2','true_object_supported_weight_m2','realized_true_object_weight_m2','true_object_support_fraction')
            m=dict(mean_candidate={k:avg([x[k]for x in current])for k in keys},
                legacy_positive_track_mean_candidate={k:avg([x[k]for x in baseline])for k in keys},
                mean_per_object={name:{k:avg([x['per_object'][name][k]for x in current])for k in ('supported_weight_m2','realized_weight_m2')}for name in ('open31','closed118')},
                legacy_positive_track_mean_per_object={name:{k:avg([x['per_object'][name][k]for x in baseline])for k in ('supported_weight_m2','realized_weight_m2')}for name in ('open31','closed118')},
                global_source_supported_weight_m2=avg([x['source_supported_weighted_template_area_m2']for x in primary]),
                template_rank_rho_vs_object_net=rho(template,actual),template_rank_rho_vs_object_gross=rho(template,actual_gross),
                selected=selection,support_validation=attributed[0]['support_validation'],
                baseline_admitted_tracks=attributed[0]['baseline_admitted_tracks'],
                lost_baseline_tracks=attributed[0]['lost_baseline_tracks'],new_tracks=attributed[0]['new_tracks'],
                all_retained_track_forecasts_unchanged=all(all(all(c.values())for c in x['same_track_forecast_parity'].values())for x in attributed))
            m['paired_positive_track_mean_deltas']={k:m['mean_candidate'][k]-m['legacy_positive_track_mean_candidate'][k]
                if m['mean_candidate'][k]is not None and m['legacy_positive_track_mean_candidate'][k]is not None else None for k in keys}
            models[model]=m
        state['variants'][variant]=dict(models=models,G_S_selected_candidate_differs=models['G']['selected']['candidate_id']!=models['S']['selected']['candidate_id'])
    return state

def main():
    states=[];pending=[]
    for layout,paid_states in STATES.items():
        for paid in paid_states:
            if(HERE/f'{layout}_{paid}'/'result.json').exists():states.append(summarize_state(layout,paid))
            else:pending.append([layout,paid])
    aggregate={};table=[]
    for variant in VARIANTS:
        aggregate[variant]=dict(G_S_different_winner_states=sum(s['variants'][variant]['G_S_selected_candidate_differs']for s in states),models={})
        for model in ('G','S'):
            rows=[s['variants'][variant]['models'][model]for s in states]
            if not rows:continue
            keys=rows[0]['mean_candidate']
            a=dict(mean_candidate={k:avg([r['mean_candidate'][k]for r in rows])for k in keys},
                legacy_positive_track_mean_candidate={k:avg([r['legacy_positive_track_mean_candidate'][k]for r in rows])for k in keys},
                paired_positive_track_mean_deltas={k:avg([r['paired_positive_track_mean_deltas'][k]for r in rows])for k in keys},
                mean_per_object={name:{k:avg([r['mean_per_object'][name][k]for r in rows])for k in ('supported_weight_m2','realized_weight_m2')}for name in ('open31','closed118')},
                template_rank_rho_vs_object_net=avg([r['template_rank_rho_vs_object_net']for r in rows]),
                template_rank_rho_vs_object_gross=avg([r['template_rank_rho_vs_object_gross']for r in rows]),
                rank_defined_states=sum(r['template_rank_rho_vs_object_net']is not None for r in rows),
                states_losing_baseline_tracks=sum(bool(r['lost_baseline_tracks'])for r in rows),
                all_retained_track_forecasts_unchanged=all(r['all_retained_track_forecasts_unchanged']for r in rows),
                selected_means={k:avg([r['selected'][k]for r in rows])for k in ('actual_object_net_new_m2','actual_global_net_new_m2','completed_actions','path_length_m','coverage_delta_percentage_points')},
                selected_object_recall_5cm={name:avg([r['selected']['after_object_recall_5cm'][name]for r in rows])for name in ('open31','closed118')})
            aggregate[variant]['models'][model]=a
            table.append(dict(variant=variant,model=model,**a['mean_candidate'],
                matched_baseline_true_supported_m2=a['legacy_positive_track_mean_candidate']['true_object_supported_weight_m2'],
                matched_baseline_realized_m2=a['legacy_positive_track_mean_candidate']['realized_true_object_weight_m2'],
                object_rank_rho=a['template_rank_rho_vs_object_net'],**a['selected_means'],
                states_losing_baseline_tracks=a['states_losing_baseline_tracks'],G_S_different_winner_states=aggregate[variant]['G_S_different_winner_states']))
    result=dict(status='complete_six_state_objectfit_summary'if not pending else'partial_objectfit_summary',completed_states=len(states),pending=pending,
        states=states,equal_state_weight_aggregate=aggregate,summary_script_sha256=sha(__file__),
        previous_complete_all_variant_summary=dict(path=str(V2/'audit/staged_summary.json'),sha256=sha(V2/'audit/staged_summary.json')),
        definitions=['Candidate summaries/ranks within state, then equal six-state averages. No66independent-trial claim.',
            'Matched baseline uses oldP positive-track subset only, without changing its saved predictions. Background removal is not treated as lost trueobject recall.',
            'Object support is matched to true object triangles only inside evaluator. Hypothetical weights are not unique actualarea.',
            'All actual branches/outcomes are reused. G/S selection divergence doesnot itself establish full-policy or semantic benefit.'])
    (HERE/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if table:
        with(HERE/'results.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print(json.dumps(dict(status=result['status'],completed_states=len(states),pending=pending)))

if __name__=='__main__':main()
