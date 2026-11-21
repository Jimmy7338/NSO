"""Additional counts from completed frozen logs only. No actor imports."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    summary=read(HERE/'summary.json');protocol=read(ROOT/'protocol.json')
    rows={};inputs={}
    for layout in protocol['layouts']:
        rows[layout]={}
        for arm in protocol['arms']:
            p=ROOT/'layouts'/layout/'runs'/arm/'attempt_001'
            if summary['rows'][layout][arm]['status']!='completed_independently_evaluated':
                rows[layout][arm]={'status':'not_complete_or_evaluation_failed'};continue
            t=read(p/'template_audit.json');choices=read(p/'choices.json');frames=read(p/'template_frame_receipts.json')
            inputs[str(p/'template_audit.json')]=sha(p/'template_audit.json')
            inputs[str(p/'choices.json')]=sha(p/'choices.json')
            assert len(t)==len(choices)
            unique_frames=len({c['mapper_frames']for c in t})==len(t)
            assert unique_frames,'Need incremental per-cache counters if same-frame scoring repeats'
            by_model={}
            for model_name in ('G','S'):
                sums={k:0 for k in ('gain_calls','repair_enabled_calls','original_self_occluded_rays',
                    'original_self_occluded_free_first_hits','candidate_intersections','strict_free_intersections',
                    'measured_surface_intersections','unresolved_intersections','history_conflicts_vetoed',
                    'recovered_points','recovered_expected_area_m2')}
                for c in t:
                    d=c['occlusion_diagnostics'].get(model_name,{})
                    assert d.get('gain_calls',0)==c['source_gain_new_calls'][model_name], 'Cache counter no longer fresh'
                    for k in sums:sums[k]+=d.get(k,0)
                by_model[model_name]=sums
            # Actual selection uses no template for C/J, G model for G and S model for the S ablations.
            used=None if arm in ('C','J') else 'G'if arm=='G'else'S'
            states=[]
            for c,a in zip(choices,t):
                out=c['outputs'];audit=out['candidate_audit'];scores=out['scores']
                n=len(out['candidates']);slack=audit['remaining_budget']-audit['minimum_current_return_cost']
                semantic=any(abs(x-y)>1e-10 for x,y in zip(scores['S'],scores['G']))
                def winner(mode):
                    return min(range(n),key=lambda i:(-scores[mode][i],out['candidates'][i]['cost'],out['candidates'][i]['candidate_id']))if n else None
                states.append(dict(paid=a['mapper_frames']-1,candidates=n,detour_slack=slack,
                    semantic_score_difference=semantic,winner_difference=winner('G')!=winner('S')))
            groups={}
            for name,pred in [('all',lambda x:True),('positive_detour_slack',lambda x:x['detour_slack']>0),
                              ('no_detour_slack_return_constrained',lambda x:x['detour_slack']==0)]:
                subset=[s for s in states if pred(s)]
                groups[name]=dict(states=len(subset),semantic_score_differences=sum(s['semantic_score_difference']for s in subset),
                    G_S_winner_differences=sum(s['winner_difference']for s in subset),single_candidate_states=sum(s['candidates']==1 for s in subset))
            row=dict(status='completed_log_audit',models=by_model,template_model_used_for_actual_selection=used,
                C_J_template_computation_is_counterfactual_only=used is None,
                observed_sensor_packets=len(frames),scored_states=len(t),each_scoring_state_has_unique_mapper_frame=True,
                computed_gain_calls_equal_uncached_source_calls=True,budget_strata=groups,
                raw_template_point_range=[min(c['model_audits']['S']['raw_template_points']for c in t),max(c['model_audits']['S']['raw_template_points']for c in t)],
                post_history_point_range=[min(c['model_audits']['S']['points_after_history']for c in t),max(c['model_audits']['S']['points_after_history']for c in t)],
                point_cap_activations=sum(c['model_audits']['S']['points_after_history']>3500 for c in t),
                reward_history_enabled_values=sorted({c['model_audits']['S']['history_rejection']for c in t}),
                states=states)
            rows[layout][arm]=row
    output=dict(schema='B2_independent_mechanism_counts_v1',rows=rows,summary_sha256=sha(HERE/'summary.json'),
        input_logs_sha256=inputs,script_sha256=sha(__file__),
        definitions=['Gain calls are simulated candidate camera views evaluated by the model, not executed observations.',
            'Recovered points and expected-area sums are point-view events. A point can contribute repeatedly across views/plans; neither is unique reconstructed area.',
            'C/J calculate template counterfactuals for diagnostics but do not use them in target selection.',
            'Positive-slack stratification is a post hoc diagnosis of decision opportunity, not a replacement for all observed planning states.',
            'S_no_history retains corrected self-occlusion; it removes reward-sample history filtering and can activate the original 3500-point approximation.'],
        actor_executed=False,reference_rebuilt=False)
    with (HERE/'mechanism_posthoc.json').open('x')as f:json.dump(output,f,indent=2)
    print(json.dumps({'output':str(HERE/'mechanism_posthoc.json'),'sha256':sha(HERE/'mechanism_posthoc.json')}))

if __name__=='__main__':main()
