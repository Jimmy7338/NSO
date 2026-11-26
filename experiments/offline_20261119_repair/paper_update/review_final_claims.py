#!/usr/bin/env python3
"""Read actual G/S choices and final evaluator output; no actor execution."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LAYOUTS = ('shelf_left', 'shelf_right')
ARMS = ('C', 'J', 'G', 'S', 'S_no_fit', 'S_no_history')


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def counts(rows):
    return dict(states=len(rows), score_difference_states=sum(x['score_changed'] for x in rows),
                winner_difference_states=sum(x['winner_changed'] for x in rows),
                all_template_gain_zero_states=sum(x['all_template_gain_zero'] for x in rows),
                single_candidate_states=sum(x['candidates'] == 1 for x in rows),
                zero_gain_single_candidate_states=sum(x['all_template_gain_zero'] and x['candidates'] == 1 for x in rows))


def analyze_layout(layout):
    path = ROOT / 'layouts' / layout / 'runs/S/attempt_001'
    gpath = ROOT / 'layouts' / layout / 'runs/G/attempt_001'
    sr, gr = read(path / 'result.json'), read(gpath / 'result.json')
    assert sr['status'] == gr['status'] == 'completed_execution'
    choices = read(path / 'choices.json')
    rows = []
    for c in choices:
        o, inputs = c['outputs'], c['inputs']; routes, scores = o['candidates'], o['scores']
        remaining = inputs['remaining_budget']; minimum = o['candidate_audit']['minimum_current_return_cost']
        slack = remaining - minimum
        if not routes:
            rows.append(dict(paid=256-remaining, slack=slack, candidates=0, score_changed=False,
                             winner_changed=False, all_template_gain_zero=True))
            continue
        choose = lambda mode: min(range(len(routes)), key=lambda i: (-scores[mode][i], routes[i]['cost'], routes[i]['candidate_id']))
        gw, sw = choose('G'), choose('S')
        assert o['selected'] is None or o['selected']['candidate_id'] == routes[sw]['candidate_id']
        delta = [s-g for s,g in zip(scores['S'],scores['G'])]
        competitors = []
        for j in range(len(routes)):
            if j == gw: continue
            margin = scores['G'][gw] - scores['G'][j]
            advantage = delta[j] - delta[gw]
            competitors.append(dict(candidate_id=routes[j]['candidate_id'], G_margin=margin,
                challenger_relative_semantic_advantage=advantage,
                remaining_S_margin=margin-advantage,
                advantage_over_G_margin=advantage/margin if margin > 1e-12 else None))
        strongest = max((x for x in competitors if x['advantage_over_G_margin'] is not None),
                       key=lambda x:x['advantage_over_G_margin'], default=None)
        other = min((j for j in range(len(routes)) if j != sw),
                    key=lambda j:(-scores['S'][j], routes[j]['cost'], routes[j]['candidate_id']), default=None)
        detail = None
        if other is not None:
            audit = o['score_audit']
            detail = dict(S_runner_up_candidate=routes[other]['candidate_id'],
                 S_winner_margin=scores['S'][sw]-scores['S'][other],
                 coverage_component_gap=scores['C'][sw]-scores['C'][other],
                 measured_quality_component_gap=audit[sw]['quality_total_proxy']-audit[other]['quality_total_proxy'],
                 G_template_component_gap=audit[sw]['prior_potentials']['G']-audit[other]['prior_potentials']['G'],
                 semantic_delta_gap=delta[sw]-delta[other])
        rows.append(dict(paid=256-remaining, remaining=remaining, minimum_return=minimum, slack=slack,
            candidates=len(routes), G_winner=routes[gw]['candidate_id'], S_winner=routes[sw]['candidate_id'],
            score_changed=any(abs(x)>1e-12 for x in delta), winner_changed=gw!=sw,
            all_template_gain_zero=all(abs(r['prior_potentials']['G'])<1e-12 and abs(r['prior_potentials']['S'])<1e-12 for r in o['score_audit']),
            semantic_delta_range=[min(delta),max(delta)], strongest_relative_challenger=strongest,
            winner_component_decomposition=detail))
    nonzero = [x for x in rows if x['score_changed']]
    ratios = [(x['paid'],x['strongest_relative_challenger']) for x in nonzero if x['strongest_relative_challenger'] is not None]
    strongest = max(ratios, key=lambda x:x[1]['advantage_over_G_margin'], default=None)
    return dict(actual_G_S_trajectory_identical=gr['trajectory_sha256']==sr['trajectory_sha256'],
                actual_episodes=2, diagnostic_trajectory='S only; G counterfactual scores use exactly the same acquired state and candidate set',
                overall=counts(rows), positive_return_slack=counts([x for x in rows if x['slack']>0]),
                zero_return_slack=counts([x for x in rows if x['slack']==0]),
                negative_return_slack=counts([x for x in rows if x['slack']<0]),
                strongest_relative_challenger=None if strongest is None else dict(paid=strongest[0],**strongest[1]),
                nonzero_score_state_details=nonzero,
                all_states=[{k:x[k] for k in ('paid','slack','candidates','score_changed','winner_changed','all_template_gain_zero')} for x in rows],
                evidence_sha256={str(p):sha(p) for p in [path/'choices.json',path/'result.json',gpath/'result.json']})


def main():
    rows = {layout:analyze_layout(layout) for layout in LAYOUTS}
    summary_path = ROOT / 'audit/summary.json'
    review = dict(schema='B2_final_claim_evidence_review_v1', status='completed_G_S_decision_audit_final_matrix_pending',
                  script_sha256=sha(Path(__file__)), layouts=rows,
                  evidence_scope='Two position permutations of one floorplan; states along a trajectory are not independent environment samples.',
                  interpretations=[
                    'A score change without a winner change establishes internal semantic influence only, not changed observation allocation.',
                    'Single-candidate and zero-return-slack states have no alternative-target opportunity and must be distinguished from earlier multi-candidate states.',
                    'If semantic effects are insufficient for a switch, quantify challenger advantage relative to the geometric winning margin; do not assume unknown coverage generally dominates.',
                    'Zero remaining template gain does not by itself identify which visibility gate is responsible; fixed-state stage traces have narrower scope.',
                    'The repaired self-occlusion inconsistency and its predicted-gain effects do not prove a semantic efficiency or reconstruction benefit.'
                  ], final_result_claims=None)
    if summary_path.exists():
        summary = read(summary_path)
        assert summary['status'] in ('completed_independent_12_slot_evaluation','completed_with_failures')
        assert all(set(summary['rows'][l]) == set(ARMS) for l in LAYOUTS)
        review['status'] = 'completed_twelve_slot_claim_evidence_review'
        review['final_summary_sha256'] = sha(summary_path)
        claim = dict(all_declared_slots_retained=True, floorplan_families=1, position_permutations=2,
                     independent_natural_environments=0, stochastic_replications=0, G_S={})
        for layout in LAYOUTS:
            a,b = summary['rows'][layout]['G'],summary['rows'][layout]['S']
            if 'surface_metrics' not in a or 'surface_metrics' not in b:
                claim['G_S'][layout] = dict(status='incomplete_or_failed_pair', G_status=a['status'], S_status=b['status'])
                continue
            claim['G_S'][layout] = dict(
                trajectory_identical=a['trajectory_sha256']==b['trajectory_sha256'],
                native_TSDF_identical=a['native_TSDF_mesh_sha256']==b['native_TSDF_mesh_sha256'],
                F1_delta_percentage_points=100*(b['surface_metrics']['f1_05cm']-a['surface_metrics']['f1_05cm']),
                full_coverage_actions_G=a['first_coverage_action']['1.0'], full_coverage_actions_S=b['first_coverage_action']['1.0'],
                coverage_efficiency_advantage_supported=(a['first_coverage_action']['1.0'] is not None and b['first_coverage_action']['1.0'] is not None and b['first_coverage_action']['1.0']<a['first_coverage_action']['1.0']))
        review['final_result_claims'] = claim
    matching_path = ROOT / 'replay/attempt_003/prediction_matching.json'
    if matching_path.exists():
        matching = read(matching_path)
        assert matching['status'] == 'completed_same_domain_weighted_prediction_matching'
        assert matching['actor_ever_received_source_truth'] is False
        records = []
        table_rows = []
        for state in matching['results']:
            metrics = {k:state['models']['S'][k]['primary_predicted_point_match']
                       for k in ('original','repaired','repair_increment')}
            old, new, increment = [metrics[k] for k in ('original','repaired','repair_increment')]
            records.append(dict(start_action=state['start_action'], end_action=state['end_action'],
                original_gain=old['total_weighted_template_area_m2'], repaired_gain=new['total_weighted_template_area_m2'],
                original_source_supported_fraction=old['source_supported_weight_fraction'],
                repaired_source_supported_fraction=new['source_supported_weight_fraction'],
                repaired_still_geometry_unsupported_fraction=1-new['source_supported_weight_fraction'],
                increment_source_supported_fraction=increment['source_supported_weight_fraction'],
                increment_supported_previously_unmatched_realization_fraction=increment['realization_fraction_of_source_supported_unmatched_weight']))
            table_rows.append(f"{state['start_action']}--{state['end_action']} & "
                f"{old['total_weighted_template_area_m2']:.3f} / {new['total_weighted_template_area_m2']:.3f} & "
                f"{100*old['source_supported_weight_fraction']:.2f} / {100*new['source_supported_weight_fraction']:.2f} & "
                f"{100*increment['source_supported_weight_fraction']:.2f} \\\\")
        review['fixed_route_prediction_geometry'] = dict(
            evidence_sha256=sha(matching_path), threshold_m=matching['threshold_m'], S_only=True,
            states=records, definitions=matching['definitions'],
            interpretation='The visibility repair restores some source-supported predicted samples, while substantial weighted template geometry remains unsupported in these two fixed states. This identifies residual surrogate geometry error, not the sole causal explanation of final performance or a semantic benefit.')
        tex = r'''% Generated only from attempt_003/prediction_matching.json; no new experiment.
\begin{table}[!htbp]
\centering\small
\caption{Same-domain diagnosis on two saved intervals of the original S route.
Values before/after the visibility repair are paired within each state.
Source support is the fraction of weighted hypothetical sample area within 5\,cm
of source triangles, not independently sampled true surface area or measured reconstruction accuracy.
The final column uses only the per-point gain restored by the repair.
Neither interval is a new policy execution.}\label{tab:repair-prediction-match}
\begin{tabular}{@{}rrrr@{}}
\toprule
Route interval & Predicted area ($\mathrm{m}^2$) & Source support (\%) & Increment support (\%) \\
& Original / repaired & Original / repaired & Repaired minus original \\
\midrule
''' + '\n'.join(table_rows) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
        (HERE/'prediction_matching_table.tex').write_text(tex)
    target = HERE / 'claim_evidence_review.json'
    target.write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps(dict(path=str(target),sha256=sha(target),status=review['status'],
                         per_layout={l:{k:v[k]for k in ('overall','positive_return_slack','zero_return_slack','strongest_relative_challenger')}for l,v in rows.items()})))


if __name__ == '__main__': main()
