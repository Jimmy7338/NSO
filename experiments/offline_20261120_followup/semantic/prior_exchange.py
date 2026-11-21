#!/usr/bin/env python3
"""Read-only algebraic diagnostic; counterfactual priors are never policy trials."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp
ROOT=Path(__file__).resolve().parent
B8=ROOT.parents[1]/'dual_template_uncertainty_20261120'

def read(p): return json.loads(Path(p).read_text())
def write(name,x): (ROOT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def winner(values,rows): return min(range(len(rows)),key=lambda i:(-values[i],rows[i]['cost'],rows[i]['candidate_id']))

def main():
    fixed=read(ROOT/'fixed_branch_details.json'); result=[]; effects=[]
    for plan in fixed['plans']:
        key=(plan['layout'],plan['paid_action'],plan['mode'])
        match=lambda x:(x['layout'],x['paid_action'],x['mode'])==key
        candidates=[x for x in fixed['candidates']if match(x)]
        objects=[x for x in fixed['objects']if match(x)]
        details=[x for x in fixed['per_object_candidate_gain']if match(x)]
        lookup={(x['candidate_id'],x['support_track']):x for x in details}
        scores=np.array([x['G_total']for x in candidates]);correct=scores.copy();swapped=scores.copy()
        deltas={};max_shifts={};contrasts={}
        for obj in objects:
            track=obj['support_track'];L=obj['geometry_log_bayes_factor_shelf_vs_box'];p=expit(L)
            sign=1 if obj['acquired_shelf_prior']>.5 else -1
            correct_p=expit(L+sign*np.log(9));swapped_p=expit(L-sign*np.log(9))
            # Finite 9:1 prior can shift class probability by at most 0.5.
            maxshift=max(abs(expit(L+np.log(9))-p),abs(expit(L-np.log(9))-p))
            assert maxshift<=.5+1e-12
            contrast=[]
            for i,candidate in enumerate(candidates):
                d=lookup[(candidate['candidate_id'],track)]
                gap=d['conditional_gain_difference_m2']
                # Both finite-prior versions underflow to the same class in this case.
                if gap is None:
                    assert abs(correct_p-p)<1e-12 and abs(swapped_p-p)<1e-12
                    gap=0.
                contrast.append(gap);correct[i]+=(correct_p-p)*gap;swapped[i]+=(swapped_p-p)*gap
            contrasts[track]=np.array(contrast);max_shifts[track]=maxshift
            effects.append(dict(layout=key[0],paid_action=key[1],mode=key[2],support_track=track,
                geometric_log_BF=L,acquired_shelf_prior=obj['acquired_shelf_prior'],neutral_shelf_p=float(p),
                observed_prior_shelf_p=float(correct_p),swapped_prior_shelf_p=float(swapped_p),
                max_9_to_1_probability_shift=float(maxshift),conditional_contrast_range_m2=float(np.ptp(contrast)),
                max_absolute_conditional_contrast_m2=float(np.max(np.abs(contrast)))))
        np.testing.assert_allclose(correct,[x['S_total']for x in candidates],atol=1e-10,rtol=0)
        g=winner(scores,candidates);s=winner(correct,candidates);swap=winner(swapped,candidates)
        ratios=[]
        for i,candidate in enumerate(candidates):
            if i==g:continue
            margin=scores[g]-scores[i]
            # Conservative upper bound: each object's prior sign may independently
            # take whichever of 1:9 or9:1 pushes this competitor most strongly.
            bound=sum(max_shifts[t]*abs(v[i]-v[g])for t,v in contrasts.items())
            ratios.append(dict(candidate_id=candidate['candidate_id'],G_margin=float(margin),
                finite_prior_upper_bound=float(bound),ratio=float(bound/margin)if margin>1e-12 else None))
        result.append(dict(layout=key[0],paid_action=key[1],mode=key[2],G_winner=candidates[g]['candidate_id'],
            observed_S_winner=candidates[s]['candidate_id'],swapped_prior_winner=candidates[swap]['candidate_id'],
            swapped_changes_winner=swap!=g,max_finite_prior_bound_margin_ratio=max((r['ratio']for r in ratios if r['ratio']is not None),default=None),
            competitor_bounds=ratios))
    # Follow each acquired object through all completed runs, de-duplicating an
    # identical observed input + mode + support track. The independent unit is
    # still two layouts, not these repeated observations.
    state_rows=[];seen=set()
    for layout in ('shelf_left','shelf_right'):
        for arm in ('G_fixed','G_uncertainty','S_mean_collapse'):
            base=B8/'rollouts/joint_depth/layouts'/layout/'runs'/arm/'attempt_001'
            for call in read(base/'template_audit.json'):
                mode=call['model_audits']['G']['extent_mode'];sig=call['bank_observed_input_sha256']
                for g,s in zip(call['model_audits']['G']['objects'],call['model_audits']['S']['objects']):
                    key=(sig,mode,g['support_track'])
                    if key in seen:continue
                    seen.add(key);q=np.array(g['prior_mass']);logq=np.log(q)[:,None]
                    terms={}
                    for name in ('energy_m2','surface_energy_m2','free_energy_m2'):
                        value=logsumexp(logq-np.array(g[name])/(2*.01**2),axis=0);terms[name]=float(value[1]-value[0])
                    cg=g['conditional_geometry'];prior=s['shape_prior'][1];p=g['shape_probability_mass'][1]
                    state_rows.append(dict(layout=layout,mode=mode,paid_action=call['mapper_frames']-1,support_track=g['support_track'],
                        acquired_shelf_prior=prior,neutral_shelf_p=p,points=cg['observed_points'],
                        source_component_count=cg['source_component_count'],yaw_observed=cg['geometry_fit_audit']['orientation_observed'],
                        observed_layer_count=len(cg['observed_layers_audit']['observed_horizontal_planes']),
                        surface_only_log_BF=terms['surface_energy_m2'],free_only_log_BF=terms['free_energy_m2'],combined_log_BF=terms['energy_m2'],
                        high_confidence_opposes_acquired_category=bool(max(p,1-p)>.99 and ((p>.5)!=(prior>.5)))))
    plans=read(ROOT/'rollout_plans.json');u={}
    for p in plans:u.setdefault((p['observed_signature'],p['mode']),p)
    summaries={}
    for mode in ('fixed','uncertainty','mean_collapse'):
        rows=[p for p in u.values()if p['mode']==mode];competitive=[p for p in rows if p['candidates']>1]
        summaries[mode]=dict(unique_observation_states=len(rows),states_with_multiple_candidates=len(competitive),
            single_candidate_states=len(rows)-len(competitive),multiple_candidate_states_with_semantic_score_change=sum(p['semantic_score_nonzero']for p in competitive),
            multiple_candidate_states_with_zero_template_range=sum(p['G_template_candidate_range']<=1e-12 for p in competitive))
    write('prior_exchange.json',dict(status='algebraic_counterfactual_only_no_new_trial',fixed_states=result,objects=effects,
        bounds='For r=exp(L), positive9:1 shift =8r/((1+r)(1+9r)); maximum0.5 at r=1/3. Negative prior is symmetric. Bounds use current conditionally inferred geometry and existing visibility.',
        original_S_score_reconstruction_passed=True,no_policy_or_model_changes=True,
        source_sha256=hashlib.sha256((ROOT/'fixed_branch_details.json').read_bytes()).hexdigest()))
    write('evidence_terms.json',dict(objects=state_rows,
        note='Removing a loss term is a diagnostic algebraic counterfactual, not a new trained model or trial. Log evidences are not additive after particle marginalization.',
        repeated_observations_are_not_independent=True))
    write('opportunity_summary.json',summaries)
    print(json.dumps(dict(opportunities=summaries,swapped_prior_winner_changes=sum(x['swapped_changes_winner']for x in result),
        maximum_possible_finite_prior_bound=max(x['max_finite_prior_bound_margin_ratio']for x in result),
        rows=[{k:v for k,v in x.items()if k!='competitor_bounds'}for x in result]),indent=2))

if __name__=='__main__':main()
