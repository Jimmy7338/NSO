#!/usr/bin/env python3
"""Read-only B8 semantic transfer and candidate-margin diagnosis.

No actor, policy, geometry fitting or weight change is executed.
"""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parent
B8=ROOT.parents[1]/'dual_template_uncertainty_20261120'
ARMS=('G_fixed','S_fixed','G_uncertainty','S_uncertainty','S_mean_collapse')
STATES={'shelf_left':(24,54,109),'shelf_right':(24,53,110)}
TOL=1e-12;INPUTS={}

def read(path):
    path=Path(path);INPUTS[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
def arrays(path):
    path=Path(path);INPUTS[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return dict(np.load(path,allow_pickle=False))
def write(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(name,rows):
    if rows:
        with(ROOT/name).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(v):return float(np.mean(v))if len(v)else None
def finite(v):return float(v)if np.isfinite(v)else None
def winner(scores,candidates):
    if not candidates:return None
    i=min(range(len(candidates)),key=lambda i:(-scores[i],candidates[i]['cost'],candidates[i]['candidate_id']))
    return i if scores[i]>0 else None

def object_diagnostic(g,s,common):
    assert g['support_track']==s['support_track']
    e=np.asarray(g['energy_m2']);q=np.asarray(g['prior_mass']);pg=np.asarray(g['joint_probability_mass']);ps=np.asarray(s['joint_probability_mass'])
    np.testing.assert_array_equal(e,s['energy_m2']);np.testing.assert_array_equal(q,s['prior_mass'])
    np.testing.assert_array_equal(g['particles'],s['particles'])
    # Compute evidence in log space, including cases whose final mass underflows.
    log_e=logsumexp(np.log(q)[:,None]-e/(2*.01**2),axis=0)
    log_bf=float(log_e[1]-log_e[0]);sg=pg.sum(axis=0);ss=ps.sum(axis=0)
    prior=np.asarray(s['shape_prior']);prior_log_odds=float(np.log(prior[1]/prior[0]))
    expect=1/(1+np.exp(-np.clip(log_bf+prior_log_odds,-700,700)))
    assert abs(expect-ss[1])<1e-10
    cond_tv=[]
    for h in (0,1):
        if sg[h]>1e-280 and ss[h]>1e-280:
            tv=float(.5*np.abs(pg[:,h]/sg[h]-ps[:,h]/ss[h]).sum());assert tv<1e-10;cond_tv.append(tv)
    expected_open=bool(prior[1]>.5) if prior[1]!=.5 else None
    agreement=(bool(sg[1]>.5)==expected_open) if expected_open is not None else None
    saturated=bool(max(sg)>.99)
    return dict(**common,support_track=g['support_track'],observed_points=g['conditional_geometry']['observed_points'],
        acquired_shelf_prior=float(prior[1]),geometry_log_bayes_factor_shelf_vs_box=log_bf,semantic_prior_log_odds=prior_log_odds,
        G_shelf_probability=float(sg[1]),S_shelf_probability=float(ss[1]),posterior_shelf_difference=float(ss[1]-sg[1]),
        joint_probability_total_variation=float(.5*np.abs(pg-ps).sum()),max_within_shape_conditional_geometry_TV=max(cond_tv,default=0.),
        neutral_structure_confidence_gt_0_99=saturated,neutral_agrees_with_acquired_category=agreement,
        saturated_and_agrees=saturated and agreement is True,saturated_and_disagrees=saturated and agreement is False,
        semantic_prior_transmission_verified=True)

def plan_diagnostics(common,candidates,scores,audits,objects):
    G=np.asarray(scores['G'],float);S=np.asarray(scores['S'],float);N=np.asarray(scores['N'],float);C=np.asarray(scores['C'],float)
    assert G.shape==S.shape==N.shape==C.shape==(len(candidates),)
    ga=np.asarray([r['G_area_m2']for r in audits]);sa=np.asarray([r['S_area_m2']for r in audits]);assert len(ga)==len(G)
    np.testing.assert_allclose(G,N+ga,atol=1e-10,rtol=0);np.testing.assert_allclose(S,N+sa,atol=1e-10,rtol=0)
    delta=S-G;gw=winner(G,candidates);sw=winner(S,candidates);rows=[]
    for i,c in enumerate(candidates):
        margin=float(G[gw]-G[i])if gw is not None else None
        advantage=float(delta[i]-delta[gw])if gw is not None else None
        rows.append(dict(**common,candidate_id=c['candidate_id'],cost=c['cost'],C_coverage=float(C[i]),N_common=float(N[i]),Q_quality=float(N[i]-C[i]),
            G_template=float(ga[i]),S_template=float(sa[i]),G_total=float(G[i]),S_total=float(S[i]),semantic_delta=float(delta[i]),
            is_G_winner=gw==i,is_S_winner=sw==i,G_winner_margin=margin,semantic_advantage_vs_G_winner=advantage,
            semantic_advantage_margin_ratio=(advantage/margin if margin is not None and margin>TOL else None),
            residual_S_winner_margin=(margin-advantage if margin is not None else None),
            common_margin=(float(N[gw]-N[i])if gw is not None else None),coverage_margin=(float(C[gw]-C[i])if gw is not None else None),
            quality_margin=(float((N-C)[gw]-(N-C)[i])if gw is not None else None),G_template_margin=(float(ga[gw]-ga[i])if gw is not None else None)))
    competitors=[x for x in rows if not x['is_G_winner'] and gw is not None]
    pressure=[x for x in competitors if x['semantic_advantage_vs_G_winner']>TOL]
    closest=min(competitors,key=lambda x:(x['residual_S_winner_margin'],x['cost'],x['candidate_id']))if competitors else None
    strongest=max(pressure,key=lambda x:x['semantic_advantage_margin_ratio']or 0)if pressure else None
    summary=dict(**common,candidates=len(candidates),objects=len(objects),G_winner=None if gw is None else candidates[gw]['candidate_id'],S_winner=None if sw is None else candidates[sw]['candidate_id'],
        winner_changed=gw!=sw,score_exact_equal=bool(np.array_equal(G,S)),semantic_score_nonzero=bool(np.any(abs(delta)>TOL)),
        max_absolute_semantic_delta=float(np.max(abs(delta)))if len(delta)else 0.,semantic_delta_candidate_range=float(np.ptp(delta))if len(delta)else 0.,
        G_template_candidate_range=float(np.ptp(ga))if len(ga)else 0.,N_common_candidate_range=float(np.ptp(N))if len(N)else 0.,C_coverage_candidate_range=float(np.ptp(C))if len(C)else 0.,
        competitors_with_positive_semantic_pressure=len(pressure),max_semantic_pressure_margin_ratio=None if strongest is None else strongest['semantic_advantage_margin_ratio'],
        closest_competitor=closest,strongest_semantic_pressure=strongest,
        G_template_changes_common_winner=(gw!=winner(N,candidates)),S_template_changes_common_winner=(sw!=winner(N,candidates)))
    return summary,rows

def rollouts():
    plans=[];candidates=[];objects=[]
    for layout in STATES:
        for arm in ARMS:
            base=B8/'rollouts/joint_depth/layouts'/layout/'runs'/arm/'attempt_001'
            result=read(base/'result.json');choices=read(base/'choices.json');calls=read(base/'template_audit.json')
            by={int(c['mapper_frames']):c for c in calls};assert len(by)==len(calls)
            for choice in choices:
                paid=result['paid_action_budget']-choice['inputs']['remaining_budget'];out=choice['outputs'];call=by.get(paid+1)
                if call is None:assert not out['candidates'];continue
                mode=call['model_audits']['G']['extent_mode'];common=dict(layout=layout,arm=arm,mode=mode,paid_action=paid,
                    observed_signature=call['bank_observed_input_sha256'],plan_number=out['plan_number'])
                g,s=call['model_audits']['G'],call['model_audits']['S'];assert len(g['objects'])==len(s['objects'])
                obs=[object_diagnostic(x,y,common)for x,y in zip(g['objects'],s['objects'])];objects+=obs
                row,cs=plan_diagnostics(common,out['candidates'],out['scores'],call['routes'],obs)
                for model,audit in [('G',g),('S',s)]:
                    before=audit['expected_area_before_history_m2'];after=audit['expected_area_after_history_m2']
                    row[model+'_expected_area_before_history_m2']=before;row[model+'_expected_area_after_history_m2']=after
                    row[model+'_remaining_history_fraction']=after/before if before else None
                plans.append(row);candidates+=cs
    return plans,candidates,objects

def fixed_branches():
    object_rows=[];plans=[];candidates=[];details=[];point_audits=[]
    for layout,states in STATES.items():
        for paid in states:
            base=B8/'replay'/f'{layout}_{paid}_attempt_001';routes=read(base/'fixed_candidates.json');costs={r['candidate_id']:r['cost']for r in routes}
            for mode in ('fixed','uncertainty'):
                cap={n:read(base/(n+'_'+mode+'_predictions.json'))for n in ('G','S')}
                a={n:arrays(base/(n+'_'+mode+'_predictions.npz'))for n in ('G','S')}
                np.testing.assert_array_equal(a['G']['G_points'],a['S']['S_points'])
                common=dict(layout=layout,mode=mode,paid_action=paid,arm='paired_'+mode,plan_number=None,observed_signature=cap['G']['common_bank_audit']['sha256'])
                g,s=cap['G']['model_audits']['G']['objects'],cap['S']['model_audits']['S']['objects'];obs=[object_diagnostic(x,y,common)for x,y in zip(g,s)];object_rows+=obs
                scores={n:[r['scores'][n]for r in cap[n]['routes']]for n in ('G','S')}
                scores['N']=[r['common_total_proxy']for r in cap['G']['routes']];scores['C']=[r['coverage_only_proxy']for r in cap['G']['routes']]
                audits=[dict(candidate_id=r['candidate_id'],G_area_m2=r['template_area']['G'],S_area_m2=t['template_area']['S'])for r,t in zip(cap['G']['routes'],cap['S']['routes'])]
                plan,cs=plan_diagnostics(common,routes,scores,audits,obs);plans.append(plan);candidates+=cs
                for oi,(go,so) in enumerate(zip(g,s)):
                    mask=a['G']['G_object_ids']==oi;shape=a['G']['G_shape_ids'];pG=np.asarray(go['shape_probability_mass']);pS=np.asarray(so['shape_probability_mass'])
                    for name in ('G','S'):
                        base_weights=a[name][name+'_base_weights'];history_weights=a[name][name+'_weights']
                        point_audits.append(dict(**common,support_track=go['support_track'],name=name,
                            area_before_history_m2=float(base_weights[mask].sum()),area_after_history_m2=float(history_weights[mask].sum()),
                            history_kept_fraction=float(history_weights[mask].sum()/base_weights[mask].sum())if base_weights[mask].sum()else None,
                            note='Combined measured/strict-free history mask; this stored mask cannot separate the two causes.'))
                    for j,r in enumerate(routes):
                        cid=r['candidate_id'];ag=a['G'][f'G_gain_{cid}'];ass=a['S'][f'S_gain_{cid}'];conditional=[]
                        for h in (0,1):
                            part=mask&(shape==h);mass=pG[h]
                            value=float(ag[part].sum()/mass)if mass>1e-280 else(float(ass[part].sum()/pS[h])if pS[h]>1e-280 else None)
                            if mass>1e-280 and pS[h]>1e-280:assert np.isclose(value,ass[part].sum()/pS[h],atol=1e-8,rtol=1e-8)
                            conditional.append(value)
                        observed=float(ass[mask].sum()-ag[mask].sum());predicted=(pS[1]-pG[1])*(conditional[1]-conditional[0])if all(x is not None for x in conditional)else 0.
                        assert np.isclose(observed,predicted,atol=1e-10,rtol=1e-8)
                        details.append(dict(**common,candidate_id=cid,support_track=go['support_track'],G_shelf_probability=float(pG[1]),S_shelf_probability=float(pS[1]),
                            posterior_difference=float(pS[1]-pG[1]),box_conditional_route_gain_m2=conditional[0],shelf_conditional_route_gain_m2=conditional[1],
                            conditional_gain_difference_m2=conditional[1]-conditional[0]if all(x is not None for x in conditional)else None,
                            observed_semantic_gain_delta_m2=observed,reconstructed_semantic_gain_delta_m2=float(predicted),decomposition_error_m2=float(observed-predicted)))
    return dict(plans=plans,candidates=candidates,objects=object_rows,per_object_candidate_gain=details,history=point_audits)

def summarize(plans,objects):
    groups={}
    for mode in ('fixed','uncertainty','mean_collapse'):
        ps=[p for p in plans if p['mode']==mode];os=[o for o in objects if o['mode']==mode]
        if not ps:continue
        pressure=[p['max_semantic_pressure_margin_ratio']for p in ps if p['max_semantic_pressure_margin_ratio']is not None]
        groups[mode]=dict(plans=len(ps),object_states=len(os),plans_without_objects=sum(p['objects']==0 for p in ps),plans_with_nonzero_score_difference=sum(p['semantic_score_nonzero']for p in ps),
            changed_winner_plans=sum(p['winner_changed']for p in ps),plans_semantic_delta_constant_across_candidates=sum(p['semantic_delta_candidate_range']<=TOL for p in ps),
            max_semantic_pressure_margin_ratio=max(pressure,default=None),mean_joint_probability_TV=mean([o['joint_probability_total_variation']for o in os]),
            saturated_agree_object_states=sum(o['saturated_and_agrees']for o in os),saturated_disagree_object_states=sum(o['saturated_and_disagrees']for o in os),
            non_saturated_object_states=sum(not o['neutral_structure_confidence_gt_0_99']for o in os),
            N_vs_G_winner_changes=sum(p['G_template_changes_common_winner']for p in ps),N_vs_S_winner_changes=sum(p['S_template_changes_common_winner']for p in ps),
            positive_semantic_pressure_plans=sum(p['competitors_with_positive_semantic_pressure']>0 for p in ps))
    return groups

def main():
    plans,cs,objs=rollouts();fixed=fixed_branches();unique_plans={};unique_objects={}
    for p in plans:unique_plans.setdefault((p['observed_signature'],p['mode']),p)
    for o in objs:unique_objects.setdefault((o['observed_signature'],o['mode'],o['support_track']),o)
    unique=list(unique_plans.values());uo=list(unique_objects.values())
    summary=dict(status='complete_read_only_B8_semantic_diagnosis',all_ten_run_summary=summarize(plans,objs),
        deduplicated_observed_prefix_mode_summary=summarize(unique,uo),fixed_branch_summary=summarize(fixed['plans'],fixed['objects']),
        unique_observed_prefix_mode_count=len(unique),all_recorded_planning_calls=len(plans),
        strongest_semantic_pressure=sorted([p for p in unique if p['max_semantic_pressure_margin_ratio']is not None],key=lambda p:p['max_semantic_pressure_margin_ratio'],reverse=True)[:12],
        exact_pointwise_gain_identity_verified=True,conditional_geometry_distribution_G_S_equal_within_structure=True,
        formula='Delta_gain(route)=sum_objects[(P_S(shelf)-P_G(shelf))*(E[gain|shelf,route]-E[gain|box,route])], for fixed and uncertainty modes.',
        no_new_actor_actions=True,no_scoring_or_weight_changes=True,
        definitions=['0.99 posterior confidence and1e-12 numerical difference are descriptive diagnostics, never new policy gates.',
            'Agreement uses the acquired controlled category input, not a new hidden-geometry query. Disagreement means confident geometric evidence opposes this cue, not necessarily correctly identified geometry.',
            'All10runs and allfixed66branches retained. Deduplication by acquired-evidence signature andmode prevents identical G/S trajectories inflating numerical explanations.',
            'Mean-collapse conditional raw posterior still shares q(k|h), but collapsed scene points may differ with class weights only through numerical geometry; exact pointwise identity is audited on fixed/uncertainty modes.',
            'Combined stored history masks do not identify measured-surface consumption separately from strict-free rejection; no such causal split is inferred here.'])
    write('rollout_plans.json',plans);table('rollout_candidates.csv',cs);table('rollout_objects.csv',objs)
    write('fixed_branch_details.json',fixed);table('fixed_object_candidates.csv',fixed['per_object_candidate_gain']);table('fixed_history.csv',fixed['history']);write('summary.json',summary)
    for path,digest in INPUTS.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    write('input_receipt.json',dict(files_sha256=INPUTS,analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_files_unchanged=True))
    print(json.dumps({k:v for k,v in summary.items()if k not in ('strongest_semantic_pressure','definitions')},indent=2))
if __name__=='__main__':main()
