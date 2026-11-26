"""Read-only endpoint/ranking audit of the one predeclared Q1 comparison."""
import json,hashlib,math
from pathlib import Path
from fractions import Fraction as F
D=Path(__file__).resolve().parent;E=D/'execution'
rows=json.loads((E/'results.json').read_text());assert [r['condition'] for r in rows]==['G','S','X']
assert len({r['initial_packet_sha256'] for r in rows})==1
steps={m:json.loads((E/f'Q1_shelf_west_96_{m}_steps.json').read_text()) for m in ['G','S','X']}
choices={m:json.loads((E/f'Q1_shelf_west_96_{m}_choices.json').read_text()) for m in ['G','S','X']}

def winner(o,m):
 r=o['candidates'];v=o['scores'][m]
 if not r:return None
 i=min(range(len(r)),key=lambda i:(-v[i],r[i]['cost'],r[i]['candidate_id']))
 return None if v[i]<=0 else i

def scalar_interval(o):
 routes=o['candidates'];audit=o['score_audit'];i=winner(o,'S');low=F(0);high=None;li=True;hi=False;impossible=False
 def add(a,b,strict):
  nonlocal low,high,li,hi,impossible
  if b==0:
   if a<0 or (a==0 and strict):impossible=True
   return
  bound=-a/b
  if b>0:
   if bound>low:low=bound;li=not strict
   elif bound==low:li=li and not strict
  else:
   if high is None or bound<high:high=bound;hi=not strict
   elif bound==high:hi=hi and not strict
 terms=[(F.from_float(float(a['common_total_proxy'])),F.from_float(float(a['prior_potentials']['G']))) for a in audit]
 if i is None:
  for a,b in terms:add(-a,-b,False)
 else:
  a,b=terms[i];add(a,b,True)
  for j,(c,d) in enumerate(terms):
   if j!=i:add(a-c,b-d,(routes[i]['cost'],routes[i]['candidate_id'],i)>(routes[j]['cost'],routes[j]['candidate_id'],j))
 empty=impossible or high is not None and (low>high or low==high and not(li and hi))
 return dict(nonnegative_uniform_generic_weight_exists=not empty,lower=str(low),upper=None if high is None else str(high),lower_inclusive=li,upper_inclusive=hi)

state_audits=[]
for mode,calls in choices.items():
 for n,c in enumerate(calls):
  o=c['outputs'];routes=o['candidates'];score=o['scores'];audit=o['score_audit']
  assert all(math.isclose(a['common_total_proxy']+a['prior_potentials']['G'],score['G'][i],abs_tol=1e-9,rel_tol=1e-12) for i,a in enumerate(audit))
  selected=winner(o,mode);assert (None if selected is None else routes[selected]['candidate_id'])==(o.get('selected') or {}).get('candidate_id')
  winners={m:None if winner(o,m) is None else routes[winner(o,m)]['candidate_id'] for m in ['G','S','X']}
  paid=96-c['inputs']['remaining_budget'];surfacecols=[float(a['prior_potentials']['G']) for a in audit];semanticcols=[float(a['prior_potentials']['S']) for a in audit]
  denominator=sum(v*v for v in surfacecols);scale=sum(a*b for a,b in zip(surfacecols,semanticcols))/denominator if denominator else None
  residual=max((abs(b-scale*a) for a,b in zip(surfacecols,semanticcols)),default=0) if scale is not None else max(map(abs,semanticcols),default=0)
  state_audits.append(dict(trajectory_mode=mode,planning_call=n+1,paid_actions=paid,candidate_count=len(routes),same_state_top1=winners,G_S_scores_equal=score['G']==score['S'],S_X_scores_equal=score['S']==score['X'],scalar_selection=scalar_interval(o),best_uniform_scale_to_match_prior_scores=scale,max_prior_score_fit_residual=residual,coverage_before=rows[0]['before_metrics']['coverage_2d'] if paid==0 else steps[mode][paid-1]['covered_area_m2']/rows[0]['after_metrics']['covered_area_m2']*rows[0]['after_metrics']['coverage_2d']))

def divergence(a,b):
 first=next((i for i,(x,y) in enumerate(zip(steps[a],steps[b]),1) if x['action']!=y['action']),None)
 return dict(first_different_action=first,all_steps_exact=steps[a]==steps[b],shared_action_prefix=(min(len(steps[a]),len(steps[b])) if first is None else first-1))
endpoints=[dict(mode=r['condition'],steps=r['steps'],collisions=r['collisions'],full_anchor_return=r['final_pose']==list(r['anchor']),termination=r['termination'],initial_coverage=r['before_metrics']['coverage_2d'],final_coverage=r['after_metrics']['coverage_2d'],coverage_gain_m2=r['coverage_gain_m2'],new_reference_visible_surface_m2=r['new_surface_m2'],final_F1_5cm=r['after_metrics']['f1_05cm'],final_F1_10cm=r['after_metrics']['f1_10cm']) for r in rows]
by={r['condition']:r for r in rows};deltas={}
for other in ['G','X']:
 s=by['S'];g=by[other];deltas['S_minus_'+other]=dict(new_reference_visible_surface_m2=s['new_surface_m2']-g['new_surface_m2'],coverage_m2=s['after_metrics']['covered_area_m2']-g['after_metrics']['covered_area_m2'],F1_5cm=s['after_metrics']['f1_05cm']-g['after_metrics']['f1_05cm'])
report=dict(schema='nso.rapid.Q1.semantic_comparison.v1',status='completed',protocol_sha256=hashlib.sha256((E/'protocol.json').read_bytes()).hexdigest(),initial_packet_equal=True,endpoints=endpoints,deltas=deltas,trajectory_divergence={a+'_vs_'+b:divergence(a,b) for a,b in [('S','G'),('S','X')]},same_state_counts=dict(total=len(state_audits),S_vs_G_different_top1=sum(r['same_state_top1']['S']!=r['same_state_top1']['G'] for r in state_audits),S_vs_X_different_top1=sum(r['same_state_top1']['S']!=r['same_state_top1']['X'] for r in state_audits),S_top1_not_reproducible_by_any_nonnegative_generic_weight=sum(not r['scalar_selection']['nonnegative_uniform_generic_weight_exists'] for r in state_audits)),state_audits=state_audits,source_unchanged_at_finish=json.load(open(E/'heartbeat.json'))['status']=='completed',limits=['Single prospectively specified synthetic Q1 parent, no detector or physical efficacy validation.','G/S prior-score changes and trajectory differences do not alone exclude geometric weighting explanations.','State counts are correlated planning states, not independent samples.','Uniform generic-weight selection constraints use exact rationals of saved binary-float coefficients; boundary floating addition caveat retained.','Coverage_before reconstructed from recorded area/constant reachable denominator; not physical area.'])
(D/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['state_audits','limits']},ensure_ascii=False,indent=2))
# One common uniform-weight intersection across all actual S planning states.
actual_S=[a for a in state_audits if a['trajectory_mode']=='S'];low=F(0);high=None;li=True;hi=False
for a in actual_S:
 z=a['scalar_selection'];lo=F(z['lower']);up=None if z['upper'] is None else F(z['upper'])
 if lo>low:low=lo;li=z['lower_inclusive']
 elif lo==low:li=li and z['lower_inclusive']
 if up is not None:
  if high is None or up<high:high=up;hi=z['upper_inclusive']
  elif up==high:hi=hi and z['upper_inclusive']
report['actual_S_common_uniform_generic_weight_interval']=dict(lower_fraction=str(low),lower=float(low),lower_inclusive=li,upper_fraction=None if high is None else str(high),upper=None if high is None else float(high),upper_inclusive=hi,scope='Recorded actualS states only; no added weighted rollout or parameter search. A positive interval prevents a claim of category-specific advantage from this comparison.')
report['G_X_steps_exact']=steps['G']==steps['X']
# Verify complete shared state at the first differing native planning choice.
shared=[]
for i,(a,b) in enumerate(zip(choices['G'],choices['S']),1):
 oa,ob=a['outputs'],b['outputs']
 if oa.get('selected')!=ob.get('selected'):
  # option_id includes mode/run identity; compare candidate IDs for selection.
  ga=(oa.get('selected')or{}).get('candidate_id');sb=(ob.get('selected')or{}).get('candidate_id')
  if ga!=sb:
   paid=96-a['inputs']['remaining_budget'];shared=dict(planning_call=i,paid_actions=paid,inputs_exact=a['inputs']==b['inputs'],candidates_exact=oa['candidates']==ob['candidates'],candidate_audit_exact=oa['candidate_audit']==ob['candidate_audit'],score_audit_exact=oa['score_audit']==ob['score_audit'],earlier_steps_exact=steps['G'][:paid]==steps['S'][:paid]);break
report['first_planning_divergence_shared_state']=shared
report['interpretation']='Positive within-parent synthetic allocation result versus nativeG/X; common genericweight feasibility remains, so category-specific advantage, natural semantic recognition, physical benefit and generalization are not established.'
(D/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('common generic weight lower',float(low),'G/X exact',report['G_X_steps_exact'],'shared',shared)
