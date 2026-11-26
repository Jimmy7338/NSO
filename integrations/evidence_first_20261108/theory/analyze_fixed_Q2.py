"""Read saved ten-run records only: primary coverage, ledger, exact affine controls."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,math,csv
D=Path(__file__).resolve().parent/'Q2_execution'
load=lambda p:json.loads(Path(p).read_text())
hashjson=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
class Interval:
 def __init__(self):self.lo=F(0);self.hi=None;self.li=True;self.ui=False;self.impossible=False
 def add(self,a,b,strict=False):
  if b==0:
   if a<0 or a==0 and strict:self.impossible=True
   return
  z=-a/b
  if b>0:
   if z>self.lo:self.lo=z;self.li=not strict
   elif z==self.lo:self.li=self.li and not strict
  else:
   if self.hi is None or z<self.hi:self.hi=z;self.ui=not strict
   elif z==self.hi:self.ui=self.ui and not strict
 def merge(self,r):
  self.add(-r.lo,F(1),not r.li)
  if r.hi is not None:self.add(r.hi,F(-1),not r.ui)
  self.impossible|=r.impossible
 def empty(self):return self.impossible or self.hi is not None and (self.lo>self.hi or self.lo==self.hi and not(self.li and self.ui))
 def out(self):return dict(empty=self.empty(),lower=float(self.lo),upper=None if self.hi is None else float(self.hi),lower_fraction=str(self.lo),upper_fraction=None if self.hi is None else str(self.hi),lower_inclusive=self.li,upper_inclusive=self.ui)
def winner(o,mode):
 r=o['candidates'];scores=o['scores'][mode]
 if not r:return None
 i=min(range(len(r)),key=lambda i:(-scores[i],r[i]['cost'],r[i]['candidate_id']))
 return i if scores[i]>0 else None
def intervals(o):
 routes=o['candidates'];terms=[(F.from_float(float(a['common_total_proxy'])),F.from_float(float(a['prior_potentials']['G']))) for a in o['score_audit']];j=winner(o,'S');z=Interval()
 if j is None:
  for a,b in terms:z.add(-a,-b)
 else:
  a,b=terms[j];z.add(a,b,True)
  for k,(c,d) in enumerate(terms):
   if k!=j:z.add(a-c,b-d,(routes[j]['cost'],routes[j]['candidate_id'],j)>(routes[k]['cost'],routes[k]['candidate_id'],k))
 ranking=Interval();ranking.merge(z)
 order=sorted(range(len(routes)),key=lambda i:(-o['scores']['S'][i],routes[i]['cost'],routes[i]['candidate_id'],i))
 for j,k in zip(order,order[1:]):
  a,b=terms[j];c,d=terms[k];ranking.add(a-c,b-d,(routes[j]['cost'],routes[j]['candidate_id'],j)>(routes[k]['cost'],routes[k]['candidate_id'],k))
 return z,ranking
protocol=load(D/'protocol.json');rows=load(D/'results.json');assert len(rows)==10
assert [(r['arrangement'],r['condition']) for r in rows]==[tuple(v) for v in protocol['run_order']]
summary=[];curve=[];states=[];per_S={a:Interval() for a in ['shelf_west','shelf_east']};all_S=Interval();max_midpoint=0.;control_checks=0;shared_hashes={};data={}
for r in rows:
 arr,mode=r['arrangement'],r['condition'];key='Q2_'+arr+'_96_'+mode
 if r['status']!='completed':summary.append(dict(arrangement=arr,condition=mode,status=r['status'],error=r.get('error')));continue
 steps=load(D/(key+'_steps.json'));calls=load(D/(key+'_choices.json'));ledger=load(D/(key+'_ledger.json'))
 assert len(steps)==r['steps']==ledger['snapshot']['paid_actions']
 assert ledger['snapshot']['total_budget']==96
 observed_ids=[e.get('action_id') for e in ledger['events'] if e.get('event_type')=='paid_observation'];assert observed_ids==list(range(1,r['steps']+1))
 shared_hashes.setdefault(arr,set()).add(r['initial_packet_sha256']);data[(arr,mode)]=(steps,calls)
 values=[r['before_metrics']['coverage_2d']]+[s['coverage_fraction'] for s in steps];values+=([values[-1]]*(97-len(values)));assert len(values)==97
 auc=sum((a+b)/2 for a,b in zip(values,values[1:]))/96
 surface=[0.]+[s['new_visible_surface_area_m2'] for s in steps];surface+=([surface[-1]]*(97-len(surface)))
 for i,(a,b) in enumerate(zip(values,surface)):curve.append(dict(arrangement=arr,condition=mode,paid_action=i,coverage_fraction=a,new_reference_visible_surface_m2=b,actual_observation=i<=r['steps']))
 returned=r['final_pose']==r['anchor'];success=returned and not r['termination']['failed'] and r['collisions']==0
 path=0.;prev=r['anchor'][:2]
 for s in steps:path+=math.dist(prev,s['position'])*.2;prev=s['position']
 summary.append(dict(arrangement=arr,condition=mode,status='completed',paid_actions=r['steps'],observations=r['steps']+1,full_anchor_return=returned,method_failed=r['termination']['failed'],termination_reason=r['termination']['reason'],collisions=r['collisions'],valid_for_efficiency_claim=success,initial_coverage=r['before_metrics']['coverage_2d'],final_coverage=r['after_metrics']['coverage_2d'],coverage_AUC_over96=auc,coverage_at32=values[32],coverage_at64=values[64],coverage_at96=values[96],first90=next((i for i,v in enumerate(values[:r['steps']+1]) if v>=.90),None),first95=next((i for i,v in enumerate(values[:r['steps']+1]) if v>=.95),None),first99=next((i for i,v in enumerate(values[:r['steps']+1]) if v>=.99),None),coverage_gain_m2=r['coverage_gain_m2'],new_reference_visible_surface_m2=r['new_surface_m2'],F1_5cm=r['after_metrics']['f1_05cm'],F1_10cm=r['after_metrics']['f1_10cm'],discrete_translation_m=path,action_counts={a:sum(s['action']==a for s in steps) for a in ['forward','left','right']},planning_calls=len(calls)))
 for n,c in enumerate(calls,1):
  o=c['outputs'];assert hashjson(o)==c['output_sha256'];scores=o['scores'];routes=o['candidates'];audit=o['score_audit']
  for i,a in enumerate(audit):
   assert math.isclose(a['common_total_proxy']+a['prior_potentials']['G'],scores['G'][i],abs_tol=1e-9,rel_tol=1e-12)
   max_midpoint=max(max_midpoint,abs(scores['S'][i]+scores['X'][i]-2*scores['G'][i]))
   if mode=='G_lambda_dev7':assert scores['G_lambda_dev7'][i]==float(a['common_total_proxy']+7*a['prior_potentials']['G'])
  if mode=='G_lambda_dev7':assert o['mode']=='G_lambda_dev7' and o['comparator']['native_score_columns_unchanged'];control_checks+=1
  i=winner(o,mode);assert (None if i is None else routes[i]['candidate_id'])==(o.get('selected')or{}).get('candidate_id')
  z,rank=intervals(o)
  if mode=='S':per_S[arr].merge(z);all_S.merge(z)
  paid=96-c['inputs']['remaining_budget'];native={m:scores[m] for m in ['C','G','S','X']};state_key=hashjson(dict(inputs=c['inputs'],audit=o['candidate_audit'],routes=routes,scores=native))
  wins={m:None if winner(o,m) is None else routes[winner(o,m)]['candidate_id'] for m in ['C','G','S','X']}
  states.append(dict(arrangement=arr,trajectory_condition=mode,planning_call=n,paid_actions=paid,coverage_before=values[paid],candidate_count=len(routes),state_key=state_key,native_same_state_top1=wins,S_selected_option_uniform_weight=z.out(),S_full_ranking_uniform_weight=rank.out(),G_S_scores_equal=scores['G']==scores['S']))
assert all(len(h)==1 for h in shared_hashes.values());assert max_midpoint<1e-9
# Preserve actual paired trajectories; no counterfactual rollout is invented from scores.
pairs=[]
for arr in ['shelf_west','shelf_east']:
 for other in ['C','G','X','G_lambda_dev7']:
  if (arr,'S')not in data or(arr,other)not in data:continue
  a,ca=data[(arr,'S')];b,cb=data[(arr,other)];first=next((i for i,(x,y) in enumerate(zip(a,b),1) if x['action']!=y['action']),None);witness=None
  for i,(x,y) in enumerate(zip(ca,cb),1):
   sx=(x['outputs'].get('selected')or{}).get('candidate_id');sy=(y['outputs'].get('selected')or{}).get('candidate_id')
   if sx!=sy:
    paid=96-x['inputs']['remaining_budget'];witness=dict(planning_call=i,paid_actions=paid,coverage_before=([next(r for r in rows if r['arrangement']==arr and r['condition']=='S')['before_metrics']['coverage_2d']]+[s['coverage_fraction'] for s in a])[paid],inputs_exact=x['inputs']==y['inputs'],candidates_exact=x['outputs']['candidates']==y['outputs']['candidates'],candidate_audit_exact=x['outputs']['candidate_audit']==y['outputs']['candidate_audit'],score_audit_exact=x['outputs']['score_audit']==y['outputs']['score_audit'],earlier_steps_exact=a[:paid]==b[:paid]);break
  pairs.append(dict(arrangement=arr,comparison='S_vs_'+other,first_different_action=first,all_steps_exact=a==b,first_planning_divergence=witness))
lookup={(r['arrangement'],r['condition']):r for r in summary};deltas=[]
for arr in ['shelf_west','shelf_east']:
 for other in ['C','G','X','G_lambda_dev7']:
  s=lookup[(arr,'S')];g=lookup[(arr,other)]
  if s['status']!='completed' or g['status']!='completed':continue
  deltas.append(dict(arrangement=arr,comparison='S_minus_'+other,both_returned_without_failure=s['valid_for_efficiency_claim']and g['valid_for_efficiency_claim'],coverage_AUC=s['coverage_AUC_over96']-g['coverage_AUC_over96'],final_coverage=s['final_coverage']-g['final_coverage'],new_reference_visible_surface_m2=s['new_reference_visible_surface_m2']-g['new_reference_visible_surface_m2'],F1_5cm=s['F1_5cm']-g['F1_5cm']))
verification=dict(initial_packet_matches_within_arrangements=True,ledger_counts_and_contiguous_paid_ids_match=True,archived_call_output_hashes_match=True,control_selected_by_exact_common_plus7generic=True,control_planning_calls_checked=control_checks,native_midpoint_max_abs_error=max_midpoint,runner_source_sha256=hashlib.sha256((D/'executed_runner.py').read_bytes()).hexdigest(),sources_frozen_unchanged=load(D/'heartbeat.json')['status']=='completed_all_ten')
assert verification['runner_source_sha256']==protocol['harness_sha256']
report=dict(schema='nso.Q2.ten_run_evidence.v1',status='completed_all_records_audited',statistical_unit='one Q2syntheticparent/two related placements, not ten independent scenes',primary='Unknown coverage and successful full-pose return; surface/F1 secondary',summary=summary,deltas=deltas,pairwise_trajectories=pairs,scalar_equivalence=dict(recorded_planning_calls=len(states),unique_recorded_state_keys=len({s['state_key'] for s in states}),selected_option_unreproducible_calls=sum(s['S_selected_option_uniform_weight']['empty'] for s in states),full_ranking_unreproducible_calls=sum(s['S_full_ranking_uniform_weight']['empty'] for s in states),actual_S_common_intervals={a:z.out() for a,z in per_S.items()},all_actual_S_common_interval=all_S.out(),boundary='Exact rational inequalities on savedbinaryfloat coefficients; floataddition boundary caveat. Counterfactual savedstate choices are not extra trajectories or independent samples.'),verification=verification,states=states,limits=['All10fixedconditions retained; no outcome-based rerun or selection.','Known truth is used only by evaluation; syntheticmarkerlabels and exact simulated poses are not natural perception evidence.','Originalsimradius.20m differs from physical.7212186m; no physical configuration changed.','Positive secondarymetrics or late differences do not establish earlyunknowncoverage efficiency.','G_lambda_dev7 is an explicitly named comparison, not a native NSOmode or policydeployment.'])
report['scalar_equivalence'].update(
 actual_S_unreproducible_selected_option_calls=sum(s['trajectory_condition']=='S' and s['S_selected_option_uniform_weight']['empty'] for s in states),
 counterfactual_S_on_other_trajectory_unreproducible_calls=sum(s['trajectory_condition']!='S' and s['S_selected_option_uniform_weight']['empty'] for s in states),
 unique_unreproducible_selected_option_state_keys=len({s['state_key'] for s in states if s['S_selected_option_uniform_weight']['empty']}))
(D/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with open(D/'coverage_curves.csv','w')as f:
 w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
with open(D/'summary.csv','w')as f:
 flat=[{k:v for k,v in r.items()if not isinstance(v,dict)}for r in summary];w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print(json.dumps(dict(summary=summary,deltas=deltas,scalar_equivalence=report['scalar_equivalence'],verification=verification),ensure_ascii=False,indent=2))
