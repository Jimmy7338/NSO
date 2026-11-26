#!/usr/bin/env python3
"""Evaluate frozen prediction/actual alignment; no actor, score or route changes."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent
B9=HERE.parents[1]/'dual_template_followup_20261120'
M=B9/'semantic/mechanism_v2_meshfix'
INPUTS={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);INPUTS[str(p)]=sha(p);return json.loads(p.read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvwrite(p,rows):
 with Path(p).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rho(x,y):return float(spearmanr(x,y).statistic)if len(x)>1 and np.ptp(x)>1e-12 and np.ptp(y)>1e-12 else None
METRICS=['object_gross_m2','object_net_m2','global_gross_m2','global_net_m2','coverage_gain_pp']

def observed_rows():
 d=read(B9/'structure/replay_audit/summary.json');out=[]
 for state in d['states']:
  for arm,a in state['arms'].items():
   for x in a['all_candidates']:
    v=x['actual']
    execution=read(x['execution_receipt']['path'])
    assert INPUTS[x['execution_receipt']['path']]==x['execution_receipt']['sha256']
    # These 66 branches happen to have no false-occupied reference-free cells.
    # Verify before calling their recorded known-coverage delta correct-free.
    for endpoint in ('initial_coverage','final_coverage'):
     c=execution[endpoint]
     assert c['false_occupied_cells']==0
     assert c['coverage_fraction']==c['free_correct_fraction']
    out.append(dict(study='six_prefixes',layout=state['layout'],state_action=state['state_action'],temperature=16,cue='acquired',arm=arm,candidate_id=x['candidate_id'],selected=x['candidate_id']==a['new_selected_candidate'],template_gain_m2=x['new_template_gain_m2'],common_score=x['common_score'],total_score=x['new_total_score'],score_route_cost=x['cost'],outbound_actions=v['completed_actions'],outbound_path_m=v['path_length_m'],object_gross_m2=v['actual_object_gross_new_m2'],object_lost_m2=v['actual_object_lost_m2'],object_net_m2=v['actual_object_net_new_m2'],global_gross_m2=v['actual_global_gross_new_m2'],global_lost_m2=v['actual_global_lost_m2'],global_net_m2=v['actual_global_net_new_m2'],coverage_gain_pp=v['coverage_delta_percentage_points'],endpoint_reached=v['endpoint_reached'],collisions=v['simulated_collisions'],unique_actual_branch=f"{state['layout']}_{state['state_action']}_{x['candidate_id']}"))
 assert len(out)==132 and len({x['unique_actual_branch']for x in out})==66
 return out

def mechanism_rows():
 d=read(M/'audit_attempt_001/summary.json');branch={(x['layout'],x['candidate_id']):x for x in d['branches']};out=[]
 for c in d['conditions']:
  lay,T,cue=c['layout'],c['temperature'],c['cue'];p=read(M/'candidate_attempt_002'/lay/f'T{T}_{cue}_predictions.json')
  for arm in('G','S'):
   for x in p['candidates']:
    a=branch[lay,x['candidate_id']];f=a['surface']['all']
    out.append(dict(study='mechanism',layout=lay,state_action=0,temperature=T,cue=cue,arm=arm,candidate_id=x['candidate_id'],selected=x['candidate_id']==p[arm+'_winner'],template_gain_m2=x[arm+'_gain'],common_score=x['common'],total_score=x[arm+'_total'],score_route_cost=x['cost'],outbound_actions=a['completed_actions'],outbound_path_m=a['path_length_m'],object_gross_m2=a['object_new_area_m2'],object_lost_m2=a['object_new_area_m2']-a['object_net_area_m2'],object_net_m2=a['object_net_area_m2'],global_gross_m2=f['new_correct_area_m2'],global_lost_m2=f['lost_correct_area_m2'],global_net_m2=f['net_correct_area_m2'],coverage_gain_pp=100*a['correct_free_coverage_increment'],endpoint_reached=a['endpoint_reached'],collisions=a['collisions'],unique_actual_branch=f"{lay}_mechanism_{x['candidate_id']}"))
 assert len(out)==96 and len({x['unique_actual_branch']for x in out})==8
 return out

def summarize(rows):
 keys=['study','layout','state_action','temperature','cue','arm'];groups={}
 for x in rows:
  k=tuple(x[n]for n in keys);groups.setdefault(k,[]).append(x)
 outputs=[]
 for k,xs in groups.items():
  selected=[x for x in xs if x['selected']];assert len(selected)==1;s=selected[0]
  assert s==min(xs,key=lambda x:(-x['total_score'],x['score_route_cost'],x['candidate_id']))
  r=dict(zip(keys,k));r.update(candidates=len(xs),selected_candidate=s['candidate_id'],mean_template_gain_m2=float(np.mean([x['template_gain_m2']for x in xs])),selected_outbound_actions=s['outbound_actions'],selected_path_m=s['outbound_path_m'])
  for target in METRICS:
   y=np.array([x[target]for x in xs]);best=max(xs,key=lambda x:(x[target],-x['outbound_actions'],-x['candidate_id']))
   r['template_rho_vs_'+target]=rho([x['template_gain_m2']for x in xs],y)
   r['total_rho_vs_'+target]=rho([x['total_score']for x in xs],y)
   r['selected_'+target]=s[target];r['oracle_'+target]=best[target];r['oracle_candidate_'+target]=best['candidate_id'];r['selected_regret_'+target]=best[target]-s[target]
  for target in ('object_gross_m2','object_net_m2'):
   e=np.array([x['template_gain_m2']-x[target]for x in xs]);r['prediction_bias_vs_'+target]=float(e.mean());r['prediction_MAE_vs_'+target]=float(abs(e).mean())
  # Descriptive Pareto check only, not a new policy filter or scalar objective.
  dims=['object_net_m2','global_net_m2','coverage_gain_pp'];dom=[]
  for x in xs:
   if x is s:continue
   weak=all(x[t]>=s[t]-1e-10 for t in dims) and x['outbound_actions']<=s['outbound_actions'] and x['outbound_path_m']<=s['outbound_path_m']+1e-10
   strict=any(x[t]>s[t]+1e-10 for t in dims) or x['outbound_actions']<s['outbound_actions'] or x['outbound_path_m']<s['outbound_path_m']-1e-10
   if weak and strict:dom.append(x['candidate_id'])
  r['selected_pareto_dominated_by']=';'.join(map(str,dom));outputs.append(r)
 return outputs

def main():
 rows=observed_rows()+mechanism_rows()
 for r in rows:
  assert abs(r['object_gross_m2']-r['object_lost_m2']-r['object_net_m2'])<1e-8
  assert abs(r['global_gross_m2']-r['global_lost_m2']-r['global_net_m2'])<1e-8
  r['pred_minus_object_gross_m2']=r['template_gain_m2']-r['object_gross_m2'];r['pred_minus_object_net_m2']=r['template_gain_m2']-r['object_net_m2']
 groups=summarize(rows);csvwrite(HERE/'all_candidate_alignment.csv',rows);csvwrite(HERE/'state_condition_summary.csv',groups)
 agg={}
 for arm in('G','S'):
  a=[r for r in groups if r['study']=='six_prefixes'and r['arm']==arm]
  metrics={key:float(np.mean([r[key]for r in a if r[key] is not None]))for key in a[0]if key.startswith(('mean_','template_rho_','total_rho_','selected_regret_','prediction_')) and any(r[key]is not None for r in a)}
  agg[arm]=dict(equal_weight_states=len(a),**metrics,dominated_selected_states=sum(bool(r['selected_pareto_dominated_by'])for r in a))
 output=dict(status='complete_fixed_output_metric_alignment',fixed_prefixes=6,fixed_predictions=132,fixed_actual_branches=66,mechanism_predictions_including_duplicate_G=96,mechanism_conditions=12,mechanism_actual_branches=8,no_new_sensor_actions=True,aggregate_fixed_states=agg,all_state_conditions=groups,metric_definitions={'template_gain':'Expected unconsumed hypothetical template area under alternative masses and original visibility, not unique physical source area.','object_gross':'Unique evaluator object source area within5cm after branch but not before.','object_net':'Object gross minus source area that loses5cm support after native TSDF integration.','global_gross_net':'Same definitions over all reference geometry, not just modeled objects.','error':'Template minus actual object area is a surrogate discrepancy, not a statistically calibrated predictive error. Total score includes coverage/quality and is not compared in absolute square metres to global surface area.','regret':'Per-metric best fixed candidate minus chosen; evaluator-only hindsight, no multiobjective optimum or policy-change claim.','repetition':'66 and8 actual branches reused across matched modes/cues; rows are dependent. Fixed summaries average equally over6states; mechanism conditions separately retained, not pooled as independent.'},inputs_sha256=INPUTS,script_sha256=sha(__file__))
 write(HERE/'summary.json',output)
 print(json.dumps(agg,indent=2))
if __name__=='__main__':main()
