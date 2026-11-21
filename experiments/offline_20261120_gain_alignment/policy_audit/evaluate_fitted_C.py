#!/usr/bin/env python3
"""Independent binding of C predictions to all11 retained actual L54 branches."""
import argparse,importlib.util,json,hashlib,csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;B=HERE.parent;F=B/'execution_audit/acquisition_forecast';P=F.parent/'acquisition_prefix/shelf_left_54_attempt_001'
s=importlib.util.spec_from_file_location('prior_alignment',HERE/'diagnose_alignment.py');D=importlib.util.module_from_spec(s);s.loader.exec_module(D)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resource(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rho(x,y):return float(spearmanr(x,y).statistic)if np.ptp(x)>1e-12 and np.ptp(y)>1e-12 else None

def main(path,output):
 data=read(path/'result.json');assert data['status']=='completed_fitted_acquisition_prediction_all_original_candidates'
 assert data['hypothesis_candidate_pairs']==44 and data['original_candidate_count']==11
 assert data['new_acquisition_actions']==0 and data['current_native_and_past_unchanged']
 protocol=read(F/'protocol.json');assert data['protocol']==resource(F/'protocol.json')
 for p,h in protocol['files_sha256'].items():assert sha(p)==h,p
 actual=[r for r in D.observed_rows()if r['layout']=='shelf_left'and r['state_action']==54];assert len(actual)==22
 actualmap={(r['arm'],r['candidate_id']):r for r in actual};routes={x['candidate_id']:x for x in read(P/'fixed_candidates.json')}
 hypotheses=read(P/'conditional_hypotheses.json')['objects'];prefix=read(P/'receipt.json');hyper=[];allfiles={}
 for res in data['hypothesis_outputs']:
  assert sha(res['path'])==res['sha256'];h=read(res['path']);allfiles[res['path']]=res['sha256']
  obj=hypotheses[h['object_index']];hi=h['shape_id']
  for a in ('G','S'):assert h['shape_probability_mass'][a]==obj['shape_probability_mass'][a][hi]
  rr=h['arrays'];assert sha(rr['path'])==rr['sha256'];allfiles[rr['path']]=rr['sha256']
  with np.load(rr['path'],allow_pickle=False)as z:
   assert len(h['rows'])==11 and {r['candidate_id']for r in h['rows']}==set(routes)
   for r in h['rows']:
    cid=r['candidate_id'];gain=z[f'candidate_{cid}_forward_gain'];loss=z[f'candidate_{cid}_forward_loss']
    assert np.all(gain>=0)and np.all(loss>=0)
    assert np.all(gain[z['strict_free']]==0)and np.all(gain[~z['native_keep']]==0)
    assert abs(gain.sum()-r['predicted_gross_new_m2'])<1e-12
    assert abs(loss.sum()-r['predicted_lost_m2'])<1e-12
    assert abs(gain.sum()-loss.sum()-r['predicted_net_new_m2'])<1e-12
    assert abs(z[f'candidate_{cid}_fine_original'].sum()-r['matched_fine_original_visibility_m2'])<1e-12
    assert abs(z[f'candidate_{cid}_fine_native'].sum()-r['matched_fine_native_visibility_m2'])<1e-12
    assert r['forecast_receipt']['before_vertices_sha256']==prefix['actual_native_replay']['vertices_sha256']
    assert r['forecast_receipt']['before_triangles_sha256']==prefix['actual_native_replay']['triangles_sha256']
  hyper.append(h)
 for obj in hypotheses:
  for arm in ('G','S'):assert abs(sum(obj['shape_probability_mass'][arm])-1)<1e-12
 rows=[];groups={}
 for arm in ('G','S'):
  for mode in ('original_T16','fine_original','fine_native','forward'):
   if mode=='original_T16':rr=[dict(candidate_id=r['candidate_id'],template_gain_m2=r['template_gain_m2'],score=r['total_score'],common_total_proxy=r['common_score'],cost=r['score_route_cost'])for r in actual if r['arm']==arm]
   else:rr=data['arms'][arm]['rows']if mode=='forward'else data['comparators'][mode][arm]['rows']
   assert len(rr)==11;winner=min(rr,key=lambda r:(-r['score'],r['cost'],r['candidate_id']))['candidate_id']
   if mode!='original_T16':assert winner==(data['arms'][arm]if mode=='forward'else data['comparators'][mode][arm])['selected_candidate']
   group=[]
   for r in rr:
    cid=r['candidate_id'];a=actualmap[arm,cid];route=routes[cid]
    assert abs(r['common_total_proxy']-a['common_score'])<1e-12 and r['cost']==a['score_route_cost']
    assert abs(r['common_total_proxy']+r['template_gain_m2']-r['score'])<1e-12
    if mode!='original_T16':
     key={'forward':'predicted_gross_new_m2','fine_original':'matched_fine_original_visibility_m2','fine_native':'matched_fine_native_visibility_m2'}[mode]
     val=sum(h['shape_probability_mass'][arm]*next(x for x in h['rows']if x['candidate_id']==cid)[key]for h in hyper)
     assert abs(val-r['template_gain_m2'])<1e-12
    x=dict(mode=mode,arm=arm,candidate_id=cid,selected=cid==winner,template_gain_m2=r['template_gain_m2'],score=r['score'],common_score=a['common_score'],cost=r['cost'])
    for key in ('object_gross_m2','object_lost_m2','object_net_m2','global_gross_m2','global_lost_m2','global_net_m2','coverage_gain_pp','outbound_actions','outbound_path_m','endpoint_reached','collisions','unique_actual_branch'):x[key]=a[key]
    if mode=='forward':x['forecast_lost_m2']=r['predicted_lost_m2'];x['forecast_net_m2']=r['predicted_net_new_m2']
    else:x['forecast_lost_m2']=None;x['forecast_net_m2']=None
    group.append(x);rows.append(x)
   selected=next(x for x in group if x['selected']);g=dict(selected_candidate=winner,selected_actual=selected,candidates=11)
   for metric in ('object_gross_m2','object_net_m2','global_gross_m2','global_net_m2','coverage_gain_pp'):
    g['template_rho_vs_'+metric]=rho([x['template_gain_m2']for x in group],[x[metric]for x in group]);g['selected_regret_'+metric]=max(x[metric]for x in group)-selected[metric]
   for metric in ('object_gross_m2','object_net_m2'):
    err=np.array([x['template_gain_m2']-x[metric]for x in group]);g['MAE_vs_'+metric]=float(abs(err).mean());g['bias_vs_'+metric]=float(err.mean())
   if mode=='forward':
    err=np.array([x['forecast_net_m2']-x['object_net_m2']for x in group]);g['forecast_net_MAE_vs_object_net_m2']=float(abs(err).mean())
   groups[mode+'_'+arm]=g
 output.mkdir(exist_ok=False)
 with (output/'all_candidates.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result=dict(status='complete_independent_fitted_forward_evaluation',layout='shelf_left',state_action=54,independent_prefixes=1,reused_actual_branches=11,unique_forecasts=44,comparison_rows=len(rows),groups=groups,inputs=dict(prediction=resource(path/'result.json'),protocol=resource(F/'protocol.json'),prefix=resource(P/'receipt.json'),actual_summary=resource(D.B9/'structure/replay_audit/summary.json')),actual_execution_bindings=D.INPUTS,prediction_bindings=allfiles,code=resource(__file__),limits=['Single development prefix,all11 original candidates retained;actualoutcomes reused,not11 independent scenes.','Forecast sums mutually exclusive per-structure conditional-mean own-surface increments;absolute comparison target is objectgross,notglobal area.','Original T16 versus fine controls also changes geometry integration/basis;forward versus fine controls isolates acquired-fusion forecast oncommon geometry/basis.','Gross scoring unchanged;forecastnet separately reported,notnewscoreterm.','Unmodeled occluders and shape errors remain;no fullC policy executed.'])
 write(output/'summary.json',result);print(json.dumps(dict(status=result['status'],summary=resource(output/'summary.json'),groups=groups)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prediction',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.prediction,a.output)
