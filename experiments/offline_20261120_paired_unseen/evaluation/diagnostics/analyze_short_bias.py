"""Posthoc descriptive audit of frozen 24 actual branches, no actor or fitting."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
inputs={}
def read(p):
 p=Path(p);inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 summary=read(ROOT/'evaluation/all_short_attempt_001/table/summary.json');pre=read(ROOT/'performance/observed_prefix_preflight.json')['rows'];rows=[];states=[];parts=[];pairs=[]
 for record in summary['state_rows']:
  l,prefix=record['layout'],record['prefix_id'];path=ROOT/'evaluation/short_results'/f'{l}_{prefix}_attempt_001/result.json';state=read(path);pred=read(state['prediction_binding']['path']);assert inputs[state['prediction_binding']['path']]==state['prediction_binding']['sha256']
  directory=Path(state['prediction_binding']['path']).parent;forecast=read(directory/'hypothesis_forecasts.json');prefix_data=read(directory/'prefix.json')
  # Use pre-outcome preparation records for bank order only; replaced p1
  # states have their own cost-only record. No geometry is reconstructed.
  if prefix_data['prefix_override_applied']:
   ov=read(ROOT/'preparation_v2/prefix_override.json')['overrides'][l][prefix];prep=read(ov['preparation_record']['path']);tracks=prep['selected']['fit_support_tracks']
  else:tracks=next(x['fit_support_tracks']for x in pre if x['layout']==l and x['prefix']==prefix)
  owners=defaultdict(set)
  for frame in prefix_data['owner_receipts']:
   for entry in frame['association']['instances']:
    if entry['assigned_track']>0:owners[entry['assigned_track']].add(entry['instance_index'])
  fs={r['candidate_id']:r for r in forecast['rows']}
  lookup={c:{r['candidate_id']:r for r in pred['conditions'][c]['rows']}for c in pred['conditions']}
  for cid in pred['selected_candidate_ids']:
   a=state['actual_by_candidate'][str(cid)];assert a['status']=='completed'
   for prior in ('G','S'):
    old=lookup['visibility_'+prior][cid]['predicted_gain_m2'];new=lookup['acquisition_'+prior][cid]['predicted_gain_m2'];truth=a['object_gross_new_m2']
    row=dict(layout=l,prefix=prefix,prior=prior,candidate_id=cid,old_prediction_m2=old,new_prediction_m2=new,actual_object_gross_m2=truth,actual_object_lost_m2=a['object_lost_m2'],actual_object_net_m2=a['object_net_new_m2'],old_signed_error_m2=old-truth,new_signed_error_m2=new-truth,old_abs_error_m2=abs(old-truth),new_abs_error_m2=abs(new-truth),abs_error_delta_m2=abs(new-truth)-abs(old-truth),actual_global_gross_m2=a['global_gross_new_m2'],actual_global_net_m2=a['global_net_new_m2'],actual_actions=a['actual_actions'])
    for name in ('open31','closed118'):
     z=a['fixed_reference_increment'][name]
     for metric in ('gross_new','lost','net_new'):row[f'actual_{name}_{metric}_m2']=z[metric+'_area_m2']
    rows.append(row)
    sums=defaultdict(float);loss=defaultdict(float)
    for t in fs[cid]['terms']:
     sums[t['object_index']]+=t['posterior_mass'][prior]*t['own_area_gross_new_m2'];loss[t['object_index']]+=t['posterior_mass'][prior]*t['own_area_lost_m2']
    assert math.isclose(sum(sums.values()),new,abs_tol=1e-10)
    for oi,gross in sums.items():
     track=tracks[oi];owner=sorted(owners[track]);object_key='open31'if owner==[1]else 'closed118'if owner==[2]else 'ambiguous_unassigned'
     parts.append(dict(layout=l,prefix=prefix,prior=prior,candidate_id=cid,model_object_index=oi,observed_support_track=track,observed_owner_indices=owner,source_object_for_diagnostic=object_key,forecast_gross_m2=gross,forecast_lost_m2=loss[oi],forecast_net_m2=gross-loss[oi],association_basis='existing sensor association receipts and pre-outcome bank order, not source fitting'))
  for prior in ('G','S'):
   subset=[r for r in rows if r['layout']==l and r['prefix']==prefix and r['prior']==prior];old=state['condition_summaries']['visibility_'+prior];new=state['condition_summaries']['acquisition_'+prior]
   assert math.isclose(sum(r['old_abs_error_m2']for r in subset)/3,old['MAE_m2'],abs_tol=1e-12);assert math.isclose(sum(r['new_abs_error_m2']for r in subset)/3,new['MAE_m2'],abs_tol=1e-12)
   states.append(dict(layout=l,prefix=prefix,prior=prior,old_MAE_m2=old['MAE_m2'],new_MAE_m2=new['MAE_m2'],MAE_delta_m2=new['MAE_m2']-old['MAE_m2'],old_bias_m2=old['bias_m2'],new_bias_m2=new['bias_m2'],new_overestimates=sum(r['new_signed_error_m2']>0 for r in subset),new_underestimates=sum(r['new_signed_error_m2']<0 for r in subset)))
   oi,ni=old['selected_candidate'],new['selected_candidate'];oa,na=state['actual_by_candidate'][str(oi)],state['actual_by_candidate'][str(ni)]
   pairs.append(dict(layout=l,prefix=prefix,prior=prior,old_selected=oi,new_selected=ni,selected_changed=oi!=ni,old_selected_prediction_m2=lookup['visibility_'+prior][oi]['predicted_gain_m2'],new_selected_prediction_m2=lookup['acquisition_'+prior][ni]['predicted_gain_m2'],actual_gross_delta_m2=na['object_gross_new_m2']-oa['object_gross_new_m2'],actual_loss_delta_m2=na['object_lost_m2']-oa['object_lost_m2'],actual_net_delta_m2=na['object_net_new_m2']-oa['object_net_new_m2'],actual_global_net_delta_m2=na['global_net_new_m2']-oa['global_net_new_m2'],old_object_gross_m2=oa['object_gross_new_m2'],new_object_gross_m2=na['object_gross_new_m2'],old_object_lost_m2=oa['object_lost_m2'],new_object_lost_m2=na['object_lost_m2'],old_object_net_m2=oa['object_net_new_m2'],new_object_net_m2=na['object_net_new_m2']))
 aggregate={}
 for prior in ('G','S'):
  subset=[r for r in rows if r['prior']==prior];aggregate[prior]=dict(actual_branches=24,old_MAE_m2=sum(r['old_abs_error_m2']for r in subset)/24,new_MAE_m2=sum(r['new_abs_error_m2']for r in subset)/24,old_bias_m2=sum(r['old_signed_error_m2']for r in subset)/24,new_bias_m2=sum(r['new_signed_error_m2']for r in subset)/24,new_overestimates=sum(r['new_signed_error_m2']>0 for r in subset),new_underestimates=sum(r['new_signed_error_m2']<0 for r in subset),absolute_error_increased_routes=sum(r['abs_error_delta_m2']>0 for r in subset),absolute_error_reduced_routes=sum(r['abs_error_delta_m2']<0 for r in subset))
 for name,data in [('all_48_prior_route_rows.csv',rows),('all_16_state_prior_rows.csv',states),('acquisition_per_track.csv',parts),('selected_pair_tradeoffs.csv',pairs)]:
  with(HERE/name).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 denominator=aggregate['S']['new_MAE_m2']-aggregate['S']['old_MAE_m2'];dominant=next(x for x in states if x['layout']=='cycle_b'and x['prefix']=='p2'and x['prior']=='S')
 output=dict(status='completed_posthoc_existing_24_branch_bias_audit',aggregate=aggregate,dominant_state=dominant,dominant_state_share_of_total_net_S_MAE_change=dominant['MAE_delta_m2']/8/denominator,negative_net_actual_routes=[r for r in rows if r['prior']=='S'and r['actual_object_net_m2']<0],selected_tradeoffs=pairs,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_actions=0,new_fitting=0,new_metrics_mesh_sampling=0,boundaries=['Descriptive attribution of observed error, not a controlled causal identification of shape versus visibility error.','Forecast component area is integrated on fitted hypothesis surfaces, whereas actual gross/net is independently sampled source-to-native surface area.','Runtime tracks can fragment one physical object; their forecast contributions are reported separately, not claimed to be independent objects or proven overlapping surfaces.','Four layouts, eight dependent states and 24 dependent branch executions; G/S share the same24 actual branches.','No frozen source or result was changed.'])
 for p,h in inputs.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 write(HERE/'short_bias_summary.json',output);print(json.dumps({'aggregate':aggregate,'dominant_share':output['dominant_state_share_of_total_net_S_MAE_change']}))
if __name__=='__main__':main()
