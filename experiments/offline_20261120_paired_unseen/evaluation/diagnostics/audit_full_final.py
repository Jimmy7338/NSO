"""Final read-only integrity/aggregation review of all frozen B11 full episodes."""
from pathlib import Path
import argparse,hashlib,json,time
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SUMMARY=ROOT/'evaluation/full_policy/attempt_001/summary.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def close(a,b,tol=1e-12):assert abs(a-b)<=tol,(a,b)
def audit(x):
 assert not x['pending'];assert x['design']['layout_units']==4 and x['design']['room_connectivity_topologies']==2 and len(x['table'])==8
 inputs={};run_checks=[];pairs={};failures=x['failures'];expected_units=0
 for l,arms in x['rows'].items():
  expected_units+=len(arms)
  for arm,row in arms.items():
   attempt=Path(row['attempt']);raw=read(attempt/'result.json');initial=read(attempt/'initial_evaluation.json');trace=read(attempt/'steps.json');manifest=read(attempt/'artifact_manifest.json')
   assert raw['paid_actions']==row['paid_actions']==len(trace)<=256
   assert [r['step']for r in trace]==list(range(1,len(trace)+1))
   d=row['final_coverage']['reference_free_cells'];assert d>0
   for z in trace:
    assert z['reference_free_cells']==d
    close(z['correctly_free_cells']/d,z['free_correct_fraction'])
    close(z['known_cells']/d,z['coverage_fraction'])
   calculated={}
   for key,threshold in [('A95',.95),('A100',1.)]:
    hit=0 if initial['free_correct_fraction']>=threshold else next((z['step']for z in trace if z['free_correct_fraction']>=threshold),None)
    assert row[key]==hit;calculated[key]=hit
   assert row['censoring_actions']==row['paid_actions']
   anchor=read(ROOT/'execution/layouts'/l/'prepared/initial_state.json')['anchor'];returned=raw['final_pose']==anchor
   assert row['returned_to_anchor']==raw['final_pose_equals_initial_anchor']==returned
   assert row['native_mesh_sha256']==manifest['files_sha256']['final_tsdf_mesh.ply']==sha(attempt/'final_tsdf_mesh.ply')
   for path,h in row['input_sha256'].items():
    assert path not in inputs or inputs[path]==h
    inputs[path]=h
   m=row['surface_metrics'];recs=row['object_recall_5cm'];close(sum(recs.values())/2,row['object_macro_recall_5cm'])
   for obj in ('open31','closed118'):
    for cm in (5,10):close(row[f'object_recall_{cm}cm'][obj],m['target_surface_metrics'][obj][f'recall_{cm:02d}cm'])
   for cm in (5,10):
    tag=f'{cm:02d}cm';p,r=m['precision_'+tag],m['recall_'+tag];close(m['f1_'+tag],2*p*r/(p+r) if p+r else 0.)
   run_checks.append(dict(layout=l,arm=arm,paid_actions=row['paid_actions'],A95=calculated['A95'],A100=calculated['A100'],returned=returned,collisions=row['collisions'],native_sha256=row['native_mesh_sha256'],object_recall_5cm=row['object_recall_5cm'],scene_F1_5cm=m['f1_05cm']))
  if len(arms)==2:
   old,new=arms['S_visibility'],arms['S_acquisition'];assert old['initial_packet_sha256']==new['initial_packet_sha256']and old['reference_sha256']==new['reference_sha256']
   reported=x['paired_differences'][l];assert reported['A95_pair']==[old['A95'],new['A95']] and reported['A100_pair']==[old['A100'],new['A100']]
   for key in ('A95','A100'):
    delta=new[key]-old[key] if new[key]is not None and old[key]is not None else None
    assert reported['deltas'][key]==delta
   for name in ('open31','closed118'):
    for cm in (5,10):close(reported['deltas'][f'{name}_recall_{cm}cm'],new[f'object_recall_{cm}cm'][name]-old[f'object_recall_{cm}cm'][name])
   for cm in (5,10):close(reported['deltas'][f'scene_F1_{cm}cm'],new['surface_metrics'][f'f1_{cm:02d}cm']-old['surface_metrics'][f'f1_{cm:02d}cm'])
   pairs[l]=dict(**reported,elapsed_ratio_new_over_old=new['elapsed_s']/old['elapsed_s'],decision_time_ratio_new_over_old=new['decision_wall_s_total']/old['decision_wall_s_total'])
 assert expected_units==x['design']['completed_runs']
 for key,aggregate in x['paired_equal_layout_means'].items():
  values=[p['deltas'][key]for p in pairs.values()if p['deltas'][key]is not None];assert aggregate['complete_pairs']==len(values)and aggregate['all_four_layout_pairs']==(len(values)==4)
  if values:close(aggregate['mean_delta'],sum(values)/len(values))
  else:assert aggregate['mean_delta']is None
 for p,h in inputs.items():assert sha(p)==h,p
 freeze=read(ROOT/'evaluation/full_policy/freeze.json')
 for p,h in freeze['files_sha256'].items():assert sha(p)==h,p
 return dict(status='passed_final_integrity_and_aggregation_review' if not failures and expected_units==8 else 'reviewed_terminal_results_with_failures_retained',summary_sha256=sha(SUMMARY),evaluator_freeze_sha256=sha(ROOT/'evaluation/full_policy/freeze.json'),execution_freeze_sha256=sha(ROOT/'execution/freeze.json'),complete_run_count=expected_units,paired_layout_count=len(pairs),failures=failures,per_run_checks=run_checks,paired_checks=pairs,unique_input_hashes_rechecked=len(inputs),negative_outcomes_and_null_thresholds_preserved=True,raw_native_metric_formula_and_file_provenance_verified=True,original_native_metric_computation_not_repeated=True,actor_truth_boundary='Frozen source/interface declaration audit, not OS-level access tracing',time_comparison='Shared concurrent batch, not realtime benchmark',no_new_actor_actions=True,no_new_parameters_or_fitting=True,source_sha256=sha(Path(__file__)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 while True:
  x=read(SUMMARY)
  if not x['pending']:break
  if not a.watch:print(json.dumps(dict(status='waiting_for_eight_terminal_episodes',remaining=len(x['pending']))));return
  time.sleep(30)
 receipt=audit(x);out=HERE/'full_final_review.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=receipt['status'],receipt=str(out),sha256=sha(out))))
if __name__=='__main__':main()
