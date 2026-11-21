"""Read-only eight-run B11 evaluator. Never creates a world or launches an actor."""
from pathlib import Path
import argparse,csv,json,time,traceback
import numpy as np
from evaluate_full_run import Evaluator,evaluate,read,write,sha,resource,ROOT,HERE
LAYOUTS=('chain_a','cycle_a','chain_b','cycle_b')
ARMS=('S_visibility','S_acquisition')

def inventory(parent):
 rows=[]
 for p in sorted(parent.glob('attempt_*')):
  if not p.is_dir():continue
  result=read(p/'result.json') if (p/'result.json').exists() else None
  binding=(p/'B11_execution_binding.json').exists()
  complete=False
  if binding and (p/'artifact_manifest.json').exists():
   manifest=read(p/'artifact_manifest.json')
   complete=result is not None and result.get('status')=='completed_execution' and 'B11_execution_binding.json' in manifest.get('files_sha256',{})
  state='complete' if complete else 'failed_or_interrupted' if (p/'failure.json').exists() else 'partial_or_running'
  saved={f.name:sha(f) for f in p.glob('*.json') if f.name in ('attempt.json','failure.json','result.json','artifact_manifest.json','B11_execution_binding.json','partial_result.json')}
  rows.append(dict(attempt=p.name,path=str(p),status=state,paid_actions=result.get('paid_actions') if result else None,receipt_sha256=saved))
 return rows

def flat_row(layout,arm,row=None,status='missing',error=None):
 out=dict(layout=layout,topology=layout.split('_')[0],arm=arm,status=status,error=error)
 if row is None:return out
 m=row['surface_metrics'];f=row['final_coverage']
 out.update(paid_actions=row['paid_actions'],A95=row['A95'],A100=row['A100'],censoring_actions=row['censoring_actions'],known_reachable_fraction=f['coverage_fraction'],correct_free_fraction=f['free_correct_fraction'],false_occupied_fraction=f['false_occupied_fraction'],path_length_m=row['path_length_m'],collisions=row['collisions'],returned=row['returned_to_anchor'],elapsed_s=row['elapsed_s'],decision_wall_s_total=row['decision_wall_s_total'],scene_F1_5cm=m['f1_05cm'],scene_F1_10cm=m['f1_10cm'],scene_precision_5cm=m['precision_05cm'],scene_recall_5cm=m['recall_05cm'],object_macro_recall_5cm=row['object_macro_recall_5cm'],object_area_weighted_recall_5cm=row['object_area_weighted_recall_5cm'],trajectory_sha256=row['trajectory_sha256'],native_mesh_sha256=row['native_mesh_sha256'])
 for obj in ('open31','closed118'):
  for cm in (5,10):out[f'{obj}_recall_{cm}cm']=row[f'object_recall_{cm}cm'][obj]
 return out

def pair(old,new):
 assert old['layout']==new['layout']
 assert old['initial_packet_sha256']==new['initial_packet_sha256']
 assert old['reference_sha256']==new['reference_sha256']
 assert old['budget']==new['budget']==256
 a=flat_row(old['layout'],ARMS[0],old);b=flat_row(new['layout'],ARMS[1],new)
 keys=('A95','A100','known_reachable_fraction','correct_free_fraction','false_occupied_fraction','path_length_m','collisions','elapsed_s','decision_wall_s_total','scene_F1_5cm','scene_F1_10cm','open31_recall_5cm','closed118_recall_5cm','open31_recall_10cm','closed118_recall_10cm','object_macro_recall_5cm','object_area_weighted_recall_5cm')
 delta={k:b[k]-a[k] if a[k] is not None and b[k] is not None else None for k in keys}
 return dict(direction='acquisition minus visibility',deltas=delta,A95_pair=[old['A95'],new['A95']],A100_pair=[old['A100'],new['A100']],returned_pair=[old['returned_to_anchor'],new['returned_to_anchor']],same_trajectory=old['trajectory_sha256']==new['trajectory_sha256'],same_native_mesh=old['native_mesh_sha256']==new['native_mesh_sha256'])

def summarize(rows,failures,pending,inventories):
 pairs={l:pair(rows[l][ARMS[0]],rows[l][ARMS[1]]) for l in LAYOUTS if all(a in rows.get(l,{}) for a in ARMS)}
 means={}
 if pairs:
  for k in next(iter(pairs.values()))['deltas']:
   v=[p['deltas'][k] for p in pairs.values() if p['deltas'][k] is not None]
   means[k]=dict(mean_delta=float(np.mean(v)) if v else None,complete_pairs=len(v),all_four_layout_pairs=len(v)==4)
 table=[flat_row(l,a,rows.get(l,{}).get(a),'evaluated' if a in rows.get(l,{}) else 'evaluation_or_actor_failed' if f'{l}/{a}' in failures else 'pending',failures.get(f'{l}/{a}',{}).get('error')) for l in LAYOUTS for a in ARMS]
 return dict(status='pending_eight_run_evaluation' if pending else 'completed_with_failures' if failures else 'completed_independent_eight_run_evaluation',rows=rows,paired_differences=pairs,paired_equal_layout_means=means,failures=failures,pending=pending,all_attempts=inventories,table=table,design=dict(planned_runs=8,completed_runs=sum(map(len,rows.values())),layout_units=4,room_connectivity_topologies=2,arms_per_layout=2,stochastic_replicates=0,scene_state_or_arm_not_an_independent_layout=True,procedural_layouts_not_natural_scene_sample=True),boundaries=['Correct-free coverage uses frozen start-reachable cells. Known-but-falsely-occupied cells are not correct free.', 'A95/A100 are first observed correct-free hits. Unreached thresholds are null, censored at actually executed actions, never imputed as budget.', 'Only complete declared attempt_001 runs enter paired full-run comparisons. Other attempts and partial/failed episodes remain listed; choosing a replacement requires an explicit dated amendment.', 'Native TSDF source reference metrics use fixed independently sampled visible source surfaces, no alignment. Object recall is completeness, not per-instance precision or semantic accuracy.', 'Four specified layouts in two topology families are the paired aggregation units; eight arms and eight short prefixes are not eight independent scenes.', 'Ideal first-hit semantic support, noise-free sensors and exact simulation poses do not demonstrate natural detection, physical control or real-world SLAM accuracy.', 'Runtime declarations, frozen actor code and file hashes establish the observed-input interface; these receipts are not an operating-system access trace.', 'Wall time comes from a shared three-process batch, not an isolated realtime benchmark.'],actor_actions_executed_by_evaluator=0)

def verify_freeze():
 frozen=read(HERE/'freeze.json')
 for p,h in frozen['files_sha256'].items():assert sha(p)==h,p
 return sha(HERE/'freeze.json')

def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);frozen_sha=verify_freeze();rows={};failures={};pending=[f'{l}/{s}' for l in LAYOUTS for s in ARMS]
 while True:
  inventories={}
  for layout in LAYOUTS:
   for arm in ARMS:
    key=f'{layout}/{arm}';parent=ROOT/'execution/layouts'/layout/'runs'/arm;inv=inventory(parent);inventories[key]=inv
    if key not in pending:continue
    entry=next((x for x in inv if x['attempt']=='attempt_001'),None)
    if entry is None:continue
    if entry['status']=='failed_or_interrupted':
     failures[key]=dict(status=entry['status'],error='Actor did not complete the declared episode',attempt=entry);pending.remove(key)
    elif entry['status']=='complete':
     out=a.output/'runs'/layout/arm/'attempt_001';out.mkdir(parents=True,exist_ok=False)
     try:
      start=time.monotonic();result=evaluate(Path(entry['path']),layout,arm);result['evaluation_wall_s']=time.monotonic()-start;write(out/'result.json',result);rows.setdefault(layout,{})[arm]=result
     except Exception as e:failures[key]=dict(status='independent_evaluation_failed',error=repr(e),traceback=traceback.format_exc(),attempt=entry);write(out/'failure.json',failures[key])
     pending.remove(key)
  result=summarize(rows,failures,pending,inventories);result['evaluation_freeze_sha256']=frozen_sha;result['execution_freeze_sha256']=sha(ROOT/'execution/freeze.json');write(a.output/'summary.json',result)
  table=result['table'];fields=list(dict.fromkeys(k for r in table for k in r))
  with (a.output/'all_runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(table)
  print(json.dumps(dict(status=result['status'],completed=result['design']['completed_runs'],pending=len(pending),failures=len(failures))),flush=True)
  if not pending or not a.watch:break
  time.sleep(15)
 verify_freeze()
 write(a.output/'evaluation_receipt.json',dict(status=result['status'],summary=resource(a.output/'summary.json'),table=resource(a.output/'all_runs.csv'),freeze_sha256=frozen_sha,actor_actions=0))
if __name__=='__main__':main()
