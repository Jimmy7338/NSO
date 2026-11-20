#!/usr/bin/env python3
"""Independently re-query acquired meshes and bind all replay outcomes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,hashlib,json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;B10=HERE.parent;BASE=B10.parent
B9=BASE/'dual_template_followup_20261120';B8=BASE/'dual_template_uncertainty_20261120';V1=BASE/'dual_template_geometry_20261120'
INPUTS={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):p=Path(p);INPUTS[str(p)]=sha(p);return json.loads(p.read_text())
def checked(p,h):p=Path(p);assert sha(p)==h,p;INPUTS[str(p)]=h;return p
def arrays(p):p=Path(p);INPUTS[str(p)]=sha(p);return dict(np.load(p,allow_pickle=False))
def rho(a,b):return float(spearmanr(a,b).statistic)if np.ptp(a)>0 and np.ptp(b)>0 else None
def main():
 freeze=read(B10/'replay_attempt_001/freeze.json')
 for p,h in freeze['source_hashes'].items():checked(p,h)
 original=read(B9/'structure/replay_audit/summary.json');reported=read(B10/'replay_attempt_001/summary.json')
 csvpath=B10/'replay_attempt_001/all_candidates.csv';INPUTS[str(csvpath)]=sha(csvpath)
 csvrows=list(csv.DictReader(csvpath.open()));by={(r['layout'],int(r['paid']),r['arm'],int(r['candidate_id'])):r for r in csvrows}
 assert len(by)==132
 states=[];max_distance_error=0.;max_gain_error=0.;source_rows=0
 for st in original['states']:
  layout,paid=st['layout'],st['state_action'];prefix=read(B9/'structure/replay_attempt_001'/f'{layout}_{paid}/result.json')
  meshpath=V1/'replay'/f'{layout}_{paid}_attempt_001/prefix_native_tsdf_mesh.ply';checked(meshpath,prefix['native_mesh_sha256'])
  mesh=o3d.io.read_triangle_mesh(str(meshpath));scene=o3d.t.geometry.RaycastingScene(nthreads=1)
  scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
  evaluator=read(B8/'audit/particle_replay'/f'{layout}_{paid}/result.json')
  for p,h in evaluator['input_sha256'].items():checked(p,h)
  assert evaluator['threshold_m']==.05 and evaluator['no_actor_executed'] and evaluator['source_truth_loaded_only_by_evaluator']
  actual={s['candidate_id']:s for s in evaluator['stages']if s['variant']=='G_uncertainty'}
  masks=[];state=dict(layout=layout,paid=paid,arms={})
  for arm,a in st['arms'].items():
   oldentry=prefix['arms'][arm]['arrays'];old=arrays(checked(oldentry['path'],oldentry['sha256']))
   new=arrays(B10/'replay_attempt_001'/f'{layout}_{paid}'/f'{arm}_predictions.npz')
   for field in ('points','candidate_ids'):np.testing.assert_array_equal(old[field],new[field])
   np.testing.assert_array_equal(old['route_union_gain_matrix'],new['old_gain'])
   d=scene.compute_distance(o3d.core.Tensor(np.asarray(old['points'],np.float32)),nthreads=1).numpy().astype(float)
   keep=d>float(np.float32(.05));masks.append(keep)
   np.testing.assert_array_equal(d,new['measured_mesh_distance_m']);np.testing.assert_array_equal(keep,new['native_keep'])
   expected=old['route_union_gain_matrix']*keep
   np.testing.assert_array_equal(expected,new['new_gain']);assert np.all(new['new_gain']<=new['old_gain'])
   assert np.all(new['new_gain'][:,~old['history_keep']]==0)
   max_distance_error=max(max_distance_error,float(np.max(abs(d-new['measured_mesh_distance_m']),initial=0)))
   max_gain_error=max(max_gain_error,float(np.max(abs(expected-new['new_gain']),initial=0)))
   candidates=[]
   for i,cid in enumerate(old['candidate_ids']):
    source_rows+=1;cid=int(cid);oldrow=next(x for x in a['all_candidates']if x['candidate_id']==cid);r=by[layout,paid,arm,cid]
    e=actual[cid];v=e['actual_fixed_reference_increment'];assert e['execution_receipt']['sha256']==oldrow['execution_receipt']['sha256']
    checked(e['execution_receipt']['path'],e['execution_receipt']['sha256'])
    execution=read(e['execution_receipt']['path'])
    metrics=dict(actual_object_gross_m2=sum(v[n]['gross_new_correct_surface_area_estimate_m2']for n in ('open31','closed118')),
     actual_object_net_m2=sum(v[n]['net_correct_surface_area_estimate_m2']for n in ('open31','closed118')),
     actual_global_net_m2=v['all']['net_correct_surface_area_estimate_m2'],actual_actions=execution['completed_actions'],actual_path_m=execution['path_length_m'],
     actual_coverage_pp=100*(execution['final_coverage']['coverage_fraction']-execution['initial_coverage']['coverage_fraction']))
    for k,value in metrics.items():assert np.isclose(float(r[k]),value,rtol=0,atol=1e-10),(layout,paid,cid,k)
    oldgain=float(new['old_gain'][i].sum());newgain=float(expected[i].sum());common=oldrow['common_score']
    for k,value in dict(old_gain_m2=oldgain,new_gain_m2=newgain,consumed_gain_m2=oldgain-newgain,common_score=common,old_score=common+oldgain,new_score=common+newgain,cost=oldrow['cost']).items():
     assert np.isclose(float(r[k]),value,rtol=0,atol=1e-12),(layout,paid,cid,k)
    candidates.append(dict(candidate_id=cid,old=oldgain,new=newgain,common=common,cost=oldrow['cost'],**metrics))
   oldwinner=min(candidates,key=lambda x:(-x['common']-x['old'],x['cost'],x['candidate_id']))['candidate_id']
   newwinner=min(candidates,key=lambda x:(-x['common']-x['new'],x['cost'],x['candidate_id']))['candidate_id']
   report=next(s for s in reported['states']if s['layout']==layout and s['paid']==paid)['arms'][arm]
   assert report['old_selected']==oldwinner and report['new_selected']==newwinner
   summary=dict(old_selected=oldwinner,new_selected=newwinner,target_changed=oldwinner!=newwinner)
   for kind in ('old','new'):
    for metric in ('object_gross','object_net','global_net'):
     x=np.array([c[kind]for c in candidates]);y=np.array([c['actual_'+metric+'_m2']for c in candidates])
     summary[kind+'_MAE_'+metric+'_m2']=float(abs(x-y).mean())
     summary[kind+'_rho_'+metric]=rho(x,y)
   for key in ('old_MAE_object_gross_m2','new_MAE_object_gross_m2','old_MAE_object_net_m2','new_MAE_object_net_m2','old_rho_object_gross','new_rho_object_gross'):
    assert np.isclose(summary[key],report[key],rtol=0,atol=1e-12),key
   state['arms'][arm]=summary
  np.testing.assert_array_equal(masks[0],masks[1]);states.append(state)
 aggregate={arm:{k:float(np.mean([s['arms'][arm][k]for s in states]))for k in states[0]['arms'][arm]if k.startswith(('old_MAE','new_MAE','old_rho','new_rho'))}for arm in ('G','S')}
 for p,h in INPUTS.items():checked(p,h)
 out=dict(status='passed_independent_native_replay_review',states=states,aggregate_equal_state_mean=aggregate,prediction_rows=source_rows,reused_actual_branches=66,new_actor_actions=0,max_distance_error_m=max_distance_error,max_gain_error_m2=max_gain_error,target_changes=sum(s['arms'][a]['target_changed']for s in states for a in ('G','S')),source_hashes=INPUTS,script_sha256=sha(__file__),limitations=['Reused six-prefix development observations, not independent held-out scenes.','Global-net surrogate discrepancies are reported for diagnosis only: the template term estimates object gain, not all-scene area.','Matched unchanged targets imply unchanged selected branch outcomes in this replay, not guaranteed unchanged full-policy trajectories.','No score weights, geometry, original posterior/history or candidate changes occur in this diagnostic.'])
 (HERE/'independent_replay_review.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=out['status'],aggregate=aggregate,target_changes=out['target_changes']),indent=2))
if __name__=='__main__':main()
