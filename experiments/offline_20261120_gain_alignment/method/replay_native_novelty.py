"""Single-change all-candidate B9 fixed replay using acquired native mesh only.

No new visibility queries are needed: original per-point route union commutes
with the route-invariant novelty mask. Reused candidate endpoints are retained
as development evidence; no independent test is claimed.
"""
from pathlib import Path
import csv,hashlib,importlib.util,json,sys,time
import numpy as np
import open3d as o3d
from scipy.stats import spearmanr
import measured_novelty as method
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
B9=BASE/'dual_template_followup_20261120'
B3=BASE/'dual_template_geometry_20261120'
STATES={'shelf_left':[24,54,109],'shelf_right':[24,53,110]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def rho(a,b):
 return float(spearmanr(a,b).statistic) if len(set(a))>1 and len(set(b))>1 else None
def main():
 out=ROOT/'replay_attempt_001';out.mkdir(exist_ok=False)
 source=B9/'structure/replay_audit/summary.json';parent=read(source)
 inputs={str(source):sha(source),str(Path(__file__)):sha(Path(__file__)),str(Path(method.__file__)):sha(method.__file__)}
 for st in parent['states']:
  folder=B9/'structure/replay_attempt_001'/f"{st['layout']}_{st['state_action']}"
  for arm in ['G','S']:inputs[str(folder/(arm+'_predictions.npz'))]=sha(folder/(arm+'_predictions.npz'))
  mesh=B3/'replay'/f"{st['layout']}_{st['state_action']}_attempt_001/prefix_native_tsdf_mesh.ply";inputs[str(mesh)]=sha(mesh)
 write(out/'freeze.json',{'status':'frozen_development_replay_before_execution','source_hashes':inputs,'tolerance_m':.05,'new_acquisition_actions':0,'prediction_rows':132,'candidate_branches_reused':66,'score_weights_candidates_common_unchanged':True})
 rows=[];states=[];started=time.monotonic()
 for st in parent['states']:
  layout,paid=st['layout'],st['state_action'];folder=B9/'structure/replay_attempt_001'/f'{layout}_{paid}'
  meshfile=B3/'replay'/f'{layout}_{paid}_attempt_001/prefix_native_tsdf_mesh.ply';mesh=o3d.io.read_triangle_mesh(str(meshfile))
  scene=None
  if len(mesh.triangles):
   scene=o3d.t.geometry.RaycastingScene(nthreads=1);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
  state={'layout':layout,'paid':paid,'arms':{}};mask_shared=None
  for arm in ['G','S']:
   z=dict(np.load(folder/(arm+'_predictions.npz'),allow_pickle=False));keep,dist=method.unmeasured_mask(z['points'],scene)
   if mask_shared is None:mask_shared=keep
   else:np.testing.assert_array_equal(mask_shared,keep)
   old=z['route_union_gain_matrix'];new=method.consume(old,keep);assert np.all(new<=old)
   np.testing.assert_allclose(old.sum(1)-new.sum(1),old[:,~keep].sum(1),rtol=1e-12,atol=1e-12)
   group=st['arms'][arm];prev=group['all_candidates'];assert len(prev)==len(new)
   armrows=[]
   for i,x in enumerate(prev):
    assert x['candidate_id']==int(z['candidate_ids'][i])
    a=x['actual'];gain=float(new[i].sum());row={'layout':layout,'paid':paid,'arm':arm,'candidate_id':x['candidate_id'],'old_gain_m2':x['new_template_gain_m2'],'new_gain_m2':gain,'consumed_gain_m2':float(old[i,~keep].sum()),'common_score':x['common_score'],'old_score':x['new_total_score'],'new_score':x['common_score']+gain,'cost':x['cost'],'actual_object_gross_m2':a['actual_object_gross_new_m2'],'actual_object_net_m2':a['actual_object_net_new_m2'],'actual_global_net_m2':a['actual_global_net_new_m2'],'actual_actions':a['completed_actions'],'actual_path_m':a['path_length_m'],'actual_coverage_pp':a['coverage_delta_percentage_points']}
    np.testing.assert_allclose(row['old_gain_m2'],old[i].sum(),rtol=1e-12,atol=1e-12);armrows.append(row);rows.append(row)
   oldwinner=min(armrows,key=lambda x:(-x['old_score'],x['cost'],x['candidate_id']));newwinner=min(armrows,key=lambda x:(-x['new_score'],x['cost'],x['candidate_id']))
   state['arms'][arm]={'old_selected':oldwinner['candidate_id'],'new_selected':newwinner['candidate_id'],'target_changed':oldwinner['candidate_id']!=newwinner['candidate_id'],'selected_actual_delta':{k:newwinner[k]-oldwinner[k] for k in ['actual_object_gross_m2','actual_object_net_m2','actual_global_net_m2','actual_actions','actual_path_m','actual_coverage_pp']},'mean_removed_m2':float(np.mean([x['consumed_gain_m2']for x in armrows])),'old_MAE_object_gross_m2':float(np.mean([abs(x['old_gain_m2']-x['actual_object_gross_m2'])for x in armrows])),'new_MAE_object_gross_m2':float(np.mean([abs(x['new_gain_m2']-x['actual_object_gross_m2'])for x in armrows])),'old_MAE_object_net_m2':float(np.mean([abs(x['old_gain_m2']-x['actual_object_net_m2'])for x in armrows])),'new_MAE_object_net_m2':float(np.mean([abs(x['new_gain_m2']-x['actual_object_net_m2'])for x in armrows])),'old_rho_object_gross':rho([x['old_gain_m2']for x in armrows],[x['actual_object_gross_m2']for x in armrows]),'new_rho_object_gross':rho([x['new_gain_m2']for x in armrows],[x['actual_object_gross_m2']for x in armrows])}
   dest=out/f'{layout}_{paid}';dest.mkdir(exist_ok=True)
   np.savez_compressed(dest/(arm+'_predictions.npz'),points=z['points'],measured_mesh_distance_m=dist,native_keep=keep,old_gain=old,new_gain=new,candidate_ids=z['candidate_ids'])
  states.append(state)
 aggregate={}
 for arm in ['G','S']:
  group=[x['arms'][arm]for x in states];keys=[k for k in group[0]if k.startswith(('old_MAE','new_MAE','old_rho','new_rho','mean_removed'))]
  aggregate[arm]={k:float(np.mean([x[k]for x in group if x[k]is not None])) for k in keys}
  aggregate[arm]['target_changes']=sum(x['target_changed']for x in group)
 for path,h in inputs.items():assert sha(path)==h,path
 write(out/'summary.json',{'status':'complete_single_change_development_replay','states':states,'aggregate_equal_prefixes':aggregate,'new_acquisition_actions':0,'reused_branches':66,'prediction_rows':132,'independent_test':False,'elapsed_s':time.monotonic()-started})
 with (out/'all_candidates.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'aggregate_equal_prefixes':aggregate,'changes':[(s['layout'],s['paid'],a,x['old_selected'],x['new_selected'])for s in states for a,x in s['arms'].items()if x['target_changed']]}))
if __name__=='__main__':main()
