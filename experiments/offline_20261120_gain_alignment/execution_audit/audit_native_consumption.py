#!/usr/bin/env python3
"""Post-hoc observation-consumption audit. No actor import or execution."""
from pathlib import Path
import os,json,hashlib,csv,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import open3d as o3d
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
B=BASE/'dual_template_exploration_20261119'
OLD=BASE/'dual_template_geometry_20261120/replay'
PREDS=BASE/'dual_template_followup_20261120/structure/replay_attempt_001'
METRICS=BASE/'dual_template_uncertainty_20261120/audit/particle_replay'
STATES={'shelf_left':[24,54,109],'shelf_right':[24,53,110]}
TOL=.05

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def scene(p):
 m=o3d.io.read_triangle_mesh(str(p));s=o3d.t.geometry.RaycastingScene(nthreads=1)
 if len(m.triangles):s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
 return s

def dist(s,p):return s.compute_distance(o3d.core.Tensor(np.asarray(p,np.float32)),nthreads=1).numpy()
def main():
 t=time.monotonic();rows=[];states=[];inputs={}
 for layout,paids in STATES.items():
  ref=B/'audit/references'/layout;receipt=json.loads((ref/'RECEIPT.json').read_text())
  source=ref/'source_mesh.ply';assert sha(source)==receipt['source_mesh_sha256'];inputs[str(source)]=sha(source);truth=scene(source)
  for paid in paids:
   original=OLD/f'{layout}_{paid}_attempt_001';pred=PREDS/f'{layout}_{paid}'
   manifest=json.loads((original/'manifest.json').read_text());result=json.loads((pred/'result.json').read_text())
   evalpath=METRICS/f'{layout}_{paid}/result.json';evaluation=json.loads(evalpath.read_text());inputs[str(evalpath)]=sha(evalpath)
   entries={int(s['candidate_id']):s for s in manifest['stages'] if s['variant']=='B2'}
   p=original/'prefix_native_tsdf_mesh.ply';assert sha(p)==result['native_mesh_sha256'];inputs[str(p)]=sha(p)
   before=scene(p);arrays={};input_signatures={}
   for arm in ['G','S']:
    r=result['arms'][arm]['arrays'];p=Path(r['path']);assert sha(p)==r['sha256'];inputs[str(p)]=sha(p)
    arrays[arm]=dict(np.load(p,allow_pickle=False));input_signatures[arm]=sha(p)
   np.testing.assert_array_equal(arrays['G']['points'],arrays['S']['points']);np.testing.assert_array_equal(arrays['G']['history_keep'],arrays['S']['history_keep'])
   points=arrays['G']['points'];before_d=dist(before,points);source_d=dist(truth,points)
   already=before_d<=TOL;real=source_d<=TOL
   out=HERE/f'{layout}_{paid}';out.mkdir(exist_ok=False)
   np.savez_compressed(out/'native_consumption_mask.npz',already_native_5cm=already,history_keep=arrays['G']['history_keep'],before_distance_m=before_d,source_distance_m=source_d)
   for cid,entry in sorted(entries.items()):
    for key in ['after_mesh','execution_receipt']:
     r=entry[key];assert sha(r['path'])==r['sha256'];inputs[r['path']]=r['sha256']
    after_d=dist(scene(entry['after_mesh']['path']),points);now=after_d<=TOL
    actual=next(x for x in evaluation['stages'] if x['variant']=='G_uncertainty' and x['candidate_id']==cid)['actual_fixed_reference_increment']
    for arm in ['G','S']:
     data=arrays[arm];idx=np.flatnonzero(data['candidate_ids']==cid);assert len(idx)==1
     gain=data['route_union_gain_matrix'][int(idx[0])];assert np.all(gain[~data['history_keep']]==0)
     gain=float(1)*gain;total=float(gain.sum());kept_total=float(data['weights'].sum())
     m=lambda mask:float(gain[mask].sum())
     row=dict(layout=layout,paid_action=paid,arm=arm,candidate_id=cid,
      predicted_gain_m2=total,native_already_before_gain_m2=m(already),native_already_fraction=m(already)/total if total else None,
      source_true_already_before_gain_m2=m(real&already),source_false_already_before_gain_m2=m(~real&already),
      source_false_geometry_gain_m2=m(~real),source_true_new_realized_gain_m2=m(real&~already&now),
      source_true_new_unrealized_gain_m2=m(real&~already&~now),native_already_before_now_lost_gain_m2=m(already&~now),
      native5cm_dedup_remaining_gain_m2=m(~already),
      actual_object_gross_new_m2=sum(actual[n]['gross_new_correct_surface_area_estimate_m2']for n in ['open31','closed118']),
      actual_object_lost_m2=sum(actual[n]['lost_correct_surface_area_estimate_m2']for n in ['open31','closed118']),
      actual_object_net_new_m2=sum(actual[n]['net_correct_surface_area_estimate_m2']for n in ['open31','closed118']),
      actual_global_net_new_m2=actual['all']['net_correct_surface_area_estimate_m2'])
     assert np.isclose(row['source_false_geometry_gain_m2']+row['source_true_already_before_gain_m2']+row['source_true_new_realized_gain_m2']+row['source_true_new_unrealized_gain_m2'],total)
     rows.append(row)
   state=dict(layout=layout,paid_action=paid,candidates=len(entries),points=len(points),before_mesh_sha256=result['native_mesh_sha256'],prediction_npz_sha256=input_signatures,
    mask_file=str(out/'native_consumption_mask.npz'),mask_sha256=sha(out/'native_consumption_mask.npz'),arms={})
   for arm in ['G','S']:
    d=arrays[arm];w=d['weights'];base=d['base_weights'];kept=d['history_keep']
    state['arms'][arm]=dict(history_retained_area_m2=float(w.sum()),native_already_retained_area_m2=float(w[already].sum()),
      native_already_retained_fraction=float(w[already].sum()/w.sum()) if w.sum() else None,
      source_true_already_retained_area_m2=float(w[already&real].sum()),
      native_unmatched_history_removed_area_m2=float(base[~already&~kept].sum()),
      native_already_history_removed_area_m2=float(base[already&~kept].sum()))
   states.append(state);write(out/'state.json',state)
   print(json.dumps({'layout':layout,'state':paid,'rows':2*len(entries),'elapsed_s':time.monotonic()-t}),flush=True)
 for p,h in inputs.items():assert sha(p)==h,p
 with (HERE/'candidate_consumption.csv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 aggregate={}
 for arm in ['G','S']:
  selected=[r for r in rows if r['arm']==arm]
  fields=[k for k,v in selected[0].items() if k.endswith('_m2')]
  per_state=[]
  for s in states:
   rr=[r for r in selected if r['layout']==s['layout'] and r['paid_action']==s['paid_action']]
   per_state.append({f:float(np.mean([r[f] for r in rr])) for f in fields})
  mean={f:float(np.mean([r[f]for r in per_state])) for f in fields}
  mean['ratio_of_state_mean_native_already_to_gain']=mean['native_already_before_gain_m2']/mean['predicted_gain_m2'] if mean['predicted_gain_m2'] else None
  mean['mean_per_candidate_native_already_fraction']=float(np.mean([r['native_already_fraction']for r in selected if r['native_already_fraction'] is not None]))
  aggregate[arm]=mean
 write(HERE/'summary.json',dict(status='complete_posthoc_native_consumption_audit',states=states,aggregate_equal_weight_states=aggregate,candidate_rows=len(rows),actual_branches=66,independent_new_trials=0,
  threshold_m=TOL,inputs_sha256=inputs,script_sha256=sha(__file__),elapsed_s=time.monotonic()-t,
  no_actor_executed=True,no_old_artifact_modified=True,no_new_score_selection_or_rollout=True,
  interpretation=['Template weights are expected quadrature area, not unique physical surface area.','Native5cm mask is an acquired-map diagnostic; ground truth used only to partition errors in this audit.','History-removed native-unmatched area includes legitimate strict-free contradictions; do not label all of it missed reward.','Native already-before does not alone identify omitted keyframes as cause: tolerance/projection and TSDF interpolation can also contribute.','Within-state candidates are dependent; aggregate equally weights the six prefixes.']))
 print(json.dumps(aggregate,indent=2),flush=True)
if __name__=='__main__':main()
