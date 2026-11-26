"""Independent B11 scene/reference freeze validation; no actor execution."""
from pathlib import Path
import hashlib,json
import numpy as np
import open3d as o3d
HERE=Path(__file__).resolve().parent;S=HERE.parent/'scene'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 freeze=read(S/'freeze.json');assert freeze['new_estimator_scores_computed']==freeze['new_branches_executed']==freeze['new_policy_runs']==0
 for p,h in freeze['code_and_data_sha256'].items():assert sha(p)==h,p
 rows=[]
 for layout in freeze['layout_order']:
  prepared=read(S/'prepared'/layout/'receipt.json');initial=read(S/'prepared'/layout/'initial_state.json');ref=S/'evaluator_references'/layout;meta=read(ref/'RECEIPT.json');points=dict(np.load(ref/'fixed_reference.npz',allow_pickle=False));mesh=o3d.io.read_triangle_mesh(str(ref/'source_mesh.ply'));scene=o3d.t.geometry.RaycastingScene(nthreads=1);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
  assert len(prepared['prefixes'])==2 and not prepared['prefix_selected_using_gain'] and not prepared['source_truth_returned_in_actor_packet']
  assert not prepared['candidate_scores_computed']and not prepared['policy_executed']
  assert prepared['graph_cycle_rank']==(0 if layout.startswith('chain')else 1)
  assert sha(ref/'fixed_reference.npz')==meta['reference_npz_sha256'];assert sha(ref/'source_mesh.ply')==meta['source_mesh_sha256']
  checks={}
  for key in ['all','open31','closed118']:
   p=points[key];m=meta['metadata'][key];assert p.ndim==2 and p.shape[1]==3 and np.isfinite(p).all();assert m['visible_samples']==len(p);assert m['drawn_samples']>=len(p)>0
   unit=m['sampled_source_area_m2']/m['drawn_samples'];assert np.isclose(unit,m['area_per_sample_m2'],atol=1e-12,rtol=0)
   assert np.isclose(unit*len(p),m['estimated_visible_area_m2'],atol=1e-9,rtol=0)
   dist=scene.compute_distance(o3d.core.Tensor(p),nthreads=1).numpy();assert float(dist.max())<=.001  # Existing frozen reference visibility tolerance;not the5cm reconstruction metric
   checks[key]=dict(points=len(p),area_unit_m2=unit,reference_on_source_max_distance_m=float(dist.max()))
  assert all(k not in initial for k in ['evaluator_truth','source_mesh','reference_points','source_objects'])
  rows.append(dict(layout=layout,cycle_rank=prepared['graph_cycle_rank'],actual_room_edges=prepared['actual_room_edges'],prefixes=prepared['prefixes'],reference=checks,initial_actor_descriptor_keys=list(initial),source_object_count=len(prepared['source_objects'])))
 result=dict(status='independent_scene_reference_freeze_review_passed',scene_freeze_sha256=sha(S/'freeze.json'),layout_count=len(rows),distinct_cycle_ranks=sorted(set(r['cycle_rank']for r in rows)),predeclared_prefixes=sum(len(r['prefixes'])for r in rows),hash_checked_files=len(freeze['code_and_data_sha256']),rows=rows,source=__file__,source_sha256=sha(__file__),no_new_policy_or_branch_executed=True,source_support_audit_tolerance_m=.001,initial_diagnostic_note='An exploratory0.1mm support check found one global point at0.116mm due to float32 point-to-triangle numerics; allpoints satisfy the original frozen1mm source visibility tolerance. No reference resampling or reconstruction metric change.',limitations=['Constructed geometry supplies renderer and physics;actor methods must still receive only acquired maps/frames.','Reference review verifies fixedsource sample support/area,not image detection or physical SLAM accuracy.','Four layouts reuse authored furniture assets;no unseenasset or naturalenvironment claim.'])
 (HERE/'independent_scene_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['status','layout_count','distinct_cycle_ranks','predeclared_prefixes','hash_checked_files']}))
if __name__=='__main__':main()
