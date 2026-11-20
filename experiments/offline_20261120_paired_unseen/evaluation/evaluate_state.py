"""Evaluate B11 paired short branches from immutable scalar/mesh manifests.

Truth/reference meshes are opened here only, after actor predictions and actual
execution are recorded. This evaluator never launches or calls a policy/world.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,sys,time,traceback
import numpy as np
import open3d as o3d
from paired_metrics import surface_increment,paired_summary,summarize_condition,summarize_layout,winner
HERE=Path(__file__).resolve().parent
METRICS_SOURCE=Path('/home/ouc/nso_robot/analyses/dual_template_exploration_20261119/audit/surface_metrics.py')
spec=importlib.util.spec_from_file_location('B11_frozen_native_surface_metrics',METRICS_SOURCE);surface=importlib.util.module_from_spec(spec);spec.loader.exec_module(surface)
CONDITIONS=('visibility_G','visibility_S','acquisition_G','acquisition_S')
REFERENCE_KEYS=('all','open31','closed118')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def resource(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def checked(ref,inputs):
 p=Path(ref['path']).resolve();actual=sha(p)
 if actual!=ref['sha256']:raise ValueError('Changed frozen input:'+str(p))
 inputs[str(p)]=actual;return p

def distance(ray,points):
 return np.full(len(points),np.inf)if ray is None else ray.compute_distance(o3d.core.Tensor(np.asarray(points,np.float32)),nthreads=1).numpy()
def ray(mesh):
 if not len(mesh.triangles):return None
 s=o3d.t.geometry.RaycastingScene(nthreads=1);s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));return s

class Evaluator:
 def __init__(self):self.inputs={};self.references={};self.meshes={}
 def reference(self,directory):
  directory=Path(directory).resolve();key=str(directory)
  if key not in self.references:
   receipt=read(directory/'RECEIPT.json');self.inputs[str(directory/'RECEIPT.json')]=sha(directory/'RECEIPT.json')
   for name,field in [('fixed_reference.npz','reference_npz_sha256'),('source_mesh.ply','source_mesh_sha256')]:checked(dict(path=str(directory/name),sha256=receipt[field]),self.inputs)
   for p,h in receipt.get('source_inputs_sha256',{}).items():checked(dict(path=p,sha256=h),self.inputs)
   if receipt.get('protocol_sha256')and(directory/'PROTOCOL.json').exists():checked(dict(path=str(directory/'PROTOCOL.json'),sha256=receipt['protocol_sha256']),self.inputs)
   points=dict(np.load(directory/'fixed_reference.npz',allow_pickle=False));mesh=o3d.io.read_triangle_mesh(str(directory/'source_mesh.ply'));weights={}
   for name in REFERENCE_KEYS:
    meta=receipt['metadata'][name];unit=meta['sampled_source_area_m2']/meta['drawn_samples'];weights[name]=np.full(len(points[name]),unit)
   self.references[key]=dict(points=points,weights=weights,mesh=mesh,receipt=receipt)
  return self.references[key]
 def native(self,ref,reference):
  p=checked(ref,self.inputs);key=(ref['sha256'],reference['receipt']['reference_npz_sha256'])
  if key not in self.meshes:
   mesh=o3d.io.read_triangle_mesh(str(p));r=ray(mesh)
   self.meshes[key]=dict(mesh=mesh,distances={name:distance(r,reference['points'][name])for name in REFERENCE_KEYS},metrics=surface.evaluate_surface(mesh,reference['mesh'],reference['points']))
  return self.meshes[key]
 def state(self,manifest_path):
  started=time.monotonic();manifest_path=Path(manifest_path);manifest=read(manifest_path);self.inputs[str(manifest_path.resolve())]=sha(manifest_path)
  prediction_path=checked(manifest['predictions'],self.inputs);prediction=read(prediction_path)
  ref=self.reference(manifest['reference_directory']);before=self.native(manifest['before_mesh'],ref)
  conditions=prediction['conditions'];assert set(conditions)==set(CONDITIONS)
  ids=prediction['selected_candidate_ids'];assert len(ids)==3 and len(set(ids))==3
  for name,group in conditions.items():
   if [r['candidate_id']for r in group['rows']]!=ids:raise ValueError('All four prior/estimator arms must use ordered shared three candidates')
   if winner(group['rows'])!=group['selected_candidate']:raise ValueError('Prediction winner inconsistent:'+name)
  actual={};branch_bindings={}
  for branch in manifest['branches']:
   cid=int(branch['candidate_id'])
   if cid not in ids or cid in actual:raise ValueError('Unexpected or duplicate actual branch')
   execution_path=checked(branch['execution'],self.inputs);execution=read(execution_path);endpoint=bool(execution.get('endpoint_reached',False))
   raw_status=execution.get('status','unknown');status='completed'if endpoint and raw_status not in ('failed','interrupted','external_interruption')else'partial'if branch.get('after_mesh')else'failed'
   row=dict(candidate_id=cid,status=status,raw_execution_status=raw_status,endpoint_reached=endpoint,
    actual_actions=execution.get('completed_actions',execution.get('paid_actions')),planned_actions=execution.get('planned_actions'),path_length_m=execution.get('path_length_m'),collisions=execution.get('simulated_collisions',execution.get('collisions')),elapsed_s=execution.get('elapsed_s'),
    object_gross_new_m2=None,object_lost_m2=None,object_net_new_m2=None,global_gross_new_m2=None,global_lost_m2=None,global_net_new_m2=None)
   if branch.get('after_mesh'):
    after=self.native(branch['after_mesh'],ref);increments={name:surface_increment(before['distances'][name],after['distances'][name],ref['weights'][name])for name in REFERENCE_KEYS}
    for suffix,key in [('gross_new','gross_new_area_m2'),('lost','lost_area_m2'),('net_new','net_new_area_m2')]:
     row['object_'+suffix+'_m2']=sum(increments[t][key]for t in ('open31','closed118'));row['global_'+suffix+'_m2']=increments['all'][key]
    row.update(fixed_reference_increment=increments,after_native_metrics=after['metrics'])
   else:row.update(fixed_reference_increment=None,after_native_metrics=None)
   actual[cid]=row;branch_bindings[str(cid)]=dict(execution=branch['execution'],after_mesh=branch.get('after_mesh'))
  # All attempted candidate IDs are represented, even if the actor failed to
  # produce an execution receipt. Missing measurements remain null.
  for cid in ids:
   if cid not in actual:actual[cid]=dict(candidate_id=cid,status='missing',object_gross_new_m2=None,object_net_new_m2=None)
  paired={};summaries={}
  for prior in ('G','S'):
   paired[prior]=paired_summary(conditions['visibility_'+prior]['rows'],conditions['acquisition_'+prior]['rows'],actual)
  for name,group in conditions.items():
   summary=summarize_condition(group['rows'],actual)
   summary['net_comparison']=summarize_condition(group['rows'],actual,metric='object_net_new_m2')
   summary['source_wall_s']=group.get('elapsed_s');summaries[name]=summary
  category={}
  for method in ('visibility','acquisition'):
   g,s=summaries[method+'_G'],summaries[method+'_S'];category[method]=dict(neutral_selected=g['selected_candidate'],category_selected=s['selected_candidate'],target_changed=g['selected_candidate']!=s['selected_candidate'],selected_object_gain_delta_m2=(s['selected_observed_area_m2']-g['selected_observed_area_m2']if g['selected_observed_area_m2']is not None and s['selected_observed_area_m2']is not None else None))
  return dict(status='complete_independent_short_state_evaluation',layout=manifest['layout'],topology=manifest['topology'],prefix_id=manifest['prefix_id'],candidate_ids=ids,actual_by_candidate=actual,condition_summaries=summaries,paired_by_prior=paired,category_prior_diagnostic=category,before_native_metrics=before['metrics'],
   prediction_binding=manifest['predictions'],before_mesh=manifest['before_mesh'],actual_bindings=branch_bindings,reference_sha256=ref['receipt']['reference_npz_sha256'],actor_given_evaluation_truth=False,new_actor_actions=0,evaluator_wall_s=time.monotonic()-started)
 def verify(self):
  for p,h in self.inputs.items():
   if sha(p)!=h:raise ValueError('Input mutated during evaluation:'+p)

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);e=Evaluator()
 try:
  result=e.state(a.manifest);e.verify();result['input_sha256']=e.inputs;result['evaluator_sha256']={str(x):sha(x)for x in [Path(__file__),HERE/'paired_metrics.py',METRICS_SOURCE]};write(a.output/'result.json',result)
  rows=[]
  predictions=read(read(a.manifest)['predictions']['path'])
  for name,group in predictions['conditions'].items():
   for r in group['rows']:
    cid=r['candidate_id'];actual=result['actual_by_candidate'][cid];row=dict(condition=name,**r)
    for k in ('object_gross_new_m2','object_lost_m2','object_net_new_m2','global_gross_new_m2','global_lost_m2','global_net_new_m2','status','endpoint_reached','actual_actions','planned_actions','path_length_m','collisions','elapsed_s'):row[k]=actual.get(k)
    rows.append(row)
  with(a.output/'all_candidate_conditions.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(json.dumps(dict(status=result['status'],layout=result['layout'],prefix_id=result['prefix_id'],output=str(a.output))))
 except Exception as ex:write(a.output/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
