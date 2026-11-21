#!/usr/bin/env python3
"""Independent same-point gain partition against objects and native meshes.

Post-hoc only: truth is used here, never by a policy. Alternatives retain their
probability mass; weighted prediction support is not unique physical area.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import open3d as o3d
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
B9=BASE/'dual_template_followup_20261120'
M=B9/'semantic/mechanism_v2_meshfix'
INPUTS={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def remember(p):p=Path(p);INPUTS[str(p)]=sha(p);return p
def read(p):return json.loads(remember(p).read_text())
def resource(r):p=remember(r['path']);assert sha(p)==r['sha256'];return p
def load(name,p):
 p=remember(p);s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def scene(mesh):
 r=o3d.t.geometry.RaycastingScene(nthreads=1)
 if len(mesh.triangles):r.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
 return r
def meshscene(p):return scene(o3d.io.read_triangle_mesh(str(p)))
def distance(r,p):return r.compute_distance(o3d.core.Tensor(np.asarray(p,np.float32)),nthreads=1).numpy()
def partition(gain,objects,global_d,before,after):
 g=np.asarray(gain,float);assert np.all(np.isfinite(g))and np.all(g>=0)
 real=objects<=.05;was=before<=.05;now=after<=.05
 f=lambda m:float(g[m].sum())
 row=dict(predicted_gain_m2=float(g.sum()),unsupported_object_gain_m2=f(~real),
  supported_object_already_before_m2=f(real&was),supported_object_new_realized_m2=f(real&~was&now),
  supported_object_new_unrealized_m2=f(real&~was&~now),
  unsupported_object_but_on_other_scene_m2=f(~real&(global_d<=.05)),
  unsupported_object_native_already_before_m2=f(~real&was),
  native_already_before_m2=f(was),native_near_5_to_10cm_m2=f((before>.05)&(before<=.10)))
 assert np.isclose(sum(row[k]for k in ('unsupported_object_gain_m2','supported_object_already_before_m2','supported_object_new_realized_m2','supported_object_new_unrealized_m2')),g.sum())
 return row
def fixed(rows):
 helper=load('B10_object_source_eval',BASE/'dual_template_objectfit_20261120/audit/evaluate_objectfit.py');ev=helper.Evaluator()
 for layout,paids in {'shelf_left':[24,54,109],'shelf_right':[24,53,110]}.items():
  rays=ev.reference_objects(layout);global_ray=ev.core.reference(layout)['ray']
  for paid in paids:
   result=read(B9/'structure/replay_attempt_001'/f'{layout}_{paid}/result.json')
   manifest=read(BASE/'dual_template_geometry_20261120/replay'/f'{layout}_{paid}_attempt_001/manifest.json')
   entries={int(x['candidate_id']):x for x in manifest['stages']if x['variant']=='B2'}
   arrays={arm:dict(np.load(resource(result['arms'][arm]['arrays']),allow_pickle=False))for arm in ('G','S')}
   p=arrays['G']['points'];np.testing.assert_array_equal(p,arrays['S']['points'])
   obj=np.minimum.reduce([distance(r,p)for r in rays.values()]);gd=distance(global_ray,p)
   before=distance(meshscene(resource(next(iter(entries.values()))['before_mesh'])),p)
   for cid,entry in sorted(entries.items()):
    after=distance(meshscene(resource(entry['after_mesh'])),p)
    for arm,a in arrays.items():
     i=np.flatnonzero(a['candidate_ids']==cid);assert len(i)==1
     rows.append(dict(study='six_prefixes',layout=layout,state_action=paid,temperature=16,cue='acquired',arm=arm,candidate_id=cid,**partition(a['route_union_gain_matrix'][int(i[0])],obj,gd,before,after)))
   print(json.dumps(dict(study='six_prefixes',layout=layout,state=paid,rows=len(rows))),flush=True)
 INPUTS.update(ev.core.inputs)
def mechanism(rows):
 sys.path.insert(0,str(M));mod=load('B10_mechanism_source_only',M/'scene.py');mod.verify()
 for layout in mod.SPEC['layouts']:
  world=mod.MechanismWorld(layout);source,_,metadata=world.source_scene_for_evaluator()
  vertices=[];faces=[];labels=[];offset=0
  for node in source.graph.nodes_geometry:
   tr,name=source.graph[node];part=source.geometry[name];xyz=mod.trimesh.transform_points(part.vertices,tr)
   vertices.append(xyz);faces.append(np.asarray(part.faces,np.int32)+offset);labels.extend([metadata[node]['owner']]*len(part.faces));offset+=len(xyz)
  mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(vertices)),o3d.utility.Vector3iVector(np.concatenate(faces)))
  saved=o3d.io.read_triangle_mesh(str(remember(M/'audit_attempt_001'/layout/'source_mesh.ply')))
  np.testing.assert_array_equal(np.asarray(mesh.triangles),np.asarray(saved.triangles));np.testing.assert_allclose(np.asarray(mesh.vertices),np.asarray(saved.vertices),rtol=0,atol=1e-12)
  selected=np.asarray(mesh.triangles)[np.asarray(labels)>0]
  objray=scene(o3d.geometry.TriangleMesh(mesh.vertices,o3d.utility.Vector3iVector(selected)));gr=scene(mesh)
  manifest=read(M/'branches_attempt_001'/layout/'manifest.json');entries={x['candidate_id']:x for x in manifest['stages']}
  baseline=dict(np.load(remember(M/'candidate_attempt_002'/layout/'T1_correct_predictions.npz'),allow_pickle=False));p=baseline['G_points']
  obj=distance(objray,p);gd=distance(gr,p);before=distance(meshscene(resource(next(iter(entries.values()))['before_mesh'])),p)
  after={cid:distance(meshscene(resource(x['after_mesh'])),p)for cid,x in entries.items()}
  for T in (1,16):
   for cue in ('correct','neutral','swapped'):
    stem=M/'candidate_attempt_002'/layout/f'T{T}_{cue}_predictions'
    receipt=read(stem.with_suffix('.json'));a=dict(np.load(remember(stem.with_suffix('.npz')),allow_pickle=False))
    for arm in ('G','S'):
     np.testing.assert_array_equal(p,a[arm+'_points'])
     matrix=a[arm+'_route_union_gain_matrix'];assert len(matrix)==len(receipt['candidates'])
     for i,c in enumerate(receipt['candidates']):
      assert np.isclose(matrix[i].sum(),c[arm+'_gain']);cid=c['candidate_id']
      rows.append(dict(study='mechanism',layout=layout,state_action=0,temperature=T,cue=cue,arm=arm,candidate_id=cid,**partition(matrix[i],obj,gd,before,after[cid])))
  print(json.dumps(dict(study='mechanism',layout=layout,rows=len(rows))),flush=True)
def main():
 start=time.monotonic();rows=[];fixed(rows);mechanism(rows);assert len(rows)==228
 groups={}
 for r in rows:
  key=(r['study'],r['layout'],r['state_action'],r['temperature'],r['cue'],r['arm']);groups.setdefault(key,[]).append(r)
 states=[]
 for key,rs in groups.items():
  d=dict(zip(('study','layout','state_action','temperature','cue','arm'),key))
  d.update({k:float(np.mean([r[k]for r in rs]))for k in rs[0]if k.endswith('_m2')});states.append(d)
 aggregate={}
 for arm in ('G','S'):
  ss=[s for s in states if s['study']=='six_prefixes'and s['arm']==arm]
  aggregate[arm]={k:float(np.mean([s[k]for s in ss]))for k in ss[0]if k.endswith('_m2')}
  a=aggregate[arm]
  for k in list(a):
   if k!='predicted_gain_m2':a[k.replace('_m2','_fraction')]=a[k]/a['predicted_gain_m2']if a['predicted_gain_m2']else None
 for p,h in INPUTS.items():assert sha(p)==h,p
 with (HERE/'point_support_partition.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 out=dict(status='complete_object_support_partition',rows=len(rows),actual_branches=74,new_trials=0,threshold_m=.05,
  fixed_equal_state_mean=aggregate,states=states,inputs_sha256=INPUTS,script_sha256=sha(__file__),elapsed_s=time.monotonic()-start,
  limitations=['Weighted alternative-template quadrature is not unique physical surface area.','Object support is distance to raw source object triangles; this does not establish visibility or reference-surface recall.','Native distance is not the same operation as original projected-depth history veto.','No policy executes and no sensor observations are acquired.','Mechanism G repeats across cues; all rows retained, not treated as independent trials.'])
 (HERE/'point_support_summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(aggregate,indent=2),flush=True)
if __name__=='__main__':main()
