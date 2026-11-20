"""Independent actual source references; forbidden to planning actors."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys,time
import numpy as np
import open3d as o3d
import trimesh
from unseen_world import build_scene,LAYOUTS,sha
HERE=Path(__file__).resolve().parent;B=HERE.parents[1]/'dual_template_exploration_20261119';sys.path.insert(0,str(B/'audit'))
from surface_metrics import unique_sampling_faces,subset_mesh,fixed_camera_poses,visible_reference,surface_samples,ray_scene

def source_mesh(layout):
 scene,records,metadata=build_scene(layout);vertices=[];triangles=[];labels=[];nodes=[];offset=0
 for node in scene.graph.nodes_geometry:
  transform,name=scene.graph[node];part=scene.geometry[name];points=trimesh.transform_points(part.vertices,transform);faces=np.asarray(part.faces,np.int32);iid=int(metadata[node]['source_instance_id']);assert iid in(0,31,118)
  vertices.append(points);triangles.append(faces+offset);labels.extend([iid]*len(faces));offset+=len(points);nodes.append(dict(node=node,source_instance=iid,triangles=len(faces)))
 mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(vertices)),o3d.utility.Vector3iVector(np.concatenate(triangles)));return mesh,np.asarray(labels,np.int16),nodes,records

def prepare(layout):
 start=time.monotonic();prepared=HERE/'prepared'/layout;initial=json.loads((prepared/'initial_state.json').read_text());config=SimpleNamespace(**initial['actual_config']);truth=np.load(prepared/'evaluator_truth.npz',allow_pickle=False);reachable=truth['reachable'];out=HERE/'evaluator_references'/layout;out.mkdir(parents=True,exist_ok=False)
 protocol=json.loads((HERE/'REFERENCE_PROTOCOL.json').read_text());hashes={str(p):sha(p)for p in [Path(__file__),HERE/'unseen_world.py',HERE/'REFERENCE_PROTOCOL.json',prepared/'evaluator_truth.npz',prepared/'initial_state.json',B/'audit/surface_metrics.py']}
 mesh,labels,nodes,records=source_mesh(layout);sampling,label_subset,duplicates=unique_sampling_faces(mesh,labels);poses,cells=fixed_camera_poses(reachable,config,initial['anchor'][:2]);ray=ray_scene(mesh);reference={'camera_poses':poses};metadata={};index=LAYOUTS.index(layout)
 for name,target in [('all',None),('open31',31),('closed118',118)]:
  selected=sampling if target is None else subset_mesh(sampling,label_subset==target);count=100000 if target is None else 10000;seed=protocol['base_seed']+index*1000+(0 if target is None else target);points,_=surface_samples(selected,count,seed);visible,first,rays=visible_reference(points,ray,poses,config,tolerance_m=.001);reference[name]=points[visible].astype(np.float32);reference[name+'_first_pose']=first[visible];assert visible.any(),name;area=selected.get_surface_area();metadata[name]=dict(sampled_source_area_m2=area,drawn_samples=count,visible_samples=int(visible.sum()),estimated_visible_area_m2=float(area*visible.mean()),area_per_sample_m2=float(area/count),seed=seed,first_hit_rays=rays)
 np.savez_compressed(out/'fixed_reference.npz',**reference);assert o3d.io.write_triangle_mesh(str(out/'source_mesh.ply'),mesh)
 receipt=dict(status='fixed_source_only_surface_reference_prepared',layout=layout,source_inputs_sha256=hashes,reference_npz_sha256=sha(out/'fixed_reference.npz'),source_mesh_sha256=sha(out/'source_mesh.ply'),metadata=metadata,raw_source_triangles=len(mesh.triangles),duplicate_sampling_handling=duplicates,reachable_cells=int(reachable.sum()),camera_lattice_cells=cells,camera_pose_count=len(poses),source_object_records=records,target_triangle_nodes=[n for n in nodes if n['source_instance']],actor_trajectory_read=False,actor_executed=False,reference_source_is_native_geometry_not_template=True,elapsed_s=time.monotonic()-start)
 assert all(sha(Path(p))==h for p,h in hashes.items());(out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(layout=layout,status=receipt['status'],elapsed_s=receipt['elapsed_s'],visible_samples={k:v['visible_samples']for k,v in metadata.items()})),flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--layout',required=True,choices=LAYOUTS);prepare(p.parse_args().layout)
