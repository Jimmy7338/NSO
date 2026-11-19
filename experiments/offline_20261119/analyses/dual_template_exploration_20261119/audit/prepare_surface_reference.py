#!/usr/bin/env python3
"""Freeze one layout's evaluator reference from prepared geometry, never runs."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import open3d as o3d
import trimesh
from surface_metrics import (unique_sampling_faces,subset_mesh,fixed_camera_poses,
    visible_reference,surface_samples,ray_scene,OLD_EVALUATION)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source')
sys.path.insert(0,str(ROOT/'scene'))
from controlled_world import build_scene
from env.virtual3d import camera_pose

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def source_mesh(layout):
    scene,records,metadata=build_scene(layout)
    vertices=[];triangles=[];labels=[];nodes=[];offset=0
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node];part=scene.geometry[name]
        points=trimesh.transform_points(part.vertices,transform)
        faces=np.asarray(part.faces,np.int32)
        instance=int(metadata[node]['source_instance_id'])
        assert instance in (0,31,118)
        vertices.append(points);triangles.append(faces+offset);labels.extend([instance]*len(faces))
        offset+=len(points);nodes.append({'node':node,'source_instance':instance,'triangles':len(faces)})
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(vertices)),
        o3d.utility.Vector3iVector(np.concatenate(triangles)))
    return mesh,np.asarray(labels,np.int16),nodes,records

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout',required=True,choices=['shelf_left','shelf_right'])
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    reference_protocol=read(HERE/'reference_protocol.json');study_protocol=read(ROOT/'protocol.json')
    freeze=read(args.prepared/'freeze.json')
    for field in ('code_sha256','geometry_cache_sha256'):
        for path,h in freeze[field].items():assert sha(path)==h,path
    for path,h in freeze['frozen_scientific_source_sha256'].items():assert sha(SOURCE/path)==h,path
    for path,h in freeze['source_asset_sha256'].items():assert sha(OLD_EVALUATION.parent/'assets/replicacad'/path)==h,path
    for name,h in freeze['prepared_files_sha256'].items():assert sha(args.prepared/name)==h,name
    initial=read(args.prepared/'initial_state.json')
    config=SimpleNamespace(**initial['actual_config'])
    assert initial['actual_config']==study_protocol['environment']['config']
    with np.load(args.prepared/'evaluator_truth.npz',allow_pickle=False) as z:
        reachable=z['reachable'].copy()
    with np.load(args.prepared/'initial_sensor.npz',allow_pickle=False) as z:
        pose=z['world_from_camera'].copy();K=z['intrinsic'].copy()
    assert tuple(initial['shape'])==reachable.shape
    assert np.allclose(pose,camera_pose(initial['anchor'][:2],initial['anchor'][2],config,reachable.shape[0]),atol=1e-6)
    f=config.width_px/(2*np.tan(np.deg2rad(config.fov_deg/2)))
    assert np.allclose(K,[[f,0,(config.width_px-1)/2],[0,f,(config.height_px-1)/2],[0,0,1]],atol=1e-6)
    inputs=[ROOT/'protocol.json',ROOT/'scene/controlled_world.py',args.prepared/'initial_state.json',
        args.prepared/'evaluator_truth.npz',args.prepared/'initial_sensor.npz',HERE/'reference_protocol.json',
        Path(__file__),HERE/'surface_metrics.py',OLD_EVALUATION/'surface_reference.py',
        OLD_EVALUATION.parent/'scene_adapter/replicacad_world.py',
        OLD_EVALUATION.parent/'assets/replicacad/assemble_apt0.py',
        OLD_EVALUATION.parent/'assets/replicacad/download_receipt.json']
    if (args.prepared/'freeze.json').exists():inputs.append(args.prepared/'freeze.json')
    hashes={str(p.resolve()):sha(p)for p in inputs}
    hashes.update(freeze['geometry_cache_sha256'])
    protocol={**reference_protocol,'layout':args.layout,'actual_camera_config':initial['actual_config'],
        'prepared_directory':str(args.prepared.resolve()),'source_inputs_sha256':hashes}
    (args.output/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    mesh,labels,nodes,records=source_mesh(args.layout)
    sampling,label_subset,duplicates=unique_sampling_faces(mesh,labels)
    poses,cells=fixed_camera_poses(reachable,config,initial['anchor'][:2]);ray=ray_scene(mesh)
    reference={'camera_poses':poses};metadata={}
    for name,target_id in [('all',None),('open31',31),('closed118',118)]:
        target=sampling if target_id is None else subset_mesh(sampling,label_subset==target_id)
        count=reference_protocol['global_area_samples'] if target_id is None else reference_protocol['per_target_area_samples']
        seed=reference_protocol['source_seed'] if target_id is None else reference_protocol['target_seeds'][name]
        points,_=surface_samples(target,count,seed)
        visible,first,rays=visible_reference(points,ray,poses,config,tolerance_m=reference_protocol['first_hit_tolerance_m'])
        reference[name]=points[visible].astype(np.float32);reference[name+'_first_pose']=first[visible]
        assert len(reference[name])>0,'No fixed visible reference for '+name
        metadata[name]={'sampled_source_area_m2':target.get_surface_area(),'drawn_samples':len(points),
            'visible_samples':int(visible.sum()),'estimated_visible_area_m2':target.get_surface_area()*float(visible.mean()),
            'seed':seed,'first_hit_rays':rays}
    np.savez_compressed(args.output/'fixed_reference.npz',**reference)
    assert o3d.io.write_triangle_mesh(str(args.output/'source_mesh.ply'),mesh)
    assert hashes=={p:sha(p)for p in hashes},'reference inputs changed during preparation'
    receipt={'status':'fixed_surface_reference_prepared','layout':args.layout,'source_inputs_sha256':hashes,
        'protocol_sha256':sha(args.output/'PROTOCOL.json'),'reference_npz_sha256':sha(args.output/'fixed_reference.npz'),
        'source_mesh_sha256':sha(args.output/'source_mesh.ply'),'metadata':metadata,
        'raw_source_triangles':len(mesh.triangles),'duplicate_sampling_handling':duplicates,
        'reachable_cells':int(reachable.sum()),'camera_lattice_cells':cells,'camera_pose_count':len(poses),
        'source_object_records':records,'target_triangle_nodes':[n for n in nodes if n['source_instance']],
        'actor_trajectory_read':False,'actor_executed':False,'reference_source_is_native_geometry_not_template':True,
        'elapsed_s':time.monotonic()-start}
    (args.output/'RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'layout':args.layout,'visible_samples':{k:len(reference[k])for k in ['all','open31','closed118']},
        'poses':len(poses),'reference_sha256':receipt['reference_npz_sha256'],'elapsed_s':receipt['elapsed_s']}),flush=True)

if __name__=='__main__':main()
