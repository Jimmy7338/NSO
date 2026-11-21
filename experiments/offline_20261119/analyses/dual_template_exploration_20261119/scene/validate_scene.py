#!/usr/bin/env python3
"""Geometry/first-hit acceptance only. Never import or execute an NSO policy."""
from collections import deque
from dataclasses import asdict
import json
import time
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from controlled_world import (ControlledFunctionalWorld,ControlledConfig,build_scene,LAYOUTS,SPEC,HERE,
    PREVIOUS,ASSETS,sha,canonical_digest,DIRECTIONS)
from nso.cpu_sensor_contract_v10 import SensorPacket,GridTransform
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10


def write(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n')


def packet(world,frame):
    return SensorPacket('controlled_'+world.layout,'geometry_acceptance','frame0',0,frame,world.scan(),
        world.position,world.heading,'actual_textured_mesh_and_partial_visible_oracle',
        'exact_discrete_geometry_pose').validate(GridTransform(world.shape,.2),world.config)


def distances(world):
    d=np.full(world.shape,-1,int);d[world.start]=0;q=deque([world.start])
    while q:
        c=q.popleft()
        for h,(dr,dc) in enumerate(DIRECTIONS):
            t=(c[0]+dr,c[1]+dc)
            if world._edge_valid[c+(h,)] and d[t]<0:d[t]=d[c]+1;q.append(t)
    return d


def structural_probes(world):
    rows=[]
    for obj in world._records:
        ray=o3d.t.geometry.RaycastingScene(nthreads=1);nodes={}
        for node in obj['source_mesh_nodes']:
            transform,name=world._source_scene.graph[node];mesh=world._source_scene.geometry[name]
            v=trimesh.transform_points(mesh.vertices,transform).astype(np.float32)
            gid=ray.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(np.asarray(mesh.faces,np.uint32)))
            nodes[gid]=node
        lo,hi=np.asarray(obj['bounds_world_xyz_m']);cx,cy=(lo[:2]+hi[:2])/2
        if obj['source_instance_id']==31:zlo,zhi=.10,1.85
        else:zlo,zhi=.25,min(.70,hi[2]-.08)
        zz=np.linspace(zlo,zhi,60);xx=np.linspace(lo[0]+.08,hi[0]-.08,80)
        x,z=np.meshgrid(xx,zz);n=x.size
        front=np.c_[x.ravel(),np.full(n,lo[1]-.2),z.ravel(),np.zeros(n),np.ones(n),np.zeros(n)].astype(np.float32)
        hits=ray.cast_rays(o3d.core.Tensor(front),nthreads=1);distance=hits['t_hit'].numpy();gids=hits['geometry_ids'].numpy()
        finite=np.isfinite(distance);row={'source_instance':obj['source_instance_id'],'slot':obj['slot'],
            'bounds_world_xyz_m':obj['bounds_world_xyz_m'],'extent_xyz_m':obj['extent_xyz_m'],
            'front_probe_rays':n,'front_probe_z_range_m':[zlo,zhi],
            'through_opening_fraction':float(np.mean(~finite)),
            'first_surface_within_front_8cm_fraction':float(np.mean(finite&(distance<=.28))),
            'first_hit_node_counts':{nodes[g]:int(np.sum(gids==g)) for g in nodes},
            'source_meshes_unmodified':True,'mesh_scaling_applied':False}
        if obj['source_instance_id']==118:
            back=front.copy();back[:,1]=hi[1]+.2;back[:,4]=-1
            dist=ray.cast_rays(o3d.core.Tensor(back),nthreads=1)['t_hit'].numpy()
            row['back_probe_hit_fraction']=float(np.mean(np.isfinite(dist)))
            row['back_surface_within_8cm_fraction']=float(np.mean(np.isfinite(dist)&(dist<=.28)))
            # Full body, rear, top/bottom and two closed door source links are
            # present, unlike the earlier front-only semantic reference.
            row['complete_source_nodes']=obj['source_mesh_nodes']
            assert row['first_surface_within_front_8cm_fraction']>.90,row
            assert row['back_probe_hit_fraction']>.99,row
            assert len(obj['source_mesh_nodes'])>=3,row
        else:
            assert row['through_opening_fraction']>.70,row
        rows.append(row)
    return rows


def main():
    begun=time.monotonic();out=HERE/'acceptance';out.mkdir(exist_ok=True)
    protocol=dict(SPEC,config=asdict(ControlledConfig()),
        acceptance={'initial_known_reachable_fraction_max':.50,
                    'both_objects_must_have_reachable_front_back_left_right_fixed_views':True,
                    'each_target_fixed_view_min_visible_pixels':20,
                    'full_closed_cabinet_min_front8cm_hit_fraction':.90,
                    'full_closed_cabinet_min_back_hit_fraction':.99,
                    'shelf_front_open_through_fraction_min':.70},
        frozen_before_policy_execution=True,source_dependencies={str(f):sha(f) for f in
        [HERE/'controlled_world.py',HERE/'validate_scene.py',PREVIOUS/'scene_adapter/replicacad_world.py',
         PREVIOUS/'assets/textured_raycast.py',ASSETS/'assemble_apt0.py',ASSETS/'download_receipt.json',
         ASSETS/'objects/frl_apartment_wall_cabinet_02.glb',ASSETS/'urdf/cabinet/cabinet.urdf',
         ASSETS/'urdf/cabinet/cabinet.glb',ASSETS/'urdf/cabinet/door.glb']})
    write(HERE/'PROTOCOL.json',protocol)
    results=[];initial_hashes=[]
    for layout in LAYOUTS:
        world=ControlledFunctionalWorld(layout=layout);row=dict(world.geometry_receipt)
        frame=world.sense();p=packet(world,frame);initial_hashes.append(p.sha256())
        mapper=ObservedRuntimeMapperV10(world.shape,world.config)
        assert np.all(mapper.belief==-1);mapper.update(frame,world.scan())
        initial_known=int(np.sum((mapper.belief!=-1)&world.reachable))
        initial_fraction=initial_known/int(world.reachable.sum())
        assert initial_fraction<=.5,initial_fraction
        assert int((frame.semantic>0).sum())==0
        row.update(initial_known_reachable_cells=initial_known,initial_known_reachable_fraction=initial_fraction,
            initial_packet_sha256=p.sha256(),initial_semantic_pixels=0,initial_footprint_conflict=bool(mapper.current_footprint_conflict),
            structure_probes=structural_probes(world),object_records=world._records)
        assert not row['initial_footprint_conflict']
        np.savez_compressed(out/(layout+'_evaluator_truth.npz'),**world.to_evaluator_truth())
        frame.save(out/(layout+'_initial_rgbd.npz'));world.scan().save(out/(layout+'_initial_scan.npz'))
        Image.fromarray(frame.color_rgb).save(out/(layout+'_initial.png'))
        d=distances(world);views=[]
        for obj in world._records:
            cx,cy=obj['controlled_center_xy']
            fixed=[('front',(cx,3.7),0),('rear',(cx,6.7),2),('left',(cx-1.4,5.1),1),('right',(cx+1.4,5.1),3)]
            for name,xy,heading in fixed:
                cell=(world.shape[0]-1-int(np.floor(xy[1]/.2)),int(np.floor(xy[0]/.2)))
                assert world.reachable[cell],(layout,obj['slot'],name,cell)
                world.position=cell;world.heading=heading
                f=world.sense();packet(world,f)
                owner=world.sense_visible_instances();assert not np.any(owner[f.depth_m==0])
                pixels=int(np.sum(owner==obj['owner']))
                assert pixels>=20,(layout,obj['slot'],name,pixels)
                delta=np.asarray(world.xy(cell))-np.asarray(obj['controlled_center_xy'])
                views.append({'object_source_instance':obj['source_instance_id'],'slot':obj['slot'],'view':name,
                    'cell':list(cell),'heading':heading,'position_xy_m':list(world.xy(cell)),
                    'actual_swept_forward_steps_from_start':int(d[cell]),'target_visible_pixels':pixels,
                    'relative_object_xy_m':delta.tolist()})
                if name in ('front','rear'):
                    Image.fromarray(f.color_rgb).save(out/(layout+'_'+obj['slot']+'_'+name+'.png'))
        row['four_side_views']=views;row['all_geometric_acceptance_passed']=True;results.append(row)
    # Compare initial actual observations, excluding packet scene IDs only.
    a=np.load(out/(LAYOUTS[0]+'_initial_rgbd.npz'),allow_pickle=False)
    b=np.load(out/(LAYOUTS[1]+'_initial_rgbd.npz'),allow_pickle=False)
    raw_equal={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
    receipt={'status':'passed','layout_count':2,'controlled_topology_families':1,'independent_natural_scenes':0,
             'results':results,'initial_RGBD_arrays_equal_across_swapped_layouts':raw_equal,
             'source_dependencies':protocol['source_dependencies'],'protocol_sha256':sha(HERE/'PROTOCOL.json'),
             'elapsed_s':time.monotonic()-begun,'policy_runs':0,'hardware_or_ROS_calls':0,
             'limits':['New synthetic three-branch building and authored object relocation; no realistic distribution claim',
               'Original source assets retained without scaling; imperfect match to V3 primitive dimensions/backboard',
               'Exact pose, noiseless opaque depth, no glass or physics dynamics',
               'Fixed reachable front/side/rear observations prove geometric opportunities, not policy selection or G/S advantage',
               'Full object original first-hit semantics/visible owners are an ideal perception condition, not a trained detector']}
    write(HERE/'acceptance.json',receipt)
    print(json.dumps({'status':'passed','layouts':[{k:r[k] for k in ('layout','reachable_cells','initial_known_reachable_fraction','all_geometric_acceptance_passed')} for r in results],
                      'receipt':str(HERE/'acceptance.json'),'elapsed_s':receipt['elapsed_s']},indent=2))

if __name__=='__main__':main()
