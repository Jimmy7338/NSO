"""Controlled three-branch environment with original textured ReplicaCAD objects.

New architecture/placements are synthetic, not the official apartment and not a
measured semantic distribution. No NSO planner is called by this module. Source
object meshes and articulation at zero displacement are retained without scaling.
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import trimesh
import open3d as o3d

HERE=Path(__file__).resolve().parent
PREVIOUS=Path('/home/ouc/nso_robot/analyses/semantic_scene_redesign_20261119')
for directory in (PREVIOUS/'scene_adapter',PREVIOUS/'assets',PREVIOUS/'assets/replicacad'):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
from replicacad_world import (ReplicaCADWorld,ReplicaCADConfig,MeshMotionGeometry,
    GEOMETRY_VERSION,COLLISION_BAND_M,FLOOR_BAND_M,FLOOR_NORMAL_MIN,camera_pose,DIRECTIONS)
from textured_raycast import TexturedRayRenderer
from assemble_apt0 import build,placement
from utils.rgbd_contract import RGBDFrame

ASSETS=PREVIOUS/'assets/replicacad'
LAYOUTS=('shelf_left','shelf_right')
YUP_TO_ZUP=np.array([[1.,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
SPEC={
 'schema':'controlled_functional_three_branch_v1',
 'scene_role':'controlled_synthetic_architecture_and_furniture_placement',
 'selection_basis':'Fixed geometric design before any NSO policy execution; A/B swap objects, never select by score or winner',
 'width_m':10.,'height_m':8.,'resolution_m':.2,'robot_radius_m':.2,
 'shape':[40,50],'start':[34,25],'heading':0,
 'wall_thickness_m':.2,'wall_height_m':2.4,
 'wall_boxes_xyz_size':[
  [0,0,0,10,.2,2.4],[0,7.8,0,10,.2,2.4],[0,.2,0,.2,7.6,2.4],[9.8,.2,0,.2,7.6,2.4],
  [.2,2.6,0,4.,.2,2.4],[5.8,2.6,0,4.,.2,2.4],
  [4.2,2.8,0,.2,1.6,2.4],[4.2,6.,0,.2,1.8,2.4],
  [5.6,2.8,0,.2,1.6,2.4],[5.6,6.,0,.2,1.8,2.4]],
 'floor_box_xyz_size':[0,0,-.02,10,8,.02],
 'slots':{'left':[2.2,5.2],'right':[7.8,5.2]},
 'layouts':{'shelf_left':{'left':31,'right':118},'shelf_right':{'left':118,'right':31}},
 'object_front_world':[0.,-1.,0.],
 'assets':{'31':'objects/frl_apartment_wall_cabinet_02.glb','118':'urdf/cabinet/cabinet.urdf; body and two doors at zero displacement'},
 'asset_transform_rule':'Undo original scene-instance placement, retain mesh/link transforms, rotate Y-up to Z-up, centre horizontal bounds at slot, set lowest mesh point to Z=0. No scaling.',
 'labels':'Current first-hit source31->open class2, complete source118->closed class3; architecture0. Observed owner1/2 only where valid depth.',
 'template_limits':'Original V3 clips estimated height0.9..1.5m and min horizontal dims0.9x0.65m. Source shelf heightabout1.98m and cabinetheightabout.778m are not exact templates; no dimension or likelihood-threshold edit.',
 'physical_claim':False,'natural_distribution_claim':False,'planner_used_to_select_layout':False,
 'prior_natural_negative_case':str(PREVIOUS/'exploration_evaluation/summary_five_arms.json')}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical_digest(data):return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class ControlledConfig(ReplicaCADConfig):
    pass


def build_scene(layout='shelf_left'):
    """Evaluator/world-only assembly; returns Z-up scene, object records, node metadata."""
    if layout not in LAYOUTS:raise ValueError('Unknown frozen controlled layout')
    original,original_records,original_meta=build()
    scene=trimesh.Scene();records=[];metadata={}
    def architecture(name,geometry,color):
        x,y,z,sx,sy,sz=geometry
        mesh=trimesh.creation.box(extents=(sx,sy,sz),transform=trimesh.transformations.translation_matrix([x+sx/2,y+sy/2,z+sz/2]))
        mesh.visual.face_colors=np.array([*color,255],np.uint8)
        scene.add_geometry(mesh,geom_name=name,node_name=name)
        metadata[name]={'instance_id':0,'source_instance_id':0,'semantic_class':0,'owner':0,'kind':'synthetic_architecture'}
    architecture('controlled_floor',SPEC['floor_box_xyz_size'],[142,145,147])
    for i,wall in enumerate(SPEC['wall_boxes_xyz_size']):architecture(f'controlled_wall_{i:02d}',wall,[205,207,209])
    for slot,instance_id in SPEC['layouts'][layout].items():
        record=original_records[instance_id]
        inverse=np.linalg.inv(placement(record['source_pose']))
        parts=[];all_vertices=[]
        for node in original.graph.nodes_geometry:
            if original_meta[node]['instance_id']!=instance_id:continue
            local,name=original.graph[node];mesh=original.geometry[name]
            normalized=YUP_TO_ZUP@inverse@local
            all_vertices.append(trimesh.transform_points(mesh.vertices,normalized))
            parts.append((node,mesh,normalized))
        all_vertices=np.concatenate(all_vertices);lo=all_vertices.min(0);hi=all_vertices.max(0)
        center=(lo+hi)/2;cx,cy=SPEC['slots'][slot]
        translate=trimesh.transformations.translation_matrix([cx-center[0],cy-center[1],-lo[2]])
        category=2 if instance_id==31 else 3;owner=1 if category==2 else 2
        new_vertices=[];source_nodes=[]
        for node,mesh,normalized in parts:
            name=f'{slot}/'+node
            transform=translate@normalized
            scene.add_geometry(mesh,geom_name=name,node_name=name,transform=transform)
            metadata[name]={'instance_id':instance_id,'source_instance_id':instance_id,'semantic_class':category,
                'owner':owner,'kind':'original_source_furniture','source_asset':original_meta[node]['source_asset'],
                'source_node':node,'asset_to_controlled_world':transform.tolist()}
            new_vertices.append(trimesh.transform_points(mesh.vertices,transform));source_nodes.append(name)
        points=np.concatenate(new_vertices);bounds=[points.min(0).tolist(),points.max(0).tolist()]
        records.append({'slot':slot,'source_instance_id':instance_id,'source_template':record['source_template'],
            'category':category,'owner':owner,'object_front_world':[0,-1,0],
            'bounds_world_xyz_m':bounds,'extent_xyz_m':(points.max(0)-points.min(0)).tolist(),
            'source_mesh_nodes':source_nodes,'original_source_record':record,
            'articulation':'original zero joint displacement','rescaled':False,
            'controlled_center_xy':SPEC['slots'][slot]})
    return scene,records,metadata


class ControlledFunctionalWorld(ReplicaCADWorld):
    """Same action/sensor interface as ReplicaCADWorld; geometry is evaluator-only."""
    def __init__(self,config=None,seed=20261119,layout='shelf_left'):
        self.config=config or ControlledConfig();c=self.config
        if not np.isclose(c.resolution_m,.2) or not np.isclose(c.robot_radius_m,.2):
            raise ValueError('Frozen geometry requires .2m grid/radius')
        self.seed=int(seed);self.layout=layout;self.shape=tuple(SPEC['shape'])
        self.width=SPEC['width_m'];self.height=SPEC['height_m']
        self.start=tuple(SPEC['start']);self.position=self.start;self.heading=SPEC['heading']
        self.step_count=self.collisions=self.moves=0
        self.world_from_source=np.eye(4);self.source_from_world=np.eye(4)
        begun=time.monotonic()
        self._source_scene,self._records,self._node_meta=build_scene(layout)
        self._renderer=TexturedRayRenderer(self._source_scene);self._ray=self._renderer.scene
        self._geometry=MeshMotionGeometry(self._source_scene,np.eye(4))
        self._node_for_gid={p['geometry_id']:p['node'] for p in self._renderer.stats}
        self._label_for_gid={gid:self._node_meta[node]['semantic_class'] for gid,node in self._node_for_gid.items()}
        self._owner_for_gid={gid:self._node_meta[node]['owner'] for gid,node in self._node_for_gid.items()}
        self._build_evaluator_topology()
        if not self._robot_valid[self.start]:raise ValueError('Fixed controlled start invalid; no automatic relocation')
        self._reachable=self._flood(self.start)
        f=c.width_px/(2*np.tan(np.deg2rad(c.fov_deg/2)))
        self.intrinsic=np.array([[f,0,(c.width_px-1)/2],[0,f,(c.height_px-1)/2],[0,0,1.]])
        self._frame_cache={};self._owner_cache={}
        self.geometry_receipt=dict(self._geometry.stats,layout=layout,shape=list(self.shape),start=list(self.start),
            heading=self.heading,robot_valid_cells=int(self._robot_valid.sum()),reachable_cells=int(self._reachable.sum()),
            directed_swept_edges=int(self._edge_valid.sum()),topology_cache_key=self._topology_key,
            collision_model=GEOMETRY_VERSION,collision_height_m=list(COLLISION_BAND_M),
            floor_support_height_m=list(FLOOR_BAND_M),world_from_source=self.world_from_source.tolist(),
            source_meshes_unchanged=True,controlled_placements=True,elapsed_s=time.monotonic()-begun)

    def _build_evaluator_topology(self):
        # A distinct cache, keyed by the complete new scene protocol/source; no
        # old apartment cache can match a changed layout or construction script.
        key=dict(spec=SPEC,layout=self.layout,source_code=sha(__file__),
            parent_motion_code=sha(PREVIOUS/'scene_adapter/replicacad_world.py'),
            assembler=sha(ASSETS/'assemble_apt0.py'),assets=sha(ASSETS/'download_receipt.json'))
        self._topology_key=canonical_digest(key);cache=HERE/'precomputed_assets';cache.mkdir(exist_ok=True)
        path=cache/(self._topology_key+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as d:
                self._occupancy=d['occupancy'].copy();self._robot_valid=d['robot_valid'].copy();self._edge_valid=d['edge_valid'].copy()
            return
        self._occupancy=np.ones(self.shape,bool);self._robot_valid=np.zeros(self.shape,bool)
        self._edge_valid=np.zeros((*self.shape,4),bool)
        radius=self.config.robot_radius_m
        for cell in np.ndindex(self.shape):
            xy=self.xy(cell);self._occupancy[cell]=not self._geometry.valid(xy,radius=0.)
            self._robot_valid[cell]=self._geometry.valid(xy,radius=radius)
        for r,c in np.argwhere(self._robot_valid):
            cell=(int(r),int(c))
            for h in (1,2):
                dr,dc=DIRECTIONS[h];target=(cell[0]+dr,cell[1]+dc)
                if (0<=target[0]<self.shape[0] and 0<=target[1]<self.shape[1] and self._robot_valid[target]
                    and self._geometry.valid(self.xy(cell),self.xy(target),radius)):
                    self._edge_valid[cell+(h,)]=True;self._edge_valid[target+((h+2)%4,)]=True
        np.savez_compressed(path,occupancy=self._occupancy,robot_valid=self._robot_valid,edge_valid=self._edge_valid)
        (cache/(self._topology_key+'.json')).write_text(json.dumps(key,indent=2)+'\n')

    def _render(self):
        c=self.config;key=(self.position,self.heading,c.width_px,c.height_px,c.max_depth_m)
        if key not in self._frame_cache:
            camera=camera_pose(self.position,self.heading,c,self.shape[0])
            result=self._renderer.render(self.intrinsic,camera,c.width_px,c.height_px,max_depth_m=c.max_depth_m,min_depth_m=.15)
            depth=result['depth_m'];gids=result['reference_geometry_id'];valid=depth>0
            labels=np.zeros(depth.shape,np.uint8);owner=np.zeros(depth.shape,np.uint16)
            for gid,label in self._label_for_gid.items():
                mask=(gids==gid)&valid;labels[mask]=label;owner[mask]=self._owner_for_gid[gid]
            self._frame_cache[key]=(depth.copy(),result['rgb'].copy(),labels,camera)
            self._owner_cache[key]=owner
        return self._frame_cache[key]

    def sense_visible_instances(self):
        """Controlled ideal CURRENT visible support, not a global object map.

        owner1=shelf,owner2=closed cabinet. IDs are arbitrary per-scene instance
        support; callers must not derive hidden geometry or classes from IDs.
        Same support array is shared by every score-mode arm, including no-label.
        """
        self._render();c=self.config
        key=(self.position,self.heading,c.width_px,c.height_px,c.max_depth_m)
        owner=self._owner_cache[key].copy()
        if c.depth_sigma_m or c.dropout:owner[self.sense().depth_m==0]=0
        return owner

    def to_evaluator_truth(self):return self.evaluator_truth()

    def source_scene_for_evaluator(self):
        return self._source_scene,self._records,self._node_meta


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout',choices=LAYOUTS,default='shelf_left')
    parser.add_argument('--prepare',action='store_true',help='Build and inspect first packet only; never execute a policy')
    args=parser.parse_args()
    if args.prepare:
        world=ControlledFunctionalWorld(layout=args.layout);frame=world.sense()
        print(json.dumps(dict(world.geometry_receipt,initial_semantic_pixels=int((frame.semantic>0).sum())),indent=2))
