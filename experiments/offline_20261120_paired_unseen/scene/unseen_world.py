"""Four predeclared unseen furniture layouts; simulator/evaluator geometry only.

The actor receives the inherited current RGB-D/scan/visible-instance packets.
No policy, score, candidate or observed outcome selects these layouts/prefixes.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
import trimesh
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'dual_template_exploration_20261119/scene/controlled_world.py'
spec=importlib.util.spec_from_file_location('B11_frozen_sensor_motion_parent',OLD);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
sha=old.sha;canonical_digest=old.canonical_digest
LAYOUTS=('chain_a','chain_b','cycle_a','cycle_b')
# Coordinates and topology committed before any new candidate score or outcome.
RAW={
 'chain_a':dict(width_m=10.,height_m=8.,split_x=5.,split_y=4.,topology='chain',door_bottom_y=2.1,door_top_y=6.1,door_right_x=7.1,door_left_x=None,shelf=[2.1,3.1,0.],cabinet=[8.1,2.9,90.],prefix_end_x=7.1),
 'chain_b':dict(width_m=10.4,height_m=8.8,split_x=4.4,split_y=4.6,topology='chain',door_bottom_y=2.1,door_top_y=6.9,door_right_x=7.7,door_left_x=None,shelf=[2.3,3.5,25.],cabinet=[8.3,3.3,-20.],prefix_end_x=6.5),
 'cycle_a':dict(width_m=9.6,height_m=9.2,split_x=4.8,split_y=4.2,topology='cycle',door_bottom_y=2.1,door_top_y=6.7,door_right_x=7.3,door_left_x=2.3,shelf=[2.1,3.1,-25.],cabinet=[7.9,3.1,15.],prefix_end_x=6.7),
 'cycle_b':dict(width_m=11.2,height_m=8.4,split_x=6.,split_y=3.8,topology='cycle',door_bottom_y=2.1,door_top_y=6.3,door_right_x=8.7,door_left_x=2.5,shelf=[3.5,3.0,15.],cabinet=[9.5,2.9,-30.],prefix_end_x=8.1),
}

def xy_cell(x,y,height):return (int(round(height/.2-.5-y/.2)),int(round(x/.2-.5)))
def gap_segments(start,end,center,width=1.4):
 return [(start,center-width/2),(center+width/2,end)] if center is not None else [(start,end)]
def layout_spec(layout):
 q=dict(RAW[layout]);W,H,X,Y=q['width_m'],q['height_m'],q['split_x'],q['split_y'];t=.2;z=2.4
 walls=[[0,0,0,W,t,z],[0,H-t,0,W,t,z],[0,t,0,t,H-2*t,z],[W-t,t,0,t,H-2*t,z]]
 for low,high,center in [(t,Y-t/2,q['door_bottom_y']),(Y+t/2,H-t,q['door_top_y'])]:
  for a,b in gap_segments(low,high,center):walls.append([X-t/2,a,0,t,b-a,z])
 for low,high,center in [(t,X-t/2,q['door_left_x']),(X+t/2,W-t,q['door_right_x'])]:
  for a,b in gap_segments(low,high,center):walls.append([a,Y-t/2,0,b-a,t,z])
 # Script uses only fixed legal geometry waypoints. Prefixes are final stops,
 # not adaptive views chosen from prediction/gain or future acquisitions.
 # Start (1.1,1.1,N), first (3.1,2.1,N), second (end_x,2.1,N).
 first=['forward']*5+['right']+['forward']*10+['left'];second=first+['right']+['forward']*int(round((q['prefix_end_x']-3.1)/.2))+['left']
 q.update(layout_id=layout,resolution_m=.2,robot_radius_m=.2,shape=[int(round(H/.2)),int(round(W/.2))],start=list(xy_cell(1.1,1.1,H)),heading=0,wall_boxes_xyz_size=walls,floor_box_xyz_size=[0,0,-.02,W,H,.02],door_clear_width_m=1.4,
 room_graph={'rooms':['A_southwest','B_southeast','C_northeast','D_northwest'],'edges':[['A_southwest','B_southeast'],['B_southeast','C_northeast'],['C_northeast','D_northwest']]+([['D_northwest','A_southwest']]if q['topology']=='cycle'else[])},
 sealed_prefixes=[dict(prefix_id='p1',actions=first,paid_actions=len(first),expected_xy=[3.1,2.1],expected_heading=0),dict(prefix_id='p2',actions=second,paid_actions=len(second),expected_xy=[q['prefix_end_x'],2.1],expected_heading=0)])
 return q
SPECS={name:layout_spec(name)for name in LAYOUTS}
ControlledConfig=old.ControlledConfig

def build_scene(layout):
 q=SPECS[layout];original,original_records,original_meta=old.build();scene=trimesh.Scene();metadata={};records=[]
 def add_arch(name,geometry,color):
  x,y,z,sx,sy,sz=geometry;mesh=trimesh.creation.box(extents=[sx,sy,sz],transform=trimesh.transformations.translation_matrix([x+sx/2,y+sy/2,z+sz/2]));mesh.visual.face_colors=np.array([*color,255],np.uint8);scene.add_geometry(mesh,geom_name=name,node_name=name);metadata[name]=dict(instance_id=0,source_instance_id=0,semantic_class=0,owner=0,kind='new_synthetic_architecture')
 add_arch('floor',q['floor_box_xyz_size'],[142,145,147])
 for i,wall in enumerate(q['wall_boxes_xyz_size']):add_arch(f'wall_{i:02d}',wall,[205,207,209])
 for role,source,category,owner in [('shelf',31,2,1),('cabinet',118,3,2)]:
  rec=original_records[source];inv=np.linalg.inv(old.placement(rec['source_pose']));parts=[];verts=[]
  for node in original.graph.nodes_geometry:
   if original_meta[node]['instance_id']!=source:continue
   local,name=original.graph[node];mesh=original.geometry[name];normal=old.YUP_TO_ZUP@inv@local;parts.append((node,mesh,normal));verts.append(trimesh.transform_points(mesh.vertices,normal))
  vv=np.concatenate(verts);lo,hi=vv.min(0),vv.max(0);center=(lo+hi)/2;cx,cy,yaw=q[role];R=trimesh.transformations.rotation_matrix(np.deg2rad(yaw),[0,0,1]);place=trimesh.transformations.translation_matrix([cx,cy,0])@R@trimesh.transformations.translation_matrix([-center[0],-center[1],-lo[2]]);new=[];nodes=[]
  for node,mesh,normal in parts:
   name=role+'/'+node;transform=place@normal;scene.add_geometry(mesh,geom_name=name,node_name=name,transform=transform);metadata[name]=dict(instance_id=source,source_instance_id=source,semantic_class=category,owner=owner,kind='original_source_furniture_new_pose',source_asset=original_meta[node]['source_asset'],source_node=node,asset_to_controlled_world=transform.tolist());new.append(trimesh.transform_points(mesh.vertices,transform));nodes.append(name)
  vv=np.concatenate(new);front=(R[:3,:3]@np.array([0,-1,0])).tolist();records.append(dict(slot=role,source_instance_id=source,source_template=rec['source_template'],category=category,owner=owner,object_front_world=front,bounds_world_xyz_m=[vv.min(0).tolist(),vv.max(0).tolist()],extent_xyz_m=(vv.max(0)-vv.min(0)).tolist(),source_mesh_nodes=nodes,original_source_record=rec,articulation='original zero joint displacement',rescaled=False,controlled_center_xy=[cx,cy],yaw_degrees=yaw))
 return scene,records,metadata

class ControlledFunctionalWorld(old.ControlledFunctionalWorld):
 def __init__(self,config=None,seed=20261120,layout='chain_a'):
  self.config=config or ControlledConfig();c=self.config
  if not np.isclose(c.resolution_m,.2)or not np.isclose(c.robot_radius_m,.2):raise ValueError('Retained .2m resolution/robot radius required')
  self.spec=SPECS[layout];self.seed=int(seed);self.layout=layout;self.shape=tuple(self.spec['shape']);self.width=self.spec['width_m'];self.height=self.spec['height_m'];self.start=tuple(self.spec['start']);self.position=self.start;self.heading=self.spec['heading'];self.step_count=self.collisions=self.moves=0;self.world_from_source=np.eye(4);self.source_from_world=np.eye(4);begin=time.monotonic()
  self._source_scene,self._records,self._node_meta=build_scene(layout);self._renderer=old.TexturedRayRenderer(self._source_scene);self._ray=self._renderer.scene;self._geometry=old.MeshMotionGeometry(self._source_scene,np.eye(4));self._node_for_gid={p['geometry_id']:p['node']for p in self._renderer.stats};self._label_for_gid={gid:self._node_meta[node]['semantic_class']for gid,node in self._node_for_gid.items()};self._owner_for_gid={gid:self._node_meta[node]['owner']for gid,node in self._node_for_gid.items()};self._build_evaluator_topology()
  if not self._robot_valid[self.start]:raise ValueError('Predeclared start invalid: do not relocate automatically')
  self._reachable=self._flood(self.start);f=c.width_px/(2*np.tan(np.deg2rad(c.fov_deg/2)));self.intrinsic=np.array([[f,0,(c.width_px-1)/2],[0,f,(c.height_px-1)/2],[0,0,1.]]);self._frame_cache={};self._owner_cache={}
  self.geometry_receipt=dict(self._geometry.stats,layout=layout,shape=list(self.shape),start=list(self.start),heading=self.heading,robot_valid_cells=int(self._robot_valid.sum()),reachable_cells=int(self._reachable.sum()),directed_swept_edges=int(self._edge_valid.sum()),topology_cache_key=self._topology_key,collision_model=old.GEOMETRY_VERSION,collision_height_m=list(old.COLLISION_BAND_M),floor_support_height_m=list(old.FLOOR_BAND_M),world_from_source=self.world_from_source.tolist(),source_meshes_unchanged=True,controlled_placements=True,room_graph=self.spec['room_graph'],elapsed_s=time.monotonic()-begin)
 def _build_evaluator_topology(self):
  key=dict(spec=self.spec,source_code=sha(__file__),parent_world=sha(OLD),parent_motion=sha(old.PREVIOUS/'scene_adapter/replicacad_world.py'),assembler=sha(old.ASSETS/'assemble_apt0.py'),assets=sha(old.ASSETS/'download_receipt.json'));self._topology_key=canonical_digest(key);cache=HERE/'precomputed_assets';cache.mkdir(exist_ok=True);path=cache/(self._topology_key+'.npz')
  if path.exists():
   with np.load(path,allow_pickle=False)as d:self._occupancy=d['occupancy'].copy();self._robot_valid=d['robot_valid'].copy();self._edge_valid=d['edge_valid'].copy()
   return
  self._occupancy=np.ones(self.shape,bool);self._robot_valid=np.zeros(self.shape,bool);self._edge_valid=np.zeros((*self.shape,4),bool);r=self.config.robot_radius_m
  for cell in np.ndindex(self.shape):self._occupancy[cell]=not self._geometry.valid(self.xy(cell),radius=0.);self._robot_valid[cell]=self._geometry.valid(self.xy(cell),radius=r)
  for rr,cc in np.argwhere(self._robot_valid):
   cell=(int(rr),int(cc))
   for h in (1,2):
    dr,dc=old.DIRECTIONS[h];target=(cell[0]+dr,cell[1]+dc)
    if 0<=target[0]<self.shape[0]and 0<=target[1]<self.shape[1]and self._robot_valid[target]and self._geometry.valid(self.xy(cell),self.xy(target),r):self._edge_valid[cell+(h,)]=True;self._edge_valid[target+((h+2)%4,)]=True
  np.savez_compressed(path,occupancy=self._occupancy,robot_valid=self._robot_valid,edge_valid=self._edge_valid);(cache/(self._topology_key+'.json')).write_text(json.dumps(key,indent=2)+'\n')

def make_world(layout_id,seed=20261120,config=None):return ControlledFunctionalWorld(config=config,seed=seed,layout=layout_id)
def sealed_prefix_actions(layout_id,prefix_id):return list(next(p['actions']for p in SPECS[layout_id]['sealed_prefixes']if p['prefix_id']==prefix_id))
