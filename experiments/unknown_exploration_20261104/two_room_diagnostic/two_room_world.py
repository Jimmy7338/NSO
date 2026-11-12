"""Fixed mechanism-oriented two-room Q0-derived world, not a random scene."""
import numpy as np
import open3d as o3d
from scipy.ndimage import label
from env.virtual3d_competition_v9 import CompetitionWorldV9,get_competition_context
from env.virtual3d_inspection_v4 import union_surface_from_boxes
from utils.grid_geometry import inflated_obstacles

WALL_BOXES=((.2,3.6,0.,2.8,.2,2.4,1),(4.6,3.6,0.,3.6,.2,2.4,1))
class TwoRoomWorld(CompetitionWorldV9):
 def __init__(self):
  super().__init__(get_competition_context('Q0'),'shelf_west')
  for x,y,z,sx,sy,sz,cl in WALL_BOXES:
   mesh=o3d.geometry.TriangleMesh.create_box(sx,sy,sz).translate((x,y,z))
   gid=self._ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));self._geometry_labels[gid]=1
   self._solid_primitives.append((x,y,z,sx,sy,sz,cl));self.boxes.append((x,y,sx,sy))
   c0=max(0,int(np.floor((x+1e-8)/.2)));c1=min(self.shape[1],int(np.ceil((x+sx-1e-8)/.2)))
   r0=max(0,int(np.floor((self.height-y-sy+1e-8)/.2)));r1=min(self.shape[0],int(np.ceil((self.height-y-1e-8)/.2)))
   self.occupancy[r0:r1,c0:c1]=True
  self.mesh,self.triangle_classes,audit=union_surface_from_boxes(self._solid_primitives)
  self.competition_truth=dict(generator='two_room_q0_derived_fixed',context_id='TWO_ROOM_01',arrangement='shelf_west',union_surface=audit,synthetic_asset_prior=True,reward_class_weighting=False)
  self.inspection_truth=self.competition_truth
  self._blocked=inflated_obstacles(self.occupancy,self.config.robot_radius_m/self.config.resolution_m)
  components,_=label(~self._blocked)
  if not components[self.start]:raise ValueError('fixed start blocked')
  self.reachable=components==components[self.start]
