"""CPU facility-inspection world with deliberately controlled information gaps.

This is simulator/evaluator code, never a planner dependency. World truth and
audit viewpoints are not part of RGBDFrame / PlanarScan. Semantic interventions
only change visible labels; geometry, RGB, depth, laser and noise stay fixed.
"""
from dataclasses import dataclass
import math
import numpy as np
import open3d as o3d
from scipy.ndimage import label

from env.virtual3d_v2 import VirtualConfigV2, VirtualWorldV2
from utils.grid_geometry import inflated_obstacles


@dataclass(frozen=True)
class InspectionConfigV4(VirtualConfigV2):
    bays_per_side: int = 2
    bay_width_m: float = 4.8
    bay_depth_m: float = 5.2
    corridor_width_m: float = 2.4
    transfer_length_m: float = 4.0
    entrance_buffer_m: float = 2.0
    doorway_width_m: float = 1.6
    complex_fraction: float = .5
    observation_condition: str = 'ambiguous'  # ambiguous | open
    semantic_relation: str = 'correlated'  # correlated | independent | reversed
    hidden_parts: str = 'shelves'  # shelves | vertical_baffles
    voxel_m: float = .03
    max_steps: int = 480

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.bays_per_side,int) or not 1 <= self.bays_per_side <= 12:
            raise ValueError('bays_per_side must be an integer in [1,12]')
        if self.resolution_m != .2:
            raise ValueError('inspection world uses a 0.2m grid')
        if self.bay_width_m < 4.0 or self.bay_depth_m < 4.8:
            raise ValueError('work cells need room to inspect behind the object')
        if self.corridor_width_m < 2*self.robot_radius_m + .8:
            raise ValueError('corridor too narrow for footprint')
        if not 2*self.robot_radius_m + .8 <= self.doorway_width_m <= self.bay_width_m-1.2:
            raise ValueError('doorway width cannot support the requested footprint')
        if self.transfer_length_m < 0 or self.entrance_buffer_m < 1.2:
            raise ValueError('invalid transfer length or entrance buffer')
        if not 0 <= self.complex_fraction <= 1:
            raise ValueError('complex_fraction must be in [0,1]')
        if self.observation_condition not in ('ambiguous','open'):
            raise ValueError('invalid observation condition')
        if self.semantic_relation not in ('correlated','independent','reversed'):
            raise ValueError('invalid semantic relation')
        if self.hidden_parts not in ('shelves','vertical_baffles'):
            raise ValueError('invalid hidden-parts control')
        dimensions=(self.bay_width_m,self.bay_depth_m,self.corridor_width_m,
                    self.transfer_length_m,self.entrance_buffer_m,self.doorway_width_m)
        if any(not np.isfinite(v) or not np.isclose(v/.2,round(v/.2)) for v in dimensions):
            raise ValueError('layout dimensions must be finite multiples of 0.2m')
        width=.4+2*self.entrance_buffer_m+self.bays_per_side*self.bay_width_m+(self.bays_per_side-1)*self.transfer_length_m
        height=.4+2*self.bay_depth_m+self.corridor_width_m
        object.__setattr__(self,'width_m',width)
        object.__setattr__(self,'height_m',height)
        object.__setattr__(self,'occluded_objects',self.observation_condition=='ambiguous')


class InspectionWorldV4(VirtualWorldV2):
    """Work cells connected by a common corridor; no GT-derived planner inputs."""
    def __init__(self,config=None,seed=1,semantic_condition='aligned'):
        config=InspectionConfigV4() if config is None else config
        if not isinstance(config,InspectionConfigV4):
            raise TypeError('InspectionConfigV4 required')
        if semantic_condition not in ('aligned','shuffled','absent'):
            raise ValueError('invalid visible-label intervention')
        self.config=config;self.seed=int(seed);self.semantic_condition=semantic_condition
        self.width,self.height=config.width_m,config.height_m
        self.shape=(round(self.height/.2),round(self.width/.2))
        self.mesh=o3d.geometry.TriangleMesh();self.triangle_classes=[]
        self.boxes=[];self.objects=[]
        self._ray=o3d.t.geometry.RaycastingScene(nthreads=1);self._geometry_labels={}
        self.inspection_truth=[]  # evaluator-only audit metadata, never returned by sense()

        def box(x,y,z,sx,sy,sz,true_category=1,visible_category=None):
            if min(sx,sy,sz)<=0:return
            mesh=o3d.geometry.TriangleMesh.create_box(sx,sy,sz).translate((x,y,z))
            mesh.paint_uniform_color([.6,.6,.6]);mesh.compute_vertex_normals()
            gid=self._ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            self._geometry_labels[gid]=true_category if visible_category is None else visible_category
            self.mesh+=mesh;self.triangle_classes.extend([true_category]*len(mesh.triangles))
            if z<1.2 and z+sz>.15:self.boxes.append((x,y,sx,sy))

        w,h=self.width,self.height;t=.2;depth=config.bay_depth_m
        corridor_low=t+depth;corridor_high=corridor_low+config.corridor_width_m
        box(0,0,-.1,w,h,.1)
        box(0,0,0,w,t,2.4);box(0,h-t,0,w,t,2.4)
        box(0,0,0,t,h,2.4);box(w-t,0,0,t,h,2.4)
        starts=[t+config.entrance_buffer_m+i*(config.bay_width_m+config.transfer_length_m)
                for i in range(config.bays_per_side)]
        # Fill unused service volumes so there are no disconnected fake rooms.
        cursor=t
        for x in starts+[w-t]:
            if x>cursor:
                box(cursor,t,0,x-cursor,depth,2.4)
                box(cursor,corridor_high,0,x-cursor,depth,2.4)
            cursor=x+config.bay_width_m
        n=2*config.bays_per_side
        geometry_rng=np.random.default_rng(np.random.SeedSequence([self.seed,404]))
        ranks=geometry_rng.permutation(n)
        complex_mask=ranks<int(round(config.complex_fraction*n))
        # Independent stream: the cue does not consume or alter geometry RNG.
        cue_rng=np.random.default_rng(np.random.SeedSequence([self.seed,919]))
        independent_cues=cue_rng.integers(2,4,n)
        for side in range(2):
            for bay,x in enumerate(starts):
                index=side*config.bays_per_side+bay;center_x=x+config.bay_width_m/2
                y0=t if side==0 else corridor_high
                front_y=corridor_low-t if side==0 else corridor_high
                box(x,y0,0,t,depth,2.4);box(x+config.bay_width_m-t,y0,0,t,depth,2.4)
                wing=(config.bay_width_m-config.doorway_width_m)/2
                box(x,front_y,0,wing,t,2.4)
                box(center_x+config.doorway_width_m/2,front_y,0,wing,t,2.4)
                # Identical external envelope and front panel for both types.
                # Geometric detail is discoverable by physically going around.
                sx,sy,sz=1.2,.8,1.4
                ox=center_x-sx/2
                oy=corridor_low-1.8-sy if side==0 else corridor_high+1.8
                category=3 if complex_mask[index] else 2
                cue=category
                if config.semantic_relation=='independent':cue=int(independent_cues[index])
                elif config.semantic_relation=='reversed':cue=5-category
                self.objects.append(dict(id=index,category=category,box=[ox,oy,0,sx,sy,sz]))
                board_y=oy+sy-.08 if side==0 else oy
                if config.observation_condition=='ambiguous':
                    box(ox,board_y,0,sx,.08,sz,category,cue)
                if category==2:
                    box(ox,oy,0,sx,sy,sz,category,cue)
                else:
                    detail_rng=np.random.default_rng(np.random.SeedSequence([self.seed,index,718]))
                    count=int(detail_rng.integers(3,5))
                    if config.hidden_parts=='shelves':
                        for z in np.linspace(.16,sz-.16,count):
                            box(ox,oy,float(z),sx,sy,.08,category,cue)
                    else:
                        # Shape-prior mismatch: visible class stays the same,
                        # but the hidden assembly has vertical, not shelf parts.
                        for dx in np.linspace(.12,sx-.20,count):
                            box(ox+float(dx),oy,.12,.08,sy,sz-.24,category,cue)
                    for dx in (0,sx-.08):
                        for dy in (0,sy-.08):box(ox+dx,oy+dy,0,.08,.08,sz,category,cue)
                # Simulator-only viewpoints make first-face ambiguity testable.
                facing=2 if side==0 else 0
                front_position=self._cell(center_x,oy+sy+1.4 if side==0 else oy-1.4)
                rear_position=self._cell(center_x,oy-1.2 if side==0 else oy+sy+1.2)
                self.inspection_truth.append(dict(object_index=index,true_category=category,
                    visible_category=cue,front_pose=(front_position,facing),
                    rear_pose=(rear_position,(facing+2)%4),hidden_parts=config.hidden_parts))

        self.triangle_classes=np.asarray(self.triangle_classes,dtype=np.uint8)
        self.occupancy=np.zeros(self.shape,bool)
        for x,y,sx,sy in self.boxes:
            c0,c1=max(0,int(math.floor(x/.2+1e-8))),min(self.shape[1],int(math.ceil((x+sx)/.2-1e-8)))
            r0,r1=max(0,int(math.floor((h-y-sy)/.2+1e-8))),min(self.shape[0],int(math.ceil((h-y)/.2-1e-8)))
            self.occupancy[r0:r1,c0:c1]=True
        self._blocked=inflated_obstacles(self.occupancy,config.robot_radius_m/.2)
        self.start=self._cell(1.,(corridor_low+corridor_high)/2)
        components,_=label(~self._blocked)
        if not components[self.start]:raise ValueError('invalid inspection start')
        self.reachable=components==components[self.start]
        if np.any((~self._blocked)&~self.reachable):
            raise ValueError('layout creates disconnected traversable floor')
        for item in self.inspection_truth:
            if not all(self.reachable[item[key][0]] for key in ('front_pose','rear_pose')):
                raise ValueError('requested footprint cannot reach inspection viewpoints')
        self.position=self.start;self.heading=1;self.step_count=self.collisions=self.moves=0
        fx=config.width_px/(2*np.tan(np.deg2rad(config.fov_deg/2)))
        self.intrinsic=np.array([[fx,0,(config.width_px-1)/2],[0,fx,(config.height_px-1)/2],[0,0,1.]])

    def _cell(self,x,y):
        return (self.shape[0]-1-int(math.floor(y/.2)),int(math.floor(x/.2)))
