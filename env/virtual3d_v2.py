"""Larger procedural RGB-D worlds. Simulator truth never enters policy APIs."""
from dataclasses import dataclass,replace
import numpy as np
import open3d as o3d
from scipy.ndimage import label
from env.virtual3d import VirtualConfig,VirtualWorld
from utils.grid_geometry import inflated_obstacles


@dataclass(frozen=True)
class VirtualConfigV2(VirtualConfig):
    width_m: float=16.
    height_m: float=12.
    truncation_m: float=.12
    pose_noise_m: float=0.
    occluded_objects: bool=False


class VirtualWorldV2(VirtualWorld):
    def __init__(self,config,seed=1,layout='rooms',semantic_condition='aligned'):
        if layout not in ('rooms','warehouse') or semantic_condition not in ('aligned','shuffled','absent'):
            raise ValueError('invalid scene condition')
        self.config=config;self.seed=seed;self.semantic_condition=semantic_condition
        self.width,self.height=config.width_m,config.height_m
        self.shape=(round(self.height/config.resolution_m),round(self.width/config.resolution_m))
        if self.width<10 or self.height<8 or config.resolution_m!=.2:
            raise ValueError('v2 requires >=10x8m on 0.2m grid')
        self.mesh=o3d.geometry.TriangleMesh();self.triangle_classes=[];self.boxes=[];self.objects=[]
        self._ray=o3d.t.geometry.RaycastingScene(nthreads=1);self._geometry_labels={}
        def box(x,y,z,sx,sy,sz,category=1):
            mesh=o3d.geometry.TriangleMesh.create_box(sx,sy,sz).translate((x,y,z))
            mesh.paint_uniform_color([.6,.6,.6]);mesh.compute_vertex_normals()
            gid=self._ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            self._geometry_labels[gid]=category
            self.mesh+=mesh;self.triangle_classes.extend([category]*len(mesh.triangles))
            if z<1.2 and z+sz>.15:self.boxes.append((x,y,sx,sy))
        w,h=self.width,self.height
        box(0,0,-.1,w,h,.1)
        box(0,0,0,w,.2,2.4);box(0,h-.2,0,w,.2,2.4)
        box(0,0,0,.2,h,2.4);box(w-.2,0,0,.2,h,2.4)
        if layout=='rooms':
            # Four rooms connected through a cross-shaped opening; no room ID
            # is provided to the planner. Door width is not a tuned outcome.
            for y0,y1 in ((.2,h/2-1.),(h/2+1.,h-.2)):
                box(w/2-.1,y0,0,.2,y1-y0,2.4)
            for x0,x1 in ((.2,w/2-1.),(w/2+1.,w-.2)):
                box(x0,h/2-.1,0,x1-x0,.2,2.4)
        rng=np.random.default_rng(seed)
        # Dense and sparse semantic regions, with varying geometry within each
        # category. GT evaluation is uniform surface area, never class weighted.
        probabilities=rng.permutation([.2,.8,.8,.2])
        xs=np.linspace(1.7,w-2.7,4);ys=np.array([1.5,h/2-2.0,h/2+1.4,h-2.5])
        for j,y0 in enumerate(ys):
            for i,x0 in enumerate(xs):
                x=x0+float(rng.uniform(-.2,.2));y=y0+float(rng.uniform(-.15,.15))
                region=(int(y>h/2)*2+int(x>w/2))
                complex_geometry=rng.random()<probabilities[region]
                sx=float(rng.uniform(.8,1.1));sy=float(rng.uniform(.5,.75))
                height=float(rng.uniform(.9,1.4))
                category=3 if complex_geometry else 2
                self.objects.append(dict(box=[x,y,0,sx,sy,height],category=category))
                if not complex_geometry:
                    box(x,y,0,sx,sy,height,category)
                else:
                    levels=int(rng.integers(2,5));thick=float(rng.uniform(.06,.12))
                    for z in np.linspace(.15,height-.15,levels):box(x,y,float(z),sx,sy,thick,category)
                    for dx in (0,sx-.08):
                        for dy in (0,sy-.08):box(x+dx,y+dy,0,.08,.08,height,category)
                    if rng.random()<.7:box(x+.25*sx,y+.2*sy,.25,.4*sx,.5*sy,.25,category)
                    if config.occluded_objects:
                        # Backboards toward the central aisle make the first
                        # observed geometry less informative about hidden shelf
                        # structure. Generic open shelves remain a control.
                        board_y=y+sy-.08 if y<h/2 else y
                        box(x,board_y,0,sx,.08,height,category)
        self.triangle_classes=np.asarray(self.triangle_classes)
        self.occupancy=np.zeros(self.shape,bool)
        r=config.resolution_m
        for x,y,sx,sy in self.boxes:
            c0,c1=max(0,int(np.floor(x/r))),min(self.shape[1],int(np.ceil((x+sx)/r)))
            r0,r1=max(0,int(np.floor((h-y-sy)/r))),min(self.shape[0],int(np.ceil((h-y)/r)))
            self.occupancy[r0:r1,c0:c1]=True
        self._blocked=inflated_obstacles(self.occupancy,config.robot_radius_m/r)
        self.start=(self.shape[0]//2,5);self.heading=1
        components,_=label(~self._blocked)
        # In rooms the horizontal divider occupies the west starting row;
        # project to the first free cell in a fixed 1m neighborhood.
        candidates=[(self.start[0]+dr,self.start[1]) for dr in (0,1,-1,2,-2,3,-3,4,-4,5,-5)]
        self.start=next(cell for cell in candidates if components[cell]>0)
        self.reachable=components==components[self.start]
        self.position=self.start;self.step_count=self.collisions=self.moves=0
        fx=config.width_px/(2*np.tan(np.deg2rad(config.fov_deg/2)))
        self.intrinsic=np.array([[fx,0,(config.width_px-1)/2],[0,fx,(config.height_px-1)/2],[0,0,1.]])

    def _pose_offset(self):
        # Bounded smooth pose-input error; not a simulated SLAM estimator.
        t=self.step_count
        return self.config.pose_noise_m*np.array([np.sin(t/31+self.seed),np.cos(t/37+self.seed),.3*np.sin(t/23)])

    def sense(self):
        frame=super().sense()
        # Material image independent of category labels and label perturbation.
        rgb=np.repeat(np.where(frame.depth_m>0,153,0).astype(np.uint8)[...,None],3,axis=2)
        pose=frame.world_from_camera.copy();pose[:3,3]+=self._pose_offset()
        return replace(frame,color_rgb=rgb,world_from_camera=pose).validate()

    def scan(self):
        scan=super().scan();pose=scan.world_from_laser.copy();pose[:3,3]+=self._pose_offset()
        return replace(scan,world_from_laser=pose)
