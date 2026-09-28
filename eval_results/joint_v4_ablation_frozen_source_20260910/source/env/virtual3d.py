"""Procedural metric 3D world and CPU RGB-D camera; GT stays inside simulator.

World: x=east, y=north, z=up. Grid: row increases south, column east.
Robot heading: N/E/S/W = 0/1/2/3. Camera: x=right,y=down,z=forward.
"""
from dataclasses import dataclass
import math
import numpy as np
import open3d as o3d
from scipy.ndimage import label
from utils.grid_geometry import inflated_obstacles, DIRECTIONS
from utils.rgbd_contract import RGBDFrame, PlanarScan


@dataclass(frozen=True)
class VirtualConfig:
    resolution_m: float = .2
    robot_radius_m: float = .2
    camera_height_m: float = .8
    width_px: int = 96
    height_px: int = 72
    fov_deg: float = 90.
    max_depth_m: float = 4.
    depth_sigma_m: float = .01
    dropout: float = .01
    action_duration_s: float = 1.
    voxel_m: float = .06
    max_steps: int = 120
    laser_height_m: float = .25
    laser_rays: int = 180

    def __post_init__(self):
        if min(self.resolution_m,self.camera_height_m,self.max_depth_m,self.action_duration_s,self.voxel_m)<=0:
            raise ValueError('metric parameters must be positive')
        if self.robot_radius_m<0 or self.depth_sigma_m<0 or not 0<=self.dropout<1:
            raise ValueError('invalid noise or robot radius')
        if not 0<self.fov_deg<180 or self.width_px<8 or self.height_px<8 or self.max_steps<1:
            raise ValueError('invalid camera or budget')


def camera_pose(position, heading, config, grid_height):
    r,c=position; dr,dc=DIRECTIONS[heading]
    forward=np.array([dc,-dr,0.],float)
    right=np.cross(forward,[0.,0.,1.])
    pose=np.eye(4);pose[:3,:3]=np.column_stack([right,[0,0,-1],forward])
    pose[:3,3]=[(c+.5)*config.resolution_m,(grid_height-r-.5)*config.resolution_m,config.camera_height_m]
    return pose


class VirtualWorld:
    def __init__(self, config, seed=1, layout='rooms', semantic_condition='aligned'):
        if layout not in ('rooms','clutter') or semantic_condition not in ('aligned','shuffled','absent'):
            raise ValueError('unknown scene or semantic condition')
        self.config=config;self.seed=seed;self.semantic_condition=semantic_condition
        self.shape=(30,40);self.height=6.;self.width=8.
        if not np.isclose(self.shape[0]*config.resolution_m,self.height) or not np.isclose(self.shape[1]*config.resolution_m,self.width):
            raise ValueError('prototype world requires 0.2m grid')
        self.mesh=o3d.geometry.TriangleMesh();self.triangle_classes=[];self.boxes=[]
        self._ray=o3d.t.geometry.RaycastingScene(nthreads=1)
        self._geometry_labels={};self.objects=[]
        def box(x,y,z,sx,sy,sz,category=1,object_id=None):
            mesh=o3d.geometry.TriangleMesh.create_box(sx,sy,sz).translate((x,y,z))
            colors={1:[.65,.65,.65],2:[.3,.6,.8],3:[.9,.5,.2]}
            mesh.paint_uniform_color(colors[category]);mesh.compute_vertex_normals()
            gid=self._ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            self._geometry_labels[gid]=category
            self.mesh+=mesh;self.triangle_classes.extend([category]*len(mesh.triangles))
            if z<1.2 and z+sz>.15:self.boxes.append((x,y,sx,sy))
            if object_id is not None:self.objects.append(dict(id=object_id,category=category,box=[x,y,z,sx,sy,sz]))
        box(0,0,-.1,8,6,.1)
        box(0,0,0,8,.2,2.4);box(0,5.8,0,8,.2,2.4)
        box(0,0,0,.2,6,2.4);box(7.8,0,0,.2,6,2.4)
        if layout=='rooms':
            box(3.8,.2,0,.2,2.,2.4);box(3.8,3.8,0,.2,2.,2.4)
        rng=np.random.default_rng(seed)
        # Geometry varies with seed. Objects include flat/simple and occluded,
        # multi-level structures; visible semantics are synthetic class labels.
        positions=[(1.2,1.2),(5.5,1.0),(1.3,4.2),(5.4,4.0)]
        for i,(x,y) in enumerate(positions):
            x+=float(rng.uniform(-.2,.2));y+=float(rng.uniform(-.2,.2))
            if (i+seed)%2:
                box(x,y,0,1.0,.6,.9,2,i)
            else:
                for z in (.2,.65,1.1):box(x,y,z,1.0,.6,.08,3,i)
                for dx in (0,.92):
                    for dy in (0,.52):box(x+dx,y+dy,0,.08,.08,1.3,3,i)
                box(x+.3,y+.2,.28,.3,.3,.25,3,i)
        self.triangle_classes=np.asarray(self.triangle_classes)
        self.occupancy=np.zeros(self.shape,bool)
        for x,y,sx,sy in self.boxes:
            # Conservative grid-square overlap with obstacle horizontal extent.
            c0=max(0,int(math.floor(x/config.resolution_m)))
            c1=min(self.shape[1],int(math.ceil((x+sx)/config.resolution_m)))
            r0=max(0,int(math.floor((self.height-y-sy)/config.resolution_m)))
            r1=min(self.shape[0],int(math.ceil((self.height-y)/config.resolution_m)))
            self.occupancy[r0:r1,c0:c1]=True
        self._blocked=inflated_obstacles(self.occupancy,config.robot_radius_m/config.resolution_m)
        self.start=(15,5);self.heading=1
        components,_=label(~self._blocked)
        if not components[self.start]:raise ValueError('invalid start')
        self.reachable=components==components[self.start]
        self.position=self.start;self.step_count=self.collisions=self.moves=0
        self.intrinsic=np.array([[config.width_px/(2*np.tan(np.deg2rad(config.fov_deg/2))),0,(config.width_px-1)/2],
             [0,config.width_px/(2*np.tan(np.deg2rad(config.fov_deg/2))),(config.height_px-1)/2],[0,0,1.]])

    def sense(self):
        c=self.config;pose=camera_pose(self.position,self.heading,c,self.shape[0])
        v,u=np.mgrid[:c.height_px,:c.width_px]
        directions=np.stack([(u-self.intrinsic[0,2])/self.intrinsic[0,0],
                             (v-self.intrinsic[1,2])/self.intrinsic[1,1],np.ones_like(u)],axis=-1)
        rays=np.empty((c.height_px,c.width_px,6),np.float32)
        rays[...,:3]=pose[:3,3];rays[...,3:]=directions@pose[:3,:3].T
        hits=self._ray.cast_rays(o3d.core.Tensor(rays),nthreads=1)
        # Rays have optical z=1, so t_hit is axial depth (not Euclidean range).
        depth=hits['t_hit'].numpy();valid=np.isfinite(depth)&(depth>.15)&(depth<=c.max_depth_m)
        labels=np.zeros(depth.shape,np.uint8)
        for gid,category in self._geometry_labels.items():labels[(hits['geometry_ids'].numpy()==gid)&valid]=category
        if self.semantic_condition=='shuffled':
            labels=np.where(labels==2,3,np.where(labels==3,2,labels)).astype(np.uint8)
        elif self.semantic_condition=='absent':labels[:]=0
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.step_count,739]))
        noisy=np.where(valid,depth,0.)+rng.normal(0,c.depth_sigma_m,depth.shape)
        valid&=(rng.random(depth.shape)>=c.dropout)&(noisy>.15)&(noisy<=c.max_depth_m)
        noisy=np.where(valid,noisy,0.).astype(np.float32);labels[~valid]=0
        palette=np.array([[0,0,0],[165,165,165],[76,153,204],[230,128,51]],np.uint8)
        return RGBDFrame(self.step_count*c.action_duration_s,noisy,palette[labels],self.intrinsic.copy(),pose,labels).validate()

    def step(self, action):
        if action not in ('forward','left','right','stop'):raise ValueError('unknown action')
        collision=False;self.step_count+=1
        if action=='forward':
            dr,dc=DIRECTIONS[self.heading];target=(self.position[0]+dr,self.position[1]+dc)
            if not (0<=target[0]<self.shape[0] and 0<=target[1]<self.shape[1]) or self._blocked[target]:
                collision=True;self.collisions+=1
            else:self.position=target;self.moves+=1
        elif action in ('left','right'):self.heading=(self.heading+(1 if action=='right' else -1))%4
        return self.sense(),collision,action=='stop' or self.step_count>=self.config.max_steps

    def scan(self):
        c=self.config;camera=camera_pose(self.position,self.heading,c,self.shape[0])
        pose=np.eye(4);pose[:3,0]=camera[:3,2];pose[:3,1]=-camera[:3,0]
        pose[:3,2]=[0,0,1];pose[:3,3]=camera[:3,3];pose[2,3]=c.laser_height_m
        angles=-np.pi+np.arange(c.laser_rays)*2*np.pi/c.laser_rays
        local=np.column_stack([np.cos(angles),np.sin(angles),np.zeros_like(angles)])
        rays=np.column_stack([np.tile(pose[:3,3],(len(angles),1)),local@pose[:3,:3].T]).astype(np.float32)
        ranges=self._ray.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
        ranges=np.minimum(ranges,c.max_depth_m).astype(np.float32)
        return PlanarScan(self.step_count*c.action_duration_s,ranges,-np.pi,2*np.pi/c.laser_rays,c.max_depth_m,pose)
