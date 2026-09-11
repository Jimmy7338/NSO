"""Sensor-only 2D occupancy, TSDF reconstruction and surface-view evidence."""
import numpy as np
import open3d as o3d
from env.grid_exploration import GridObservation, readonly_copy
from utils.grid_geometry import supercover_line


class SensorMapper:
    def __init__(self, shape, config):
        self.shape=shape;self.config=config
        self.belief=np.full(shape,-1,np.int8)
        self.visible=np.zeros(shape,bool)
        self.surface={}
        self.volume=o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=config.voxel_m,sdf_trunc=config.voxel_m*4,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        self.frames=0

    def grid_cell(self,point):
        return (self.shape[0]-1-int(np.floor(point[1]/self.config.resolution_m)),
                int(np.floor(point[0]/self.config.resolution_m)))

    def update(self, frame, scan=None):
        frame.validate();c=self.config
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(frame.color_rgb,np.uint8)),
            o3d.geometry.Image(np.ascontiguousarray(frame.depth_m,np.float32)),
            depth_scale=1.,depth_trunc=c.max_depth_m,convert_rgb_to_intensity=False)
        intrinsic=o3d.camera.PinholeCameraIntrinsic(frame.depth_m.shape[1],frame.depth_m.shape[0],
            frame.intrinsic[0,0],frame.intrinsic[1,1],frame.intrinsic[0,2],frame.intrinsic[1,2])
        self.volume.integrate(rgbd,intrinsic,np.linalg.inv(frame.world_from_camera))
        self.frames+=1;self.visible[:]=False
        origin=self.grid_cell(frame.world_from_camera[:3,3])
        if scan is not None:
            origin=self.grid_cell(scan.world_from_laser[:3,3])
            angles=scan.angle_min_rad+np.arange(len(scan.ranges_m))*scan.angle_increment_rad
            for angle,distance in zip(angles,scan.ranges_m):
                if not np.isfinite(distance) or distance<=0:continue
                local=np.array([np.cos(angle)*distance,np.sin(angle)*distance,0.])
                target=self.grid_cell(scan.world_from_laser[:3,:3]@local+scan.world_from_laser[:3,3])
                ray=supercover_line(target[0]-origin[0],target[1]-origin[1])
                for dr,dc in ray[:-1]:
                    r,col=origin[0]+dr,origin[1]+dc
                    if 0<=r<self.shape[0] and 0<=col<self.shape[1]:
                        if self.belief[r,col]!=1:self.belief[r,col]=0
                        self.visible[r,col]=True
                r,col=target
                if 0<=r<self.shape[0] and 0<=col<self.shape[1]:
                    self.visible[r,col]=True
                    if distance<scan.range_max_m-1e-3:self.belief[r,col]=1
                    elif self.belief[r,col]!=1:self.belief[r,col]=0
        points,labels=frame.points(stride=2)
        # Depth contributes obstacles at heights a single planar laser can miss.
        obstacle=points[(points[:,2]>.15)&(points[:,2]<1.2)]
        for p in obstacle:
            r,col=self.grid_cell(p)
            if 0<=r<self.shape[0] and 0<=col<self.shape[1]:
                self.belief[r,col]=1;self.visible[r,col]=True
        if 0<=origin[0]<self.shape[0] and 0<=origin[1]<self.shape[1]:
            self.belief[origin]=0;self.visible[origin]=True
        # One update per evidence voxel per frame. View bins, not raw duplicate
        # frame counts, track whether new viewpoints add angular diversity.
        keys=np.floor(points/.15).astype(np.int32)
        _,indices=np.unique(keys,axis=0,return_index=True)
        for i in indices:
            key=tuple(map(int,keys[i]));p=points[i]
            az=np.arctan2(frame.world_from_camera[1,3]-p[1],frame.world_from_camera[0,3]-p[0])
            view=int(np.floor((az+np.pi)/(2*np.pi)*8))%8
            old=self.surface.get(key)
            if old is None:self.surface[key]=[p.copy(),1<<view,int(labels[i])]
            else:
                old[1]|=1<<view
                if labels[i]:old[2]=int(labels[i])

    def observation(self,position,heading,step,collision=False):
        return GridObservation(readonly_copy(self.belief),readonly_copy(self.visible),
                               tuple(position),int(heading),int(step),bool(collision))

    def mesh(self):
        mesh=self.volume.extract_triangle_mesh();mesh.compute_vertex_normals()
        return mesh

    def evidence(self):
        rows=list(self.surface.values())
        if not rows:return np.empty((0,3)),np.empty(0,int),np.empty(0,int)
        return np.stack([r[0] for r in rows]),np.asarray([r[1] for r in rows]),np.asarray([r[2] for r in rows])
