"""Validation bridge from blind analytic measurements to the production certificate."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from evidence import AccumulatedEvidence


ENCLOSURE = (((-2., -2.), (-2., 2.)), ((2., -2.), (2., 2.)),
             ((-2., -2.), (2., -2.)), ((-2., 2.), (2., 2.)))
SIDE_PX = 256
CAMERA_VIEWPOINTS = ((-1.25, -.55, .90), (-1.25, .55, .90), (1.45, 0., .90),
                    (.15, -1.45, .90), (.15, 1.45, .90))


def _cross(a, b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


@dataclass
class AnalyticDepthPacket:
    depth_m: np.ndarray
    intrinsic: np.ndarray
    world_from_camera: np.ndarray
    points_xyz_m: np.ndarray

    def points(self, stride=1):
        return self.points_xyz_m[::stride], np.zeros(len(self.points_xyz_m[::stride]),dtype=np.uint16)


def _analytic_depth(camera_origin, target, walls, points):
    view = target-camera_origin
    view /= np.linalg.norm(view)
    right = np.cross(view,[0.,0.,1.])
    right /= np.linalg.norm(right)
    down = np.cross(view,right)
    pose = np.eye(4)
    pose[:3,:3] = np.column_stack((right,down,view))
    pose[:3,3] = camera_origin
    f = SIDE_PX/2
    center = (SIDE_PX-1)/2
    intrinsic = np.array(((f,0.,center),(0.,f,center),(0.,0.,1.)))
    row,col = np.indices((SIDE_PX,SIDE_PX))
    optical = np.stack(((col-center)/f,(row-center)/f,np.ones_like(row)),axis=-1)
    directions = optical @ pose[:3,:3].T
    distance = np.full(row.shape,np.inf)
    for p, q in walls:
        diff = np.asarray(q)-p
        delta = np.asarray(p)-camera_origin[:2]
        den = _cross(directions[...,:2],diff)
        crossing = np.abs(den)>1e-10
        t = np.divide(_cross(delta,diff),den,out=np.full(row.shape,np.inf),where=crossing)
        s = np.divide(_cross(delta,directions[...,:2]),den,out=np.full(row.shape,np.inf),where=crossing)
        height = camera_origin[2] + np.multiply(t,directions[...,2],out=np.zeros(row.shape),where=np.isfinite(t))
        distance = np.minimum(distance,np.where(crossing&(t>.15)&(s>=0)&(s<=1)&(height>=0)&(height<=2.),t,np.inf))
    for plane in (0.,2.):
        direction_z = directions[...,2]
        t = np.divide(plane-camera_origin[2],direction_z,out=np.full(row.shape,np.inf),where=np.abs(direction_z)>1e-10)
        distance = np.minimum(distance,np.where(t>.15,t,np.inf))
    depth = np.where(distance<8.,distance,0.).astype(np.float32)
    return AnalyticDepthPacket(depth,intrinsic,pose,points)


def evaluate(packet):
    start,end = packet["start_xyz_m"],packet["end_xyz_m"]
    heading_angle = np.arctan2(end[1]-start[1],end[0]-start[0])
    cosine,sine = np.cos(heading_angle),np.sin(heading_angle)
    rotation = np.array(((cosine,-sine),(sine,cosine)))
    transform = np.eye(4)
    transform[:2,:2] = rotation
    transform[:3,3] = packet["laser_origin_xyz_m"]
    azimuths = packet["laser_azimuths_rad"]-heading_angle
    scan = SimpleNamespace(ranges_m=packet["laser_ranges_m"],world_from_laser=transform,
                           angle_min_rad=float(azimuths[0]),angle_increment_rad=float(azimuths[1]-azimuths[0]),
                           range_max_m=packet["laser_range_max_m"])
    walls = packet.get("room_and_fixture_wall_segments_xy")
    if walls is None:
        walls = np.asarray(ENCLOSURE) @ rotation.T+start[:2]
    points = np.vstack([packet[key] for key in ("floor_surface_xyz_m","rgbd_obstacle_points_xyz_m","rgbd_drop_points_xyz_m")])
    cameras = np.asarray(CAMERA_VIEWPOINTS)
    cameras[:,:2] = cameras[:,:2] @ rotation.T + start[:2]
    cameras[:,2] += start[2]
    target = (start+end)/2 + [0.,0.,.65]
    evidence = AccumulatedEvidence()
    for origin in cameras:
        evidence.observe(_analytic_depth(origin,target,walls,points),scan)
    return evidence.certify(start[:2],end[:2],packet["radius_m"])


def admit(packet):
    return evaluate(packet)["allowed"]
