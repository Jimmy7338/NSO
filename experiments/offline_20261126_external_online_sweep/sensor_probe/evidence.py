"""Bounded-resolution evidence from accumulated registered RGB-D/laser packets.

This certificate is an offline/simulation candidate. It assumes a static scene,
a common registered frame and the declared sensor model. It does not infer free
space from no-return rays, evaluate motion, or access the simulator reference.
"""

import hashlib
from math import ceil
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree


# Sensor-model tolerances for the archived 20 cm circular-footprint simulation.
LASER_MOUNT_HEIGHT_M = .25
GROUND_HALF_BAND_M = .04
MAX_FLOOR_SAMPLE_DISTANCE_M = .025
MAX_LASER_SAMPLE_GAP_M = .08
MAX_PROBE_SPACING_M = .02
BODY_LEVELS_M = np.array([.16, .25, .40, .55, .70, .85, 1., 1.15, 1.19])
BODY_BAND_M = (.15, 1.2)
HIT_CLEARANCE_PAD_M = .005
RAY_CLEARANCE_PAD_M = .01
RANGE_LIMIT_EPS_M = .001


def _dist_to_segment(xy, p, q):
    offset = q-p
    denominator = float(offset @ offset)
    if denominator <= 1e-24:
        return np.linalg.norm(xy-p, axis=1)
    t = np.clip((xy-p) @ offset/denominator, 0., 1.)
    return np.linalg.norm(xy-(p+t[:,None]*offset), axis=1)


def _swept_xy(p, q, radius):
    lower, upper = np.minimum(p, q)-radius, np.maximum(p, q)+radius
    count = [max(1, ceil((high-low)/MAX_PROBE_SPACING_M - 1e-9)) for low, high in zip(lower, upper, strict=True)]
    axes = [lo+(np.arange(n)+.5)*((hi-lo)/n) for lo, hi, n in zip(lower, upper, count, strict=True)]
    xx, yy = np.meshgrid(*axes)
    cells = np.column_stack((xx.ravel(), yy.ravel()))
    inside = cells[_dist_to_segment(cells, p, q) <= radius + 1e-9]
    # Explicit boundary witnesses also prevent a thin edge/gap being missed
    # just because the search centers sample halfway across a footprint cell.
    angles = np.linspace(0., 2*np.pi, max(12, ceil(2*np.pi*radius/MAX_PROBE_SPACING_M)), endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    headings = q-p
    if np.linalg.norm(headings) <= 1e-12:
        return np.vstack((inside, p+radius*directions))
    along = directions @ headings
    sides = np.array([-headings[1], headings[0]]) / np.linalg.norm(headings)
    line = np.linspace(p, q, ceil(np.linalg.norm(headings)/MAX_PROBE_SPACING_M)+1)
    return np.vstack((inside, p+radius*directions[along<=0], q+radius*directions[along>=0],
                      line+sides*radius, line-sides*radius))


def _scan_hits(scan):
    ranges = np.asarray(scan.ranges_m)
    mask = np.isfinite(ranges) & (ranges > 0) & (ranges < scan.range_max_m-RANGE_LIMIT_EPS_M)
    distances = ranges[mask]
    angles = scan.angle_min_rad + np.flatnonzero(mask)*scan.angle_increment_rad
    local = np.column_stack((np.cos(angles)*distances, np.sin(angles)*distances, np.zeros(len(distances))))
    return local @ scan.world_from_laser[:3,:3].T + scan.world_from_laser[:3,3]


def _scan_support(scan, query):
    local = (query-scan.world_from_laser[:2,3]) @ scan.world_from_laser[:2,:2]
    radius = np.linalg.norm(local, axis=1)
    angle = np.arctan2(local[:,1], local[:,0])
    index = np.floor((angle-scan.angle_min_rad)/scan.angle_increment_rad).astype(int)
    count = len(scan.ranges_m)
    left, right = (scan.ranges_m[i % count] for i in (index,index+1))
    limit = scan.range_max_m-RANGE_LIMIT_EPS_M
    gap = 2*radius*np.sin(scan.angle_increment_rad/2)
    return ((left < limit) & (right < limit) & (left > radius+RAY_CLEARANCE_PAD_M)
            & (right > radius+RAY_CLEARANCE_PAD_M) & (gap <= MAX_LASER_SAMPLE_GAP_M))


def _depth_support(frame, query):
    local = (query-frame.world_from_camera[:3,3]) @ frame.world_from_camera[:3,:3]
    uv = (local @ frame.intrinsic.T)[:,:2] / np.where(local[:,2,None] > 0,local[:,2,None],np.inf)
    u, v = np.floor(uv).astype(int).T
    rows, cols = frame.depth_m.shape
    mask = (local[:,2] > .15) & (u >= 0) & (v >= 0) & (u+1 < cols) & (v+1 < rows)
    ids = np.flatnonzero(mask)
    values = np.stack([frame.depth_m[v[ids]+dy,u[ids]+dx] for dy in (0,1) for dx in (0,1)])
    mask[ids] = np.all((values > local[ids,2]+RAY_CLEARANCE_PAD_M) & (values < frame.depth_limit_m), axis=0)
    return mask


def _tree(parts):
    return cKDTree(np.vstack(parts)) if parts and any(len(p) for p in parts) else None


class AccumulatedEvidence:
    """Accumulate a single scene's registered measured observations."""

    def __init__(self):
        self._observations = []
        self._seen = set()
        self._ground_z = None
        self._scan_hits = []
        self._depth_hits = []
        self._floor_returns = []
        self._lower_returns = []
        self._trees = None
        self._certificates = {}

    def observe(self, frame, scan):
        if frame is None or scan is None:
            return
        digest = hashlib.sha256()
        for array in (frame.depth_m, frame.intrinsic, frame.world_from_camera, scan.ranges_m, scan.world_from_laser):
            digest.update(np.ascontiguousarray(array).view(np.uint8))
        identity = digest.digest()
        if identity in self._seen:
            return
        self._seen.add(identity)
        if self._ground_z is None:
            self._ground_z = float(scan.world_from_laser[2,3])-LASER_MOUNT_HEIGHT_M
        depth_limit = float(scan.range_max_m)-RANGE_LIMIT_EPS_M
        depth = SimpleNamespace(depth_m=np.asarray(frame.depth_m).copy(), intrinsic=np.asarray(frame.intrinsic).copy(),
                                world_from_camera=np.asarray(frame.world_from_camera).copy(), depth_limit_m=depth_limit)
        rays = SimpleNamespace(ranges_m=np.asarray(scan.ranges_m).copy(), world_from_laser=np.asarray(scan.world_from_laser).copy(),
                               range_max_m=scan.range_max_m, angle_min_rad=scan.angle_min_rad, angle_increment_rad=scan.angle_increment_rad)
        cloud, _ = frame.points(stride=1)
        cloud_depth = (cloud-frame.world_from_camera[:3,3]) @ frame.world_from_camera[:3,2]
        cloud = cloud[(cloud_depth > .15) & (cloud_depth < depth_limit)]
        height = cloud[:,2]-self._ground_z
        self._scan_hits.append(_scan_hits(rays)[:,:2])
        self._depth_hits.append(cloud[(height >= BODY_BAND_M[0]) & (height <= BODY_BAND_M[1]),:2])
        self._floor_returns.append(cloud[np.abs(height) <= GROUND_HALF_BAND_M,:2])
        self._lower_returns.append(cloud[height < -GROUND_HALF_BAND_M,:2])
        self._observations.append((depth,rays))
        self._trees = None
        self._certificates.clear()

    def certify(self, start_xy, end_xy, radius_m):
        p, q = np.asarray(start_xy, dtype=float), np.asarray(end_xy, dtype=float)
        radius = float(radius_m)
        if p.shape != (2,) or q.shape != (2,) or not np.isfinite([*p,*q,radius]).all() or radius <= 0.:
            raise ValueError("Expected two finite XY poses and a positive footprint radius")
        key = (*p,*q,radius)
        if key not in self._certificates:
            self._certificates[key] = self._certify(p,q,radius)
        return self._certificates[key].copy()

    def _certify(self, p, q, radius):
        if not self._observations:
            return {"allowed": False, "reason": "no_registered_sensor_packets"}
        if self._trees is None:
            self._trees = tuple(_tree(points) for points in (
                self._scan_hits, self._depth_hits, self._floor_returns, self._lower_returns))
        lidar, depth, floor, lower = self._trees
        midpoint = (p+q)/2
        query_radius = radius+np.linalg.norm(q-p)/2+.10
        near = {}
        for name, tree in (("lidar",lidar),("rgbd",depth),("lower_support",lower)):
            ids = [] if tree is None else tree.query_ball_point(midpoint, query_radius)
            near[name] = float(_dist_to_segment(tree.data[ids],p,q).min()) if ids else None
        clearance = radius+HIT_CLEARANCE_PAD_M
        if any(near[n] is not None and near[n] <= clearance for n in ("lidar","rgbd")):
            return {"allowed":False, "reason":"measured_body_hit_in_padded_capsule","nearest_measured_m":near}
        if near["lower_support"] is not None and near["lower_support"] <= radius:
            return {"allowed":False, "reason":"lower_floor_evidence_in_capsule","nearest_measured_m":near}
        xy = _swept_xy(p,q,radius)
        novel = np.linalg.norm(xy-p,axis=1) > radius+1e-9
        ground_ok = np.zeros(len(xy),dtype=bool) if floor is None else floor.query(xy)[0] <= MAX_FLOOR_SAMPLE_DISTANCE_M
        if not ground_ok.all():
            return {"allowed":False, "reason":"missing_positive_ground_support","nearest_measured_m":near,
                    "unsupported_xy_probes":int((~ground_ok).sum()),"probes":len(xy),
                    "missing_only_starting_disk":bool(ground_ok[novel].all())}
        lidar_ok = np.zeros(len(xy),dtype=bool)
        for unused,scan in reversed(self._observations):
            missing = np.flatnonzero(~lidar_ok)
            lidar_ok[missing] = _scan_support(scan,xy[missing])
            if lidar_ok.all():
                break
        if not lidar_ok.all():
            return {"allowed":False, "reason":"missing_measured_laser_clearance","nearest_measured_m":near,
                    "unsupported_xy_probes":int((~lidar_ok).sum()),"probes":len(xy),
                    "missing_only_starting_disk":bool(lidar_ok[novel].all())}
        heights = BODY_LEVELS_M+self._ground_z
        volume = np.column_stack((np.repeat(xy,len(heights),axis=0),np.tile(heights,len(xy))))
        volume_ok = np.zeros(len(volume),dtype=bool)
        for frame, unused in reversed(self._observations):
            missing = np.flatnonzero(~volume_ok)
            volume_ok[missing] = _depth_support(frame,volume[missing])
            if volume_ok.all():
                break
        camera_ok = volume_ok.reshape(len(xy),len(heights)).all(axis=1)
        if not camera_ok.all():
            return {"allowed":False, "reason":"missing_measured_rgbd_body_clearance","nearest_measured_m":near,
                    "unsupported_xy_probes":int((~camera_ok).sum()),"probes":len(xy),
                    "missing_only_starting_disk":bool(camera_ok[novel].all())}
        return {"allowed":True,"reason":"measured_full_capsule","nearest_measured_m":near,"probes":len(xy),
                "missing_only_starting_disk":False}
