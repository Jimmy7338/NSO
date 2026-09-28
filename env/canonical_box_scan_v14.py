"""Deterministic ideal lidar for the axis-aligned box-union simulator only.

Shared faces get identical float64 intersection arithmetic irrespective of
triangulation or primitive order. This is not quantization of robot scans,
sensor noise modelling, or permission to relax old byte-equality checks.
"""
import numpy as np

from env.virtual3d import camera_pose
from env.virtual3d_inspection_v4 import InspectionWorldV4
from utils.rgbd_contract import PlanarScan


def box_union_ranges(origin, directions, boxes, max_range):
    origin = np.asarray(origin, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if origin.shape != (3,) or directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError('expected origin[3] and directions[N,3]')
    if boxes.size == 0:
        boxes = boxes.reshape(0, 6)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError('boxes must be [x,y,z,sx,sy,sz]')
    if not all(np.isfinite(v).all() for v in (origin, directions, boxes)):
        raise ValueError('finite ray/box geometry required')
    if not np.isfinite(max_range) or max_range <= 0 or np.any(boxes[:, 3:] <= 0):
        raise ValueError('positive range and box dimensions required')
    if not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-10, rtol=0):
        raise ValueError('unit ray directions required')
    # Same coordinate precision as the simulator's unique union surface mesh.
    low = np.round(boxes[:, :3], 9)
    high = np.round(boxes[:, :3] + boxes[:, 3:], 9)
    result = np.full(len(directions), max_range, dtype=np.float64)
    for offset in range(0, len(directions), 256):
        rays = directions[offset:offset + 256]
        enter = np.full((len(rays), len(boxes)), -np.inf)
        leave = np.full_like(enter, np.inf)
        valid = np.ones_like(enter, bool)
        for axis in range(3):
            direction = rays[:, axis, None]
            parallel = direction == 0.
            valid &= ~parallel | ((origin[axis] >= low[:, axis]) & (origin[axis] <= high[:, axis]))
            first = np.divide(low[:, axis] - origin[axis], direction,
                              out=np.zeros_like(enter), where=~parallel)
            second = np.divide(high[:, axis] - origin[axis], direction,
                               out=np.zeros_like(enter), where=~parallel)
            enter = np.maximum(enter, np.where(parallel, -np.inf, np.minimum(first, second)))
            leave = np.minimum(leave, np.where(parallel, np.inf, np.maximum(first, second)))
        # Robot origins must be outside solids. Otherwise the nearest primitive
        # exit can be an internal union surface; reject rather than misrender it.
        if np.any(np.all((origin > low) & (origin < high), axis=1)):
            raise ValueError('lidar origin inside a solid primitive')
        hit = np.where(valid & (leave >= np.maximum(enter, 0.)), np.maximum(enter, 0.), np.inf)
        if len(boxes):
            result[offset:offset + len(rays)] = np.minimum(hit.min(axis=1), max_range)
    return result.astype(np.float32)


class CanonicalScanInspectionWorldV14(InspectionWorldV4):
    """Explicit new simulator version; camera renderer and geometry unchanged."""
    def scan(self):
        c = self.config
        camera = camera_pose(self.position, self.heading, c, self.shape[0])
        pose = np.eye(4)
        pose[:3, 0] = camera[:3, 2]
        pose[:3, 1] = -camera[:3, 0]
        pose[:3, 2] = [0, 0, 1]
        pose[:3, 3] = camera[:3, 3]
        pose[2, 3] = c.laser_height_m
        angles = -np.pi + np.arange(c.laser_rays) * 2 * np.pi / c.laser_rays
        directions = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)]) @ pose[:3, :3].T
        boxes = np.asarray(self._solid_primitives, dtype=float)[:, :6]
        ranges = box_union_ranges(pose[:3, 3], directions, boxes, c.max_depth_m)
        return PlanarScan(self.step_count * c.action_duration_s, ranges, -np.pi,
                          2 * np.pi / c.laser_rays, c.max_depth_m, pose)
