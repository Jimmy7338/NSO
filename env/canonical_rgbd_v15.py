"""Versioned ideal RGB-D/scan renderer for artificial box-union facilities.

Rendering may read simulator geometry/materials; planners receive only frames.
No post-hoc rounding of old sensor logs or weakened matching tolerance is used.
"""
import numpy as np

from env.canonical_box_scan_v14 import CanonicalScanInspectionWorldV14, box_union_ranges
from env.virtual3d import camera_pose
from env.virtual3d_inspection_v4 import MARKER_COLORS, read_inspection_markers_rgb
from utils.rgbd_contract import RGBDFrame


def render_axial_depth(boxes, intrinsic, pose, height, width, max_depth):
    intrinsic = np.asarray(intrinsic, dtype=float)
    pose = np.asarray(pose, dtype=float)
    if (not isinstance(height, int) or not isinstance(width, int) or height < 1 or width < 1
            or not np.isfinite(max_depth) or max_depth <= .15):
        raise ValueError('positive image size and valid axial range required')
    if intrinsic.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError('camera matrix shape mismatch')
    if not np.isfinite(intrinsic).all() or not np.isfinite(pose).all():
        raise ValueError('finite camera calibration required')
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError('positive focal lengths required')
    if not np.allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3), atol=1e-10, rtol=0):
        raise ValueError('rigid camera rotation required')
    v, u = np.mgrid[:height, :width]
    optical = np.stack([(u-intrinsic[0,2])/intrinsic[0,0],
                        (v-intrinsic[1,2])/intrinsic[1,1], np.ones_like(u)], axis=-1)
    direction = optical @ pose[:3,:3].T
    norm = np.linalg.norm(direction, axis=2)
    # A no-hit sentinel must map beyond the axial far plane for EVERY pixel.
    ranges = box_union_ranges(pose[:3,3], (direction/norm[...,None]).reshape(-1,3),
                              boxes, (max_depth + 1.) * float(norm.max()))
    axial = ranges.reshape(height,width).astype(float) / norm
    valid = (axial > .15) & (axial <= max_depth)
    depth = np.where(valid, axial, 0.).astype(np.float32)
    points = pose[:3,3] + direction * depth[...,None]
    return depth, points


def render_marker_rgb(depth, points, markers):
    rgb = np.repeat(np.where(depth>0,153,0).astype(np.uint8)[...,None],3,axis=2)
    for marker in markers:
        mask = ((depth>0) & (np.abs(points[...,1]-marker['y'])<1e-5)
                & (points[...,0]>marker['x0']) & (points[...,0]<marker['x1'])
                & (points[...,2]>marker['z0']) & (points[...,2]<marker['z1']))
        rgb[mask] = MARKER_COLORS[marker['physical_class']]
    return rgb


class CanonicalRGBDInspectionWorldV15(CanonicalScanInspectionWorldV14):
    def __init__(self, config, seed=1, semantic_condition='aligned'):
        if config.depth_sigma_m != 0 or config.dropout != 0 or config.pose_noise_m != 0:
            raise ValueError('V15 ideal renderer requires zero sensor/pose noise; no silent noise removal')
        if config.semantic_source != 'rgb_marker':
            raise ValueError('V15 requires RGB-derived marker semantics')
        super().__init__(config, seed=seed, semantic_condition=semantic_condition)

    def sense(self):
        c = self.config
        pose = camera_pose(self.position,self.heading,c,self.shape[0])
        boxes = np.asarray(self._solid_primitives,dtype=float)[:,:6]
        depth, points = render_axial_depth(boxes,self.intrinsic,pose,c.height_px,c.width_px,c.max_depth_m)
        rgb = render_marker_rgb(depth,points,self._physical_markers if c.appearance=='marked' else [])
        semantic = read_inspection_markers_rgb(rgb)
        if self.semantic_condition=='shuffled':
            semantic = np.where(semantic==2,3,np.where(semantic==3,2,semantic)).astype(np.uint8)
        elif self.semantic_condition=='absent':
            semantic[:] = 0
        return RGBDFrame(self.step_count*c.action_duration_s,depth,rgb,self.intrinsic.copy(),pose,semantic).validate()
