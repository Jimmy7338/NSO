"""Small analytic sensor fixture; never a development scene or performance run.

Each observation ray-intersects the same two fixed, finite rectangles in world
coordinates. This file is a test-data producer; online modules receive only its
PaidRGBDObservationV40 return value, never fixture geometry or labels as fields.
"""
import math

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40


ANALYTIC_MARKER_COLOR_V42 = (40, 100, 220)


def analytic_plane_observation_v42(paid_step, pose_xyyaw_rad=(.75, .75, 0.), *, frame_id=None):
    """Render a closed-form finite plane/label, optical axial depth in metres.

    The fixture is deliberately noiseless with known camera pose. Outside the
    finite rectangles depth is missing (zero), not free-space evidence. The
    camera is upright at 0.9 m; a turn or translation changes actual ray hits.
    """
    pose = np.asarray(pose_xyyaw_rad, dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("finite x,y,yaw_rad required")
    if type(paid_step) is not int or paid_step < 0:
        raise ValueError("nonnegative integer paid_step required")
    height, width = 72, 96
    intrinsic = np.array([[48., 0., 47.5], [0., 48., 35.5], [0., 0., 1.]])
    forward = np.array([math.cos(pose[2]), math.sin(pose[2]), 0.])
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((np.cross(forward, [0., 0., 1.]), [0., 0., -1.], forward))
    transform[:3, 3] = [pose[0], pose[1], .9]
    yy, xx = np.indices((height, width))
    rays_camera = np.stack(((xx-47.5)/48., (yy-35.5)/48., np.ones_like(xx)), axis=-1)
    rays_world = rays_camera @ transform[:3, :3].T
    origin = transform[:3, 3]
    depth = np.full((height, width), np.inf)
    rgb = np.full((height, width, 3), 165, dtype=np.uint8)
    # Plane and physical label plate are fixed across the entire sequence.
    # Bounds are fixture-generator internals, never part of online packets.
    surfaces = ((1.85, .15, 1.35, .1, 1.7, (165, 165, 165)),
                (1.83, .59, .91, .75, 1.05, ANALYTIC_MARKER_COLOR_V42))
    for plane_x, ymin, ymax, zmin, zmax, color in surfaces:
        axial = np.full((height, width), np.inf)
        moving = np.abs(rays_world[..., 0]) > 1e-12
        np.divide(plane_x-origin[0], rays_world[..., 0], out=axial, where=moving)
        # Avoid inf*0 at parallel rays; those never satisfy the valid mask.
        finite_axial = np.where(np.isfinite(axial), axial, 0.)
        hit_y = origin[1] + finite_axial*rays_world[..., 1]
        hit_z = origin[2] + finite_axial*rays_world[..., 2]
        visible = (moving & (axial >= .1) & (axial <= 4.) & (axial < depth)
                   & (hit_y >= ymin) & (hit_y <= ymax) & (hit_z >= zmin) & (hit_z <= zmax))
        depth[visible] = axial[visible]
        rgb[visible] = color
    depth[~np.isfinite(depth)] = 0.
    return PaidRGBDObservationV40(frame_id or f"analytic-plane-v42-{paid_step}", paid_step,
                                  rgb, depth, intrinsic, transform)


def analytic_forward_sequence_v42():
    """One initial sensor grant and one actual 0.25 m forward observation.

    This does not execute a simulator or claim a complete planner trajectory.
    The second frame expands the ledger's initially radius-limited support;
    that is not a claim that these surfaces were occluded in the first image.
    """
    return (analytic_plane_observation_v42(0),
            analytic_plane_observation_v42(1, (1., .75, 0.)))
