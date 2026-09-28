"""Pixel acquisition for the frozen V29 scene; scripted 1 m primitives.

This is a simulator, never an ANS planning interface. References and safe-floor
coverage are evaluation-only. All poses, solids and reference windows receive
the same translation; no semantic labels enter the measured mapper.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
from scipy.ndimage import label

from env.virtual3d import VirtualConfig, camera_pose
from env.canonical_rgbd_v15 import render_axial_depth, render_marker_rgb
from env.canonical_box_scan_v14 import box_union_ranges
from env.facility_documentation_v19 import stereo_depth_v19
from nso.box_union_geometry_v30 import BoxV30, union_exterior_faces_v30
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket
from utils.rgbd_contract import RGBDFrame, PlanarScan


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'configs/virtual3d/v29_information_scene_20260917.json'
SHIFT = np.array([5.5, .5, 0.])


def mesh_from_faces(faces):
    vertices, triangles = [], []
    for face in faces:
        offset = len(vertices)
        vertices.extend(face.vertices())
        triangles.extend(((offset, offset+1, offset+2), (offset, offset+2, offset+3)))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, float).reshape(-1, 3))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, np.int32).reshape(-1, 3))
    mesh.remove_duplicated_vertices()
    return mesh


def translated_box(bounds, owner=None):
    b = np.array(bounds, float).reshape(3, 2) + SHIFT[:, None]
    return BoxV30(tuple(b.ravel()), owner)


def scene_boxes(hypothesis):
    cfg = json.loads(SCENE.read_text())
    if hypothesis not in (0, 1):
        raise ValueError('only the two frozen assignments are supported')
    boxes = [translated_box(b) for b in cfg['class_independent_wall_bounds']]
    facilities = []
    for index, station in enumerate(cfg['stations']):
        members = []
        for key in ['body_relative_bounds'] + (['complex_attachment_relative_bounds']
                if cfg['hypotheses'][hypothesis][index] == 'complex' else []):
            b = np.array(cfg[key], float)
            b[:2] += station['center_x']
            members.append(translated_box(b, index))
        facilities.append(members)
        boxes.extend(members)
    # Same class-independent ground across both assignments. Contact at z=0
    # does not add an internal surface to the facility reference.
    boxes.append(BoxV30((0., 11., 0., 8., -.1, 0.), None))
    return cfg, boxes, facilities


def swept_clear(start_xy, end_xy, boxes, radius=.2):
    """Exact axis-aligned segment / rectangle Euclidean footprint clearance."""
    start, end = np.asarray(start_xy, float), np.asarray(end_xy, float)
    if start.shape != (2,) or end.shape != (2,) or not np.isfinite([start, end]).all():
        raise ValueError('finite XY segment required')
    if np.count_nonzero(np.abs(start-end) > 1e-9) > 1:
        raise ValueError('driver only permits axis-aligned primitives')
    lo, hi = np.minimum(start, end), np.maximum(start, end)
    for box in boxes:
        b = np.asarray(box.bounds).reshape(3, 2)
        if b[2, 1] <= .15 or b[2, 0] >= 1.2:
            continue
        gap = np.maximum(np.maximum(b[:2, 0]-hi, lo-b[:2, 1]), 0.)
        if float(gap @ gap) <= radius**2 + 1e-12:
            return False
    return True


class InformationPixelWorldV30:
    def __init__(self, hypothesis, episode_id):
        self.hypothesis, self.episode_id = hypothesis, episode_id
        self.source, self.boxes, self.facilities = scene_boxes(hypothesis)
        self.config = VirtualConfig(camera_height_m=.9, depth_sigma_m=0., dropout=0.,
            voxel_m=.04, max_steps=48)
        self.shape = (40, 55)
        self.transform = GridTransform(self.shape, .2)
        self.position = self.transform.world_to_cell(SHIFT[:2])
        self.start = self.position
        self.heading = self.step_count = self.collisions = 0
        b = np.asarray([box.bounds for box in self.boxes]).reshape(-1, 3, 2)
        self.primitives = np.concatenate((b[:, :, 0], b[:, :, 1]-b[:, :, 0]), axis=1)
        self.intrinsic = np.array([[48., 0., 47.5], [0., 48., 35.5], [0., 0., 1.]])
        v, u = np.mgrid[:72, :96]
        self.ray_norm = np.sqrt(1 + ((u-47.5)/48)**2 + ((v-35.5)/48)**2)
        self.markers = []
        for index, station in enumerate(self.source['stations']):
            x = station['center_x'] + SHIFT[0]
            self.markers.append(dict(x0=x-.35, x1=x+.35, y=2.+SHIFT[1], z0=.5, z1=1.2,
                physical_class=3 if self.source['hypotheses'][hypothesis][index]=='complex' else 2))
        self.last_clean_depth = None

    def packet(self, action=None, collision=False):
        pose = camera_pose(self.position, self.heading, self.config, self.shape[0])
        clean, points = render_axial_depth(self.primitives, self.intrinsic, pose, 72, 96, 4.)
        clean = np.where(clean*self.ray_norm <= 4., clean, 0.).astype(np.float32)
        rgb = render_marker_rgb(clean, points, self.markers)
        depth = stereo_depth_v19(clean, model='iid_025px', parent_index=0,
            step=self.step_count, noise_seed=1901, max_depth_m=4.)
        depth[depth*self.ray_norm > 4.] = 0.
        self.last_clean_depth = clean
        frame = RGBDFrame(float(self.step_count), depth, rgb, self.intrinsic.copy(), pose,
                          np.zeros(depth.shape, np.uint8)).validate()
        laser = np.eye(4)
        laser[:3, 0] = pose[:3, 2]
        laser[:3, 1] = -pose[:3, 0]
        laser[:3, 3] = pose[:3, 3]
        laser[2, 3] = .25
        angles = -np.pi + np.arange(180)*2*np.pi/180
        directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(180))) @ laser[:3, :3].T
        ranges = box_union_ranges(laser[:3, 3], directions, self.primitives, 4.).astype(np.float32)
        scan = PlanarScan(float(self.step_count), ranges, -np.pi, 2*np.pi/180, 4., laser)
        return SensorPacket('v30-frozen-two-stations', self.episode_id,
            f'{self.episode_id}:{self.step_count:03d}', self.step_count, frame, scan,
            self.position, self.heading, 'v30-canonical-pixel-euclidean4-iid025',
            'simulator-exact-discrete-pose', action=action, collision=collision,
            done=self.step_count==48).validate(self.transform, self.config)

    def step(self, action):
        if self.step_count >= 48 or action not in ('forward', 'left', 'right'):
            raise ValueError('invalid or over-budget primitive')
        collision = False
        if action == 'forward':
            directions = ((-5, 0), (0, 5), (5, 0), (0, -5))
            target = tuple(int(a+b) for a, b in zip(self.position, directions[self.heading]))
            inside = all(0 <= x < n for x, n in zip(target, self.shape))
            collision = not inside or not swept_clear(self.transform.cell_to_world(self.position),
                self.transform.cell_to_world(target), self.boxes)
            if collision:
                self.collisions += 1
            else:
                self.position = target
        else:
            self.heading = (self.heading + (1 if action == 'right' else -1)) % 4
        self.step_count += 1
        return self.packet(action, collision)

    def evaluation_reference(self):
        """Only call after all prediction snapshots and physical actions end."""
        objects, meshes = [], {}
        for index, station in enumerate(self.source['stations']):
            x = station['center_x'] + SHIFT[0]
            objects.append(dict(id=index, evaluation_bounds=[[x-1.75, 2., .01], [x+1.75, 7., 2.1]]))
            meshes[index] = mesh_from_faces(union_exterior_faces_v30(self.facilities[index]))
        reference = SimpleNamespace(objects=objects, instance_mesh=lambda index: meshes[index])
        safe = np.zeros(self.shape, bool)
        for index in np.ndindex(self.shape):
            xy = self.transform.cell_to_world(index)
            safe[index] = swept_clear(xy, xy, self.boxes)
        groups, _ = label(safe)
        if not groups[self.start]:
            raise ValueError('evaluation start must be safe')
        return reference, groups == groups[self.start]
