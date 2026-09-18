"""Pixel-only bridge for frozen V33r1; truth never enters a planner packet.

An explicit query renders one RGB-D frame and one independent8m planar scan.
No implicit initial/successor observations, TSDF, inferred surfaces or Q exist.
Known exact synthetic marker colours are a public controlled observation
contract. Colour removal is not a natural-image marker detector: only RGB
matches are neutralized, preserving the observed marker's location and extent.
"""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

from env.virtual3d import VirtualConfig, camera_pose
from env.canonical_rgbd_v15 import render_axial_depth
from env.canonical_box_scan_v14 import box_union_ranges
from env.facility_documentation_v19 import stereo_depth_v19
from nso.box_union_geometry_v30 import BoxV30, union_exterior_faces_v30
from nso.cpu_sensor_contract_v10 import GridTransform
from nso.sensor_contract_v34 import SensorPacketV34
from utils.rgbd_contract import RGBDFrame, PlanarScan

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT/'configs/virtual3d/v33_direction_scene_r1_20260917.json'
SCENE_SHA256 = '55bc23b57de6e13472970eb1b6670a6e6040db74d3b61e155fe84f9fa08cd7c9'
MARKER_COLORS_V34 = {2: (40, 100, 220), 3: (220, 60, 40)}
NEUTRAL_MARKER_RGB = (127, 127, 127)
NOISE_MODELS_V34 = ('clean', 'iid_025px')
COUNTER_KEYS = ('worlds', 'packets', 'clean_depth_queries', 'scan_queries', 'step_calls')
_COUNTERS = {key: 0 for key in COUNTER_KEYS}
EPS = 1e-7
HEADINGS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def sensor_counts_v34():
    """Readonly snapshot of actual call counts in this process; no reset API."""
    return MappingProxyType(dict(_COUNTERS))


def marker_codes_v34(rgb):
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError('uint8 HxWx3 RGB required')
    codes = np.zeros(pixels.shape[:2], np.uint8)
    for code, color in MARKER_COLORS_V34.items():
        codes[np.all(pixels == np.asarray(color, np.uint8), axis=-1)] = code
    return codes


def nonsemantic_rgb(rgb):
    """Public RGB-only colour removal; no depth, labels, world or GT mask."""
    result = np.array(rgb, copy=True)
    result[marker_codes_v34(rgb) > 0] = NEUTRAL_MARKER_RGB
    return result


def cue_from_rgb(frame):
    """Read only artificial visible RGB; depth support is reported separately.

    A16-pixel colour quorum identifies the controlled class. This is not an
    instance association, pose-persistence or calibrated semantic confidence.
    It does not inspect frame.semantic or the world's hidden hypothesis.
    """
    has_depth = isinstance(frame, RGBDFrame)
    codes = marker_codes_v34(frame.color_rgb if has_depth else frame)
    counts = {code: int(np.count_nonzero(codes == code)) for code in MARKER_COLORS_V34}
    depth_counts = None
    if has_depth:
        depth = np.asarray(frame.depth_m)
        if depth.shape != codes.shape or not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError('aligned finite nonnegative measured depth required')
        depth_counts = {code: int(np.count_nonzero((codes == code) & (depth > 0))) for code in counts}
    supported = [code for code, count in counts.items() if count >= 16]
    # Mixed visible classes stay ambiguous even when only one reaches quorum.
    selected = supported[0] if len(supported) == 1 and sum(v > 0 for v in counts.values()) == 1 else None
    return dict(class_id=selected, type={2: 'type_A', 3: 'type_B'}.get(selected),
        pixel_count=counts.get(selected, 0), pixel_counts=counts,
        valid_depth_pixel_counts=depth_counts, minimum_pixels=16,
        source='public_exact_artificial_RGB_colors_only', natural_semantic_network=False)


@dataclass(frozen=True)
class InformationConfigV34(VirtualConfig):
    camera_height_m: float = .9
    depth_sigma_m: float = 0.
    dropout: float = 0.
    voxel_m: float = .04
    truncation_m: float = .12
    max_steps: int = 42
    laser_range_m: float = 8.
    width_m: float = 7.
    height_m: float = 5.


def swept_clear_v34(start_xy, end_xy, boxes, radius=.2):
    """Whole XY footprints; axis-aligned sweep; V33 tangent epsilon unchanged."""
    start, end = np.asarray(start_xy, float), np.asarray(end_xy, float)
    if start.shape != (2,) or end.shape != (2,) or not np.isfinite([start, end]).all():
        raise ValueError('finite XY segment required')
    if np.count_nonzero(np.abs(start-end) > 1e-9) > 1:
        raise ValueError('only axis-aligned motion is declared')
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('finite nonnegative radius required')
    lo, hi = np.minimum(start, end), np.maximum(start, end)
    for box in boxes:
        b = np.asarray(box.bounds if hasattr(box, 'bounds') else box, float).reshape(3, 2)
        gap = np.maximum(np.maximum(b[:2, 0]-hi, lo-b[:2, 1]), 0.)
        if float(gap @ gap) < radius**2-EPS:
            return False
    return True


def _mesh_from_faces(faces):
    # Evaluation-only, invoked explicitly after prediction output is fixed.
    import open3d as o3d
    vertices, triangles = [], []
    for face in faces:
        offset = len(vertices); vertices.extend(face.vertices())
        triangles.extend(((offset, offset+1, offset+2), (offset, offset+2, offset+3)))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, float).reshape(-1, 3))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, np.int32).reshape(-1, 3))
    mesh.remove_duplicated_vertices()
    return mesh


class InformationPixelWorldV34:
    def __init__(self, parent_id, hypothesis, episode_id, noise_model='clean'):
        if parent_id not in ('P00', 'P01') or type(hypothesis) is not int or hypothesis not in (0, 1):
            raise ValueError('declared parent and integer hypothesis0/1 required')
        if not isinstance(episode_id, str) or not episode_id or noise_model not in NOISE_MODELS_V34:
            raise ValueError('episode identity and declared noise model required')
        payload = SCENE.read_bytes()
        if hashlib.sha256(payload).hexdigest() != SCENE_SHA256:
            raise ValueError('frozen V33r1 geometry changed')
        self.source = json.loads(payload)
        self.parent = next(p for p in self.source['parents'] if p['id'] == parent_id)
        self.parent_id, self.hypothesis, self.episode_id = parent_id, hypothesis, episode_id
        self.noise_model = noise_model; self.noise_seed = 1901
        self._counts = {key: 0 for key in COUNTER_KEYS}
        cells = np.asarray(self.parent['nav_cells'], int)
        low, high = cells.min(axis=0)-.5, cells.max(axis=0)+.5
        self.public_bounds_v33 = tuple(float(x) for x in (low[0], high[0], low[1], high[1]))
        self.shift = np.array([-low[0], -low[1], 0.])
        size = high-low
        self.config = InformationConfigV34(width_m=float(size[0]), height_m=float(size[1]))
        self.shape = (int(round(size[1]/.2)), int(round(size[0]/.2)))
        self.transform = GridTransform(self.shape, .2)
        self._nav_cells = frozenset(map(tuple, self.parent['nav_cells']))
        self._pose = tuple(self.parent['anchor'])
        self.position = self.pose_to_cell(self._pose)
        self.start, self.heading = self.position, self._pose[2]
        self.step_count = self.collisions = 0
        self._reference_boxes = []
        self.solid_boxes = []
        for asset in self.parent['hypotheses'][hypothesis]['assets']:
            members = [BoxV30(tuple((np.asarray(b).reshape(3, 2)+self.shift[:, None]).ravel()), asset['id'])
                       for b in asset['boxes']]
            self._reference_boxes.extend(members); self.solid_boxes.extend(members)
        for b in self.parent['background_boxes']:
            self.solid_boxes.append(BoxV30(tuple((np.asarray(b).reshape(3, 2)+self.shift[:, None]).ravel()), None))
        ground = BoxV30((0., self.config.width_m, 0., self.config.height_m, -.1, 0.), None)
        self.boxes = tuple(self.solid_boxes)+ (ground,)
        bounds = np.asarray([b.bounds for b in self.boxes]).reshape(-1, 3, 2)
        self.primitives = np.concatenate((bounds[:, :, 0], bounds[:, :, 1]-bounds[:, :, 0]), axis=1)
        self.intrinsic = np.array([[48., 0., 47.5], [0., 48., 35.5], [0., 0., 1.]])
        v, u = np.mgrid[:72, :96]
        self.ray_norm = np.sqrt(1+((u-47.5)/48)**2+((v-35.5)/48)**2)
        self.last_clean_depth = None
        self._floor_cache = None
        self._record('worlds')

    @property
    def counts(self):
        return MappingProxyType(dict(self._counts))

    @property
    def pose(self):
        return self._pose

    def _record(self, key):
        self._counts[key] += 1; _COUNTERS[key] += 1

    def v33_to_world(self, xy):
        point = np.asarray(xy, float)
        if point.shape != (2,) or not np.isfinite(point).all(): raise ValueError('finite original XY required')
        return tuple(float(x) for x in point+self.shift[:2])

    def world_to_v33(self, xy):
        point = np.asarray(xy, float)
        if point.shape != (2,) or not np.isfinite(point).all(): raise ValueError('finite metric XY required')
        return tuple(float(x) for x in point-self.shift[:2])

    def _validate_pose(self, pose):
        if (len(pose) != 3 or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in pose)
                or tuple(pose[:2]) not in self._nav_cells or int(pose[2]) not in range(4)):
            raise ValueError('declared safe integer V33 pose required')
        return tuple(int(v) for v in pose)

    def pose_to_cell(self, pose):
        pose = self._validate_pose(pose)
        cell = self.transform.world_to_cell(self.v33_to_world(pose[:2]))
        if not np.allclose(self.transform.cell_to_world(cell), self.v33_to_world(pose[:2]), rtol=0, atol=1e-10):
            raise ValueError('original integer pose not aligned to a raster cell centre')
        return cell

    def cell_to_pose(self, cell, heading):
        xy = self.world_to_v33(self.transform.cell_to_world(cell))
        rounded = tuple(int(round(v)) for v in xy)
        if not np.allclose(xy, rounded, rtol=0, atol=1e-10):
            raise ValueError('raster centre is not a declared integer motion pose')
        return self._validate_pose((*rounded, heading))

    def _rgb(self, clean, points):
        rgb = np.repeat(np.where(clean > 0, 153, 0).astype(np.uint8)[..., None], 3, axis=2)
        local = points-self.shift
        origin = np.asarray(self.parent['device_frame']['origin_xyz'], float)
        local = local-origin
        for _ in range((-self.parent['device_frame']['quarter_turns_ccw']) % 4):
            x, y = local[..., 0].copy(), local[..., 1].copy()
            local[..., 0], local[..., 1] = -y, x
        # First-hit world points alone decide where the surface decal appears.
        mask = ((clean > 0) & (np.abs(local[..., 1]-1.1) < 1e-5)
                & (local[..., 0] >= -.35) & (local[..., 0] <= .35)
                & (local[..., 2] >= .55) & (local[..., 2] <= 1.25))
        rgb[mask] = MARKER_COLORS_V34[2+self.hypothesis]
        return rgb

    def _packet(self, pose, step, noise_model, action=None, collision=False):
        pose = self._validate_pose(pose)
        if type(step) is not int or step < 0 or noise_model not in NOISE_MODELS_V34:
            raise ValueError('nonnegative query step and declared noise required')
        cell = self.pose_to_cell(pose)
        camera = camera_pose(cell, pose[2], self.config, self.shape[0])
        self._record('packets'); self._record('clean_depth_queries')
        clean, points = render_axial_depth(self.primitives, self.intrinsic, camera, 72, 96, 4.)
        clean = np.where(clean*self.ray_norm <= 4., clean, 0.).astype(np.float32)
        rgb = self._rgb(clean, points)
        if noise_model == 'clean': depth = clean.copy()
        else:
            depth = stereo_depth_v19(clean, model=noise_model, parent_index=int(self.parent_id[-1]),
                step=step, noise_seed=self.noise_seed, max_depth_m=4.)
            depth[depth*self.ray_norm > 4.] = 0.
        self.last_clean_depth = clean.copy()
        self.last_clean_depth.setflags(write=False)
        frame = RGBDFrame(float(step), depth, rgb, self.intrinsic.copy(), camera,
                          np.zeros(clean.shape, np.uint8))
        laser = np.eye(4)
        laser[:3, 0] = camera[:3, 2]; laser[:3, 1] = -camera[:3, 0]
        laser[:3, 3] = camera[:3, 3]; laser[2, 3] = .25
        angles = -np.pi+np.arange(180)*2*np.pi/180
        rays = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(180))) @ laser[:3, :3].T
        self._record('scan_queries')
        ranges = box_union_ranges(laser[:3, 3], rays, self.primitives, 8.).astype(np.float32)
        scan = PlanarScan(float(step), ranges, -np.pi, 2*np.pi/180, 8., laser)
        return SensorPacketV34('v34-'+self.parent_id, self.episode_id,
            f'{self.episode_id}:{step}:{pose[0]}:{pose[1]}:{pose[2]}', step, frame, scan,
            cell, pose[2], 'v34-first-hit-pixels-euclidean4-laser8-'+noise_model,
            'simulator-exact-discrete-pose', action=action, collision=collision,
            done=step == self.config.max_steps).validate(self.transform, self.config)

    def packet_at(self, pose, step=0, noise_model='clean'):
        """One stateless query; paid state unchanged; cached clean depth updated."""
        return self._packet(pose, step, noise_model)

    def packet(self):
        return self._packet(self._pose, self.step_count, self.noise_model)

    def step(self, action):
        if action not in ('forward', 'left', 'right') or self.step_count >= self.config.max_steps:
            raise ValueError('unsupported or over-budget primitive')
        self._record('step_calls')
        x, y, h = self._pose; collision = False
        if action == 'forward':
            dx, dy = HEADINGS[h]; target = (x+dx, y+dy, h)
            collision = ((target[:2] not in self._nav_cells) or not swept_clear_v34(
                self.v33_to_world((x, y)), self.v33_to_world(target[:2]), self.solid_boxes))
            if collision: self.collisions += 1
            else: self._pose = target
        else: self._pose = (x, y, (h+(1 if action == 'right' else -1)) % 4)
        self.position = self.pose_to_cell(self._pose); self.heading = self._pose[2]
        self.step_count += 1
        return self._packet(self._pose, self.step_count, self.noise_model, action, bool(collision))

    def evaluation_floor(self):
        """GT-only full public raster/declared-grid denominators; no sensing."""
        if self._floor_cache is None:
            safe = np.zeros(self.shape, bool)
            for cell in np.ndindex(self.shape):
                xy = self.transform.cell_to_world(cell)
                safe[cell] = swept_clear_v34(xy, xy, self.solid_boxes)
            reachable = np.zeros(self.shape, bool)
            if not safe[self.start]: raise ValueError('unsafe evaluation anchor')
            reachable[self.start] = True; todo = deque([self.start])
            while todo:
                current = todo.popleft(); start_xy = self.transform.cell_to_world(current)
                for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                    nxt = (current[0]+dr, current[1]+dc)
                    if (all(0 <= v < n for v, n in zip(nxt, self.shape)) and safe[nxt] and not reachable[nxt]
                            and swept_clear_v34(start_xy, self.transform.cell_to_world(nxt), self.solid_boxes)):
                        reachable[nxt] = True; todo.append(nxt)
            grid_cells = np.asarray([self.pose_to_cell((*p, 0)) for p in self.parent['nav_cells']], np.int32)
            for array in (safe, reachable, grid_cells): array.setflags(write=False)
            self._floor_cache = dict(safe=safe, reachable=reachable, declared_grid_cells=grid_cells,
                raster_denominator=int(reachable.sum()), grid_denominator=len(grid_cells),
                public_bounds_v33=self.public_bounds_v33, translation_xy=tuple(self.shift[:2]),
                coverage_status='denominators_only_not_actual_C', resolution_m=.2)
        return MappingProxyType(dict(self._floor_cache))

    @property
    def reachable(self):
        return self.evaluation_floor()['reachable']

    def instance_mesh(self, owner=0, vertical_only=False):
        if owner != 0: raise ValueError('one frozen task asset')
        return _mesh_from_faces(union_exterior_faces_v30(self._reference_boxes, vertical_only=vertical_only))

    def evaluation_reference(self):
        b = np.asarray([box.bounds for box in self._reference_boxes]).reshape(-1, 3, 2)
        low, high = b[:, :, 0].min(axis=0), b[:, :, 1].max(axis=0)
        objects = [dict(id=0, evaluation_bounds=[(low-.25).tolist(), (high+.25).tolist()])]
        reference = SimpleNamespace(objects=objects, instance_mesh=self.instance_mesh,
            reachable=self.reachable, config=self.config, intrinsic=self.intrinsic)
        return reference, self.reachable


World = InformationPixelWorldV34
