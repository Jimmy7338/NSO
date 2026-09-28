"""Six exterior-documentation assets with versioned stereo-depth sensitivity.

Truth belongs to the simulator/evaluator. Runtime APIs expose ordinary RGB-D,
ideal planar lidar and the existing exact discrete pose contract. These are
procedural mechanism fixtures, not photorealism or calibrated ZED simulation.
"""
from dataclasses import dataclass
import math
import numpy as np
import open3d as o3d
from scipy.ndimage import label
from env.virtual3d import camera_pose
from env.virtual3d_v2 import VirtualConfigV2
from env.canonical_box_scan_v14 import CanonicalScanInspectionWorldV14
from env.canonical_rgbd_v15 import render_axial_depth, render_marker_rgb
from env.virtual3d_inspection_v4 import union_surface_from_boxes, read_inspection_markers_rgb
from utils.grid_geometry import inflated_obstacles
from utils.rgbd_contract import RGBDFrame


SENSOR_MODELS_V19 = {'ideal': (0., 0.), 'iid_010px': (.10, 0.),
                     'iid_025px': (.25, 0.), 'bias_025px_iid_025px': (.25, .25)}
PARENTS_V19 = ('D19-P00', 'D19-P01')
ASSIGNMENTS_V19 = ('A_complex_B_simple', 'A_simple_B_complex')


@dataclass(frozen=True)
class FacilityConfigV19(VirtualConfigV2):
    depth_sigma_m: float = 0.  # The base world's additive metre noise is unused.
    dropout: float = 0.
    pose_noise_m: float = 0.
    voxel_m: float = .04
    truncation_m: float = .12
    max_depth_m: float = 5.
    max_steps: int = 1000
    appearance: str = 'marked'
    semantic_source: str = 'rgb_marker'
    stereo_model: str = 'iid_025px'
    stereo_reference_fx_px: float = 480.
    stereo_baseline_m: float = .12


def stereo_depth_v19(clean_depth, *, model='iid_025px', parent_index=0,
                     step=0, noise_seed=1901, reference_fx_px=480., baseline_m=.12,
                     max_depth_m=5.):
    """Apply reference-resolution disparity error; no label or asset identity.

    Pixel noise is repeatable at the same parent/time/seed. Acquisition at a
    different action step uses a fresh draw. Bias is constant in disparity over
    all pixels/frames, deliberately distinct from independently sampled noise.
    """
    if model not in SENSOR_MODELS_V19:
        raise ValueError('undeclared V19 stereo sensitivity model')
    for value in (parent_index, step, noise_seed):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError('nonnegative integer sensor seed identifiers required')
    if not np.isfinite([reference_fx_px, baseline_m, max_depth_m]).all() or min(reference_fx_px, baseline_m, max_depth_m) <= 0:
        raise ValueError('positive finite stereo calibration/range required')
    clean = np.asarray(clean_depth)
    if clean.ndim != 2 or not np.isfinite(clean).all() or np.any(clean < 0):
        raise ValueError('finite nonnegative axial depth image required')
    valid = (clean > .15) & (clean <= max_depth_m)
    if model == 'ideal':
        return np.where(valid, clean, 0.).astype(np.float32)
    sigma, bias = SENSOR_MODELS_V19[model]
    rng = np.random.default_rng(np.random.SeedSequence([int(parent_index), int(step), int(noise_seed), 73919]))
    fb = reference_fx_px * baseline_m
    disparity = np.divide(fb, clean, out=np.zeros(clean.shape, float), where=valid)
    disparity += bias + rng.normal(0., sigma, clean.shape)
    depth = np.divide(fb, disparity, out=np.zeros(clean.shape, float), where=disparity > 0)
    valid &= (disparity > 0) & (depth > .15) & (depth <= max_depth_m)
    return np.where(valid, depth, 0.).astype(np.float32)


class FacilityWorldV19(CanonicalScanInspectionWorldV14):
    def __init__(self, parent='D19-P00', assignment='A_complex_B_simple',
                 semantic_condition='aligned', sensor_model='iid_025px', noise_seed=1901):
        if parent not in PARENTS_V19 or assignment not in ASSIGNMENTS_V19:
            raise ValueError('undeclared V19 development parent or paired assignment')
        if semantic_condition not in ('aligned', 'shuffled', 'absent'):
            raise ValueError('invalid semantic sensor intervention')
        if sensor_model not in SENSOR_MODELS_V19:
            raise ValueError('undeclared V19 stereo sensitivity model')
        if isinstance(noise_seed, bool) or not isinstance(noise_seed, (int, np.integer)) or noise_seed < 0:
            raise ValueError('nonnegative integer sensor seed required')
        self.parent, self.assignment = parent, assignment
        self.parent_index = PARENTS_V19.index(parent)
        self.semantic_condition, self.noise_seed = semantic_condition, int(noise_seed)
        self.seed = 19000 + self.parent_index
        if self.parent_index == 0:
            width, height = 18., 12.
            fronts = [(6.6, 5.4), (11.4, 5.4), (2.2, 5.4), (15.8, 5.4), (4.6, 9.6), (13.4, 9.6)]
            separators = [(4.3, 3.2, .2, 4.4), (8.9, 3.2, .2, 4.4), (13.5, 3.2, .2, 4.4)]
            start_xy = (9., 1.4)
        else:
            width, height = 20., 14.
            fronts = [(7.2, 5.8), (12.4, 5.8), (2.4, 6.4), (17.4, 7.), (5., 11.2), (15.2, 11.2)]
            separators = [(4.7, 3.4, .2, 5.2), (9.9, 3.4, .2, 5.2), (15.1, 3.4, .2, 5.2)]
            start_xy = (10., 1.6)
        self.config = FacilityConfigV19(width_m=width, height_m=height, stereo_model=sensor_model)
        self.width, self.height = width, height
        self.shape = (round(height/.2), round(width/.2))
        self._solid_primitives = []; self.boxes = []; self._physical_markers = []
        self.objects = []; self.inspection_truth = []

        def box(x, y, z, sx, sy, sz, owner=1):
            self._solid_primitives.append((x, y, z, sx, sy, sz, owner))
            if z < 1.2 and z+sz > .15:
                self.boxes.append((x, y, sx, sy))

        box(0, 0, -.12, width, height, .12)
        box(0, 0, 0, width, .2, 2.4); box(0, height-.2, 0, width, .2, 2.4)
        box(0, .2, 0, .2, height-.4, 2.4); box(width-.2, .2, 0, .2, height-.4, 2.4)
        for x, y, sx, sy in separators:
            box(x, y, 0, sx, sy, 2.2)
        complex_flags = [assignment == ASSIGNMENTS_V19[0], assignment == ASSIGNMENTS_V19[1],
                         False, True, True, False]
        for index, ((cx, front), complex_shape) in enumerate(zip(fronts, complex_flags)):
            owner, category = 100+index, 3 if complex_shape else 2
            # Both types have a closed enclosure and common broad front face.
            # The two critical types differ only behind that initially visible
            # face. Protrusions are external, solid, >=12cm physical features.
            box(cx-1.1, front, 0, 2.2, .16, 1.6, owner)
            box(cx-.40, front+.16, 0, .80, .80, 1.6, owner)
            if complex_shape:
                box(cx-.42, front+.96, .30, .84, .28, .32, owner)
                box(cx-.30, front+.96, 1.04, .60, .28, .36, owner)
                box(cx-.62, front+.42, .52, .22, .32, .24, owner)
                box(cx+.40, front+.42, .92, .22, .32, .24, owner)
            self._physical_markers.append(dict(x0=cx-.36, x1=cx+.36, y=front,
                z0=.88, z1=1.24, physical_class=category))
            self.objects.append(dict(id=index, owner=owner, category=category,
                evaluation_bounds=[[cx-1.22, front-.12, .01], [cx+1.22, front+1.36, 1.72]],
                center=[cx, front+.62, .80], front_center=[cx, front, .80],
                name=chr(65+index), external_geometry='closed_body_with_external_attachments' if complex_shape else 'closed_body'))
        self.mesh, self.triangle_owners, self.surface_measure_audit = union_surface_from_boxes(self._solid_primitives)
        self.triangle_classes = self.triangle_owners.copy()
        for item in self.objects:
            self.triangle_classes[self.triangle_owners == item['owner']] = item['category']
        self.occupancy = np.zeros(self.shape, bool)
        for x, y, sx, sy in self.boxes:
            c0, c1 = math.floor(x/.2+1e-8), math.ceil((x+sx)/.2-1e-8)
            r0, r1 = math.floor((height-y-sy)/.2+1e-8), math.ceil((height-y)/.2-1e-8)
            self.occupancy[max(0, r0):min(self.shape[0], r1), max(0, c0):min(self.shape[1], c1)] = True
        self._blocked = inflated_obstacles(self.occupancy, self.config.robot_radius_m/.2)
        self.start = self._cell(*start_xy)
        groups, _ = label(~self._blocked)
        if groups[self.start] == 0:
            raise ValueError('unsafe initial pose')
        self.reachable = groups == groups[self.start]
        if np.any((~self._blocked) & ~self.reachable):
            raise ValueError('disconnected safe floor')
        self.position = self.start; self.heading = 0
        self.step_count = self.collisions = self.moves = 0
        c = self.config; focal = c.width_px/(2*np.tan(np.deg2rad(c.fov_deg/2)))
        self.intrinsic = np.array([[focal, 0, (c.width_px-1)/2], [0, focal, (c.height_px-1)/2], [0, 0, 1.]])
        self.sensor_metadata = dict(model=sensor_model, noise_seed=self.noise_seed,
            seed_fields=['parent_index', 'action_step', 'noise_seed', '73919'],
            reference_fx_px=c.stereo_reference_fx_px, baseline_m=c.stereo_baseline_m,
            output_fx_px=focal, pose='perfect centred discrete CPU contract',
            scope='generic stereo disparity sensitivity; no ZED calibration or material model')

    def sense(self):
        c = self.config
        pose = camera_pose(self.position, self.heading, c, self.shape[0])
        boxes = np.asarray(self._solid_primitives, dtype=float)[:, :6]
        clean, points = render_axial_depth(boxes, self.intrinsic, pose, c.height_px, c.width_px, c.max_depth_m)
        rgb = render_marker_rgb(clean, points, self._physical_markers if c.appearance == 'marked' else [])
        depth = stereo_depth_v19(clean, model=c.stereo_model, parent_index=self.parent_index,
            step=self.step_count, noise_seed=self.noise_seed, reference_fx_px=c.stereo_reference_fx_px,
            baseline_m=c.stereo_baseline_m, max_depth_m=c.max_depth_m)
        semantic = read_inspection_markers_rgb(rgb)
        if self.semantic_condition == 'shuffled':
            semantic = np.where(semantic == 2, 3, np.where(semantic == 3, 2, semantic)).astype(np.uint8)
        elif self.semantic_condition == 'absent':
            semantic[:] = 0
        semantic[depth == 0] = 0
        return RGBDFrame(self.step_count*c.action_duration_s, depth, rgb, self.intrinsic.copy(), pose, semantic).validate()

    def instance_mesh(self, object_id):
        item = next(x for x in self.objects if x['id'] == object_id)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices).copy())
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(self.mesh.triangles)[self.triangle_owners == item['owner']].copy())
        mesh.remove_unreferenced_vertices()
        return mesh
