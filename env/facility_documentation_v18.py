"""Evaluator-owned compact paired facilities; policies receive sensors only.

Wide common front panels hide narrower rear equipment at the initial pose.
These are controlled mechanism fixtures, not scanned industrial datasets.
"""
from dataclasses import dataclass
import math
import numpy as np
import open3d as o3d
from scipy.ndimage import label
from env.virtual3d_v2 import VirtualConfigV2
from env.canonical_rgbd_v15 import CanonicalRGBDInspectionWorldV15
from env.virtual3d_inspection_v4 import union_surface_from_boxes
from utils.grid_geometry import inflated_obstacles


@dataclass(frozen=True)
class FacilityConfigV18(VirtualConfigV2):
    depth_sigma_m: float = 0.
    dropout: float = 0.
    pose_noise_m: float = 0.
    voxel_m: float = .04
    max_depth_m: float = 5.
    max_steps: int = 240
    appearance: str = 'marked'
    semantic_source: str = 'rgb_marker'


class FacilityWorldV18(CanonicalRGBDInspectionWorldV15):
    def __init__(self, parent='D18-P00', assignment='A_open_B_closed', semantic_condition='aligned'):
        if parent not in ('D18-P00', 'D18-P01'):
            raise ValueError('undeclared development parent')
        if assignment not in ('A_open_B_closed', 'A_closed_B_open'):
            raise ValueError('undeclared paired assignment')
        if semantic_condition not in ('aligned', 'shuffled', 'absent'):
            raise ValueError('invalid sensor semantic intervention')
        width, height = (10., 9.) if parent == 'D18-P00' else (12., 8.8)
        self.config = FacilityConfigV18(width_m=width, height_m=height)
        self.width, self.height = width, height
        self.shape = (round(height/.2), round(width/.2))
        self.parent, self.assignment = parent, assignment
        self.semantic_condition = semantic_condition
        self.seed = 18000 if parent == 'D18-P00' else 18001
        self._solid_primitives = []; self.boxes = []; self._physical_markers = []
        self.objects = []; self.inspection_truth = []
        def box(x, y, z, sx, sy, sz, owner=1):
            self._solid_primitives.append((x, y, z, sx, sy, sz, owner))
            if z < 1.2 and z+sz > .15: self.boxes.append((x, y, sx, sy))
        box(0, 0, -.1, width, height, .1)
        box(0, 0, 0, width, .2, 2.4); box(0, height-.2, 0, width, .2, 2.4)
        box(0, .2, 0, .2, height-.4, 2.4); box(width-.2, .2, 0, .2, height-.4, 2.4)
        centers = [(2.5, 4.8), (7.5, 4.8)] if parent == 'D18-P00' else [(3., 4.4), (8.8, 5.)]
        if parent == 'D18-P01':
            box(5.6, 6.6, 0, .4, 1.2, 1.6)  # distinct service-divider topology
        open_flags = [assignment == 'A_open_B_closed', assignment != 'A_open_B_closed']
        for index, ((cx, front), opened) in enumerate(zip(centers, open_flags)):
            owner = 100 + index; category = 3 if opened else 2
            # Common protective front panel; rear structure is narrower.
            box(cx-1.1, front, 0, 2.2, .08, 1.5, owner)
            if opened:
                for z in (.20, .55, .90, 1.25): box(cx-.4, front+.08, z, .8, .5, .06, owner)
                for dx in (-.4, .34):
                    for dy in (.08, .52): box(cx+dx, front+dy, 0, .06, .06, 1.5, owner)
            else:
                box(cx-.4, front+.08, 0, .8, .5, 1.5, owner)
            box(cx-.25, front-.02, .94, .5, .02, .24, owner)
            self._physical_markers.append(dict(x0=cx-.25, x1=cx+.25, y=front-.02,
                z0=.94, z1=1.18, physical_class=category))
            self.objects.append(dict(id=index, owner=owner, category=category,
                evaluation_bounds=[[cx-1.22, front-.14, .01], [cx+1.22, front+.70, 1.62]],
                center=[cx, front+.29, .75]))
        self.mesh, self.triangle_owners, self.surface_measure_audit = union_surface_from_boxes(self._solid_primitives)
        self.triangle_classes = self.triangle_owners.copy()
        for item in self.objects:
            self.triangle_classes[self.triangle_owners == item['owner']] = item['category']
        self.occupancy = np.zeros(self.shape, bool)
        for x, y, sx, sy in self.boxes:
            c0, c1 = math.floor(x/.2+1e-8), math.ceil((x+sx)/.2-1e-8)
            r0, r1 = math.floor((height-y-sy)/.2+1e-8), math.ceil((height-y)/.2-1e-8)
            self.occupancy[max(0,r0):min(self.shape[0],r1), max(0,c0):min(self.shape[1],c1)] = True
        self._blocked = inflated_obstacles(self.occupancy, self.config.robot_radius_m/.2)
        self.start = self._cell(width/2, 1.4)
        groups, _ = label(~self._blocked)
        if groups[self.start] == 0: raise ValueError('unsafe initial pose')
        self.reachable = groups == groups[self.start]
        if np.any((~self._blocked) & ~self.reachable): raise ValueError('disconnected safe floor')
        self.position = self.start; self.heading = 0
        self.step_count = self.collisions = self.moves = 0
        c = self.config; focal = c.width_px/(2*np.tan(np.deg2rad(c.fov_deg/2)))
        self.intrinsic = np.array([[focal,0,(c.width_px-1)/2],[0,focal,(c.height_px-1)/2],[0,0,1.]])

    def instance_mesh(self, object_id):
        item = next(x for x in self.objects if x['id'] == object_id)
        mask = self.triangle_owners == item['owner']
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices).copy())
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(self.mesh.triangles)[mask].copy())
        mesh.remove_unreferenced_vertices()
        return mesh
