"""Two swapped industrial assets sharing a paid, occluded RGB-D prefix.

This is a deliberately synthetic asset-marker positive control. Categories name
physical assets, never a reward or a route. All scene metadata/GT stay in the
simulator; policy inputs are only RGBDFrame and PlanarScan. Old worlds are intact.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import label

from env.virtual3d import camera_pose
from env.virtual3d_v2 import VirtualConfigV2, VirtualWorldV2
from env.virtual3d_inspection_v4 import (
    MARKER_COLORS, read_inspection_markers_rgb, union_surface_from_boxes,
)
from utils.grid_geometry import inflated_obstacles


DEFAULT_CONTEXT_MANIFEST = Path(__file__).resolve().parents[1] / "configs/virtual3d/competition_v8_contexts.json"
COMPETITION_ARRANGEMENTS = ("shelf_west", "shelf_east")
ASSET_CATEGORIES = {2: "storage_shelf", 3: "closed_equipment_cabinet"}
PREFIX_SEGMENTS_P0 = (
    ("right", 4), ("left", 1), ("forward", 17), ("right", 1),
    ("forward", 15), ("right", 4), ("right", 1), ("forward", 17),
    ("right", 4), ("forward", 18), ("right", 4), ("right", 1),
    ("forward", 15), ("right", 4), ("right", 1), ("forward", 18),
    ("right", 1), ("forward", 20), ("right", 4),
)
PREFIX_ACTIONS_P0 = tuple(action for action, count in PREFIX_SEGMENTS_P0 for _ in range(count))


def _sensor_seed(outer_seed, context_id):
    token = f"nso-competition-v8/{outer_seed}/{context_id}/common-sensor-stream".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def load_competition_contexts(path=None):
    """Read the two fixed development parents without constructing any world."""
    manifest = json.loads(Path(path or DEFAULT_CONTEXT_MANIFEST).read_text())
    if manifest["schema_version"] != "competition_v8_contexts/1":
        raise ValueError("unknown competition context contract")
    if [row["context_id"] for row in manifest["contexts"]] != ["P0", "P1"]:
        raise ValueError("the finite development parent list may not be extended implicitly")
    if manifest["prefix"]["p0_actions"] != list(PREFIX_ACTIONS_P0):
        raise ValueError("prefix implementation differs from the predeclared contract")
    if manifest["arrangement_order"] != list(COMPETITION_ARRANGEMENTS):
        raise ValueError("unknown arrangement order")
    for row in manifest["contexts"]:
        if row["outer_seed"] != {"P0": 751, "P1": 752}[row["context_id"]]:
            raise ValueError("the generator opens only the two declared development seeds")
        if row["sensor_seed"] != _sensor_seed(row["outer_seed"], row["context_id"]):
            raise ValueError("invalid common sensor stream")
    return manifest


@dataclass(frozen=True)
class CompetitionContextV8:
    context_id: str
    role: str
    outer_seed: int
    sensor_seed: int
    mirror_x: bool
    object_front_centers_xy_m: tuple
    initial_xy_m: tuple
    initial_heading: int
    final_prefix_xy_m: tuple
    final_prefix_heading: int
    prefix_step: int
    depth_sigma_m: float

    def __post_init__(self):
        if self.context_id not in ("P0", "P1") or self.role != "development_positive_control":
            raise ValueError("only P0/P1 development parents are declared")
        index = int(self.context_id[-1])
        if self.outer_seed != 751 + index or self.sensor_seed != _sensor_seed(self.outer_seed, self.context_id):
            raise ValueError("invalid parent or common sensor stream")
        if self.mirror_x != bool(index):
            raise ValueError("the second parent is the predeclared geometric mirror")
        expected_centers = ((2.1 + .2 * index, 4.3), (6.1 + .2 * index, 4.3))
        checks = ((self.object_front_centers_xy_m, expected_centers),
                  (self.initial_xy_m, (4.1 + .2 * index, .7)),
                  (self.final_prefix_xy_m, (4.1 + .2 * index, 4.7)))
        if any(not np.allclose(a, b, rtol=0, atol=1e-12) for a, b in checks):
            raise ValueError("context geometry differs from the fixed construction")
        if self.prefix_step != 150 or self.initial_heading != 0 or self.final_prefix_heading != 0:
            raise ValueError("the paid prefix has fixed actions and endpoints")
        if self.depth_sigma_m != .01:
            raise ValueError("both arrangements use the predeclared standard depth noise")


def get_competition_context(context_id, path=None):
    row = next((row for row in load_competition_contexts(path)["contexts"]
                if row["context_id"] == context_id), None)
    if row is None:
        raise ValueError(f"context {context_id!r} is not a declared development parent")
    data = dict(row)
    data["object_front_centers_xy_m"] = tuple(tuple(p) for p in data["object_front_centers_xy_m"])
    for key in ("initial_xy_m", "final_prefix_xy_m"):
        data[key] = tuple(data[key])
    return CompetitionContextV8(**data)


@dataclass(frozen=True)
class CompetitionConfigV8(VirtualConfigV2):
    width_m: float = 8.4
    height_m: float = 7.2
    appearance: str = "marked"
    semantic_source: str = "rgb_marker"
    voxel_m: float = .03
    max_steps: int = 210

    def __post_init__(self):
        super().__post_init__()
        fixed = dict(width_m=8.4, height_m=7.2, resolution_m=.2, robot_radius_m=.2,
                     camera_height_m=.8, width_px=96, height_px=72, fov_deg=90.,
                     max_depth_m=4., depth_sigma_m=.01, dropout=.01,
                     laser_height_m=.25, laser_rays=180, voxel_m=.03,
                     truncation_m=.12, pose_noise_m=0., action_duration_s=1.)
        if any(getattr(self, key) != value for key, value in fixed.items()):
            raise ValueError("V8 construction fixes geometry, sensor resolution/noise and robot scale")
        if self.appearance not in ("marked", "gray") or self.semantic_source != "rgb_marker":
            raise ValueError("only physical RGB marker interpretation is available")
        if self.max_steps < 198:
            raise ValueError("max_steps must contain the 150-action prefix and a 48-action branch")


class CompetitionWorldV8(VirtualWorldV2):
    """One shelf and one closed cabinet; only their assignment changes in a pair.

    All common primitives, for both assets, enter the ray scene before hidden
    pieces. This keeps the exact visible triangles and first-hit tie order fixed.
    The class never receives a desired outcome, candidate or policy parameter.
    """

    def __init__(self, context, arrangement="shelf_west", config=None,
                 semantic_condition="aligned"):
        if not isinstance(context, CompetitionContextV8):
            raise TypeError("use get_competition_context for a predeclared parent")
        if arrangement not in COMPETITION_ARRANGEMENTS or semantic_condition not in ("aligned", "shuffled", "absent"):
            raise ValueError("unknown physical arrangement or semantic intervention")
        self.context, self.arrangement = context, arrangement
        self.config = config or CompetitionConfigV8()
        if not isinstance(self.config, CompetitionConfigV8):
            raise TypeError("CompetitionConfigV8 required")
        self.seed, self.semantic_condition = context.sensor_seed, semantic_condition
        self.width, self.height = self.config.width_m, self.config.height_m
        self.shape = (round(self.height / .2), round(self.width / .2))
        mirror_action = {"forward": "forward", "left": "right", "right": "left"}
        self.prefix_actions = tuple(mirror_action[a] for a in PREFIX_ACTIONS_P0) if context.mirror_x else PREFIX_ACTIONS_P0
        self._ray = o3d.t.geometry.RaycastingScene(nthreads=1)
        self._geometry_labels, self.boxes, self.objects = {}, [], []
        self._solid_primitives, self._physical_markers = [], []

        def add_box(x, y, z, sx, sy, sz, physical_class=1):
            mesh = o3d.geometry.TriangleMesh.create_box(sx, sy, sz).translate((x, y, z))
            gid = self._ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            # The inherited ideal-label path receives no physical asset class.
            self._geometry_labels[gid] = 1
            self._solid_primitives.append((x, y, z, sx, sy, sz, physical_class))
            if z < 1.2 and z + sz > .15:
                self.boxes.append((x, y, sx, sy))

        w, h = self.width, self.height
        add_box(0., 0., -.1, w, h, .1)
        for box in ((0., 0., 0., w, .2, 2.4), (0., h - .2, 0., w, .2, 2.4),
                    (0., 0., 0., .2, h, 2.4), (w - .2, 0., 0., .2, h, 2.4)):
            add_box(*box)
        categories = (2, 3) if arrangement == "shelf_west" else (3, 2)
        # Each common shell is a non-overlapping box partition. The rear is
        # open above the shared plinth; front/side/top occlude the paid prefix.
        common_local = (
            (-.6, 0., 0., 1.2, 1.2, .35),
            (-.6, 0., .35, 1.2, .08, 1.25),
            (-.6, .08, .35, .08, 1.12, 1.25),
            (.52, .08, .35, .08, 1.12, 1.25),
            (-.52, .08, 1.52, 1.04, 1.12, .08),
        )
        for slot, ((cx, cy), category) in enumerate(zip(context.object_front_centers_xy_m, categories)):
            for x, y, z, sx, sy, sz in common_local:
                add_box(cx + x, cy + y, z, sx, sy, sz, category)
            self.objects.append(dict(id=slot, slot="west" if slot == 0 else "east",
                                     category=category, asset_name=ASSET_CATEGORIES[category],
                                     box=[cx - .6, cy, 0., 1.2, 1.2, 1.6]))
            self._physical_markers.append(dict(cx=cx, front_y=cy, width_m=.48,
                                                z0=.88, z1=1.2, category=category))
        self.common_primitive_count = len(self._solid_primitives)
        for (cx, cy), category in zip(context.object_front_centers_xy_m, categories):
            if category == 2:
                for z in (.55, .9, 1.25):
                    add_box(cx - .52, cy + .08, z, 1.04, 1.12, .08, category)
            else:
                # Fill only the cavity and rear opening: no new external front
                # or side triangles can perturb the common prefix first hits.
                add_box(cx - .52, cy + .08, .35, 1.04, 1.04, 1.17, category)
                add_box(cx - .52, cy + 1.12, .35, 1.04, .08, 1.17, category)

        self.mesh, self.triangle_classes, union_audit = union_surface_from_boxes(self._solid_primitives)
        self.competition_truth = dict(generator="paired_competition_v8", context_id=context.context_id,
                                     arrangement=arrangement, union_surface=union_audit,
                                     synthetic_asset_prior=True, reward_class_weighting=False)
        self.inspection_truth = self.competition_truth  # evaluator-only metadata compatibility
        self.occupancy = np.zeros(self.shape, bool)
        for x, y, sx, sy in self.boxes:
            c0 = max(0, int(np.floor((x + 1e-8) / .2)))
            c1 = min(self.shape[1], int(np.ceil((x + sx - 1e-8) / .2)))
            r0 = max(0, int(np.floor((h - y - sy + 1e-8) / .2)))
            r1 = min(self.shape[0], int(np.ceil((h - y - 1e-8) / .2)))
            self.occupancy[r0:r1, c0:c1] = True
        self._blocked = inflated_obstacles(self.occupancy, self.config.robot_radius_m / .2)
        self.start, self.heading = self._cell(*context.initial_xy_m), context.initial_heading
        components, _ = label(~self._blocked)
        if not components[self.start]:
            raise ValueError("predeclared start is blocked; do not project it to a favorable cell")
        self.reachable = components == components[self.start]
        self.position = self.start
        self.step_count = self.collisions = self.moves = 0
        fx = self.config.width_px / (2 * np.tan(np.deg2rad(self.config.fov_deg / 2)))
        self.intrinsic = np.array([[fx, 0., (self.config.width_px - 1) / 2],
                                  [0., fx, (self.config.height_px - 1) / 2], [0., 0., 1.]])
        actual = camera_pose(self.start, self.heading, self.config, self.shape[0])[:2, 3]
        if not np.allclose(actual, context.initial_xy_m, rtol=0, atol=1e-12):
            raise ValueError("predeclared start must be an exact robot grid-cell center")

    def _cell(self, x, y):
        return (self.shape[0] - 1 - int(np.floor(y / self.config.resolution_m)),
                int(np.floor(x / self.config.resolution_m)))

    def sense(self):
        frame = super().sense()
        rgb = frame.color_rgb.copy()
        if self.config.appearance == "marked":
            pose = camera_pose(self.position, self.heading, self.config, self.shape[0])
            v, u = np.mgrid[:self.config.height_px, :self.config.width_px]
            local = np.stack([(u - self.intrinsic[0, 2]) / self.intrinsic[0, 0],
                              (v - self.intrinsic[1, 2]) / self.intrinsic[1, 1], np.ones_like(u)], axis=-1)
            direction = local @ pose[:3, :3].T
            rays = np.empty((*u.shape, 6), np.float32)
            rays[..., :3], rays[..., 3:] = pose[:3, 3], direction
            hit = self._ray.cast_rays(o3d.core.Tensor(rays), nthreads=1)["t_hit"].numpy()
            finite = np.isfinite(hit)
            points = pose[:3, 3] + direction * np.where(finite, hit, 0)[..., None]
            for marker in self._physical_markers:
                visible = finite & (frame.depth_m > 0) & (np.abs(points[..., 1] - marker["front_y"]) < 1e-5)
                visible &= np.abs(points[..., 0] - marker["cx"]) < marker["width_m"] / 2
                visible &= (points[..., 2] > marker["z0"]) & (points[..., 2] < marker["z1"])
                rgb[visible] = MARKER_COLORS[marker["category"]]
        semantic = read_inspection_markers_rgb(rgb)
        if self.semantic_condition == "shuffled":
            semantic = np.where(semantic == 2, 3, np.where(semantic == 3, 2, semantic)).astype(np.uint8)
        elif self.semantic_condition == "absent":
            semantic[:] = 0
        return replace(frame, color_rgb=rgb, semantic=semantic).validate()


def create_competition_world(context_id, arrangement, config=None,
                             semantic_condition="aligned", manifest_path=None):
    return CompetitionWorldV8(get_competition_context(context_id, manifest_path), arrangement,
                              config=config, semantic_condition=semantic_condition)


def collect_competition_prefix(world):
    """Initial plus 150 real, paid actions; no candidate/GT reward evaluation."""
    if world.step_count != 0:
        raise ValueError("prefix must begin with a fresh predeclared world")
    rows = [dict(step=0, action=None, frame=world.sense(), scan=world.scan(), collision=False, done=False)]
    for action in world.prefix_actions:
        frame, collision, done = world.step(action)
        rows.append(dict(step=world.step_count, action=action, frame=frame, scan=world.scan(),
                         collision=collision, done=done))
        if collision or done:
            raise RuntimeError(f"predeclared prefix failed at step {world.step_count}; retain the failed world")
    return rows
