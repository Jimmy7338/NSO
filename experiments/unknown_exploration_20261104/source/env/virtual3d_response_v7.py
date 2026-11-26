"""Paired, CPU-only asset inspection units with an occluded common prefix.

Simulator/evaluator module. Planners receive RGBDFrame and PlanarScan, never
ResponseContextV7, ``objects``, primitives, marker renderer data or GT meshes.
The asset family is an intentionally synthetic prior, not a route/reward label.
Old worlds and their frozen evaluation contracts are not changed here.
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


DEFAULT_CONTEXT_MANIFEST = Path(__file__).resolve().parents[1] / "configs/virtual3d/response_v7_contexts.json"
RESPONSE_FAMILIES = ("storage_shelves", "ventilation_baffles")
PREFIX_ACTIONS_V7 = (
    *("forward",) * 5, "left", "right", *("forward",) * 5,
    "right", "left", "left", "right", "right", "left", "left", "right",
)


def _sensor_seed(outer_seed, context_id):
    token = f"nso-response-v7/{outer_seed}/{context_id}/common-sensor-stream".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def load_response_contexts(path=None):
    """Read the predeclared T/C manifest; does not generate any physical world."""
    manifest = json.loads(Path(path or DEFAULT_CONTEXT_MANIFEST).read_text())
    if manifest["schema_version"] != "response_v7_contexts/1":
        raise ValueError("unknown response context contract")
    expected = [f"T{i}" for i in range(8)] + [f"C{i}" for i in range(4)]
    if [row["context_id"] for row in manifest["contexts"]] != expected:
        raise ValueError("the finite T/C parent list may not be extended implicitly")
    if manifest["prefix_contract"]["actions"] != list(PREFIX_ACTIONS_V7):
        raise ValueError("prefix implementation differs from the predeclared contract")
    for row in manifest["contexts"]:
        if row["outer_seed"] not in (751, 752):
            raise ValueError("this generator contract opens only training/calibration seeds")
        if row["sensor_seed"] != _sensor_seed(row["outer_seed"], row["context_id"]):
            raise ValueError("context sensor seed does not match the fixed common stream")
    return manifest


@dataclass(frozen=True)
class ResponseContextV7:
    context_id: str
    role: str
    batch_index: int
    outer_seed: int
    sensor_seed: int
    rotation_quarter_turns: int
    offset_xy_m: tuple
    visible_scale: float
    approach_distance_m: float
    structure_variant: int
    mirror_x: bool
    depth_sigma_m: float = .01
    prefix_step: int = 20

    def __post_init__(self):
        if self.context_id not in tuple(f"T{i}" for i in range(8)) + tuple(f"C{i}" for i in range(4)):
            raise ValueError("only predeclared T/C context identifiers are supported")
        if self.outer_seed not in (751, 752) or self.sensor_seed != _sensor_seed(self.outer_seed, self.context_id):
            raise ValueError("invalid parent/common sensor stream")
        if self.role != ("train" if self.context_id.startswith("T") else "calibration"):
            raise ValueError("a parent context cannot change data role")
        if self.rotation_quarter_turns not in range(4) or self.structure_variant not in (0, 1):
            raise ValueError("unknown rotation/hidden parameterization")
        if not .9 <= self.visible_scale <= 1.1 or self.approach_distance_m not in (2.8, 3., 3.2):
            raise ValueError("parent geometry outside the finite construction range")
        if len(self.offset_xy_m) != 2 or any(not np.isclose(v / .2, round(v / .2)) for v in self.offset_xy_m):
            raise ValueError("context offsets must preserve the common 0.2m robot grid")
        if self.prefix_step != 20:
            raise ValueError("the extraction step is fixed before outcomes")


def get_response_context(context_id, path=None):
    manifest = load_response_contexts(path)
    row = next((row for row in manifest["contexts"] if row["context_id"] == context_id), None)
    if row is None:
        raise ValueError(f"context {context_id!r} is not in the finite T/C manifest")
    return ResponseContextV7(**{**row, "offset_xy_m": tuple(row["offset_xy_m"])})


@dataclass(frozen=True)
class ResponseConfigV7(VirtualConfigV2):
    width_m: float = 8.4
    height_m: float = 8.4
    appearance: str = "marked"
    semantic_source: str = "rgb_marker"
    voxel_m: float = .03
    max_steps: int = 100

    def __post_init__(self):
        super().__post_init__()
        if self.width_m != 8.4 or self.height_m != 8.4 or self.resolution_m != .2:
            raise ValueError("V7 contract fixes an 8.4m square unit and 0.2m grid")
        if self.appearance not in ("marked", "gray") or self.semantic_source != "rgb_marker":
            raise ValueError("only RGB marker interpretation is available")
        if not 0 < self.laser_height_m < .35:
            raise ValueError("the common plinth must occlude the planar laser")
        if not .35 < self.camera_height_m < 1.44 or self.pose_noise_m != 0:
            raise ValueError("the paired construction uses a frontal camera and exact input pose")
        if self.max_steps < 68:
            raise ValueError("budget must contain the paid prefix and a 48-action branch")


def _quarter_rotation(turns):
    rotations = (((1, 0), (0, 1)), ((0, -1), (1, 0)),
                 ((-1, 0), (0, -1)), ((0, 1), (-1, 0)))
    return np.asarray(rotations[turns])


class ResponseWorldV7(VirtualWorldV2):
    """One hidden physical family in a fixed paired parent context.

    ``seed`` is the context-specific sensor seed; ``context.outer_seed`` keeps
    the dataset block. Family never enters the sensor RNG. The scene constructor
    has no response label, candidate, policy, or future measurement arguments.
    """

    def __init__(self, context, family="storage_shelves", config=None,
                 semantic_condition="aligned"):
        if not isinstance(context, ResponseContextV7):
            raise TypeError("use get_response_context to resolve a predeclared parent")
        if family not in RESPONSE_FAMILIES or semantic_condition not in ("aligned", "shuffled", "absent"):
            raise ValueError("unknown physical family or semantic intervention")
        self.context, self.family = context, family
        self.config = config or ResponseConfigV7(depth_sigma_m=context.depth_sigma_m)
        self.seed, self.semantic_condition = context.sensor_seed, semantic_condition
        self.width, self.height = self.config.width_m, self.config.height_m
        self.shape = (round(self.height / self.config.resolution_m), round(self.width / self.config.resolution_m))
        self.prefix_actions = PREFIX_ACTIONS_V7
        self._ray = o3d.t.geometry.RaycastingScene(nthreads=1)
        self._geometry_labels, self.boxes, self.objects = {}, [], []
        self._solid_primitives = []
        category = 2 if family == "storage_shelves" else 3
        self._rotation = _quarter_rotation(context.rotation_quarter_turns)
        anchor = np.array([4.1, 4.3]) + np.asarray(context.offset_xy_m)
        self._front_anchor = (anchor - 4.2) @ self._rotation.T + 4.2
        self._local_x_sign = -1 if context.mirror_x else 1

        def add_box(primitive):
            x, y, z, sx, sy, sz, physical_class = primitive
            mesh = o3d.geometry.TriangleMesh.create_box(sx, sy, sz).translate((x, y, z))
            gid = self._ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            # No simulator class reaches even the inherited ideal-label field.
            self._geometry_labels[gid] = 1
            self._solid_primitives.append(tuple(primitive))
            if z < 1.2 and z + sz > .15:
                self.boxes.append((x, y, sx, sy))

        def object_box(x, y, z, sx, sy, sz):
            corners = np.array([[x, y], [x + sx, y], [x, y + sy], [x + sx, y + sy]])
            corners[:, 0] *= self._local_x_sign
            corners = corners @ self._rotation.T + self._front_anchor
            low, high = corners.min(axis=0), corners.max(axis=0)
            add_box((*low, z, *(high - low), sz, category))

        # Same primitives, order and exact triangles occur before hidden pieces.
        # Sensors raycast these physical boxes; GT is their unique exterior union.
        w, h = self.width, self.height
        add_box((0., 0., -.1, w, h, .1, 1))
        for primitive in ((0., 0., 0., w, .2, 2.4, 1), (0., h - .2, 0., w, .2, 2.4, 1),
                          (0., 0., 0., .2, h, 2.4, 1), (w - .2, 0., 0., .2, h, 2.4, 1)):
            add_box(primitive)
        side = 1.6 * context.visible_scale
        base, thick = .35, .08
        object_box(-side / 2, 0., 0., side, side, base)
        object_box(-side / 2, 0., base, side, thick, side - base)
        self.common_primitive_count = len(self._solid_primitives)
        self._marker = {"width_m": .35 * side, "z0": .55 * side,
                        "z1": .75 * side, "physical_class": category}
        count = 3 + context.structure_variant
        if family == "storage_shelves":
            object_box(-side / 2, side - thick, base, side, thick, side - base)
            for z in np.linspace(base + (side - base) / count, side - thick, count):
                object_box(-side / 2, thick, float(z), side, side - thick, thick)
        else:
            for x in np.linspace(-side / 2, side / 2 - thick, count):
                object_box(float(x), thick, base, thick, side - thick, side - base)

        self.mesh, self.triangle_classes, union_audit = union_surface_from_boxes(self._solid_primitives)
        low = np.min(np.asarray(self._solid_primitives[5:])[:, :3], axis=0)
        high = np.max(np.asarray(self._solid_primitives[5:])[:, :3] + np.asarray(self._solid_primitives[5:])[:, 3:6], axis=0)
        self.objects = [{"id": 0, "category": category, "box": [*low, *(high - low)]}]
        self.inspection_truth = {"generator": "paired_response_v7", "context_id": context.context_id,
                                 "family": family, "union_surface": union_audit,
                                 "synthetic_asset_prior": True, "reward_class_weighting": False}
        self.occupancy = np.zeros(self.shape, bool)
        resolution = self.config.resolution_m
        for x, y, sx, sy in self.boxes:
            c0 = max(0, int(np.floor((x + 1e-8) / resolution)))
            c1 = min(self.shape[1], int(np.ceil((x + sx - 1e-8) / resolution)))
            r0 = max(0, int(np.floor((h - y - sy + 1e-8) / resolution)))
            r1 = min(self.shape[0], int(np.ceil((h - y - 1e-8) / resolution)))
            self.occupancy[r0:r1, c0:c1] = True
        self._blocked = inflated_obstacles(self.occupancy, self.config.robot_radius_m / resolution)
        initial_xy = self._front_anchor + np.array([0., -context.approach_distance_m]) @ self._rotation.T
        self.start = self._cell(*initial_xy)
        self.heading = (-context.rotation_quarter_turns) % 4
        components, _ = label(~self._blocked)
        if not components[self.start]:
            raise ValueError("predeclared start is blocked; do not silently project it")
        self.reachable = components == components[self.start]
        self.position = self.start
        self.step_count = self.collisions = self.moves = 0
        fx = self.config.width_px / (2 * np.tan(np.deg2rad(self.config.fov_deg / 2)))
        self.intrinsic = np.array([[fx, 0., (self.config.width_px - 1) / 2],
                                   [0., fx, (self.config.height_px - 1) / 2], [0., 0., 1.]])
        actual_xy = camera_pose(self.start, self.heading, self.config, self.shape[0])[:2, 3]
        if not np.allclose(actual_xy, initial_xy, rtol=0, atol=1e-12):
            raise ValueError("parent start does not lie at the declared grid-cell center")

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
            object_xy = (points[..., :2] - self._front_anchor) @ self._rotation
            marker = self._marker
            visible = finite & (frame.depth_m > 0) & (np.abs(object_xy[..., 1]) < 1e-5)
            visible &= np.abs(object_xy[..., 0]) < marker["width_m"] / 2
            visible &= (points[..., 2] > marker["z0"]) & (points[..., 2] < marker["z1"])
            rgb[visible] = MARKER_COLORS[marker["physical_class"]]
        semantic = read_inspection_markers_rgb(rgb)
        if self.semantic_condition == "shuffled":
            semantic = np.where(semantic == 2, 3, np.where(semantic == 3, 2, semantic)).astype(np.uint8)
        elif self.semantic_condition == "absent":
            semantic[:] = 0
        return replace(frame, color_rgb=rgb, semantic=semantic).validate()


def create_response_world(context_id, family, config=None, semantic_condition="aligned", manifest_path=None):
    return ResponseWorldV7(get_response_context(context_id, manifest_path), family,
                           config=config, semantic_condition=semantic_condition)


def collect_response_prefix(world):
    """Collect initial plus 20 paid observations, with no candidate/GT evaluation.

    Audit/runner convenience only. Each row contains frame, scan, action, step,
    collision, done; the first action is None. Never expose ``world`` to a policy.
    """
    if world.step_count != 0:
        raise ValueError("a prefix must start from a fresh predeclared world")
    rows = [{"step": 0, "action": None, "frame": world.sense(), "scan": world.scan(),
             "collision": False, "done": False}]
    for action in world.prefix_actions:
        frame, collision, done = world.step(action)
        rows.append({"step": world.step_count, "action": action, "frame": frame,
                     "scan": world.scan(), "collision": collision, "done": done})
        if collision or done:
            raise RuntimeError(f"predeclared prefix failed at step {world.step_count}; do not replace context")
    return rows
