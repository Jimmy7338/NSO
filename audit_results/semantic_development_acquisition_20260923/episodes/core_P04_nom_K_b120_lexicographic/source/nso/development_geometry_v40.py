"""Small, deterministic DEV-only triangle assets; no World or sensor calls.

Each facility is the exterior boundary of a union of solid boxes. Shelves and
recesses are openings between finite-thickness solids, not missing mesh faces.
Instance identities, actual structures and marker planes are private metadata.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from nso.scene_contract_v40 import (canonical_json_bytes, public_planner_spec,
                                    validate_scene_spec)


PALETTE_V40 = {"cabinet": (40, 100, 220), "rack": (220, 60, 40),
               "workstation": (40, 190, 90), "machine": (210, 160, 40)}
STRUCTURES = ("planar", "recessed", "louvered", "open_frame")
MAX_ASSET_BYTES = 16 * 1024 * 1024


def _readonly(array, dtype):
    value = np.array(array, dtype=dtype, copy=True)
    value.flags.writeable = False
    return value


def union_box_mesh(boxes):
    """Exact coordinate arrangement; shared vertices and outward winding.

    boxes[K,6] use xmin,xmax,ymin,ymax,zmin,zmax. Adjacent cells of the same
    material union do not emit internal faces. No ray casting is performed.
    """
    bounds = np.asarray(boxes, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 6 or not len(bounds) or not np.isfinite(bounds).all():
        raise ValueError("finite nonempty Kx6 box bounds required")
    # Equivalent arithmetic (h-t+t versus h) must not create numerical slivers.
    bounds = np.round(bounds.reshape(-1, 3, 2), 12)
    if np.any(bounds[:, :, 1] <= bounds[:, :, 0]):
        raise ValueError("box extents must be positive")
    axes = [np.unique(bounds[:, axis, :]) for axis in range(3)]
    shape = tuple(len(axis)-1 for axis in axes)
    if np.prod(shape) > 250000:
        raise ValueError("development box arrangement exceeds bounded complexity")
    centers = [(axis[:-1]+axis[1:])*.5 for axis in axes]
    occupied = np.zeros(shape, dtype=bool)
    for box in bounds:
        intervals = [(centers[d] > box[d, 0]) & (centers[d] < box[d, 1]) for d in range(3)]
        occupied |= intervals[0][:, None, None] & intervals[1][None, :, None] & intervals[2][None, None, :]
    vertices, triangles, index = [], [], {}

    def vertex(key):
        key = tuple(key)
        if key not in index:
            index[key] = len(vertices)
            vertices.append([axes[d][key[d]] for d in range(3)])
        return index[key]

    for axis in range(3):
        remaining = [d for d in range(3) if d != axis]
        empty = np.zeros(tuple(shape[d] for d in remaining), dtype=bool)
        for plane in range(shape[axis]+1):
            lower = np.take(occupied, plane-1, axis=axis) if plane else empty
            upper = np.take(occupied, plane, axis=axis) if plane < shape[axis] else empty
            for i, j in np.argwhere(lower != upper):
                corners = []
                for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    key = [0, 0, 0]
                    key[axis], key[remaining[0]], key[remaining[1]] = plane, int(i)+di, int(j)+dj
                    corners.append(vertex(key))
                a, b, c, d = corners
                outward = 1 if lower[i, j] else -1
                if outward == (1, -1, 1)[axis]:
                    triangles.extend(((a, b, c), (a, c, d)))
                else:
                    triangles.extend(((a, c, b), (a, d, c)))
    return np.asarray(vertices, np.float64), np.asarray(triangles, np.int32)


def structure_boxes(structure, dimensions):
    """Generic development shape family, not an actual hidden test template."""
    if structure not in STRUCTURES:
        raise ValueError("unknown development structure")
    dims = np.asarray(dimensions, dtype=float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims < .6):
        raise ValueError("width/depth/height must be finite and at least 0.6 m")
    w, d, h = map(float, dims)
    if h < 1.1:
        raise ValueError("marker support requires height at least 1.1 m")
    x0, x1, y0, y1, t = -w/2, w/2, -d/2, d/2, .08
    if structure == "planar":
        boxes = [[x0, x1, y0, y1, 0., h]]
    elif structure in ("recessed", "louvered"):
        boxes = [[x0, x1, y1-t, y1, 0., h], [x0, x0+t, y0, y1, 0., h],
                 [x1-t, x1, y0, y1, 0., h], [x0, x1, y0, y1, 0., t],
                 [x0, x1, y0, y1, h-t, h]]
        if structure == "louvered":
            for height in (.25*h, .45*h, .65*h, .85*h):
                boxes.append([x0, x1, y0, y0+.16, height, height+.06])
    else:
        boxes = []
        for x in (x0, x1-t):
            for y in (y0, y1-t):
                boxes.append([x, x+t, y, y+t, 0., h])
        for height in (0., .5*h, h-t):
            boxes.append([x0, x1, y0, y1, height, height+t])
    # A common attached finite-thickness label plate. Its geometry does not
    # depend on class or actual structure. The colored decal occupies its face.
    boxes.append([x0, x0+.38, y0-.02, y0+.02, .62, .98])
    marker = {"center_local_m": [x0+.19, y0-.02, .8], "normal_local": [0., -1., 0.],
              "u_local": [1., 0., 0.], "v_local": [0., 0., 1.], "width_m": .32, "height_m": .30}
    return boxes, marker


def _profile(family):
    profiles = {
        "A": (10., 8., [(2.3, 2., 180), (7.7, 2., 180), (2.3, 6., 0), (7.7, 6., 0)],
              ["cabinet", "machine", "workstation", "cabinet"], ["recessed", "louvered", "recessed", "planar"],
              (1.25, .8, 1.6), [[4.8, 5.2, 2.8, 5.2, 0., 1.4]]),
        "B": (12., 8., [(3., 2.1, 180), (9., 2.1, 180), (3., 5.9, 0), (9., 5.9, 0)],
              ["rack", "cabinet", "rack", "workstation"], ["open_frame", "recessed", "louvered", "open_frame"],
              (1.7, .85, 1.8), [[5.3, 6.7, 3.3, 4.7, 0., 1.2]]),
        "C": (10., 9., [(2., 2., 135), (7.8, 2.4, 180), (2.3, 6.7, 0), (7.5, 6.8, 315)],
              ["workstation", "machine", "cabinet", "rack"], ["planar", "open_frame", "louvered", "recessed"],
              (1.2, .85, 1.6), [[4.4, 5.2, 3.7, 5.1, 0., 1.35]]),
        "D": (10., 8., [(2.3, 2., 180), (7.7, 2., 180), (2.3, 6., 0), (7.7, 6., 0)],
              ["cabinet", "rack", "workstation", "machine"], ["planar"]*4,
              (1.2, .8, 1.4), []),
        "E": (20., 14., [(3., 3., 180), (17., 3., 180), (3., 11., 0), (17., 11., 0)],
              ["cabinet", "rack", "machine", "workstation"], ["recessed", "open_frame", "louvered", "planar"],
              (1.2, .8, 1.6), [[9.8, 10.2, 3., 11., 0., 1.4]]),
        "F": (10., 8., [(2.3, 2., 180), (7.7, 2., 180), (2.3, 6., 0), (7.7, 6., 0)],
              ["cabinet", "rack", "workstation", "machine"], ["open_frame", "planar", "louvered", "open_frame"],
              (1.2, .8, 1.6), [[4.5, 5.3, 2.7, 5.3, 0., 1.3]])}
    if family not in profiles:
        raise ValueError("only six declared development families are supported")
    return profiles[family]


def mesh_audit(vertices, triangles, triangle_instance_id):
    v, t, owners = np.asarray(vertices), np.asarray(triangles), np.asarray(triangle_instance_id)
    if v.dtype != np.float64 or v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
        raise ValueError("finite float64 Nx3 vertices required")
    if t.dtype != np.int32 or t.ndim != 2 or t.shape[1] != 3 or not len(t) or t.min() < 0 or t.max() >= len(v):
        raise ValueError("valid int32 Mx3 triangle indices required")
    if owners.dtype != np.int32 or owners.shape != (len(t),) or np.any(owners < -1):
        raise ValueError("one int32 instance identity per triangle; background=-1")
    points = v[t]
    cross = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    areas = .5*np.linalg.norm(cross, axis=1)
    if np.any(areas <= 1e-12) or len(np.unique(np.sort(t, axis=1), axis=0)) != len(t):
        raise ValueError("degenerate or repeated indexed triangles")
    result = {}
    for owner in np.unique(owners):
        selected = t[owners == owner]
        edges = np.sort(np.concatenate((selected[:, [0, 1]], selected[:, [1, 2]], selected[:, [2, 0]])), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        if np.any(counts != 2):
            raise ValueError("each component must have a closed two-manifold edge boundary")
        p = v[selected]
        volume = float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()/6.)
        if volume <= 1e-9:
            raise ValueError("closed material boundary must have positive outward signed volume")
        result[str(int(owner))] = {"triangles": len(selected), "surface_area_m2": float(areas[owners == owner].sum()),
                                   "signed_volume_m3": volume, "closed_edge_manifold": True}
    return {"vertices": len(v), "triangles": len(t), "by_instance": result,
            "indexed_duplicate_triangles": 0, "degenerate_triangles": 0}


@dataclass(frozen=True)
class DevelopmentGeometryV40:
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_instance_id: np.ndarray
    triangle_rgb: np.ndarray
    private_instances: tuple
    marker_patches: tuple
    background_boxes: tuple
    public_workspace: dict
    audit: dict

    def arrays(self):
        return {name: getattr(self, name) for name in ("vertices", "triangles", "triangle_instance_id", "triangle_rgb")}

    def logical_sha256(self):
        digest = hashlib.sha256()
        for name, array in self.arrays().items():
            digest.update(canonical_json_bytes({"name": name, "dtype": str(array.dtype), "shape": list(array.shape)}))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()


def build_development_geometry(scene_spec, protocol):
    validate_scene_spec(scene_spec, expected_split="development", protocol=protocol)
    if scene_spec["parent_id"] != f"DEV_{scene_spec['family']}_00":
        raise ValueError("only the six registered development parent skeletons are supported")
    # Only the visible development seed is consumed. No directory scan, test
    # seed path, method performance or renderer input exists in this function.
    rng = np.random.default_rng(int(scene_spec["seed_record"]["seed_hex"], 16))
    family = scene_spec["family"]
    width, depth, poses, classes, structures, nominal, occluders = _profile(family)
    background = [[0., width, 0., depth, -.1, 0.], [0., .12, 0., depth, 0., 2.4],
                  [width-.12, width, 0., depth, 0., 2.4], [.12, width-.12, 0., .12, 0., 2.4],
                  [.12, width-.12, depth-.12, depth, 0., 2.4]] + deepcopy(occluders)
    v, t = union_box_mesh(background)
    all_v, all_t, owners, colors = [v], [t], [np.full(len(t), -1, np.int32)], [np.full((len(t), 3), 175, np.uint8)]
    instances, markers, offset = [], [], len(v)
    prior = scene_spec["public"]["structure_prior"]
    for instance_id, (pose, category, structure) in enumerate(zip(poses, classes, structures)):
        x, y, yaw = pose
        jitter = np.zeros(3) if family == "D" else rng.uniform(-.045, .045, size=3)
        dimensions = np.asarray(nominal)*(1+jitter)
        xy_shift = np.zeros(2) if family == "D" else rng.uniform(-.08, .08, size=2)
        center = np.asarray([x+xy_shift[0], y+xy_shift[1], 0.])
        angle = np.deg2rad(yaw)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        boxes, marker = structure_boxes(structure, dimensions)
        local_v, local_t = union_box_mesh(boxes)
        world_v = local_v @ rotation.T + center
        all_v.append(world_v); all_t.append(local_t+offset)
        owners.append(np.full(len(local_t), instance_id, np.int32)); colors.append(np.full((len(local_t), 3), 165, np.uint8))
        offset += len(local_v)
        instances.append({"instance_id": instance_id, "category": category, "structure": structure,
            "dimensions_m": dimensions.tolist(), "position_world_m": center.tolist(), "yaw_deg": float(yaw),
            "world_from_local_rotation": rotation.tolist(), "local_solid_boxes": boxes,
            "world_aabb_m": [world_v.min(axis=0).tolist(), world_v.max(axis=0).tolist()],
            "center_world_m": ((world_v.min(axis=0)+world_v.max(axis=0))*.5).tolist(),
            "conservative_footprint_xy_m": [[float(world_v[:, 0].min()), float(world_v[:, 1].min())],
                [float(world_v[:, 0].max()), float(world_v[:, 1].min())],
                [float(world_v[:, 0].max()), float(world_v[:, 1].max())],
                [float(world_v[:, 0].min()), float(world_v[:, 1].max())]],
            "declared_class_structure_probability": prior["probability_by_category"][category][prior["abstract_structures"].index(structure)],
            "structure_assignment_rule": "fixed_development_family_profile_not_class_code",
            "relationship_shift": family == "F", "recognition_corruption": False})
        markers.append({"instance_id": instance_id, "category": category,
            "center_world_m": (rotation @ np.asarray(marker["center_local_m"]) + center).tolist(),
            "normal_world": (rotation @ np.asarray(marker["normal_local"])).tolist(),
            "u_world": (rotation @ np.asarray(marker["u_local"])).tolist(),
            "v_world": (rotation @ np.asarray(marker["v_local"])).tolist(),
            "width_m": marker["width_m"], "height_m": marker["height_m"], "rgb": list(PALETTE_V40[category]),
            "decal_on_existing_solid_face": True})
    vertices = _readonly(np.concatenate(all_v), np.float64)
    triangles = _readonly(np.concatenate(all_t), np.int32)
    identities = _readonly(np.concatenate(owners), np.int32)
    rgb = _readonly(np.concatenate(colors), np.uint8)
    audit = mesh_audit(vertices, triangles, identities)
    # Conservative AABB separation is sufficient for these declared placements.
    bounds = [np.asarray(item["world_aabb_m"]) for item in instances]
    for i, first in enumerate(bounds):
        if np.any(first[0][:2] <= .12) or first[1][0] >= width-.12 or first[1][1] >= depth-.12:
            raise ValueError("facility intersects boundary walls")
        for second in bounds[i+1:]:
            if np.all(np.minimum(first[1], second[1])-np.maximum(first[0], second[0]) > 1e-9):
                raise ValueError("facility envelopes overlap")
        for obstacle in occluders:
            ob = np.asarray(obstacle).reshape(3, 2).T
            if np.all(np.minimum(first[1], ob[1])-np.maximum(first[0], ob[0]) > 1e-9):
                raise ValueError("facility envelope intersects an occluder")
    audit.update({"independent_facilities": len(instances), "facility_envelopes_disjoint": True,
                  "selection_uses_method_results": False, "geometry_generated": True,
                  "world_or_sensor_instantiated": False, "candidate_visibility_frozen": False})
    workspace = {"schema_version": "v40.development_workspace.v1", "bounds_xy_m": [[0., 0.], [width, depth]],
        "room_inner_bounds_xy_m": [[.12, .12], [width-.12, depth-.12]],
        "start_position_world_m": [.75, .75, .9], "start_yaw_deg": 0.,
        "marker_palette": {key: list(value) for key, value in PALETTE_V40.items()},
        "navigation_graph_status": "pending", "instance_truth_included": False,
        "source": "declared_task_workspace_not_facility_roi"}
    return DevelopmentGeometryV40(vertices, triangles, identities, rgb, tuple(instances), tuple(markers),
                                   tuple(background), workspace, audit)


def mesh_npz_bytes(geometry):
    stream = io.BytesIO()
    np.savez_compressed(stream, **geometry.arrays())
    return stream.getvalue()


def load_mesh_arrays(path):
    """Read bounded static arrays; allow_pickle=False, no rendering side effects."""
    path = Path(path)
    if path.stat().st_size > MAX_ASSET_BYTES:
        raise ValueError("mesh archive exceeds P1 asset cap")
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > MAX_ASSET_BYTES:
            raise ValueError("uncompressed mesh archive exceeds P1 asset cap")
    with np.load(path, allow_pickle=False) as source:
        names = ("vertices", "triangles", "triangle_instance_id", "triangle_rgb")
        if set(source.files) != set(names):
            raise ValueError("unexpected mesh array fields")
        arrays = {name: np.array(source[name], copy=True) for name in names}
    mesh_audit(arrays["vertices"], arrays["triangles"], arrays["triangle_instance_id"])
    if arrays["triangle_rgb"].dtype != np.uint8 or arrays["triangle_rgb"].shape != (len(arrays["triangles"]), 3):
        raise ValueError("aligned uint8 triangle_rgb required")
    for array in arrays.values():
        array.flags.writeable = False
    return arrays
