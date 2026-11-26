"""Repair scan support extraction with measured local terrain, for simulation only."""

from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.strtree import STRtree
import trimesh


_SCENE_ADAPTER = Path(__file__).resolve().parents[1] / "semantic_scene_redesign_20261119/scene_adapter"
sys.path.insert(0, str(_SCENE_ADAPTER))
import replicacad_world as native


FLOOR_BAND_M = (-.09, .09)
GROUND_NORMAL_MIN = .96
MAX_OMITTED_AREA_M2 = .006
MAX_HEIGHT_RELIEF_M = .030
MAX_LOCAL_HEIGHT_CHANGE_M = .012
PROBE_STEP_M = .005
MAX_PROJECTION_ERROR_M2 = 1e-8


def _world_parts(scene, transform):
    for node in scene.graph.nodes_geometry:
        local, name = scene.graph[node]
        source = scene.geometry[name]
        yield trimesh.transform_points(source.vertices, transform @ local), np.asarray(source.faces)


class RepairedMotionGeometry:
    """Allow observed shallow continuous support with its original swept footprint."""

    def __init__(self, scene, world_from_source):
        previous = native.FLOOR_BAND_M, native.FLOOR_NORMAL_MIN
        native.FLOOR_BAND_M, native.FLOOR_NORMAL_MIN = FLOOR_BAND_M, GROUND_NORMAL_MIN
        try:
            self.original = native.MeshMotionGeometry(scene, world_from_source)
        finally:
            native.FLOOR_BAND_M, native.FLOOR_NORMAL_MIN = previous
        self.obstacles, self.tree = self.original.obstacles, self.original.tree
        self.raw_rays = o3d.t.geometry.RaycastingScene(nthreads=1)
        raw_support, clipped_ground = [], []
        low, high = FLOOR_BAND_M
        for vertices, faces in _world_parts(scene, world_from_source):
            self.raw_rays.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                                        o3d.core.Tensor(faces.astype(np.uint32)))
            tris = vertices[faces]
            z_min = tris[:, :, 2].min(axis=1)
            z_max = tris[:, :, 2].max(axis=1)
            near = tris[(z_min <= high) & (z_max >= low)]
            normals = np.cross(near[:, 1] - near[:, 0], near[:, 2] - near[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            for triangle, normal, length in zip(near, normals, lengths, strict=True):
                if abs(normal[2]) <= 2e-12:
                    continue
                clipped = not (np.min(triangle[:, 2]) >= low and np.max(triangle[:, 2]) <= high)
                polygon = (native._clip_halfspace(native._clip_halfspace(triangle, low, True), high, False)
                           if clipped else triangle)
                if len(polygon) < 3:
                    continue
                area = Polygon(polygon[:, :2])
                if area.area < 1e-12:
                    continue
                raw_support.append(area)
                if clipped and abs(normal[2]) >= GROUND_NORMAL_MIN * length:
                    clipped_ground.append(area)
        self.floor = shapely.union_all([self.original.floor, *clipped_ground])
        self.floor_boundary = self.floor.boundary
        polygons = [g for g in shapely.get_parts(self.floor) if g.geom_type == "Polygon"]
        self.outer_ground = shapely.union_all([Polygon(polygon.exterior) for polygon in polygons])
        self.raw_support = np.asarray(raw_support, dtype=object)
        self.raw_tree = STRtree(self.raw_support)
        self.stats = self.original.stats | {"clipped_near_horizontal_parts": len(clipped_ground),
                                           "measured_band_parts": len(raw_support)}

    def clearance(self, start_xy, end_xy=None):
        line = Point(start_xy) if end_xy is None else LineString([start_xy, end_xy])
        covered = bool(self.floor.covers(line))
        return line, covered, float(line.distance(self.floor_boundary)) if covered else 0.

    def _body_clear(self, center, radius):
        x0, y0, x1, y1 = center.bounds
        ids = self.tree.query(box(x0-radius, y0-radius, x1+radius, y1+radius))
        return not len(ids) or not np.any(shapely.distance(self.obstacles[ids], center) <= radius + 1e-9)

    def _height_continuity(self, capsule, missing):
        left, bottom, right, top = capsule.bounds
        xs = np.linspace(left, right, int(np.ceil((right-left)/PROBE_STEP_M))+1)
        ys = np.linspace(bottom, top, int(np.ceil((top-bottom)/PROBE_STEP_M))+1)
        xx, yy = np.meshgrid(xs, ys)
        grid = shapely.contains_xy(capsule, xx, yy)
        interior = np.column_stack((xx[grid], yy[grid]))
        parts = [p for p in shapely.get_parts(missing) if p.area > 1e-12]
        witnesses = np.array([(p.representative_point().x, p.representative_point().y) for p in parts]).reshape(-1, 2)
        samples = np.concatenate((interior, witnesses))
        if not len(samples):
            return False
        z = native.COLLISION_BAND_M[0] - .001
        rays = np.zeros((len(samples), 6), dtype=np.float32)
        rays[:, :2] = samples
        rays[:, 2], rays[:, 5] = z, -1.
        height = z - self.raw_rays.cast_rays(o3d.core.Tensor(rays), nthreads=1)["t_hit"].numpy()
        if (not np.isfinite(height).all() or np.min(height) < FLOOR_BAND_M[0] - 1e-6
                or np.max(height) > FLOOR_BAND_M[1] + 1e-6 or np.ptp(height) > MAX_HEIGHT_RELIEF_M):
            return False
        plane = np.full(grid.shape, np.nan)
        plane[grid] = height[:len(interior)]
        for axis in (0, 1):
            delta = np.abs(np.diff(plane, axis=axis))
            if np.any(delta[np.isfinite(delta)] > MAX_LOCAL_HEIGHT_CHANGE_M):
                return False
        return True

    def valid(self, start_xy, end_xy=None, radius=.20):
        line, supported, distance = self.clearance(start_xy, end_xy)
        if not supported or not self._body_clear(line, radius):
            return False
        if distance >= radius - 1e-9:
            return True
        # Circumscribe the ideal circle in the finite-resolution check.
        capsule = line.buffer(radius / np.cos(np.pi / 128), quad_segs=32)
        if not self.outer_ground.covers(capsule):
            return False
        missing = capsule.difference(self.floor)
        # Use the same patch-area allowance for poses and full transitions.
        if missing.area > MAX_OMITTED_AREA_M2:
            return False
        hits = self.raw_tree.query(missing, predicate="intersects")
        measured = shapely.union_all(self.raw_support[hits])
        if missing.difference(measured).area > MAX_PROJECTION_ERROR_M2:
            return False
        return self._height_continuity(capsule, missing)
