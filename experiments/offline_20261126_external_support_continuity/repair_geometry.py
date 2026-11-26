"""Certify measured terrain and bridge only small interior same-level scan seams."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import open3d as o3d
import shapely


V1 = Path(__file__).resolve().parents[1] / "external_support_repair_20261126/repair_geometry.py"
module = spec_from_file_location("scanned_terrain_support_v1", V1)
v1 = module_from_spec(module)
module.loader.exec_module(v1)
native = v1.native
FLOOR_BAND_M = v1.FLOOR_BAND_M

MAX_SEAM_WIDTH_M = .001
MAX_TOTAL_UNMEASURED_M2 = .0005
MAX_MEASURED_PROJECTION_RESIDUAL_M2 = 5e-8
MAX_BANK_RISE_M = .004
BANK_REACH_M = .001
SEAM_SAMPLE_STEP_M = .003
angles = np.arange(8) * np.pi / 8
BANK_DIRECTIONS = np.column_stack((np.cos(angles), np.sin(angles))) * BANK_REACH_M


class RepairedMotionGeometry(v1.RepairedMotionGeometry):
    """Keep the existing body sweep and require scan evidence throughout the footprint."""

    def __init__(self, scene, world_from_source):
        super().__init__(scene, world_from_source)
        shapely.prepare(self.floor)
        shapely.prepare(self.outer_ground)

    def _rays(self, xy):
        z = native.COLLISION_BAND_M[0] - .001
        rays = np.zeros((len(xy), 6), dtype=np.float32)
        rays[:, :2], rays[:, 2], rays[:, 5] = xy, z, -1.
        return z - self.raw_rays.cast_rays(o3d.core.Tensor(rays), nthreads=1)["t_hit"].numpy()

    @staticmethod
    def _valid_height(height):
        return np.isfinite(height) & (height >= FLOOR_BAND_M[0] - 1e-6) & (height <= FLOOR_BAND_M[1] + 1e-6)

    def _bank_height(self, sample, measured):
        """Use two measured, opposing probes and reject incompatible bank elevations."""
        banks = sample[:, None, None, :] + np.stack((BANK_DIRECTIONS, -BANK_DIRECTIONS), axis=1)[None]
        flat = banks.reshape(-1, 2)
        covered = shapely.intersects_xy(measured, flat[:, 0], flat[:, 1]).reshape(banks.shape[:-1])
        heights = self._rays(flat).reshape(covered.shape)
        difference = np.full(covered.shape[:-1], np.inf)
        both = np.all(covered & self._valid_height(heights), axis=-1)
        chords = shapely.linestrings(banks[both])
        span = shapely.length(shapely.difference(chords, measured))
        # Test the actual chord between both banks, including any gap beyond
        # the footprint's rim. Tangent chords do not witness a missing span.
        both[both] = (span > 1e-10) & (span <= MAX_SEAM_WIDTH_M + 1e-9)
        difference[both] = np.abs(heights[both, 0] - heights[both, 1])
        idx = np.argmin(difference, axis=1)
        if np.any(difference[np.arange(len(sample)), idx] > MAX_BANK_RISE_M):
            return None
        return heights[np.arange(len(sample)), idx].mean(axis=-1)

    def _seam_receipt(self, gap, measured):
        if gap.area <= 1e-12:
            return True
        if (gap.area > MAX_TOTAL_UNMEASURED_M2 or not self.outer_ground.covers(gap)
                or not gap.buffer(-MAX_SEAM_WIDTH_M / 2).is_empty):
            return False
        probes = []
        for component in shapely.get_parts(gap):
            if component.area <= 1e-12:
                continue
            edge = component.exterior
            ticks = np.linspace(0., edge.length, max(2, int(np.ceil(edge.length / SEAM_SAMPLE_STEP_M))+1))
            probes.extend(shapely.get_coordinates(shapely.line_interpolate_point(edge, ticks)))
            probes.append(shapely.get_coordinates(component.representative_point())[0])
        if not probes:
            return True
        probes = np.asarray(probes)
        direct = self._rays(probes)
        if np.any(np.isfinite(direct) & ~self._valid_height(direct)):
            return False
        return self._bank_height(probes, measured) is not None

    def _continuity(self, capsule, strict_missing, gap, measured):
        left, bottom, right, top = capsule.bounds
        xs = np.linspace(left, right, int(np.ceil((right-left)/v1.PROBE_STEP_M))+1)
        ys = np.linspace(bottom, top, int(np.ceil((top-bottom)/v1.PROBE_STEP_M))+1)
        xx, yy = np.meshgrid(xs, ys)
        grid = shapely.contains_xy(capsule, xx, yy)
        interior = np.column_stack((xx[grid], yy[grid]))
        measured_patches = strict_missing.intersection(measured)
        witnesses = [shapely.get_coordinates(p.representative_point())[0]
                     for p in shapely.get_parts(measured_patches) if p.area > 1e-12]
        samples = np.concatenate((interior, np.asarray(witnesses).reshape(-1, 2)))
        height = self._rays(samples)
        absent = ~np.isfinite(height)
        if absent.any():
            absent_xy = samples[absent]
            if not np.all(shapely.intersects_xy(gap, absent_xy[:, 0], absent_xy[:, 1])):
                return False
            bridged = self._bank_height(absent_xy, measured)
            if bridged is None:
                return False
            height[absent] = bridged
        if not self._valid_height(height).all() or np.ptp(height) > v1.MAX_HEIGHT_RELIEF_M:
            return False
        plane = np.full(grid.shape, np.nan)
        plane[grid] = height[:len(interior)]
        for axis in (0, 1):
            delta = np.abs(np.diff(plane, axis=axis))
            if np.any(delta[np.isfinite(delta)] > v1.MAX_LOCAL_HEIGHT_CHANGE_M):
                return False
        return True

    def _directly_measured_residual(self, gap):
        """Separate small overlay residuals with direct hits from a true no-return seam."""
        if not 0. < gap.area <= MAX_MEASURED_PROJECTION_RESIDUAL_M2 or not self.outer_ground.covers(gap):
            return False
        centers = [shapely.get_coordinates(part.representative_point())[0]
                   for part in shapely.get_parts(gap) if part.area > 0.]
        return bool(centers) and bool(self._valid_height(self._rays(np.asarray(centers))).all())

    def valid(self, start_xy, end_xy=None, radius=.20):
        if radius == 0:
            return super().valid(start_xy, end_xy, radius)
        line, supported, distance = self.clearance(start_xy, end_xy)
        if not self._body_clear(line, radius):
            return False
        if supported and distance >= radius - 1e-9:
            return True
        capsule = line.buffer(radius / np.cos(np.pi / 128), quad_segs=32)
        if not self.floor.intersects(capsule):
            return False
        # Include the two banks just outside the capsule when a seam crosses its rim.
        probe_region = capsule.buffer(BANK_REACH_M + 1e-6)
        indices = self.raw_tree.query(probe_region, predicate="intersects")
        measured = shapely.union_all(self.raw_support[indices])
        gap = capsule.difference(measured)
        strict_missing = capsule.difference(self.floor)
        if self._directly_measured_residual(gap) and self._height_continuity(capsule, strict_missing):
            return True
        if self._seam_receipt(gap, measured) and self._continuity(capsule, strict_missing, gap, measured):
            return True
        # Preserve the original tiny projection tolerance only on the V1
        # path that also requires direct ground returns across the swept area.
        return gap.area <= v1.MAX_PROJECTION_ERROR_M2 and super().valid(start_xy, end_xy, radius)
