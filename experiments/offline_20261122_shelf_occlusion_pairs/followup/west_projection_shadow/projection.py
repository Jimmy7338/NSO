"""Endpoint-only float32 boundary shadow for recorded simulation packets."""

from hashlib import sha256
import inspect
from math import floor
from pathlib import Path

import numpy as np


def cell_tolerance(index):
    return float(np.spacing(np.float32(max(1, abs(index)))))


def directed_endpoint(origin, point, resolution, shape, raw):
    """Resolve one ambiguous face only if the observed ray crosses it unambiguously."""
    start = np.asarray(origin[:2], dtype=float) / resolution
    hit = np.asarray(point[:2], dtype=float) / resolution
    if not np.isfinite(start).all() or not np.isfinite(hit).all():
        return raw
    boundary = np.rint(hit).astype(int)
    tolerance = [cell_tolerance(index) for index in boundary]
    ambiguous = np.flatnonzero(abs(hit - boundary) <= tolerance)
    if ambiguous.size != 1:
        return raw
    axis = int(ambiguous[0])
    transverse = 1 - axis
    direction = hit - start
    displacement = direction[axis]
    signed_approach = boundary[axis] - start[axis]
    if abs(signed_approach) <= tolerance[axis] or signed_approach * displacement <= 0:
        return raw

    # Leave two ULPs from the plane to account for the one-ULP endpoint slab.
    # Independently require both ends of that slab to lie in the same transverse
    # cell. This declines near-corner and grazing cases without an angle knob.
    sign = 1 if displacement > 0 else -1
    offsets = boundary[axis] + np.array([-2., 2.]) * tolerance[axis]
    across_face = start[transverse] + direction[transverse] * (offsets - start[axis]) / displacement
    if any(floor(value) != floor(hit[transverse]) for value in across_face):
        return raw

    xy = [floor(value) for value in hit]
    xy[axis] = boundary[axis] - (sign < 0)
    corrected = (shape[0] - 1 - xy[1], xy[0])
    origin_cell = (shape[0] - 1 - floor(start[1]), floor(start[0]))
    if (origin_cell in (raw, corrected)
            or not all(0 <= index < size for index, size in zip(corrected, shape, strict=True))):
        return raw
    return corrected


def bind(previous_rgb_only):
    """Instrument only SensorMapper.update, with controls selectable per mapper."""
    from nso.mapping3d import SensorMapper

    path = Path(inspect.getsourcefile(SensorMapper)).resolve()
    source = path.read_bytes()
    replacements = {
        b"        for p in obstacle:\n            r,col=self.grid_cell(p)\n":
        b"        for p in obstacle:\n            r,col=shadow_depth_hit(self,p,frame)\n",
        b"                r,col=target\n                if 0<=r<self.shape[0] and 0<=col<self.shape[1]:\n":
        b"                r,col=(shadow_scan_hit(self,target,scan,local)\n"
        b"                       if distance<scan.range_max_m-1e-3 else target)\n"
        b"                if 0<=r<self.shape[0] and 0<=col<self.shape[1]:\n",
    }
    modified = source
    for before, after in replacements.items():
        if modified.count(before) != 1:
            raise ValueError(f"Endpoint source cannot be bound uniquely in {path}")
        modified = modified.replace(before, after)

    def shadow_depth_hit(mapper, point, frame):
        mode = getattr(mapper, "_endpoint_shadow_mode", "raw")
        if mode == "original_rgb_only":
            return previous_rgb_only(mapper, point)
        raw = mapper.grid_cell(point)
        if mode == "directional":
            return directed_endpoint(frame.world_from_camera[:3, 3], point,
                                     mapper.config.resolution_m, mapper.shape, raw)
        return raw

    def shadow_scan_hit(mapper, raw, scan, local):
        if getattr(mapper, "_endpoint_shadow_mode", "raw") != "directional":
            return raw
        origin = scan.world_from_laser[:3, 3]
        point = scan.world_from_laser[:3, :3] @ local + origin
        return directed_endpoint(origin, point, mapper.config.resolution_m, mapper.shape, raw)

    namespace = {"shadow_depth_hit": shadow_depth_hit, "shadow_scan_hit": shadow_scan_hit}
    exec(compile(modified, str(path) + ":directional_endpoint_shadow", "exec"), namespace)
    SensorMapper.update = namespace["SensorMapper"].update
    return {"original_update_path": str(path), "original_update_sha256": sha256(source).hexdigest(),
            "shadow_update_sha256": sha256(modified).hexdigest(),
            "changed_targets": ["RGB-D obstacle endpoints", "valid finite laser obstacle endpoints"],
            "original_laser_free_space_guard_control_tsdf_unchanged": True}
