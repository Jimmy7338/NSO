"""Simulator-only shadow binding for depth endpoint floor quantization."""

from hashlib import sha256
import inspect
from math import floor
from pathlib import Path

import numpy as np


def boundary_index(value_m, resolution_m):
    exact_index = float(value_m) / resolution_m
    nearest_integer = round(exact_index)
    # The renderer returns float32 metric depth. Treat one float32 ULP in
    # cell coordinates as numerical ambiguity adjacent to an integer boundary.
    # Outside that neighborhood the original half-open cell convention holds.
    tolerance = float(np.spacing(np.float32(max(1, abs(nearest_integer)))))
    return nearest_integer if abs(exact_index - nearest_integer) <= tolerance else floor(exact_index)


def normalized_cell(mapper, point):
    c = mapper.config.resolution_m
    return mapper.shape[0]-1-boundary_index(point[1], c), boundary_index(point[0], c)


def camera_cell(mapper, point):
    if getattr(mapper, '_simulator_camera_boundary_shadow', False):
        return normalized_cell(mapper, point)
    return mapper.grid_cell(point)


def lidar_hit_cell(mapper, raw_target, scan, local):
    if getattr(mapper, '_simulator_lidar_boundary_shadow', False):
        point = scan.world_from_laser[:3,:3] @ local + scan.world_from_laser[:3,3]
        return normalized_cell(mapper, point)
    return raw_target


def bind(include_laser=False):
    from nso.mapping3d import SensorMapper

    source = Path(inspect.getsourcefile(SensorMapper)).resolve()
    raw = source.read_bytes()
    needle = b'        for p in obstacle:\n            r,col=self.grid_cell(p)\n'
    change = b'        for p in obstacle:\n            r,col=simulator_shadow_camera_cell(self,p)\n'
    if raw.count(needle) != 1:
        raise ValueError(f'No uniquely matched RGB-D obstacle projection at {source}')
    patched_source = raw.replace(needle, change)
    namespace = {'simulator_shadow_camera_cell': camera_cell, 'simulator_shadow_lidar_hit_cell': lidar_hit_cell}
    if include_laser:
        laser_target = b'                r,col=target\n                if 0<=r<self.shape[0] and 0<=col<self.shape[1]:\n'
        laser_replaced = (
            b'                r,col=(simulator_shadow_lidar_hit_cell(self,target,scan,local)\n'
            b'                       if distance<scan.range_max_m-1e-3 else target)\n'
            b'                if 0<=r<self.shape[0] and 0<=col<self.shape[1]:\n'
        )
        if patched_source.count(laser_target) != 1:
            raise ValueError(f'No uniquely matched laser obstacle endpoint in {source}')
        patched_source = patched_source.replace(laser_target, laser_replaced)
    exec(compile(patched_source, str(source)+':camera-precision-shadow', 'exec'), namespace)
    SensorMapper.update = namespace['SensorMapper'].update
    info = {
        'source': str(source),
        'source_sha256': sha256(raw).hexdigest(),
        'isolated_update_source_sha256': sha256(patched_source).hexdigest(),
        'only_changed_statement': 'RGB-D obstacle endpoint grid_cell inside SensorMapper.update',
        'snap_tolerance': 'one float32 ULP in each scaled cell coordinate',
        'native_guard_modified': False,
        'plan_budget_laser_tsdf_and_surface_evidence_unchanged_in_source': True,
    }
    if include_laser:
        info.pop('plan_budget_laser_tsdf_and_surface_evidence_unchanged_in_source')
        info['only_changed_statement'] = 'RGB-D obstacle endpoint and actual finite laser obstacle endpoint'
        info['laser_free_rays_tsdf_surface_evidence_and_policy_unchanged_in_source'] = True
    return info
