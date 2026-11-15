"""Explicit physical-range variant; frozen research sources remain unchanged.

Camera depth, quality, aperture and shortlist spacing keep their camera range.
Only lidar forecast and canonical measured scan support use eight metres.
This adapter is not a claim of original-paper equivalence.
"""
from dataclasses import replace
import hashlib
import numpy as np

PROFILE = 'sensor_range_matched_v1'
LIDAR_RANGE_M = 8.0


def validate_profile(profile, *, policy_variant, camera_range_m):
    if profile is None:
        return
    if profile != PROFILE or policy_variant != 'exploration_joint_v1':
        raise ValueError('sensor_range_matched_v1 requires exploration_joint_v1')
    if not np.isclose(camera_range_m, 4., rtol=0, atol=1e-12):
        raise ValueError('sensor_range_matched_v1 declares camera4m/lidar8m')


def lidar_range_m(mapper):
    return (LIDAR_RANGE_M if getattr(mapper, 'sensor_range_profile', None) == PROFILE
            else mapper.config.max_depth_m)


def lidar_range_cells(mapper):
    return int(lidar_range_m(mapper) / mapper.config.resolution_m)


def radar_unknown_mask(mapper, states):
    from utils.grid_geometry import inflated_obstacles, visible_mask
    occupied = mapper.belief == 1
    unknown = (mapper.belief == -1) & ~inflated_obstacles(
        occupied, mapper.config.robot_radius_m / mapper.config.resolution_m)
    radar = np.zeros(mapper.shape, bool)
    for state in states:
        radar |= visible_mask(occupied, tuple(state[:2]), int(state[2]),
                              lidar_range_cells(mapper), 360.) & unknown
    return radar


def adapt_scan(scan, profile):
    """Finite far measurements support free rays only; missing rays stay missing.

    The preserved mapper marks hits only at distance<range_max-1e-3.
    Clipped endpoints equal range_max, so they cannot become fake obstacles.
    """
    if profile is None:
        return scan, None
    if profile != PROFILE:
        raise ValueError('unknown sensor range profile')
    ranges = np.asarray(scan.ranges_m).copy()
    finite_positive = np.isfinite(ranges) & (ranges > 0)
    if not np.isfinite(scan.range_max_m) or scan.range_max_m < LIDAR_RANGE_M:
        raise ValueError('matched lidar range requires measured range_max>=8m')
    # Do not convert corrupt, beyond-source-max returns into measured support.
    beyond_source = finite_positive & (ranges > scan.range_max_m)
    ranges[beyond_source] = np.nan
    clipped = finite_positive & ~beyond_source & (ranges > LIDAR_RANGE_M)
    ranges[clipped] = LIDAR_RANGE_M
    receipt = dict(profile=PROFILE, raw_range_max_m=float(scan.range_max_m),
        canonical_range_max_m=LIDAR_RANGE_M, clipped_finite_returns=int(clipped.sum()),
        rejected_beyond_source_max=int(beyond_source.sum()),
        missing_nonfinite_returns=int(np.count_nonzero(~np.isfinite(scan.ranges_m))),
        source_ranges_sha256=hashlib.sha256(np.asarray(scan.ranges_m).tobytes()).hexdigest(),
        canonical_ranges_sha256=hashlib.sha256(ranges.tobytes()).hexdigest(),
        endpoint_policy='finite measured >8m clipped to8m free-only; NaN/Inf remain nonfinite',
        raw_source_file_unchanged=True)
    return replace(scan, ranges_m=ranges, range_max_m=LIDAR_RANGE_M), receipt


def range_identity(profile, camera_range_m, scan_receipt=None):
    return dict(profile=profile or 'legacy_shared_camera_forecast_actual_scan_range',
        camera_depth_quality_aperture_range_m=float(camera_range_m),
        lidar_forecast_range_m=LIDAR_RANGE_M if profile == PROFILE else float(camera_range_m),
        lidar_canonical_support_range_m=LIDAR_RANGE_M if profile == PROFILE else None,
        observed_navigation_map_range_m=8. if profile == PROFILE else None,
        original_paper_equivalence=False,
        last_scan_adapter_receipt=scan_receipt)
