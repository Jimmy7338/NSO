"""Archive LiDAR-only input and a separately stored reference from one fixed simulated action trace."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d

from audit_archive import sha256
from csm_replay_metrics import wrap


SCENE = Path(__file__).resolve().parents[2] / 'scenes/scene.py'
STAMP_START_NS = 1_750_000_000 * 10**9
SUBSTEPS = 10


def read(path):
    return json.loads(path.read_text())


def capture(world, xyz_yaw):
    x, y, yaw = xyz_yaw
    angle_step = 2 * np.pi / world.config.laser_rays
    angles = -np.pi + np.arange(world.config.laser_rays) * angle_step + yaw
    rays_world = np.column_stack((np.full(angles.shape, x), np.full(angles.shape, y),
                                  np.full(angles.shape, world.config.laser_height_m),
                                  np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    source = world.source_from_world
    rays = np.column_stack((rays_world[:, :3] @ source[:3, :3].T + source[:3, 3],
                            rays_world[:, 3:] @ source[:3, :3].T)).astype(np.float32)
    return np.minimum(world._ray.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy(),
                      world.config.max_depth_m).astype(np.float32)


def pose(world):
    scan = world.scan()
    transform = scan.world_from_laser
    x, y = transform[:2, 3]
    return np.asarray((x, y, math.atan2(transform[1, 0], transform[0, 0]))), scan


def digest_array(array):
    from hashlib import sha256 as hash_bytes
    return hash_bytes(np.ascontiguousarray(array).tobytes()).hexdigest()


def archive(run, out, actions):
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('csm_audit_scene', SCENE)
    scenes = module_from_spec(spec)
    spec.loader.exec_module(scenes)
    manifest = read(run / 'manifest.json')
    initial = read(run / 'initial_raw_rgbd_scan_sha256.json')
    steps = read(run / 'steps.json')
    if len(steps) < actions:
        raise ValueError(f'Fixed replay requires {actions} paid actions, source has {len(steps)}')
    layout = manifest['layout']
    world = scenes.make_world(layout)
    current, start_scan = pose(world)
    if digest_array(start_scan.ranges_m) != initial['scan_ranges_sha256']:
        raise ValueError('Recreated initial scan differs from native action trace')
    ranges, reference, paid_ends = [start_scan.ranges_m.copy()], [current], [0]
    for row in steps[:actions]:
        previous = current
        _, collision, _ = world.step(row['action'])
        assert list(world.position) == row['position'] and world.heading == row['heading']
        assert collision == row['collision'] and row['step'] == world.step_count
        current, scan = pose(world)
        assert digest_array(scan.ranges_m) == row['scan_ranges_sha256']
        delta = current - previous
        delta[2] = wrap(delta[2])
        for substep in range(1, SUBSTEPS):
            subpose = previous + delta * (substep / SUBSTEPS)
            assert world._geometry.valid(subpose[:2], radius=world.config.robot_radius_m)
            ranges.append(capture(world, subpose))
            reference.append(subpose)
            paid_ends.append(-1)
        ranges.append(scan.ranges_m.copy())
        reference.append(current)
        paid_ends.append(row['step'])
    ranges = np.stack(ranges)
    assert np.isfinite(ranges).all() and (ranges > 0).all()
    scan_period = float(world.config.action_duration_s) / SUBSTEPS
    stamps = STAMP_START_NS + np.arange(len(ranges), dtype=np.int64) * round(scan_period * 10**9)
    out.mkdir(parents=True, exist_ok=False)
    raw = out / 'native_sensor_input.npz'
    evaluation = out / 'independent_simulator_reference.npz'
    np.savez_compressed(raw, stamp_ns=stamps, ranges_m=ranges,
                        scan_period_s=scan_period, angle_min_rad=start_scan.angle_min_rad,
                        angle_increment_rad=start_scan.angle_increment_rad,
                        range_max_m=start_scan.range_max_m, laser_height_m=world.config.laser_height_m)
    np.savez_compressed(evaluation, stamp_ns=stamps, reference_xyyaw=np.stack(reference),
                        paid_action_id=np.asarray(paid_ends, dtype=np.int32))
    evidence = {'schema': 'nso.csm_simulated_lidar_replay_inputs/1', 'layout': layout, 'arm': manifest['arm'],
                'mode': 'offline_fixed_action_trace_sensor_only_pose_estimation', 'paid_actions': actions,
                'source_run_files': {name: sha256(run / name) for name in (
                    'manifest.json', 'steps.json', 'initial_raw_rgbd_scan_sha256.json')},
                'source_run': str(run), 'scene_sha256': sha256(SCENE),
                'native_kept_scan_count': actions + 1, 'native_scan_hashes_exactly_reproduced': True,
                'intermediate_raycast_count': len(ranges) - actions - 1,
                'sensor_reference_separate': True, 'robot_io': False, 'num_scan_rays': int(ranges.shape[1]),
                'motion_model': 'linear translation or shortest in-place yaw during each original charged discrete action',
                'sensor_capture': 'native 360-degree mesh raycast at action boundaries and same frozen mesh at 9 interpolated poses per action',
                'simulator_scan_period_s': scan_period, 'raw_input_sha256': sha256(raw),
                'reference_for_evaluation_only_sha256': sha256(evaluation)}
    (out / 'trace_receipt.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--actions', type=int, default=36)
    args = parser.parse_args()
    if args.actions < 1:
        parser.error('actions must be positive')
    archive(args.run.resolve(), args.out.resolve(), args.actions)
