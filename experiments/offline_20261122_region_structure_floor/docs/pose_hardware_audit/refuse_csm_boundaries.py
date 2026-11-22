"""Compare exact and CSM map fusion on fixed RGB-D/LiDAR episode observations."""

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import open3d as o3d

from audit_archive import sha256
from prepare_simulated_scan_trace import SCENE, digest_array


ROOT = SCENE.parents[1]
HERE = Path(__file__).resolve().parent
ARMS = ('original_G', 'region_G')
POSES = ('exact', 'CSM')


def read(path):
    return json.loads(path.read_text())


def write(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def so2(position):
    x, y, yaw = position
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray([[c, -s, 0., x], [s, c, 0., y], [0., 0., 1., 0.], [0., 0., 0., 1.]])


def native_frame_hash(frame):
    """Frozen dual_template_exploration method/adapter.py _frame_hash contract."""
    digest = hashlib.sha256()
    for values in (frame.depth_m, frame.color_rgb, frame.semantic, frame.intrinsic, frame.world_from_camera):
        array = np.ascontiguousarray(values)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    digest.update(repr(float(frame.timestamp_s)).encode())
    return digest.hexdigest()


def load_estimates(directory, receipts, arm):
    poses_path = directory / 'csm_stamped_poses.json'
    poses = read(poses_path)
    assert sha256(poses_path) == receipts['files_sha256'][f'replay_{arm}/{poses_path.name}']
    with np.load(directory / 'native_sensor_input.npz', allow_pickle=False) as raw:
        stamps = raw['stamp_ns']
    by_stamp = {row['stamp_ns']: row for row in poses['ros_poses']}
    assert len(by_stamp) == len(poses['ros_poses'])
    return {paid: (int(stamp), np.asarray(by_stamp[int(stamp)]['xyyaw'], dtype=float))
            for paid, stamp in enumerate(stamps[::10])}


def fuse(arm, scenes, mapper_type, output, input_receipt):
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    directory = HERE / f'replay_{arm}'
    trace_receipt = read(directory / 'trace_receipt.json')
    run = Path(trace_receipt['source_run'])
    for filename, digest in trace_receipt['source_run_files'].items():
        assert sha256(run / filename) == digest
    assert sha256(directory / 'native_sensor_input.npz') == trace_receipt['raw_input_sha256']
    estimates = load_estimates(directory, input_receipt, arm)
    rows = read(run / 'steps.json')[:trace_receipt['paid_actions']]
    assert len(rows) + 1 == len(estimates)
    expected = [read(run / 'initial_raw_rgbd_scan_sha256.json'), *rows]
    world = scenes.make_world(trace_receipt['layout'])
    mappers = {mode: mapper_type(world.shape, world.config) for mode in POSES}
    frame, scan = world.sense(), world.scan()
    anchor = scan.world_from_laser.copy()
    anchor[2, 3] = 0.
    base_from_camera = np.linalg.inv(anchor) @ frame.world_from_camera
    base_from_laser = np.linalg.inv(anchor) @ scan.world_from_laser
    exact_check = np.load(run / 'observed_belief_trace.npz', allow_pickle=False)
    with exact_check as values:
        native_beliefs = values['belief'][:len(expected)]
        assert values['paid'][:len(expected)].tolist() == list(range(len(expected)))
    final_poses = {}
    receipts = []
    map_history = {mode: [] for mode in POSES}
    for paid, row in enumerate(expected):
        if paid:
            frame, collision, _ = world.step(row['action'])
            scan = world.scan()
            assert paid == world.step_count and bool(collision) == row['collision']
            assert list(world.position) == row['position'] and world.heading == row['heading']
            assert native_frame_hash(frame) == row['frame_sha256']
        assert isinstance(frame, RGBDFrame) and isinstance(scan, PlanarScan)
        assert digest_array(frame.depth_m) == row['depth_values_sha256']
        assert digest_array(frame.color_rgb) == row['rgb_values_sha256']
        assert digest_array(scan.ranges_m) == row['scan_ranges_sha256']
        scan_stamp, estimate = estimates[paid]
        assert np.isfinite(estimate).all()
        world_from_estimated_base = anchor @ so2(estimate)
        estimated_frame = replace(frame, world_from_camera=world_from_estimated_base @ base_from_camera)
        estimated_scan = replace(scan, world_from_laser=world_from_estimated_base @ base_from_laser)
        assert estimated_frame.depth_m is frame.depth_m and estimated_frame.color_rgb is frame.color_rgb
        assert estimated_frame.semantic is frame.semantic and estimated_frame.intrinsic is frame.intrinsic
        assert estimated_scan.ranges_m is scan.ranges_m
        mappers['exact'].update(frame, scan)
        mappers['CSM'].update(estimated_frame, estimated_scan)
        assert np.array_equal(mappers['exact'].belief, native_beliefs[paid]), f'Native map disagreement at paid {paid}'
        for mode in POSES:
            map_history[mode].append(mappers[mode].belief.copy())
        final_poses['exact'] = frame.world_from_camera
        final_poses['CSM'] = estimated_frame.world_from_camera
        receipts.append({'paid': paid, 'simulator_sensor_time_s': float(frame.timestamp_s), 'csm_stamp_ns': scan_stamp,
                         'sensor_payload_sha256': {key: row[key] for key in ('depth_values_sha256',
                                                                             'rgb_values_sha256',
                                                                             'scan_ranges_sha256')},
                         'semantic_sha256': digest_array(frame.semantic),
                         'camera_intrinsic_sha256': digest_array(frame.intrinsic),
                         'native_belief_sha256': digest_array(native_beliefs[paid]),
                         'csm_world_from_base': world_from_estimated_base.tolist()})
    arm_path = output / arm
    arm_path.mkdir()
    write(arm_path / 'fusion_input_receipts.json', receipts)
    assert {x.frames for x in mappers.values()} == {len(expected)}
    for mode, mapper in mappers.items():
        mesh = mapper.mesh()
        assert o3d.io.write_triangle_mesh(str(arm_path / f'{mode}.ply'), mesh, write_ascii=False)
        np.savez_compressed(arm_path / f'{mode}_belief.npz', belief=np.stack(map_history[mode]),
                            world_from_camera=final_poses[mode], paid=np.arange(len(expected)))
    return {'fused_frames_per_mode': len(expected), 'native_exact_grid_reproduced_at_all_paid_clocks': True,
            'historical_native_belief_sha256': sha256(run / 'observed_belief_trace.npz'),
            'planning_or_execution_policy_called': False, 'arm': arm}


def score(output, fused, frozen):
    sys.path.insert(0, str(ROOT))
    import evaluate as evaluator
    weighted = evaluator.import_file('csm_audit_frozen_map_evaluation', evaluator.WEIGHTED_METRIC)
    scene_root = Path(frozen['scene_root'])
    weighted.SCENES = scene_root
    metric = weighted.files.load_metric(read(weighted.files.SCENES / 'freeze.json'),
                                       read(weighted.files.OLD_EVAL / 'freeze.json'))
    layout = 'floor26_ordinary'
    ref = weighted.load_reference(layout, read(scene_root / 'prepared' / layout / 'initial_state.json'))
    for info in fused:
        arm_path = output / info['arm']
        pair = {}
        grids = {}
        for mode in POSES:
            with np.load(arm_path / f'{mode}_belief.npz', allow_pickle=False) as state:
                grids[mode] = state['belief']
            mesh = weighted.files.read_mesh(arm_path / f'{mode}.ply')
            pair[mode], _ = weighted.files.measure_geometry(mesh, grids[mode][-1], ref, metric)
        changed = np.count_nonzero(grids['CSM'] != grids['exact'], axis=(1, 2))
        info.update({'metrics': pair, 'final_cells_differing_from_exact': int(changed[-1]),
                     'max_cells_differing_from_exact_across_clocks': int(changed.max()),
                     'csm_minus_exact': {key: pair['CSM'][key] - pair['exact'][key] for key in (
                         'C_free', 'F1_scene_05cm', 'p_scene_05cm', 'r_scene_05cm', 'J')},
                     'output_files_sha256': {p.name: sha256(p) for p in arm_path.iterdir() if p.is_file()}})
    return ref['source_reference_receipt_sha256']


def main(output):
    from importlib.util import module_from_spec, spec_from_file_location
    freeze_path = ROOT / 'FORMAL_FREEZE.json'
    frozen = read(freeze_path)
    assert sha256(SCENE) == frozen['actor_files'][str(SCENE)]
    spec = spec_from_file_location('csm_audit_refusion_fixture', SCENE)
    scenes = module_from_spec(spec)
    spec.loader.exec_module(scenes)
    mapper_path = next(Path(path) for path in frozen['actor_files'] if path.endswith('nso/observed_runtime_mapper_v10.py'))
    assert sha256(mapper_path) == frozen['actor_files'][str(mapper_path)]
    sys.path.insert(0, str(mapper_path.parents[1]))
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
    assert Path(inspect.getfile(ObservedRuntimeMapperV10)) == mapper_path
    output.mkdir(parents=True, exist_ok=False)
    pair_path = HERE / 'paired_csm_receipt.json'
    pair_receipts = read(pair_path)['arms']
    fused = [fuse(arm, scenes, ObservedRuntimeMapperV10, output, pair_receipts[arm]) for arm in ARMS]
    reference_hash = score(output, fused, frozen)
    result = {'schema': 'nso.offline_fixed_actions_pose_refusion/1', 'layout': 'floor26_ordinary',
              'poses': list(POSES), 'frozen_mapper_path': str(mapper_path), 'frozen_mapper_sha256': sha256(mapper_path),
              'source_trial_freeze_sha256': sha256(freeze_path), 'csm_replay_receipt_sha256': sha256(pair_path),
              'evaluator_reference_receipt_sha256': reference_hash, 'max_worker_threads': 1,
              'same_sensor_payloads_intrinsics_timing_and_order': True,
              'estimated_pose_input': 'Continuous CSM boundary pose with common fixed starting anchor and sensor extrinsics',
              'simulator_ground_truth_after_initial_anchor_used_in_estimated_pose': False,
              'packet_validation_or_guard_modified': False, 'hardware_io': False,
              'estimated_pose_closed_loop': False, 'rtab_pose_graph_applied': False, 'arms': fused}
    write(output / 'map_refusion_receipt.json', result)
    print(json.dumps([{key: row[key] for key in ('arm', 'metrics', 'csm_minus_exact', 'final_cells_differing_from_exact',
                                                 'max_cells_differing_from_exact_across_clocks')} for row in fused], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    main(parser.parse_args().out.resolve())
