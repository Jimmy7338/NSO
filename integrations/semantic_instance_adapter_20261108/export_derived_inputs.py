"""Export compact measured samples for association/projection checks, not RGB-D replay."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
REPORT = Path('/home/ouc/nso_robot/reports/nso-hybrid-20261108/legacy_mount_S40_20261108_01')
sys.path.insert(0, str(WORK / 'nso_bridge'))
from nso_bridge.joint_engine import _MeasuredMapMapper
from nso_bridge.sensor_range_adapter import adapt_scan
from env.virtual3d_v2 import VirtualConfigV2
from utils.rgbd_contract import RGBDFrame, PlanarScan
from association import quality_samples, quality_keys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_pixels(frame):
    """Native quality sample validity and first sample in each 15 cm voxel."""
    depth = frame.depth_m
    v, u = np.mgrid[:depth.shape[0], :depth.shape[1]]
    local = np.stack([(u-frame.intrinsic[0, 2])*depth/frame.intrinsic[0, 0],
                      (v-frame.intrinsic[1, 2])*depth/frame.intrinsic[1, 1], depth], axis=-1)
    dx = np.roll(local, -1, 1)-np.roll(local, 1, 1)
    dy = np.roll(local, -1, 0)-np.roll(local, 1, 0)
    valid = (depth > .15) & (np.linalg.norm(np.cross(dx, dy), axis=-1) > 1e-8)
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    for axis in (0, 1):
        valid &= (np.roll(depth, 1, axis) > 0) & (np.roll(depth, -1, axis) > 0)
    rows, cols = np.nonzero(valid & (v % 3 == 0) & (u % 3 == 0))
    points = local[rows, cols] @ frame.world_from_camera[:3, :3].T + frame.world_from_camera[:3, 3]
    keys = np.floor(points/.15).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return keys[first], np.column_stack([cols[first], rows[first]]), depth[rows[first], cols[first]]


def main():
    output = HERE / 'derived_inputs.npz'
    if output.exists():
        raise FileExistsError(output)
    review = json.loads((HERE / 'review.json').read_text())
    rows = [json.loads(x) for x in (REPORT / 'joint/decisions.jsonl').read_text().splitlines()]
    observations = [r for r in rows if r['request']['method'] == 'observe'
                    and r['response'].get('ok') and r['response']['result'].get('accepted')
                    and not r['response']['result'].get('duplicate')]
    config = json.loads((REPORT / 'joint_configuration.json').read_text())['configuration']
    for name, expected in review['source_sha256'].items():
        assert sha(WORK / name) == expected
    params = dict(config['sensor_config'])
    params.update(resolution_m=.2, robot_radius_m=config['robot_radius_m'])
    native = VirtualConfigV2(**params)
    mapper = _MeasuredMapMapper(tuple(config['shape']), native)
    mapper.sensor_range_profile = config['sensor_range_profile']
    transform = np.asarray(config['planning_from_odom'])
    arrays, records = {}, []
    for row in observations:
        action = row['request']['params']['action_id']
        rec = row['response']['result']['summary']['semantic_diagnostics']['receipt']
        packet, masks = Path(row['request']['params']['path']), Path(rec['mask_file'])
        assert sha(packet) == rec['source_file_sha256'] and sha(masks) == rec['mask_file_sha256']
        with np.load(packet, allow_pickle=False) as z, np.load(masks, allow_pickle=False) as cached:
            assert not json.loads(str(z['joint_metadata_json'])).get('approved_self_mask_applied')
            valid = np.isfinite(z['depth_m']) & (z['depth_m'] > 0) & (z['depth_m'] <= native.max_depth_m)
            owner = np.zeros(valid.shape, np.int64)
            owned = np.zeros(valid.shape, bool)
            semantic = np.zeros(valid.shape, np.int16)
            items = rec['detections']
            for i in sorted(range(len(items)), key=lambda i: (-items[i]['detector_score'], i)):
                mask = cached[f'mask_{i}']
                assert hashlib.sha256(mask.tobytes()).hexdigest() == items[i]['mask_sha256']
                measured = mask & valid
                selected = measured & ~owned
                assert int(selected.sum()) == items[i]['assigned_pixels']
                owner[selected] = i+1
                if items[i]['v10_class_id'] is not None:
                    semantic[selected] = items[i]['v10_class_id']
                owned |= measured
            stride = rec['pixel_stride']
            canonical_owner = owner[::stride, ::stride]
            labels = cached['semantic'].copy()
            assert np.array_equal(semantic[::stride, ::stride], labels)
            frame = RGBDFrame(float(z['rgbd_timestamp_s']), z['depth'].copy(), z['color_rgb'].copy(),
                             z['intrinsic'].copy(), transform @ z['world_from_camera'], labels)
            scan = PlanarScan(float(z['scan_timestamp_s']), z['ranges'].copy(), float(z['angle_min']),
                              float(z['angle_increment']), float(z['range_max']), transform @ z['world_from_laser'])
            scan, _ = adapt_scan(scan, config['sensor_range_profile'])
            mapper.input_observed_grid = z['joint_belief_grid'].copy()
        keys, owners, categories = quality_samples(frame, canonical_owner)
        projected_keys, uv, depth = selected_pixels(frame)
        assert np.array_equal(keys, projected_keys)
        prefix = f'a{action:02d}_'
        arrays.update({prefix+'keys': keys, prefix+'owner': owners, prefix+'label': categories,
                       prefix+'uv': uv, prefix+'depth_m': depth, prefix+'K': frame.intrinsic,
                       prefix+'planning_from_camera': frame.world_from_camera})
        mapper.update(frame, scan)
        if action in (0, 3, 18, 22):
            evidence = mapper.quality_evidence(max_points=10000)
            arrays[prefix+'quality_keys'] = quality_keys(mapper, evidence)
            for key, value in evidence.items():
                arrays[prefix+'quality_'+key] = value
            expected = next(x for x in review['checkpoints'] if x['action'] == action)
            assert hashlib.sha256(mapper.belief.tobytes()).hexdigest() == expected['mapping_belief_sha256']
        records.append(dict(action=action, frame_id=rec['frame_id'], samples=len(keys),
                            packet_sha256=sha(packet), mask_archive_sha256=sha(masks),
                            original_mask_file=str(masks), original_packet=str(packet),
                            detections=[{k: x[k] for k in ('instance_index', 'source_class_id', 'source_label',
                                                         'detector_score', 'mask_sha256')} for x in items]))
    np.savez_compressed(output, **arrays)
    metadata = dict(schema='nso.compact_measured_association_inputs.v1',
        status='derived_samples_only_not_online_reconstruction_replay',
        data_sha256=sha(output), bytes=output.stat().st_size, report=str(REPORT), frames=records,
        checkpoints=[0, 3, 18, 22], config=dict(shape=config['shape'], resolution_m=.2),
        exporter_sha256=sha(__file__), frozen_association_sha256=sha(HERE/'association.py'),
        original_review_sha256=sha(HERE/'review.json'),
        boundary='Allows exact measured-sample projection, association, ownership and patch-eligibility checks. '
                 'No full RGB-D, complete retina masks or full scans are included. Cannot reproduce detection, '
                 'prove full-mask ownership independently, rerun complete online mapping/TSDF, or execute a policy.')
    (HERE/'derived_inputs.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(output, output.stat().st_size)


if __name__ == '__main__':
    main()
