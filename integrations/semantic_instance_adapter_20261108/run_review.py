"""One frozen offline grouping prototype; no model, planner, ROS or deployment."""
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
from nso_bridge.measured_extent_adapter import install_dispatcher
from nso_bridge.instance_support import InstanceSupportIndex
from env.virtual3d_v2 import VirtualConfigV2
from utils.rgbd_contract import RGBDFrame, PlanarScan
import nso.response_features_v7 as features
from association import MeasuredAssociation, quality_samples, quality_keys, associated_patches


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def geometry_digest(patches):
    digest = hashlib.sha256()
    for patch in patches:
        digest.update(str((patch['group'], patch['support_points'])).encode())
        for name in ('center', 'normal_out_xy', 'points', 'bits', 'observed_span'):
            digest.update(np.ascontiguousarray(patch[name]).tobytes())
    return digest.hexdigest()


def main():
    output = HERE / 'review.json'
    if output.exists():
        raise FileExistsError(output)
    proposal = json.loads((HERE / 'proposal.json').read_text())
    rows = [json.loads(line) for line in (REPORT / 'joint/decisions.jsonl').read_text().splitlines()]
    observations = [r for r in rows if r['request']['method'] == 'observe'
                    and r['response'].get('ok') and r['response']['result'].get('accepted')
                    and not r['response']['result'].get('duplicate')]
    assert [r['request']['params']['action_id'] for r in observations] == list(range(23))
    config = json.loads((REPORT / 'joint_configuration.json').read_text())['configuration']
    source = {r['path']: r['sha256'] for r in json.loads((REPORT / 'source_manifest.json').read_text())['files']}
    source_files = ['nso_bridge/nso_bridge/joint_engine.py',
                    'nso_bridge/nso_bridge/sensor_range_adapter.py',
                    'nso_bridge/nso_bridge/measured_extent_adapter.py',
                    'nso_bridge/joint_vendor/nso/mapping3d_v2.py',
                    'nso_bridge/joint_vendor/nso/response_features_v7.py']
    for name in source_files:
        assert sha(WORK / name) == source[name]
    params = dict(config['sensor_config'])
    params.update(resolution_m=.2, robot_radius_m=config['robot_radius_m'])
    native_config = VirtualConfigV2(**params)
    install_dispatcher()
    mapper = _MeasuredMapMapper(tuple(config['shape']), native_config)
    mapper.sensor_range_profile = config['sensor_range_profile']
    mapper.measured_extent_profile = config['measured_extent_profile']
    transform = np.asarray(config['planning_from_odom'])
    association, swapped = MeasuredAssociation(), MeasuredAssociation()
    checkpoints, cache_audit, belief_hashes = [], [], {}
    for row in observations:
        action = row['request']['params']['action_id']
        receipt = row['response']['result']['summary']['semantic_diagnostics']['receipt']
        packet, masks = Path(row['request']['params']['path']), Path(receipt['mask_file'])
        assert sha(packet) == receipt['source_file_sha256']
        assert sha(masks) == receipt['mask_file_sha256']
        with np.load(packet, allow_pickle=False) as z, np.load(masks, allow_pickle=False) as stored:
            metadata = json.loads(str(z['joint_metadata_json']))
            if metadata.get('approved_self_mask_applied'):
                raise ValueError('This prototype is frozen to the recorded unmasked legacy episode')
            valid = np.isfinite(z['depth_m']) & (z['depth_m'] > 0) & (z['depth_m'] <= native_config.max_depth_m)
            owner = np.zeros(valid.shape, np.int64)
            rebuilt = np.zeros(valid.shape, np.int16)
            owned = np.zeros(valid.shape, bool)
            detections = receipt['detections']
            for i in sorted(range(len(detections)), key=lambda i: (-detections[i]['detector_score'], i)):
                item = detections[i]
                mask = stored[f'mask_{i}']
                assert mask.dtype == np.bool_ and mask.shape == valid.shape
                assert array_sha(mask) == item['mask_sha256'] == str(stored[f'mask_sha256_{i}'])
                assert int(stored[f'source_id_{i}']) == item['source_class_id']
                measured = mask & valid
                assigned = measured & ~owned
                assert int(measured.sum()) == item['valid_depth_pixels']
                assert int(assigned.sum()) == item['assigned_pixels']
                owner[assigned] = i + 1
                if item['v10_class_id'] is not None:
                    rebuilt[assigned] = item['v10_class_id']
                owned |= measured
            stride = receipt['pixel_stride']
            owners = np.ascontiguousarray(owner[::stride, ::stride])
            labels = stored['semantic'].copy()
            assert np.array_equal(rebuilt[::stride, ::stride], labels)
            assert array_sha(labels) == receipt['canonical_semantic_sha256']
            frame = RGBDFrame(float(z['rgbd_timestamp_s']), z['depth'].copy(), z['color_rgb'].copy(),
                             z['intrinsic'].copy(), transform @ z['world_from_camera'], labels)
            scan = PlanarScan(float(z['scan_timestamp_s']), z['ranges'].copy(), float(z['angle_min']),
                              float(z['angle_increment']), float(z['range_max']), transform @ z['world_from_laser'])
            scan, _ = adapt_scan(scan, config['sensor_range_profile'])
            mapper.input_observed_grid = z['joint_belief_grid'].copy()
        keys, sample_owners, sample_labels = quality_samples(frame, owners)
        reference_keys, reference_owners = InstanceSupportIndex._sample(frame, owners)
        assert np.array_equal(keys, reference_keys) and np.array_equal(sample_owners, reference_owners)
        association.observe(receipt['frame_id'], keys, sample_owners, sample_labels)
        swapped_labels = np.where(sample_labels == 2, 3, np.where(sample_labels == 3, 2, 0))
        swapped.observe(receipt['frame_id'], keys, sample_owners, swapped_labels)
        assert association.tracks == swapped.tracks and association.votes == swapped.votes
        mapper.update(frame, scan)
        belief_hashes[action] = array_sha(mapper.belief)
        cache_audit.append(dict(action=action, packet_sha256=sha(packet), mask_file_sha256=sha(masks),
                                detections=len(detections), canonical_owner_sha256=array_sha(owners)))
        if action not in (0, 3, 18, 22):
            continue
        evidence = mapper.quality_evidence(max_points=10000)
        voxel_keys = quality_keys(mapper, evidence)
        patches, membership, derived = associated_patches(mapper, evidence, voxel_keys, association, features.observed_patches)
        alternate, alternate_membership, _ = associated_patches(mapper, evidence, voxel_keys, swapped, features.observed_patches)
        assert np.array_equal(membership, alternate_membership)
        assert geometry_digest(patches) == geometry_digest(alternate)
        native = features.observed_patches(mapper)
        native_geometry = geometry_digest(native)
        def summarize(patch):
            return {key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in patch.items() if key not in ('points', 'bits')}
        counters = {str(track): int(np.count_nonzero(membership == track)) for track in np.unique(membership)}
        assert sum(counters.values()) == len(evidence['point'])
        original3 = evidence['label'] == 3
        checkpoints.append(dict(action=action, quality_points=len(evidence['point']),
            native_class3_points=int(original3.sum()), associated_class3_points=int(np.count_nonzero(derived == 3)),
            membership_counts=counters, unassigned_quality_points=int(np.count_nonzero(membership == 0)),
            original_class3_points_unassigned=int(np.count_nonzero(original3 & (membership == 0))),
            original_class3_points_without_same_track_label=int(np.count_nonzero(original3 & (derived != 3))),
            native_patches=[summarize(patch) for patch in native], associated_patches=[summarize(patch) for patch in patches],
            qualified_native_marked=sum(p['marked_points'] for p in native),
            qualified_associated_marked=sum(p['marked_points'] for p in patches),
            geometry_sha256=geometry_digest(patches), swapped_class_geometry_sha256=geometry_digest(alternate),
            native_geometry_sha256=native_geometry,
            mapping_belief_sha256=belief_hashes[action]))
        print('checkpoint', action, 'patches', len(patches), 'marked', checkpoints[-1]['qualified_associated_marked'], flush=True)
    last_action = None
    verified = []
    for row in rows:
        if row['request']['method'] == 'observe':
            last_action = row['request']['params']['action_id']
        selection = row.get('response', {}).get('result', {}).get('selection')
        if selection and selection['candidate_audit']['remaining_budget'] == config['total_budget'] - last_action:
            assert belief_hashes[last_action] == selection['candidate_audit']['belief_sha256']
            verified.append(last_action)
    result = dict(name=proposal['name'], status='offline_review_prototype_not_deployed',
        proposal_sha256=sha(HERE / 'proposal.json'), report=str(REPORT), checkpoints=checkpoints,
        cache_audit=cache_audit, association_events=association.events,
        verification=dict(all_cached_masks_and_semantics_reproduced=True, native_quality_sampling_matches=True,
                          category_swap_does_not_change_tracks_membership_or_patch_geometry=True,
                          all_quality_points_preserved_exactly_once=True, native_mapping_hash_checks=sorted(set(verified))),
        source_sha256={name: sha(WORK / name) for name in source_files},
        prototype_sha256={name: sha(HERE / name) for name in ('association.py', 'run_review.py')},
        limits=proposal['not_claimed'], deployment=False, original_method_equivalence=False)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
