#!/usr/bin/env python3
"""Independent raw-prefix reconstruction of frozen V6 support descriptors.

Only the mapper is reused from the verified source archive. Morphology,
components, camera axes, frustum projection, angular aggregation and feature
mixing below do not call response_features/observed_clusters. No simulator,
future branch frame, outcome, GT reference, model fitting or navigation runs.
"""
import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


CHANNELS = ('G', 'O', 'S', 'X', 'M', 'N')
NAMES = ('radar_area_rate', 'camera_area_rate', 'quality_rate', 'inverse_cost',
         'support_angular_exposure_rate', 'support_angular_novelty_rate',
         'support_proximity_rate', 'support_opposite_face_rate',
         'class_angular_exposure_rate', 'class_angular_novelty_rate',
         'class_proximity_rate', 'class_opposite_face_rate')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def close_and_components(occupancy):
    """Independent zero-border 3x3 dilation/erosion, raster-ordered 4-CC BFS."""
    import numpy as np
    height, width = occupancy.shape
    padded = np.pad(occupancy, 1, constant_values=False)
    dilated = np.zeros_like(occupancy)
    for dy in range(3):
        for dx in range(3):
            dilated |= padded[dy:dy + height, dx:dx + width]
    padded = np.pad(dilated, 1, constant_values=False)
    closed = np.ones_like(occupancy)
    for dy in range(3):
        for dx in range(3):
            closed &= padded[dy:dy + height, dx:dx + width]
    ids = np.zeros(occupancy.shape, dtype=int)
    current = 0
    for row, col in np.argwhere(closed):
        if ids[row, col]:
            continue
        current += 1
        ids[row, col] = current
        pending = deque([(int(row), int(col))])
        while pending:
            r, c = pending.popleft()
            for nr, nc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)):
                if 0 <= nr < height and 0 <= nc < width and closed[nr, nc] and not ids[nr, nc]:
                    ids[nr, nc] = current
                    pending.append((nr, nc))
    return ids


def independent_clusters(quality, shape, resolution):
    import numpy as np
    assert resolution == .2, 'Frozen descriptor contract is a 0.2 m grid.'
    points = quality['point']
    xy = np.floor(points[:, :2] / resolution).astype(int)
    cells = np.column_stack((shape[0] - 1 - xy[:, 1], xy[:, 0]))
    inside = np.all((cells >= 0) & (cells < np.asarray(shape)), axis=1)
    occupied = np.zeros(shape, dtype=bool)
    occupied[tuple(cells[inside].T)] = True
    components = close_and_components(occupied)
    point_groups = np.zeros(len(points), dtype=int)
    point_groups[inside] = components[tuple(cells[inside].T)]
    accepted, rejected = [], []
    for group_id in np.unique(point_groups):
        if group_id == 0:
            continue
        indices = np.nonzero(point_groups == group_id)[0]
        limits = np.percentile(points[indices], [5, 95], axis=0)
        span = limits[1] - limits[0]
        row = {'group_id': int(group_id), 'points': len(indices), 'span_m': span.tolist()}
        if len(indices) < 8 or np.max(span[:2]) > 2.2 or span[2] < .25:
            rejected.append(row)
            continue
        accepted.append({'indices': indices, **row})
    return accepted, {'inside_points': int(inside.sum()), 'outside_points': int((~inside).sum()),
                      'closing_dropped_points': int(np.count_nonzero(inside & (point_groups == 0))),
                      'rejected_components': rejected}


def independent_pose(state, config, grid_height):
    import numpy as np
    row, col, heading = map(int, state)
    forward = np.array(((0., 1., 0.), (1., 0., 0.), (0., -1., 0.), (-1., 0., 0.))[heading])
    right = np.array(((1., 0., 0.), (0., -1., 0.), (-1., 0., 0.), (0., 1., 0.))[heading])
    result = np.eye(4)
    result[:3, 0], result[:3, 1], result[:3, 2] = right, (0., 0., -1.), forward
    result[:3, 3] = ((col + .5) * config.resolution_m,
                      (grid_height - row - .5) * config.resolution_m, config.camera_height_m)
    return result


def sector_of(vectors):
    import numpy as np
    angles = np.arctan2(vectors[..., 1], vectors[..., 0])
    return np.floor((angles + np.pi) * (4. / np.pi)).astype(int) % 8


def independent_descriptors(quality, clusters, routes, config, shape):
    import numpy as np
    all_states = sorted({tuple(s) for route in routes for s in route['states'][1:]})
    results = np.empty((len(routes), len(clusters), 4), float)
    diagnostics = []
    for object_id, cluster in enumerate(clusters):
        ix = cluster['indices']
        full = quality['point'][ix]
        center = np.median(full, axis=0)
        mean_normal = np.mean(quality['normal'][ix], axis=0)
        normal_length = np.linalg.norm(mean_normal)
        direction = mean_normal / max(normal_length, 1e-12)
        bits = quality['bits'][ix].astype(int)
        fractions = np.array([np.count_nonzero(bits & (2 ** b)) / len(bits) for b in range(8)])
        support = len(ix) * .0225
        selected = np.linspace(0, len(full) - 1, min(len(full), 128), dtype=int)
        cloud = full[selected]
        local_cache = {}
        mismatched_sectors = visible_points = 0
        min_depth, max_depth, min_distance, max_distance = [], [], [], []
        exposure_values, opposite_values = [], []
        pixel_focal = config.width_px / (2 * np.tan(np.deg2rad(config.fov_deg) / 2))
        for state in all_states:
            pose = independent_pose(state, config, shape[0])
            offset = cloud - pose[:3, 3]
            depth = offset @ pose[:3, 2]
            # Pixel-edge bounds (-.5, W-.5) are algebraically the frozen FOV.
            safe_depth = np.where(depth == 0, 1., depth)
            u = (offset @ pose[:3, 0]) * pixel_focal / safe_depth + (config.width_px - 1) / 2
            v = (offset @ pose[:3, 1]) * pixel_focal / safe_depth + (config.height_px - 1) / 2
            included = ((depth > .2) & (depth < config.max_depth_m) &
                        (u > -.5) & (u < config.width_px - .5) &
                        (v > -.5) & (v < config.height_px - .5))
            exposure = np.count_nonzero(included) / len(cloud)
            view_delta = pose[:3, 3] - center
            distance = float(np.sqrt(np.sum(view_delta ** 2)))
            sector = int(sector_of(view_delta))
            proximity = exposure / (1 + distance ** 2)
            opposite = exposure * max(0., float(np.dot(view_delta, direction))) / max(distance, .2)
            local_cache[state] = (sector, exposure, proximity, opposite)
            point_sectors = sector_of(pose[:3, 3] - cloud)
            mismatched_sectors += int(np.count_nonzero(included & (point_sectors != sector)))
            visible_points += int(included.sum())
            min_distance.append(distance); max_distance.append(distance)
            if included.any():
                min_depth.append(float(depth[included].min())); max_depth.append(float(depth[included].max()))
            exposure_values.append(exposure); opposite_values.append(opposite)
        for route_index, route in enumerate(routes):
            cost = route['cost']
            assert cost == len(route['states']) - 1 and 0 < cost <= 48
            per_sector = np.zeros(8)
            proximity = opposite = 0.
            # Set deduplication independently proves repeated states do not add reward.
            for state in set(tuple(s) for s in route['states'][1:]):
                b, ex, near, back = local_cache[state]
                per_sector[b] = max(per_sector[b], ex)
                proximity, opposite = max(proximity, near), max(opposite, back)
            values = (np.sum(per_sector), np.sum(per_sector * (1 - fractions)), proximity, opposite)
            results[route_index, object_id] = np.asarray(values) * support / cost
            # Bounds refer to voxel support, never unique physical surface area.
            bound = support / cost
            value = results[route_index, object_id]
            assert np.all(value >= -1e-12) and value[0] <= 8 * bound + 1e-12
            assert value[1] <= value[0] + 1e-12 and np.all(value[2:] <= bound + 1e-12)
        diagnostics.append({'object_index': object_id, 'quality_points': len(ix),
            'support_proxy_m2': float(support), 'FOV_sample_points': len(cloud),
            'center_m': center.tolist(), 'mean_normal_length_before_normalization': float(normal_length),
            'mean_normal_direction': direction.tolist(), 'history_sector_fractions': fractions.tolist(),
            'candidate_center_distance_m': [min(min_distance), max(max_distance)],
            'included_axial_depth_m': [min(min_depth), max(max_depth)] if min_depth else None,
            'exposure_range': [min(exposure_values), max(exposure_values)],
            'opposite_range': [min(opposite_values), max(opposite_values)],
            'visible_point_pose_pairs': visible_points,
            'point_azimuth_differs_from_cluster_center_pairs': mismatched_sectors,
            'point_vs_center_sector_mismatch_fraction': mismatched_sectors / max(1, visible_points)})
    return results, diagnostics


def mix_features(descriptors, clusters, quality, routes, prediction, config):
    """Independently derive label priors; use archived geometric fit errors only."""
    import numpy as np
    probabilities = {name: [] for name in CHANNELS if name != 'N'}
    supported = []
    for i, cluster in enumerate(clusters):
        labels = quality['label'][cluster['indices']]
        known = labels[(labels == 2) | (labels == 3)]
        supported.append(bool(len(known)))
        base_object = prediction['objects']['G'][i]
        assert len(labels) == base_object['observed_points']
        counts = {str(int(k)): int(np.count_nonzero(labels == k)) for k in np.unique(labels)}
        assert counts == base_object['label_counts']
        assert bool(len(known)) == base_object['marker_supported']
        errors = base_object['fit_errors_m2']
        likelihood = (errors[0] - errors[1]) / .0072
        for channel in probabilities:
            prior = .5
            if channel in ('S', 'X') and len(known):
                fraction = np.count_nonzero(known == (3 if channel == 'S' else 2)) / len(known)
                prior = .1 + .8 * fraction
            posterior = 1 / (1 + np.exp(-np.clip(np.log(prior) - np.log(1 - prior) + likelihood, -6, 6)))
            saved = prediction['objects'][channel][i]
            np.testing.assert_allclose(prior, saved['shelf_prior_probability'], rtol=0, atol=1e-14)
            np.testing.assert_allclose(posterior, saved['shelf_probability'], rtol=0, atol=1e-14)
            probabilities[channel].append(float(posterior))
    candidates = {row['candidate_id']: row for row in prediction['candidates']}
    matrices = {name: np.zeros((len(routes), 12)) for name in CHANNELS}
    marker_domain = config.appearance == 'marked' and config.semantic_source == 'rgb_marker'
    for route_index, route in enumerate(routes):
        saved = candidates[route['candidate_id']]['scores']['G']
        base = np.array([saved['radar_coverage_m2'], saved['camera_coverage_m2'],
                         saved['observed_quality_gain_per_point'], 1.]) / route['cost']
        for name in CHANNELS:
            matrices[name][route_index, :4] = base
            if name == 'N':
                continue
            for i, f in enumerate(descriptors[route_index]):
                weight = .25 if name in ('O', 'S', 'X') and marker_domain and any(supported) and not supported[i] else 1.
                probability = probabilities['G' if name in ('G', 'O') else name][i]
                matrices[name][route_index, 4:8] += weight * f
                matrices[name][route_index, 8:] += weight * (2 * probability - 1) * f
    return matrices, probabilities


def micro_checks():
    import numpy as np
    from types import SimpleNamespace
    c = SimpleNamespace(resolution_m=.2, camera_height_m=.8)
    expected_sectors = [0, 1, 2, 3, 4, 5, 6, 7]
    az = -np.pi + (np.arange(8) + .5) * np.pi / 4
    np.testing.assert_array_equal(sector_of(np.column_stack((np.cos(az), np.sin(az)))), expected_sectors)
    for heading, forward in enumerate(((0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0))):
        pose = independent_pose((8, 4, heading), c, 30)
        np.testing.assert_allclose(pose[:3, 3], [.9, 4.3, .8], atol=1e-14)
        np.testing.assert_array_equal(pose[:3, 2], forward)
        np.testing.assert_array_equal(pose[:3, :3].T @ pose[:3, :3], np.eye(3))
        assert np.linalg.det(pose[:3, :3]) == 1.
        point = pose[:3, 3] + np.asarray(forward)
        np.testing.assert_allclose((point - pose[:3, 3]) @ pose[:3, :3], [0, 0, 1], rtol=0, atol=1e-14)
    occupancy = np.zeros((8, 8), bool); occupancy[2:5, 2:5] = True; occupancy[3, 3] = False
    labels = close_and_components(occupancy)
    assert np.count_nonzero(labels) == 9 and labels.max() == 1 and labels[3, 3] == 1
    return {'cardinal_camera_axes': 4, 'angular_sector_centers': 8,
            'independent_closing_hole_fill': True, 'orthonormality_and_handedness': True}


def worker(args):
    sys.path.insert(0, str(args.snapshot))
    import numpy as np
    import open3d as o3d
    from env.virtual3d_inspection_v4 import InspectionConfigV4
    from nso.semantic_completion_v3 import SemanticHistoryMapperV3
    from nso.counterfactual_view_scoring import _mapper_snapshot
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    import nso.semantic_completion_v3 as mapper_module
    assert Path(mapper_module.__file__).resolve().is_relative_to(args.snapshot)
    assert 'nso.semantic_view_response_v6' not in sys.modules
    consumed = {}
    manifests = {folder: read(folder / 'artifact_hashes.json') for folder in (args.source, args.supplement)}
    for folder in (args.source, args.supplement):
        assert read(folder / 'metadata.json')['status'] == 'complete'
        assert read(folder / 'verification.json')['status'] == 'passed_full'
        for filename in ('artifact_hashes.json', 'metadata.json', 'verification.json'):
            consumed[str(folder / filename)] = sha(folder / filename)

    def checked(path, origin=None):
        actual = sha(path)
        if origin:
            assert manifests[origin][str(path.relative_to(origin))] == actual, path
        consumed[str(path)] = actual
        return path

    # Feature archives are covered by the transfer's pre-outcome prediction seal.
    seal = read(checked(args.transfer / 'prediction_seal.json'))
    for name, expected in seal.items():
        assert sha(checked(args.transfer / name)) == expected, name
    assert read(checked(args.transfer / 'completion.json'))['status'] == 'complete'
    summaries = []
    total_raw_frames = total_raw_scans = scalar_count = descriptor_count = 0
    worst_feature = worst_descriptor = 0.
    histories = read(checked(args.source / 'summary.json', args.source))['histories']
    for h in histories:
        started = time.monotonic()
        assert h['seed'] in (751, 752)
        fixture = args.source / h['fixture']; history_name = f"history_{h['history_step']:04d}"
        folder, extra = fixture / history_name, args.supplement / h['fixture'] / history_name
        old_routes = read(checked(folder / 'candidates.json', args.source))
        extra_routes = read(checked(extra / 'candidates.json', args.supplement))
        routes = old_routes + extra_routes
        assert [r['candidate_id'] for r in routes] == list(range(len(routes)))
        assert len({tuple(r['pose']) for r in routes}) == len(routes)
        pred = read(checked(folder / 'predictions.json', args.source))
        ep = read(checked(extra / 'predictions.json', args.supplement))
        assert pred['objects'] == ep['objects']
        assert pred['invariants']['mapper_hashes'] == ep['invariants']['mapper_hashes']
        pred['candidates'] += ep['candidates']
        c = InspectionConfigV4(**read(checked(fixture / 'fixture.json', args.source))['environment'])
        shape = (round(c.height_m / c.resolution_m), round(c.width_m / c.resolution_m))
        rows = read(checked(fixture / 'prefix/records.json', args.source))[:h['history_step'] + 1]
        frames = [RGBDFrame.load(checked(fixture / 'prefix/frames' / f'{i:04d}.npz', args.source)) for i in range(len(rows))]
        scans = [PlanarScan.load(checked(fixture / 'prefix/scans' / f'{i:04d}.npz', args.source)) for i in range(len(rows))]
        for frame, row in zip(frames, rows):
            expected_pose = independent_pose((*row['position'], row['heading']), c, shape[0])
            np.testing.assert_array_equal(expected_pose, frame.world_from_camera)
            assert c.appearance == 'marked' and c.semantic_source == 'rgb_marker'
            decoded = np.zeros(frame.semantic.shape, np.uint8)
            # RGB-only rewrite of the frozen artificial industrial marker code.
            for label_id, color in ((2, (40, 100, 220)), (3, (220, 60, 40))):
                decoded[np.all(np.abs(frame.color_rgb.astype(int) - np.asarray(color)) <= 8, axis=-1)] = label_id
            np.testing.assert_array_equal(decoded, frame.semantic)
        total_raw_frames += len(frames); total_raw_scans += len(scans)
        qualities, clusters_by_condition, condition_stats, snapshot_hashes = {}, {}, {}, {}
        aligned_keys = None
        for condition in ('aligned', 'shuffled', 'absent'):
            mapper = SemanticHistoryMapperV3(shape, c, c.truncation_m)
            for original, scan, row in zip(frames, scans, rows):
                labels = original.semantic.copy()
                if condition == 'shuffled':
                    labels = np.where(labels == 2, 3, np.where(labels == 3, 2, labels)).astype(labels.dtype)
                elif condition == 'absent':
                    labels.fill(0)
                frame = replace(original, semantic=labels)
                mapper.update(frame, scan)
                if row['collision']:
                    dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[row['heading']]
                    r, col = row['position'][0] + dr, row['position'][1] + dc
                    if 0 <= r < shape[0] and 0 <= col < shape[1]:
                        mapper.belief[r, col] = 1
            snapshot = _mapper_snapshot(mapper)
            assert snapshot['hashes'] == pred['invariants']['mapper_hashes'][condition]
            snapshot_hashes[condition] = snapshot['hashes']
            quality = mapper.quality_evidence(max_points=10000)
            assert quality is not None
            qualities[condition] = {k: value.copy() for k, value in quality.items()}
            clusters, stats = independent_clusters(quality, shape, c.resolution_m)
            clusters_by_condition[condition] = clusters
            keys = [k for k, value in mapper.quality.items() if .12 < value['point'][2] < 1.8]
            if len(keys) > 10000:
                keys = [keys[i] for i in np.linspace(0, len(keys) - 1, 10000, dtype=int)]
            assert len(keys) == len(quality['point']) and len(set(keys)) == len(keys)
            if aligned_keys is None:
                aligned_keys = keys
            else:
                assert aligned_keys == keys
                for key in quality:
                    if key != 'label':
                        np.testing.assert_array_equal(quality[key], qualities['aligned'][key])
                assert [r['indices'].tolist() for r in clusters] == [r['indices'].tolist() for r in clusters_by_condition['aligned']]
            stats.update(evidence_keys=len(keys), accepted_clusters=len(clusters),
                         repeated_observation_count_max=int(quality['n'].max()),
                         repeated_observations_total=int(quality['n'].sum()),
                         revoxelized_averaged_point_unique_keys=len(np.unique(np.floor(quality['point'] / .15).astype(int), axis=0)))
            condition_stats[condition] = stats
            del mapper, snapshot
        a, x, m = (qualities[k]['label'] for k in ('aligned', 'shuffled', 'absent'))
        np.testing.assert_array_equal(x, np.where(a == 2, 3, np.where(a == 3, 2, a)))
        assert not np.any(m)
        rebuilt = {}
        descriptor_stats = None
        for condition in qualities:
            rebuilt[condition], stats = independent_descriptors(qualities[condition], clusters_by_condition[condition], routes, c, shape)
            if condition == 'aligned':
                descriptor_stats = stats
            else:
                np.testing.assert_array_equal(rebuilt[condition], rebuilt['aligned'])
        descriptors = rebuilt['aligned']
        features, probabilities = mix_features(descriptors, clusters_by_condition['aligned'], qualities['aligned'], routes, pred, c)
        old_dir, union_dir = args.probe / h['fixture'] / history_name, args.transfer / h['fixture'] / history_name
        union_audit = read(checked(union_dir / 'feature_audit.json'))
        old_audit = read(checked(old_dir / 'feature_audit.json'))
        assert union_audit['feature_names'] == list(NAMES) == old_audit['feature_names']
        np.testing.assert_allclose(descriptors, union_audit['object_descriptors'], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(descriptors[:12], old_audit['object_descriptors'], rtol=1e-12, atol=1e-12)
        descriptor_error = float(np.max(np.abs(descriptors - union_audit['object_descriptors'])))
        worst_descriptor = max(worst_descriptor, descriptor_error); descriptor_count += descriptors.size
        errors = {}
        with np.load(checked(union_dir / 'features.npz'), allow_pickle=False) as saved, np.load(checked(old_dir / 'features.npz'), allow_pickle=False) as old:
            for name in CHANNELS:
                np.testing.assert_array_equal(saved[name][:12], old[name])
                np.testing.assert_allclose(features[name], saved[name], rtol=1e-12, atol=1e-12)
                errors[name] = float(np.max(np.abs(features[name] - saved[name])))
                worst_feature = max(worst_feature, errors[name]); scalar_count += features[name].size
                assert np.isfinite(features[name]).all()
                np.testing.assert_array_equal(features[name][:, :4], features['G'][:, :4])
        np.testing.assert_array_equal(features['G'], features['M'])
        np.testing.assert_array_equal(features['S'][:, :8], features['X'][:, :8])
        np.testing.assert_array_equal(features['S'][:, :8], features['O'][:, :8])
        assert not features['N'][:, 4:].any()
        label_changes = {f'{one}_to_{two}': np.flatnonzero(np.any(np.abs(features[one] - features[two]) > 1e-12, axis=0)).tolist()
                         for one, two in (('S', 'X'), ('S', 'M'), ('G', 'O'), ('O', 'S'))}
        target = args.output / h['fixture'] / history_name; target.mkdir(parents=True)
        np.savez_compressed(target / 'independent_features.npz', **features, object_descriptors=descriptors)
        state_counts = [{'candidate_id': r['candidate_id'], 'paid_actions': r['cost'],
                         'unique_successor_states': len(set(map(tuple, r['states'][1:]))),
                         'repeated_states_not_added': r['cost'] - len(set(map(tuple, r['states'][1:])))} for r in routes]
        record = {'fixture': h['fixture'], 'seed': h['seed'], 'depth_sigma_m': h['depth_sigma_m'], 'history_step': h['history_step'],
                  'prefix_frames': len(frames), 'candidates': len(routes), 'original_candidates': 12, 'extra_candidates': len(extra_routes),
                  'mapper_snapshots_match_frozen_all_conditions': True, 'RGB_marker_labels_independently_redecoded': True,
                  'raw_geometry_equal_after_label_intervention': True, 'descriptors_equal_after_label_intervention': True,
                  'original_72_feature_subsets_exactly_preserved': True, 'condition_stats': condition_stats,
                  'descriptor_physical_diagnostics': descriptor_stats, 'descriptor_max_absolute_error': descriptor_error,
                  'feature_max_absolute_errors': errors, 'label_intervention_changed_columns_zero_based': label_changes,
                  'inherited_geometry_likelihood_posterior_recomputed': probabilities, 'route_deduplication': state_counts,
                  'snapshot_hashes': snapshot_hashes, 'elapsed_s': time.monotonic() - started}
        write(target / 'review.json', record); summaries.append(record)
        print('reviewed', h['fixture'], h['history_step'], len(routes), 'routes', round(record['elapsed_s'], 2), 's', flush=True)
    assert len(summaries) == 6 and sum(r['candidates'] for r in summaries) == 88
    assert 'nso.semantic_view_response_v6' not in sys.modules
    for path, expected in consumed.items():
        assert sha(path) == expected, path
    result = {'status': 'passed_bounded_raw_descriptor_review', 'histories': 6, 'candidate_routes': 88,
              'old_candidates': 72, 'additional_candidates': 16, 'label_conditions_remapped': 18,
              'prefix_frames_with_history_reuse': total_raw_frames, 'prefix_scans_with_history_reuse': total_raw_scans,
              'unique_raw_frame_files': sum('/prefix/frames/' in p for p in consumed),
              'unique_raw_scan_files': sum('/prefix/scans/' in p for p in consumed),
              'independent_descriptor_scalars': descriptor_count, 'independent_feature_scalars': scalar_count,
              'max_descriptor_absolute_error': worst_descriptor, 'max_feature_absolute_error': worst_feature,
              'micro_checks': micro_checks(), 'libraries': {'numpy': np.__version__, 'open3d': o3d.__version__},
              'new_navigation_branches': 0, 'GT_or_future_outcomes_read': False, 'model_fitting_performed': False,
              'raw_mapper_independently_implemented': False, 'production_response_module_imported': False,
              'source_archive': str(args.transfer / 'sources.zip'), 'source_archive_sha256': sha(args.transfer / 'sources.zip'),
              'input_files_checked': len(consumed), 'history_reviews': [str(Path(r['fixture']) / f"history_{r['history_step']:04d}" / 'review.json') for r in summaries],
              'limits': [
                  'The frozen mapper and snapshot verifier are reused, not independently reimplemented; matching their output proves replay consistency only.',
                  'The first four route/quality features reuse archived prefix-only G scores. Their normalization is checked, their raycasting is not independently recomputed here.',
                  'Geometric shape fit errors are archived V3 inputs; label priors and posterior arithmetic are independently recomputed, not the hypothesis likelihood geometry.',
                  'Support counts persistent quality keys, not repeated observations, but is a voxel support proxy, not unique exterior surface area.',
                  'FOV descriptors ignore occlusion, use at most 128 support points, and axial depth rather than Euclidean range.',
                  'History bits refer to camera-to-point azimuth; candidate sectors refer to the cluster median, an approximation quantified in the reports.',
                  'Opposite-face is a mean-normal hemisphere proxy. The mean can cancel; no actual back-face visibility or new reconstruction is implied.',
                  'S-to-M also removes marker objectness weighting and can change columns 4:8; S-to-X changes only conditional columns 8:12.',
                  'All six histories are already viewed development data; passing this implementation review proves no semantic performance advantage or full ANS validation.']}
    write(args.output / 'summary.json', result); write(args.output / 'input_hashes.json', consumed)
    write_report(args.output, result, summaries)


def write_report(output, result, rows):
    all_objects = [d for row in rows for d in row['descriptor_physical_diagnostics']]
    mismatches = sum(d['point_azimuth_differs_from_cluster_center_pairs'] for d in all_objects)
    pairs = sum(d['visible_point_pose_pairs'] for d in all_objects)
    text = ['# V6 原始传感器描述量独立复核', '',
            '结论：6 个旧开发历史、原 72 与新增 16 共 88 候选的落盘描述量和六通道特征均通过独立复算。未发现坐标轴、朝向符号或标签串入共同几何列的实现错误；这不是语义性能通过证据。', '',
            f"共复用 {result['unique_raw_frame_files']} 个唯一 RGB-D 文件及同量雷达文件，按各历史读取 {result['prefix_frames_with_history_reuse']} 帧，分别重融合 aligned/shuffled/absent 共 18 次。所有冻结 mapper 指纹吻合；从 RGB 独立读取的工业颜色标识逐像素吻合。没有读取未来分支观测、outcome 或 GT。", '',
            f"独立重算 {result['independent_descriptor_scalars']} 个对象描述标量、{result['independent_feature_scalars']} 个六通道特征标量；最大绝对差分别 {result['max_descriptor_absolute_error']:.3g}、{result['max_feature_absolute_error']:.3g}。原 72 特征在旧池和联合池中逐数组完全相同。", '',
            '|历史|候选|实测簇|描述最大差|特征最大差|', '|---|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"|{row['seed']}/{row['depth_sigma_m']}/{row['history_step']}|{row['candidates']}|{row['condition_stats']['aligned']['accepted_clusters']}|{row['descriptor_max_absolute_error']:.3g}|{max(row['feature_max_absolute_errors'].values()):.3g}|")
    text += ['', '独立实现范围：以冻结 mapper 的 quality_evidence 为边界，另写零边界形态闭运算、四邻接连通分量、分位数过滤、网格坐标、四朝向相机变换、像素边界 FOV、八方向最大值聚合及条件列组合。没有导入或调用 V6 response_features/observed_clusters。原 mapper、快照核验器属于复用组件；前四列采用原 G 路线评分，仅独立检查成本归一化；V3 几何拟合误差也作为已有输入，只重算标签先验和后验公式。', '',
             '标签干预：2/3 交换不改变 RGB-D、雷达、非语义地图、聚类、对象描述及前八列，只改变最后四个条件列。O 与 S 的前八列相同；M 完整等于 G。S→M 还移除对象性降权，因此可能同时改变第 5—8 列，不能错误地要求缺失语义也只改类别四列。', '',
             '物理边界与待改的建模近似：', '',
             '- 支撑按持久化 15 cm 证据键去重，n 次观测没有乘入支撑；同一路线重复状态采用最大值而非累加。角度暴露上界为 8×支撑/动作，角度新颖性不超过暴露，邻近和反面项不超过支撑/动作。数值界全部通过。支撑仍不是唯一实体外表面积。',
             '- FOV 使用完整像素边缘和轴向深度上限，未进行遮挡判断；128 点采样及每方向最大暴露不能表达真实新面并集。这是既有描述的限制，并未从命名推断成真实可见面积。',
             f'- 历史方向位按每个实测点定义，候选方向却按簇中位中心定义。候选视野内点—位姿对中 {mismatches}/{pairs}（{100*mismatches/max(1,pairs):.2f}%）的点方向扇区与簇中心不同。因此方向新颖性存在参考点近似；这是传感器支撑诊断，不能直接换算收益误差。',
             '- 反面项是候选位于平均内法向半空间的投影，不证明能看到物体背面。平均法向会相消；详细报告保留归一化前模长。V3 shelf_probability 仍为未校准的形状后验，独立公式复算不使其变成已学习、已校准的响应概率。', '',
             '本次仅补全原始描述提取的实现证据，没有新导航、重新拟合、改门槛、打开 753/754 或独立确认集；不能据此声称完整 ANS 或四模块协同优势。', '']
    (output / 'RAW_GEOMETRY_REVIEW.md').write_text('\n'.join(text))


def run(args):
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    started = time.monotonic()
    metadata = {'status': 'running', 'script_sha256': sha(__file__), 'workers': 1,
                'thread_limits': {'OMP_NUM_THREADS': 1, 'OPENBLAS_NUM_THREADS': 1, 'MKL_NUM_THREADS': 1},
                'runtime_estimate_minutes': [1, 2], 'maximum_output_bytes': 5000000,
                'new_navigation_branches': 0, 'raw_or_frozen_artifacts_modified': False}
    write(args.output / 'metadata.json', metadata)
    try:
        with tempfile.TemporaryDirectory(prefix='nso_raw_response_review_') as name:
            snapshot = Path(name)
            protocol = read(args.transfer / 'completion.json')
            with zipfile.ZipFile(args.transfer / 'sources.zip') as archive:
                for member in archive.infolist():
                    target = (snapshot / member.filename).resolve()
                    assert target.is_relative_to(snapshot)
                    if not member.is_dir():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(member))
            for filename, expected in protocol['source_sha256'].items():
                assert sha(snapshot / filename) == expected, filename
            env = os.environ | {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                                'NUMEXPR_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--snapshot', str(snapshot)]
            for field in ('source', 'supplement', 'probe', 'transfer', 'output'):
                command += ['--' + field, str(getattr(args, field))]
            subprocess.run(command, cwd=snapshot, env=env, check=True)
        assert sha(__file__) == metadata['script_sha256']
        metadata.update(status='complete', elapsed_s=time.monotonic() - started)
    except Exception as error:
        metadata.update(status='failed', error=repr(error), elapsed_s=time.monotonic() - started)
        write(args.output / 'metadata.json', metadata)
        raise
    write(args.output / 'metadata.json', metadata)
    write(args.output / 'artifact_hashes.json', {str(p.relative_to(args.output)): sha(p) for p in args.output.rglob('*') if p.is_file()})
    total = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
    assert total <= 5000000, total
    print(json.dumps({'status': 'complete', 'elapsed_s': metadata['elapsed_s'], 'output_bytes': total}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('source', 'supplement', 'probe', 'transfer', 'output'):
        parser.add_argument('--' + field, required=True, type=lambda s: Path(s).resolve())
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    worker(args) if args.worker else run(args)
