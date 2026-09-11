#!/usr/bin/env python3
"""Independent T0 V7 feature mathematics using only sealed prefix sensors.

Reuses a frozen mapper and Open3D ray intersections on its measured mesh.
Does not instantiate a world/evaluator, read outcomes, execute candidate actions,
or call the production V7 response_features/observed_patches calculation.
"""
import argparse
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/review_response_v6_raw_geometry.py'
CHANNELS = ('G', 'O', 'S', 'X', 'M', 'N', 'G_capacity')


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper(path):
    spec = importlib.util.spec_from_file_location('independent_v6_geometry_math', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def matched_camera(quality_points, keyframes, tolerance):
    """Project with explicit dot products; retain first strictly greatest count."""
    import numpy as np
    best_count, best_frame, counts = -1, None, []
    for frame in keyframes:
        offset = quality_points - frame.world_from_camera[:3, 3]
        z = offset @ frame.world_from_camera[:3, 2]
        u = np.rint((offset @ frame.world_from_camera[:3, 0]) / np.maximum(z, .01) * frame.intrinsic[0, 0] + frame.intrinsic[0, 2]).astype(int)
        v = np.rint((offset @ frame.world_from_camera[:3, 1]) / np.maximum(z, .01) * frame.intrinsic[1, 1] + frame.intrinsic[1, 2]).astype(int)
        mask = (z > .15) & (u >= 0) & (u < frame.depth_m.shape[1]) & (v >= 0) & (v < frame.depth_m.shape[0])
        ids = np.flatnonzero(mask)
        observed = frame.depth_m[v[ids], u[ids]]
        count = int(np.count_nonzero((observed > 0) & (np.abs(observed - z[ids]) <= tolerance)))
        counts.append(count)
        if count > best_count:
            best_count, best_frame = count, frame
    if best_frame is None: raise ValueError('no retained measured camera')
    return best_frame.world_from_camera[:3, 3].copy(), best_count, counts


def make_patches(mapper, feature_config, helper):
    import numpy as np
    q = mapper.quality_evidence(max_points=10000)
    if q is None: return [], []
    assert feature_config['min_support_points'] == 8 and feature_config['max_horizontal_extent_m'] == 2.2
    assert feature_config['min_vertical_extent_m'] == .25 and feature_config['quality_voxel_m'] == .15
    groups, _ = helper.independent_clusters(q, mapper.shape, mapper.config.resolution_m)
    patches, details = [], []
    for group in groups:
        ids = group['indices']; points = q['point'][ids]; center = np.median(points, axis=0)
        camera, count, camera_counts = matched_camera(points, mapper.keyframes, mapper.config.truncation_m)
        normal = np.mean(q['normal'][ids, :2], axis=0)
        original_length = float(np.linalg.norm(normal))
        fallback = original_length < feature_config['normal_fallback_threshold']
        reference_side = camera[:2] - center[:2]
        if fallback: normal = reference_side.copy()
        elif np.dot(normal, reference_side) < 0: normal = -normal
        length = np.linalg.norm(normal)
        normal = normal / length if length >= 1e-10 else np.array([1., 0.])
        categories = q['label'][ids]
        n2, n3 = int(np.count_nonzero(categories == 2)), int(np.count_nonzero(categories == 3))
        selected = np.linspace(0, len(ids) - 1, min(len(ids), feature_config['max_descriptor_points']), dtype=int)
        patch = {'group': group['group_id'], 'center': center, 'normal_out_xy': normal,
            'points': points[selected], 'bits': q['bits'][ids][selected],
            'support_points': len(ids), 'support_m2_proxy': len(ids) * feature_config['quality_voxel_m']**2,
            'marked_points': n2 + n3, 'class_vote': (n3 - n2) / (n2 + n3) if n2 + n3 else 0.,
            'normal_fallback': bool(fallback), 'normal_sign_depth_support': count,
            'orientation_available': count > 0, 'observed_span': np.asarray(group['span_m'])}
        patches.append(patch)
        details.append({'group': group['group_id'], 'camera_support_counts': camera_counts,
                        'selected_camera_xyz': camera.tolist(), 'normal_xy_length_before_orientation': original_length})
    return patches, details


def frustum(points, pose, c, minimum=.15, inclusive_max=True):
    import numpy as np
    offset = points - pose[:3, 3]
    depth = offset @ pose[:3, 2]
    focal = c.width_px / (2 * np.tan(np.deg2rad(c.fov_deg / 2)))
    denominator = np.where(depth == 0, 1., depth)
    u = (offset @ pose[:3, 0]) * focal / denominator + (c.width_px - 1) / 2
    v = (offset @ pose[:3, 1]) * focal / denominator + (c.height_px - 1) / 2
    valid = (depth > minimum) & ((depth <= c.max_depth_m) if inclusive_max else (depth < c.max_depth_m))
    return valid & (u > -.5) & (u < c.width_px - .5) & (v > -.5) & (v < c.height_px - .5)


def point_response(patch, pose, c, helper):
    import numpy as np
    displacement = pose[:3, 3] - patch['points']
    range_squared = np.sum(displacement**2, axis=1)
    horizontal_range = np.sqrt(np.sum(displacement[:, :2]**2, axis=1))
    alignment = np.clip((displacement[:, :2] * patch['normal_out_xy']).sum(axis=1) /
                        np.maximum(horizontal_range, 1e-10), -1., 1.)
    per_point = np.zeros((len(displacement), 4))
    if patch['orientation_available']:
        per_point[:, 0] = np.maximum(alignment, 0)
        per_point[:, 1] = np.sqrt(np.maximum(1 - alignment * alignment, 0))
        per_point[:, 2] = np.maximum(-alignment, 0)
    sectors = helper.sector_of(displacement)
    per_point[:, 3] = ((patch['bits'].astype(int) >> sectors) & 1) == 0
    attenuation = 1 / (1 + range_squared / c.max_depth_m**2)
    per_point *= (frustum(patch['points'], pose, c) * attenuation)[:, None]
    assert np.isfinite(per_point).all() and np.all((per_point >= 0) & (per_point <= 1 + 1e-12))
    return per_point


def independent_inflation(obstacles, radius):
    import numpy as np
    threshold = radius + np.sqrt(2) / 2
    padding = int(np.ceil(threshold))
    padded = np.pad(obstacles, padding, constant_values=True)
    result = np.zeros_like(obstacles)
    for dy in range(-padding, padding + 1):
        for dx in range(-padding, padding + 1):
            if dx * dx + dy * dy <= threshold * threshold:
                result |= padded[padding + dy:padding + dy + result.shape[0], padding + dx:padding + dx + result.shape[1]]
    return result


@lru_cache(maxsize=16)
def independent_ray_offsets(radius, fov, heading):
    """Exact rational grid boundary crossings, including both cells at corners."""
    import numpy as np
    count = max(3, math.ceil(math.radians(fov) * radius * 2) + 1)
    angles = np.linspace(heading * math.pi / 2 - math.radians(fov) / 2,
                         heading * math.pi / 2 + math.radians(fov) / 2, count)
    endpoints = sorted({(round(-radius * math.cos(a)), round(radius * math.sin(a))) for a in angles})
    lines = []
    for row, col in endpoints:
        nr, nc = abs(row), abs(col)
        sr, sc = (1 if row > 0 else -1 if row < 0 else 0), (1 if col > 0 else -1 if col < 0 else 0)
        r = c = ir = ic = 0; line = [(0, 0)]
        while ir < nr or ic < nc:
            yr = Fraction(2 * ir + 1, 2 * nr) if ir < nr else None
            xc = Fraction(2 * ic + 1, 2 * nc) if ic < nc else None
            if yr == xc:
                line.extend(((r + sr, c), (r, c + sc))); r += sr; c += sc; ir += 1; ic += 1
            elif xc is None or (yr is not None and yr < xc): r += sr; ir += 1
            else: c += sc; ic += 1
            line.append((r, c))
        lines.append(line)
    width = max(map(len, lines)); offsets = np.zeros((len(lines), width, 2), int); valid = np.zeros((len(lines), width), bool)
    for i, line in enumerate(lines): offsets[i, :len(line)] = line; valid[i, :len(line)] = True
    valid &= np.sum(offsets**2, axis=2) <= radius**2
    return offsets, valid


def grid_visibility(obstacles, state, radius, fov):
    import numpy as np
    offsets, valid = independent_ray_offsets(radius, fov, state[2])
    points = offsets + np.asarray(state[:2]); row, col = points[..., 0], points[..., 1]
    inside = valid & (row >= 0) & (row < obstacles.shape[0]) & (col >= 0) & (col < obstacles.shape[1])
    r, c = np.clip(row, 0, obstacles.shape[0] - 1), np.clip(col, 0, obstacles.shape[1] - 1)
    blocked = ~inside | obstacles[r, c]
    after_obstacle = np.zeros_like(blocked)
    after_obstacle[:, 1:] = np.maximum.accumulate(blocked, axis=1)[:, :-1]
    chosen = inside & ~after_obstacle
    result = np.zeros_like(obstacles); result[r[chosen], c[chosen]] = True
    return result


def quality_response(q, pose, c, ray, helper):
    """Independent arithmetic for the frozen common nonsemantic quality proxy."""
    import numpy as np
    import open3d as o3d
    if q is None: return np.zeros(0)
    delta = pose[:3, 3] - q['point']; distance = np.sqrt((delta**2).sum(axis=1))
    valid = frustum(q['point'], pose, c, minimum=.2, inclusive_max=False)
    ids = np.flatnonzero(valid)
    if ray is not None and len(ids):
        queries = np.concatenate((np.broadcast_to(pose[:3, 3], (len(ids), 3)), -delta[ids]), axis=1).astype(np.float32)
        t = ray.cast_rays(o3d.core.Tensor(queries), nthreads=1)['t_hit'].numpy()
        valid[ids] &= t >= 1 - .12 / np.maximum(distance[ids], .2)
    sector = helper.sector_of(delta)
    novel = ((q['bits'].astype(int) >> sector) & 1) == 0
    counts = np.asarray([int(value).bit_count() for value in q['bits']])
    completion = novel * np.maximum(2 - counts, 0) * .5
    incidence = np.abs(np.sum(q['normal'] * delta, axis=1)) / np.maximum(distance, .2)
    information = incidence**2 / np.maximum(distance**2, .25)
    gain = np.clip((information - q['information']) / np.maximum(q['information'], .1), 0, 2) * .5
    uncertainty = np.clip(.25 / np.sqrt(q['n']) + q['residual'] / .001 + q['normal_dispersion'] * .25, 0, 1)
    complexity = 1 + .5 * np.clip(q['normal_dispersion'], 0, 1)
    return .65 * (valid * completion * complexity) + .35 * (valid * gain * uncertainty * complexity)


def matrices(mapper, routes, feature_config, helper):
    import numpy as np
    import open3d as o3d
    c = mapper.config; patches, diagnostics = make_patches(mapper, feature_config, helper)
    obstacles = mapper.belief == 1
    radar_target = (mapper.belief == -1) & ~independent_inflation(obstacles, c.robot_radius_m / c.resolution_m)
    camera_target = ~mapper.camera_seen & ~obstacles
    q = mapper.quality_evidence(); mesh = mapper.mesh(); ray = None
    if len(mesh.triangles):
        ray = o3d.t.geometry.RaycastingScene(nthreads=1)
        ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    states = sorted({tuple(s) for route in routes for s in route['states'][1:]})
    cache = {}
    for state in states:
        pose = helper.independent_pose(state, c, mapper.shape[0])
        cache[state] = (grid_visibility(obstacles, state, int(c.max_depth_m / c.resolution_m), 360) & radar_target,
                        grid_visibility(obstacles, state, int(c.max_depth_m / c.resolution_m), c.fov_deg) & camera_target,
                        quality_response(q, pose, c, ray, helper), [point_response(p, pose, c, helper) for p in patches])
    values = {name: [] for name in CHANNELS}; descriptions = []
    for route in routes:
        length = route['cost']; assert length == len(route['states']) - 1 and 0 < length <= 48
        entries = [cache[s] for s in sorted(set(map(tuple, route['states'][1:])))]
        radar = np.logical_or.reduce([r[0] for r in entries]); camera = np.logical_or.reduce([r[1] for r in entries])
        quality = np.maximum.reduce([r[2] for r in entries])
        base = np.asarray([radar.sum() * c.resolution_m**2 / c.max_depth_m**2,
                           camera.sum() * c.resolution_m**2 / c.max_depth_m**2,
                           quality.sum() / max(1, len(quality)), 1.]) / length
        unscaled = np.array([np.maximum.reduce([r[3][i] for r in entries]).mean(axis=0) * p['support_m2_proxy']
                             for i, p in enumerate(patches)]).reshape(-1, 4)
        for p, row in zip(patches, unscaled):
            assert np.all(row >= 0) and np.all(row <= p['support_m2_proxy'] + 1e-12)
        d = unscaled / c.max_depth_m**2
        geom = d.sum(axis=0) / length
        weights = np.asarray([1. if p['marked_points'] else feature_config['unmarked_objectness'] for p in patches])
        votes = np.asarray([p['class_vote'] for p in patches])
        obj = (d * weights[:, None]).sum(axis=0) / length
        cond = (d * (weights * votes)[:, None]).sum(axis=0) / length
        quadratic = (unscaled**2).sum(axis=0) / (c.max_depth_m**4 * length)
        zeros = np.zeros(4)
        additions = {'G': (geom, zeros), 'O': (obj, zeros), 'S': (obj, cond), 'X': (obj, -cond),
                     'M': (geom, zeros), 'N': (zeros, zeros), 'G_capacity': (geom, quadratic)}
        for name in values: values[name].append(np.concatenate((base, *additions[name])))
        descriptions.append({'candidate_id': route['candidate_id'], 'cost': length,
                             'unscaled_patch_descriptors': unscaled.tolist(),
                             'unique_successor_states': len(entries), 'repeated_states': length - len(entries)})
    return {k: np.asarray(v) for k, v in values.items()}, patches, descriptions, diagnostics


def negative_checks(mapper, helper):
    """Independent expected values, then a black-box test of the frozen low-level function."""
    import numpy as np
    from nso.response_features_v7 import state_patch_descriptor
    c = mapper.config
    patch = {'points': np.array([[2.1, 2.1, .8]]), 'normal_out_xy': np.array([0., 1.]),
             'bits': np.array([1 << 6]), 'orientation_available': True}
    # Use explicit world translation/optical axes, independent of map dimensions.
    cases = [('front', (2.1, 3.1, .8), (0, -1, 0), [16/17, 0, 0, 0]),
             ('side', (3.1, 2.1, .8), (-1, 0, 0), [0, 16/17, 0, 16/17]),
             ('back', (2.1, 1.1, .8), (0, 1, 0), [0, 0, 16/17, 16/17])]
    for name, origin, forward, expected in cases:
        pose = np.eye(4); pose[:3, 3] = origin; pose[:3, 2] = forward
        pose[:3, 0] = np.cross(forward, (0., 0., 1.)); pose[:3, 1] = (0., 0., -1.)
        independent = point_response(patch, pose, c, helper)
        np.testing.assert_allclose(independent[0], expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(state_patch_descriptor(patch, pose, c), independent, atol=1e-12, rtol=0)
    dead_frames = [replace(f, depth_m=np.zeros_like(f.depth_m)) for f in mapper.keyframes]
    origin, count, _ = matched_camera(patch['points'], dead_frames, c.truncation_m)
    assert count == 0
    disabled = dict(patch, orientation_available=count > 0)
    own = point_response(disabled, pose, c, helper)
    np.testing.assert_array_equal(own[:, :3], np.zeros((1, 3))); assert own[0, 3] > 0
    np.testing.assert_allclose(state_patch_descriptor(disabled, pose, c), own, atol=1e-12, rtol=0)
    return {'front_side_back_analytic': True, 'zero_depth_direction_disabled_novelty_retained': True,
            'frozen_low_level_function_used_only_as_test_subject': True,
            'production_high_level_feature_function_called': False}


def worker(args):
    sys.path.insert(0, str(args.snapshot))
    import numpy as np
    from env.virtual3d_response_v7 import ResponseConfigV7
    from nso.semantic_completion_v3 import SemanticHistoryMapperV3
    from nso.counterfactual_view_scoring import _mapper_snapshot
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    helper = load_helper(args.helper)
    manifest = read(args.run / 'artifact_hashes.json'); consumed = {}
    def checked(path):
        rel = str(path.relative_to(args.run))
        assert 'candidate_' not in rel.split('/')[-2] if len(rel.split('/')) > 1 else True
        assert 'outcome' not in path.name and path.name not in ('summary.json', 'verification.json', 'reference.npz')
        value = sha(path); assert manifest[rel] == value, rel
        consumed[rel] = value; return path
    summaries, paired = [], []
    for family in ('storage_shelves', 'ventilation_baffles'):
        folder = args.run / family; fixture = read(checked(folder / 'fixture.json'))
        assert fixture['context']['context_id'] == 'T0' and fixture['context']['role'] == 'train'
        config = ResponseConfigV7(**fixture['config']); shape = (42, 42)
        rows = read(checked(folder / 'prefix/records.json')); assert len(rows) == 21
        frames = [RGBDFrame.load(checked(folder / 'prefix/frames' / f'{i:04d}.npz')) for i in range(21)]
        scans = [PlanarScan.load(checked(folder / 'prefix/scans' / f'{i:04d}.npz')) for i in range(21)]
        routes = read(checked(folder / 'candidates.json')); assert len(routes) == 6
        saved_audit = read(checked(folder / 'feature_audit.json'))
        legacy = read(checked(folder / 'legacy_predictions.json'))
        rebuilt_by_mode = {}; snapshots = {}; current_summary = None
        for mode in ('aligned', 'shuffled', 'absent'):
            mapper = SemanticHistoryMapperV3(shape, config, config.truncation_m)
            for frame, scan, row in zip(frames, scans, rows):
                labels = frame.semantic.copy()
                if mode == 'shuffled': labels = np.where(labels == 2, 3, np.where(labels == 3, 2, labels)).astype(labels.dtype)
                elif mode == 'absent': labels.fill(0)
                mapper.update(replace(frame, semantic=labels), scan)
                assert not row['collision'], 'T0 prefix expected no collision'
            snapshots[mode] = _mapper_snapshot(mapper)['hashes']
            assert snapshots[mode] == legacy['invariants']['mapper_hashes'][mode]
            calculated, patches, descriptions, diagnostics = matrices(mapper, routes, saved_audit['feature_config'], helper)
            rebuilt_by_mode[mode] = calculated
            if mode == 'aligned':
                patch_records = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in p.items()
                                  if k not in ('points', 'bits')} for p in patches]
                assert len(patch_records) == len(saved_audit['patches'])
                for actual, expected in zip(patch_records, saved_audit['patches']):
                    assert actual.keys() == expected.keys()
                    for name in actual: np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=1e-12)
                for actual, expected in zip(descriptions, saved_audit['routes']):
                    assert actual['candidate_id'] == expected['candidate_id'] and actual['cost'] == expected['cost']
                    np.testing.assert_allclose(actual['unscaled_patch_descriptors'], expected['unscaled_patch_descriptors'], atol=1e-12, rtol=1e-12)
                errors = {}
                with np.load(checked(folder / 'features.npz'), allow_pickle=False) as saved:
                    for name in CHANNELS:
                        np.testing.assert_allclose(calculated[name], saved[name], atol=1e-12, rtol=1e-12)
                        errors[name] = float(np.max(np.abs(calculated[name] - saved[name])))
                current_summary = {'family': family, 'candidate_count': 6, 'patches': patch_records,
                    'patch_diagnostics': diagnostics, 'routes': descriptions, 'max_feature_errors': errors,
                    'negative_checks': negative_checks(mapper, helper)}
                target = args.output / family; target.mkdir()
                np.savez_compressed(target / 'independent_features.npz', **calculated)
            del mapper
        aligned, swapped, absent = (rebuilt_by_mode[k] for k in ('aligned', 'shuffled', 'absent'))
        np.testing.assert_array_equal(aligned['X'], swapped['S'])
        np.testing.assert_array_equal(aligned['M'], absent['G']); np.testing.assert_array_equal(aligned['G'], aligned['M'])
        for mode in rebuilt_by_mode:
            for name in ('G', 'M', 'N', 'G_capacity'): np.testing.assert_array_equal(aligned[name], rebuilt_by_mode[mode][name])
        np.testing.assert_array_equal(aligned['S'][:, :8], aligned['X'][:, :8])
        np.testing.assert_array_equal(aligned['S'][:, :8], aligned['O'][:, :8])
        current_summary.update(mapper_snapshots=snapshots, label_swap_and_absent_exact=True,
            area_scale_m2=config.max_depth_m**2, G_capacity_formula='sum_p(D_unscaled[p,k]^2)/(R^4*paid_cost), not square of the rate')
        write(args.output / family / 'review.json', current_summary); summaries.append(current_summary); paired.append(aligned)
        print('independent mathematics passed', family, flush=True)
    for name in ('G', 'O', 'M', 'N', 'G_capacity'): np.testing.assert_array_equal(paired[0][name], paired[1][name])
    np.testing.assert_array_equal(paired[0]['S'][:, :8], paired[1]['S'][:, :8])
    np.testing.assert_array_equal(paired[0]['S'][:, 8:], -paired[1]['S'][:, 8:])
    for name, value in consumed.items(): assert sha(args.run / name) == value
    assert all(not ('outcome' in name or '/candidate_' in name) for name in consumed)
    worst = max(max(row['max_feature_errors'].values()) for row in summaries)
    summary = {'status': 'passed_independent_T0_feature_math_review', 'context_id': 'T0', 'histories': 2,
        'candidate_routes': 12, 'prefix_frames': 42, 'prefix_scans': 42, 'label_remaps': 6,
        'feature_scalars_compared': 12 * 12 * 7, 'max_feature_absolute_error': worst,
        'new_physical_routes': 0, 'future_frames_read': False, 'outcomes_read': False, 'GT_reference_read': False,
        'raw_mapper_independently_implemented': False, 'high_level_V7_feature_functions_called': False,
        'frozen_low_level_descriptor_black_box_tests': True, 'all_label_and_pair_invariants_passed': True,
        'common_route_features_independently_recomputed': True, 'retained_measured_mesh_raycast_backend_reused': 'Open3D',
        'source_archive_sha256': sha(args.run / 'sources.zip'), 'input_files_checked': len(consumed),
        'limits': ['The TSDF/quality mapper is reused from frozen sources; this is not independent SLAM.',
            'Common quality occlusion uses Open3D ray intersections on the reconstructed prefix mesh, never GT.',
            'Grid beam discretization and descriptor formulas are independently reimplemented, but reproduce the same declared approximations.',
            'Measured support and candidate frustum are not unique new exterior area or hidden visibility certificates.',
            'Zero-depth behavior is tested synthetically; both actual T0 patches have positive depth support.',
            'Only T0 is audited, not all T/C contexts; no fitting, response outcome or semantic efficacy is evaluated.']}
    write(args.output / 'summary.json', summary); write(args.output / 'input_hashes.json', consumed)
    (args.output / 'FEATURE_MATH_REVIEW.md').write_text(
        '# T0 V7 特征数学独立复核\n\n'
        f'两类设备、12 个封存候选、7 通道共 1008 个特征标量通过独立复算，最大绝对差 {worst:.3g}。'
        '仅读取初始加 20 动作的前缀；没有读取收益、未来分支帧或真值参考。\n\n'
        '独立实现包括：形态闭运算和聚类、逐关键帧深度支持及法向定向、逐点 FOV/距离/方向新颖性、'
        '二维栅格射线及并集、实测旧面质量公式、逐点全路径最大值、类别票与 12 列聚合。'
        'mapper 仍复用冻结实现，已测网格的射线求交仍使用 Open3D；这不等于独立 SLAM。\n\n'
        '实际交换与缺失标签均重新融合：X 等于交换输入的 S，M 等于缺失输入的 G；'
        'G/M/N/G_capacity 不受标签变化影响，两家族共同几何与 O 特征相同，仅 S 类别条件列反号。\n\n'
        '有向面零深度支持时前三列禁用，新方向列保留。独立前/侧/后解析值及零深度负例也核对了冻结低层函数，'
        '没有调用完整 response_features 或 observed_patches 生成审计答案。\n\n'
        '面积按 R² 缩放，路线长度只除一次；G_capacity 为各 patch 面积描述平方之和除以 R⁴×L，'
        '没有误写为已经除 L 的特征再平方。重复状态通过最大值/并集去重。\n\n'
        '本次没有发现数学实现、朝向或类别隔离错误。FOV 无遮挡保证、支撑并非实体表面积等建模限制仍保留；'
        '通过审计不能证明语义收益或推广到未审阅上下文。\n')


def run(args):
    assert not args.output.exists(); args.output.mkdir(parents=True)
    started = time.monotonic(); own_hash, helper_hash = sha(__file__), sha(HELPER)
    status = {'status': 'running', 'workers': 1, 'source_run': str(args.run), 'script_sha256': own_hash,
              'independent_geometry_helper_sha256': helper_hash, 'outcomes_read': False, 'new_physical_routes': 0}
    write(args.output / 'metadata.json', status)
    try:
        metadata = read(args.run / 'metadata.json'); assert metadata['status'] == 'complete'
        assert metadata['context']['context_id'] == 'T0'
        with tempfile.TemporaryDirectory(prefix='response-v7-math-review-') as name:
            temp = Path(name); snapshot = temp / 'sources'; snapshot.mkdir()
            with zipfile.ZipFile(args.run / 'sources.zip') as archive:
                assert set(archive.namelist()) == set(metadata['source_sha256'])
                for filename in archive.namelist():
                    target = (snapshot / filename).resolve(); assert target.is_relative_to(snapshot)
                    payload = archive.read(filename); assert hashlib.sha256(payload).hexdigest() == metadata['source_sha256'][filename]
                    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
            copied_helper = temp / 'independent_math.py'; shutil.copyfile(HELPER, copied_helper)
            assert sha(copied_helper) == helper_hash
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--run', str(args.run), '--output', str(args.output),
                       '--snapshot', str(snapshot), '--helper', str(copied_helper)]
            subprocess.run(command, cwd=temp, env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1'}, check=True)
        assert sha(__file__) == own_hash and sha(HELPER) == helper_hash
        status.update(status='complete', elapsed_s=time.monotonic() - started)
    except Exception as error:
        status.update(status='failed', error=str(error), traceback=traceback.format_exc(), elapsed_s=time.monotonic() - started)
        write(args.output / 'metadata.json', status); raise
    write(args.output / 'metadata.json', status)
    write(args.output / 'artifact_hashes.json', {str(p.relative_to(args.output)): sha(p) for p in args.output.rglob('*') if p.is_file()})
    print(json.dumps({'status': 'complete', 'elapsed_s': status['elapsed_s']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument('--output', type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--snapshot', type=Path); parser.add_argument('--helper', type=Path)
    args = parser.parse_args(); worker(args) if args.worker else run(args)
