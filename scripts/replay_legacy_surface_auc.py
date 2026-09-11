#!/usr/bin/env python3
"""Replay recorded checkpoints against legacy and frozen union GT, without policy."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from io import BytesIO
import json
from pathlib import Path
import resource
import sys
import time
import traceback
import zipfile

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.virtual3d_v2 import VirtualConfigV2
from nso.semantic_completion_v3 import SemanticHistoryMapperV3
from utils.rgbd_contract import RGBDFrame, PlanarScan
from utils.reconstruction_metrics import ReconstructionEvaluator, ray_scene
from utils.grid_geometry import DIRECTIONS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def mesh_from_arrays(vertices, triangles):
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(triangles))


def compare_metrics(actual, expected, tolerance=1e-9):
    maximum = 0.
    for key, value in actual.items():
        if value is None:
            if expected[key] is not None:
                raise AssertionError(f'{key}: incompatible missing value')
        else:
            error = abs(float(value) - expected[key])
            if error > tolerance:
                raise AssertionError(f'{key}: {value} != {expected[key]} (error {error})')
            maximum = max(maximum, error)
    return maximum


def integrated_curve(steps, values, budget):
    return float(np.trapz(np.interp(np.arange(budget + 1), steps, values)) / budget)


def replay_one(endpoint, endpoint_dir, reference_record, output_dir):
    started = time.monotonic()
    episode = Path(endpoint['episode_path'])
    run = episode.parent
    inputs = {}

    def read_bytes(path):
        path = Path(path).resolve()
        value = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(value).hexdigest()
        return value

    config = json.loads(read_bytes(run / 'config.json'))
    original = json.loads(read_bytes(episode / 'episode.json'))
    entries = []
    for entry in config['scenes']:
        c = VirtualConfigV2(**(config['environment'] | entry.get('environment', {})))
        key = (f"{entry['layout']}_{entry['seed']}_{entry.get('semantic_condition', 'aligned')}"
               f'_d{c.depth_sigma_m}_p{c.pose_noise_m}_opaque{int(c.occluded_objects)}')
        if key == endpoint['scene']:
            entries.append((entry, c))
    if len(entries) != 1 or not config.get('object_hypotheses'):
        raise AssertionError('expected unique original V3/V4 history-mapper configuration')
    _, c = entries[0]
    thresholds = config['thresholds_m']
    if thresholds != [.02, .05, .1]:
        raise AssertionError('unexpected evaluation thresholds')
    original_checkpoints = [json.loads(line) for line in read_bytes(episode / 'metrics.jsonl').splitlines()]
    steps = [json.loads(line) for line in read_bytes(episode / 'steps.jsonl').splitlines()]
    checkpoint_by_step = {row['step']: row for row in original_checkpoints}
    if len(checkpoint_by_step) != len(original_checkpoints):
        raise AssertionError('duplicate original checkpoint')
    with np.load(BytesIO(read_bytes(episode / 'maps.npz')), allow_pickle=False) as data:
        original_known = data['known_packed'].copy()
        original_poses = data['poses'].copy()
        original_final_belief = data['final_belief'].copy()
        map_digest = array_digest(*(data[key] for key in sorted(data.files)))
    if map_digest != endpoint['maps_array_sha256']:
        raise AssertionError('maps no longer match frozen endpoint audit')
    legacy = ReconstructionEvaluator.__new__(ReconstructionEvaluator)
    old_reference_path = run / (endpoint['scene'] + '_reference.npz')
    with np.load(BytesIO(read_bytes(old_reference_path)), allow_pickle=False) as data:
        legacy.reference, legacy.classes = data['points'].copy(), data['classes'].copy()
        reachable = data['reachable'].copy()
        legacy.truth = ray_scene(mesh_from_arrays(data['vertices'], data['triangles']))
    if array_digest(legacy.reference, legacy.classes) != endpoint['legacy_reference_array_sha256']:
        raise AssertionError('legacy reference no longer matches endpoint audit')
    union = ReconstructionEvaluator.__new__(ReconstructionEvaluator)
    new_reference_path = Path(endpoint_dir) / reference_record['path']
    raw_reference = read_bytes(new_reference_path)
    if hashlib.sha256(raw_reference).hexdigest() != reference_record['sha256']:
        raise AssertionError('union reference changed since endpoint freeze')
    with np.load(BytesIO(raw_reference), allow_pickle=False) as data:
        union.reference, union.classes = data['points'].copy(), data['classes'].copy()
        union.truth = ray_scene(mesh_from_arrays(data['vertices'], data['triangles']))
        if not np.array_equal(reachable, data['reachable']):
            raise AssertionError('legacy and union reachable regions differ')
    if array_digest(union.reference, union.classes) != reference_record['reference_array_sha256']:
        raise AssertionError('union reference arrays differ from endpoint freeze')
    shape = reachable.shape
    known = np.unpackbits(original_known, axis=1)[:, :reachable.size].reshape(-1, *shape).astype(bool)
    expected_coverage = np.sum(known & reachable, axis=(1, 2)) / reachable.sum()
    if len(steps) != len(known) or len(original_poses) != len(known):
        raise AssertionError('map, pose and step lengths disagree')
    np.testing.assert_array_equal([row['step'] for row in steps], np.arange(len(steps)))
    np.testing.assert_allclose(expected_coverage, [row['coverage'] for row in steps], rtol=0, atol=1e-12)
    if original['step'] != len(steps) - 1 or original['frames'] != len(steps):
        raise AssertionError('recorded final step/frame count mismatch')
    np.testing.assert_allclose(expected_coverage[-1], original['coverage_2d'], rtol=0, atol=1e-12)
    coverage_auc = integrated_curve(np.arange(len(steps)), expected_coverage, c.max_steps)
    np.testing.assert_allclose(coverage_auc, original['coverage_auc'], rtol=0, atol=1e-12)
    for row in original_checkpoints:
        np.testing.assert_allclose(row['coverage_2d'], expected_coverage[row['step']], rtol=0, atol=1e-12)
    frames = sorted((episode / 'frames').glob('*.npz'))
    scans = sorted((episode / 'scans').glob('*.npz'))
    expected_names = [f'{index:04d}.npz' for index in range(len(steps))]
    if [p.name for p in frames] != expected_names or [p.name for p in scans] != expected_names:
        raise AssertionError('saved frame/scan sequence has missing or extra observations')
    mapper = SemanticHistoryMapperV3(shape, c, c.truncation_m)
    checkpoints = []
    last_timestamp = -float('inf')
    max_old_error = 0.
    collisions_applied = 0
    replay_coverage = []
    for index, (frame_path, scan_path) in enumerate(zip(frames, scans)):
        frame = RGBDFrame.load(BytesIO(read_bytes(frame_path)))
        scan = PlanarScan.load(BytesIO(read_bytes(scan_path)))
        if frame.timestamp_s <= last_timestamp:
            raise AssertionError('frame timestamps are not strictly increasing')
        last_timestamp = frame.timestamp_s
        mapper.update(frame, scan)
        # Original loop adds a contact obstacle after sensor integration.
        # This replay consumes the recorded contact/pose; it never asks a policy.
        collision_increment = steps[index]['collisions'] - (steps[index - 1]['collisions'] if index else 0)
        if collision_increment not in (0, 1):
            raise AssertionError('invalid cumulative collision sequence')
        if collision_increment:
            row, col, heading = map(int, original_poses[index])
            dr, dc = DIRECTIONS[heading]
            r, c_index = row + dr, col + dc
            if 0 <= r < shape[0] and 0 <= c_index < shape[1]:
                mapper.belief[r, c_index] = 1
            collisions_applied += 1
        current_known = mapper.belief != -1
        if not np.array_equal(current_known, known[index]):
            raise AssertionError(f'{index}: replayed known cells differ from recorded map')
        current_coverage = float(np.count_nonzero(current_known & reachable) / reachable.sum())
        replay_coverage.append(current_coverage)
        if index in checkpoint_by_step:
            mesh = mapper.mesh()
            old = legacy.evaluate(mesh, current_coverage, thresholds)
            error = compare_metrics(old, checkpoint_by_step[index])
            max_old_error = max(max_old_error, error)
            new = union.evaluate(mesh, current_coverage, thresholds)
            checkpoints.append(dict(step=index, coverage_2d=current_coverage, legacy=old, union=new,
                                    legacy_reproduction_max_absolute_error=error))
            del mesh
    if len(checkpoints) != len(original_checkpoints):
        raise AssertionError('not all original checkpoints were replayed')
    if not np.array_equal(mapper.belief, original_final_belief):
        raise AssertionError('replayed final belief differs from recorded belief')
    final_mesh = mapper.mesh()
    replay_vertices, replay_triangles = np.asarray(final_mesh.vertices), np.asarray(final_mesh.triangles)
    with np.load(BytesIO(read_bytes(episode / 'final_mesh.npz')), allow_pickle=False) as data:
        original_vertices, original_triangles = data['vertices'].copy(), data['triangles'].copy()
    if array_digest(original_vertices, original_triangles) != endpoint['prediction_array_sha256']:
        raise AssertionError('saved final prediction differs from endpoint freeze')
    if replay_vertices.shape != original_vertices.shape or not np.array_equal(replay_triangles, original_triangles):
        raise AssertionError('replayed final mesh topology differs from original')
    vertex_error = float(np.max(np.abs(replay_vertices - original_vertices))) if len(replay_vertices) else 0.
    if vertex_error > 1e-8:
        raise AssertionError('replayed final vertices do not reproduce original mesh')
    compare_metrics(checkpoints[-1]['legacy'], endpoint['legacy'])
    compare_metrics(checkpoints[-1]['union'], endpoint['union'])
    max_old_error = max(max_old_error, compare_metrics(checkpoints[-1]['legacy'], original))
    auc = {mode: {f'joint_auc_{round(threshold * 100):02d}cm': integrated_curve(
        [r['step'] for r in checkpoints],
        [r[mode][f'joint_{round(threshold * 100):02d}cm'] for r in checkpoints], c.max_steps)
        for threshold in thresholds} for mode in ('legacy', 'union')}
    for key, value in auc['legacy'].items():
        np.testing.assert_allclose(value, original[key], rtol=0, atol=1e-12)
    np.testing.assert_allclose(replay_coverage, expected_coverage, rtol=0, atol=1e-12)
    name = endpoint['scene'] + '_' + endpoint['method']
    output = Path(output_dir) / 'episodes'
    input_path = output / (name + '_inputs.json')
    write_json(input_path, inputs)
    result = dict(scene=endpoint['scene'], method=endpoint['method'], source_episode=str(episode),
                  frames=len(frames), checkpoint_count=len(checkpoints), budget=c.max_steps,
                  final_step=original['step'], coverage_2d=replay_coverage[-1], coverage_auc=coverage_auc,
                  coverage_by_step=replay_coverage, legacy=auc['legacy'], union=auc['union'], checkpoints=checkpoints,
                  old_checkpoint_max_absolute_error=max_old_error,
                  final_mesh=dict(topology_identical=True, vertices_bitwise_identical=bool(np.array_equal(replay_vertices, original_vertices)),
                                  maximum_vertex_coordinate_error_m=vertex_error,
                                  replay_array_sha256=array_digest(replay_vertices, replay_triangles),
                                  recorded_array_sha256=endpoint['prediction_array_sha256']),
                  validation=dict(every_replayed_known_mask_matches=True, every_recorded_coverage_matches=True,
                                  final_belief_identical=True, old_auc_matches=True, frozen_union_endpoint_matches=True,
                                  original_collision_updates_applied=collisions_applied),
                  legacy_reference_array_sha256=endpoint['legacy_reference_array_sha256'],
                  union_reference_array_sha256=reference_record['reference_array_sha256'],
                  input_manifest=str(input_path.relative_to(Path(output_dir))), input_manifest_sha256=sha256(input_path),
                  elapsed_s=time.monotonic() - started,
                  worker_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    write_json(output / (name + '.json'), result)
    return result


def summarize(rows):
    metrics = [f'joint_auc_{tag}cm' for tag in ('02', '05', '10')]
    methods = sorted({row['method'] for row in rows})
    means = {}
    for method in methods:
        selected = [row for row in rows if row['method'] == method]
        means[method] = dict(n=len(selected), coverage_auc=float(np.mean([r['coverage_auc'] for r in selected])),
                            **{mode: {key: float(np.mean([r[mode][key] for r in selected])) for key in metrics}
                               for mode in ('legacy', 'union')})
    lookup = {(row['scene'], row['method']): row for row in rows}
    scenes = sorted({row['scene'] for row in rows})
    comparisons = []
    def sign(value):
        return 1 if value > 1e-12 else -1 if value < -1e-12 else 0
    for method in methods:
        if method == 'full_v4':
            continue
        pairs = []
        for scene in scenes:
            full, control = lookup[scene, 'full_v4'], lookup[scene, method]
            delta = {mode: {key: full[mode][key] - control[mode][key] for key in metrics}
                     for mode in ('legacy', 'union')}
            pairs.append(dict(scene=scene, **delta,
                              sign_changed={key: sign(delta['legacy'][key]) != sign(delta['union'][key]) for key in metrics}))
        means_delta = {mode: {key: float(np.mean([row[mode][key] for row in pairs])) for key in metrics}
                       for mode in ('legacy', 'union')}
        comparisons.append(dict(control=method, n=len(pairs), mean_full_minus_control=means_delta,
                                mean_sign_changed={key: sign(means_delta['legacy'][key]) != sign(means_delta['union'][key]) for key in metrics},
                                paired_scenes=pairs))
    return dict(method_means=means, comparisons=comparisons,
                scope='Descriptive sensitivity on existing trajectories; only two seed blocks, not independent confirmation.')


def report(summary, metadata):
    lines = ['# 原轨迹检查点回放：唯一外表面联合 AUC 敏感性复核', '',
             f"完成 {metadata['unique_episode_count']} 个唯一 scene-method、{metadata['replayed_frames']} 帧、"
             f"{metadata['checkpoint_count']} 个原检查点；原跨批次 8 个重复终点不再增加样本。", '',
             '先按旧 GT/参考验证原检查点指标、逐步覆盖和最终网格，再使用终点复核中冻结的相同唯一外表面参考。'
             '没有新导航、没有 GT 重采样，没有保存重复原始帧或检查点网格。', '',
             '## 各方法联合 AUC 均值', '',
             '| 方法 | N | 覆盖 AUC | 旧 J-AUC@5cm | 新 J-AUC@5cm |',
             '|---|---:|---:|---:|---:|']
    for method, value in summary['method_means'].items():
        lines.append(f"| {method} | {value['n']} | {value['coverage_auc']:.6f} | "
                     f"{value['legacy']['joint_auc_05cm']:.6f} | {value['union']['joint_auc_05cm']:.6f} |")
    lines += ['', '## 完整 V4 减去对照', '',
              '| 对照 | 阈值 | 旧 ΔJ-AUC | 新 ΔJ-AUC | 均值变号 |',
              '|---|---:|---:|---:|---|']
    for value in summary['comparisons']:
        for tag in ('02', '05', '10'):
            key = f'joint_auc_{tag}cm'
            lines.append(f"| {value['control']} | {int(tag)}cm | "
                         f"{value['mean_full_minus_control']['legacy'][key]:+.6f} | "
                         f"{value['mean_full_minus_control']['union'][key]:+.6f} | "
                         f"{'是' if value['mean_sign_changed'][key] else '否'} |")
    lines += ['', '## 验证和解释范围', '',
              f"- 旧检查点指标最大绝对差：{metadata['old_checkpoint_max_absolute_error']:.12g}；最终网格最大顶点坐标差："
              f"{metadata['final_mesh_max_coordinate_error_m']:.12g} 米，三角拓扑全部一致。",
              '- 每一步回放的已知栅格与原 maps 一致；覆盖、检查点覆盖、覆盖 AUC 与原记录独立核对。',
              '- 联合曲线保持原检查点间线性插值、提前结束后保持终值至原 280 步预算的规则。',
              '- 每个重放检查点仍按原预测表面 12000 点、种子 812 评价。GT 直接复用终点 posthoc 的冻结参考，没有再次生成 GT 样本。',
              '- 保留旧与新口径并列结果。变化包括从旧参考到冻结新参考的权重和有限采样变化，不是新规划策略的增益。',
              '- 这是原开发数据的评价敏感性复核；不增加场景或独立种子，不证明完整方案的全部模块有效。',
              '- metadata.json 记录源码；input_hashes.json 与每回合 inputs 清单记录原帧、雷达、地图、网格、参考和配置；原冻结文件未改。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        raise ValueError('one or two CPU workers only')
    endpoint_dir, output = args.endpoints.resolve(), args.output.resolve()
    endpoint_metadata = json.loads((endpoint_dir / 'metadata.json').read_text())
    if endpoint_metadata['status'] != 'complete':
        raise ValueError('requires completed endpoint sensitivity audit')
    if output.exists() or output.is_relative_to(endpoint_dir) or any(
            output.is_relative_to(Path(run)) for run in endpoint_metadata['source_runs']):
        raise ValueError('output must be a new independent directory')
    endpoints = json.loads((endpoint_dir / 'endpoints.json').read_text())
    keys = {(row['scene'], row['method']) for row in endpoints}
    if len(keys) != len(endpoints) or len(endpoints) != endpoint_metadata['unique_endpoint_count']:
        raise AssertionError('endpoint matrix is duplicated or incomplete')
    references = {row['scene']: row for row in json.loads((endpoint_dir / 'references.json').read_text())}
    inputs = json.loads((endpoint_dir / 'input_hashes.json').read_text())
    for name in ('metadata.json', 'endpoints.json', 'references.json', 'input_hashes.json', 'duplicates.json'):
        inputs[str(endpoint_dir / name)] = sha256(endpoint_dir / name)
    for record in references.values():
        inputs[str(endpoint_dir / record['path'])] = record['sha256']
    for path, value in inputs.items():
        if sha256(path) != value:
            raise AssertionError(f'endpoint frozen input changed: {path}')
    legacy_sources = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(ROOT):
            name = str(path.relative_to(ROOT))
            if Path(name).parts[0] in ('env', 'nso', 'utils') and path.suffix == '.py':
                legacy_sources[name] = sha256(path)
    sources = legacy_sources | {'scripts/replay_legacy_surface_auc.py': sha256(Path(__file__)),
                               'requirements-3d.lock.txt': sha256(ROOT / 'requirements-3d.lock.txt')}
    for run in endpoint_metadata['source_runs']:
        run = Path(run)
        metadata = json.loads((run / 'run_metadata.json').read_text())
        if metadata['status'] != 'complete':
            raise AssertionError('source run is not completed')
        with zipfile.ZipFile(run / 'sources.zip') as archive:
            for name, value in legacy_sources.items():
                if metadata['files_sha256'].get(name) != value or hashlib.sha256(archive.read(name)).hexdigest() != value:
                    raise AssertionError(f'loaded replay dependency differs from frozen runtime: {name}')
    output.mkdir(parents=True)
    (output / 'episodes').mkdir()
    with zipfile.ZipFile(output / 'replay_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(sources):
            archive.write(ROOT / name, name)
    started = time.monotonic()
    metadata = dict(status='running', scope='Posthoc original-trajectory checkpoint replay, no new navigation',
                    source_endpoint_audit=str(endpoint_dir), source_runs=endpoint_metadata['source_runs'],
                    source_sha256=sources, source_archive_sha256=sha256(output / 'replay_sources.zip'),
                    workers=args.workers, cpu_only=True, expected_unique_episodes=len(endpoints),
                    excluded_duplicate_episode_count=endpoint_metadata['duplicate_endpoint_count'],
                    prediction_sampling=dict(count=12000, seed=812),
                    reference_sampling='Reuse endpoint frozen reference bytes; no GT resampling in this run',
                    thresholds_m=[.02, .05, .1], auc_rule='Original checkpoint trapezoid; hold final value after early stop to budget',
                    saved_raw_frames=False, saved_checkpoint_meshes=False, new_navigation_episodes=0,
                    frozen_results_modified=False)
    write_json(output / 'metadata.json', metadata)
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(replay_one, row, str(endpoint_dir), references[row['scene']], str(output)) for row in endpoints]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                metadata.update(completed_unique_episodes=len(rows), elapsed_s=time.monotonic() - started)
                write_json(output / 'metadata.json', metadata)
                print(f"{len(rows)}/{len(endpoints)}", row['scene'], row['method'],
                      'old/new AUC05', round(row['legacy']['joint_auc_05cm'], 6), round(row['union']['joint_auc_05cm'], 6),
                      'old error', row['old_checkpoint_max_absolute_error'], 'seconds', round(row['elapsed_s'], 1), flush=True)
    except Exception:
        metadata.update(status='failed', error=traceback.format_exc(), elapsed_s=time.monotonic() - started)
        write_json(output / 'metadata.json', metadata)
        raise
    for row in rows:
        manifest = output / row['input_manifest']
        if sha256(manifest) != row['input_manifest_sha256']:
            raise AssertionError('per-episode input manifest changed')
        for path, value in json.loads(manifest.read_text()).items():
            if path in inputs and inputs[path] != value:
                raise AssertionError('replayed bytes differ from frozen endpoint input')
            inputs[path] = value
    for path, value in inputs.items():
        if sha256(path) != value:
            raise AssertionError(f'input changed during replay: {path}')
    if any(sha256(ROOT / name) != value for name, value in sources.items()):
        raise AssertionError('replay source changed during run')
    if len(rows) != len(keys) or {(r['scene'], r['method']) for r in rows} != keys:
        raise AssertionError('completed replay matrix mismatch')
    summary = summarize(rows)
    metadata.update(status='complete', unique_episode_count=len(rows), replayed_frames=sum(r['frames'] for r in rows),
                    checkpoint_count=sum(r['checkpoint_count'] for r in rows), elapsed_s=time.monotonic() - started,
                    old_checkpoint_max_absolute_error=max(r['old_checkpoint_max_absolute_error'] for r in rows),
                    final_mesh_max_coordinate_error_m=max(r['final_mesh']['maximum_vertex_coordinate_error_m'] for r in rows),
                    all_final_vertices_bitwise_identical=all(r['final_mesh']['vertices_bitwise_identical'] for r in rows),
                    worker_peak_rss_mib=max(r['worker_peak_rss_mib'] for r in rows), input_file_count=len(inputs))
    write_json(output / 'input_hashes.json', inputs)
    write_json(output / 'summary.json', summary)
    write_json(output / 'metadata.json', metadata)
    (output / 'results.md').write_text(report(summary, metadata))


if __name__ == '__main__':
    main()
