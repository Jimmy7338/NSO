#!/usr/bin/env python3
"""Posthoc endpoint sensitivity with immutable recorded predictions; no policy."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.virtual3d_v2 import VirtualConfigV2, VirtualWorldV2
from env.virtual3d_inspection_v4 import union_surface_from_boxes
from utils.reconstruction_metrics import ReconstructionEvaluator, ray_scene
from scripts.audit_legacy_surface_reference import recover_box_primitives, sha256


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


def load_mesh(path):
    with np.load(path, allow_pickle=False) as data:
        vertices, triangles = data['vertices'].copy(), data['triangles'].copy()
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                    o3d.utility.Vector3iVector(triangles))
    return mesh, array_digest(vertices, triangles)


def match_metrics(actual, recorded):
    errors = []
    for key, value in actual.items():
        if value is None:
            if recorded[key] is not None:
                raise AssertionError(f'{key}: incompatible missing metric')
        else:
            error = abs(float(value) - recorded[key])
            if error > 1e-9:
                raise AssertionError(f'{key}: old endpoint mismatch {value} != {recorded[key]}')
            errors.append(error)
    return max(errors, default=0.)


def scene_key(entry, config):
    c = VirtualConfigV2(**(config['environment'] | entry.get('environment', {})))
    return (f"{entry['layout']}_{entry['seed']}_{entry.get('semantic_condition', 'aligned')}"
            f'_d{c.depth_sigma_m}_p{c.pose_noise_m}_opaque{int(c.occluded_objects)}')


def sign(value):
    return 1 if value > 1e-12 else -1 if value < -1e-12 else 0


def summarize(rows):
    methods = sorted({row['method'] for row in rows})
    metrics = [f'{quantity}_{tag}cm' for tag in ('02', '05', '10')
               for quantity in ('precision', 'recall', 'f1', 'joint')]
    means = {}
    for method in methods:
        selected = [row for row in rows if row['method'] == method]
        means[method] = dict(n=len(selected), coverage_2d=float(np.mean([r['coverage_2d'] for r in selected])),
                            **{mode: {key: float(np.mean([r[mode][key] for r in selected]))
                                      for key in metrics} for mode in ('legacy', 'union')})
    by_key = {(row['scene'], row['method']): row for row in rows}
    comparisons = []
    scenes = sorted({row['scene'] for row in rows})
    for control in methods:
        if control == 'full_v4':
            continue
        paired = []
        for scene in scenes:
            full, other = by_key[scene, 'full_v4'], by_key[scene, control]
            delta = {mode: {key: full[mode][key] - other[mode][key] for key in metrics}
                     for mode in ('legacy', 'union')}
            paired.append(dict(scene=scene, **delta,
                               sign_changed={key: sign(delta['legacy'][key]) != sign(delta['union'][key])
                                             for key in metrics}))
        mean_delta = {mode: {key: float(np.mean([r[mode][key] for r in paired])) for key in metrics}
                      for mode in ('legacy', 'union')}
        comparisons.append(dict(control=control, n=len(paired), mean_full_minus_control=mean_delta,
                                mean_sign_changed={key: sign(mean_delta['legacy'][key]) != sign(mean_delta['union'][key])
                                                   for key in metrics}, paired_scenes=paired))
    return dict(method_means=means, comparisons=comparisons,
                statistical_scope='Eight existing scene conditions but only two seed blocks; descriptive posthoc sensitivity, no new confirmation.')


def report(summary, metadata):
    lines = ['# 旧参考与唯一外表面 GT：原轨迹终点敏感性复核', '',
             '本结果只替换评价 GT，使用原预测网格、原 12000 点预测采样及种子 812。'
             '没有新导航、没有重新融合，也没有重算 joint-AUC。', '',
             f"核验 {metadata['recorded_episode_count']} 个有效记录，合并为 {metadata['unique_endpoint_count']} 个唯一 scene-method 终点；"
             f"另 {metadata['duplicate_endpoint_count']} 个重复终点已核验预测、地图和旧指标一致，不增加样本量。", '',
             '## 5 厘米阈值下的方法均值', '',
             '| 方法 | N | C | 旧 P | 新 P | 旧 R | 新 R | 旧 F1 | 新 F1 | 旧 C×F1 | 新 C×F1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, row in summary['method_means'].items():
        values = [row['coverage_2d']]
        for key in ('precision_05cm', 'recall_05cm', 'f1_05cm', 'joint_05cm'):
            values.extend([row['legacy'][key], row['union'][key]])
        lines.append(f"| {method} | {row['n']} | " + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines += ['', '## 完整 V4 减去对照的终点差值', '',
              '正值表示完整 V4 更高。表中只讨论跨 8 场景均值，逐场景变化及全部 2/5/10 厘米阈值在 summary.json。', '',
              '| 对照 | 旧 ΔF1@5cm | 新 ΔF1@5cm | 旧 Δ(C×F1)@5cm | 新 Δ(C×F1)@5cm | 联合均值变号 |',
              '|---|---:|---:|---:|---:|---|']
    for row in summary['comparisons']:
        d = row['mean_full_minus_control']
        lines.append(f"| {row['control']} | {d['legacy']['f1_05cm']:+.6f} | {d['union']['f1_05cm']:+.6f} | "
                     f"{d['legacy']['joint_05cm']:+.6f} | {d['union']['joint_05cm']:+.6f} | "
                     f"{'是' if row['mean_sign_changed']['joint_05cm'] else '否'} |")
    lines += ['', '## 解释边界', '',
              '- 旧口径的所有返回指标均与原记录在 1e-9 绝对容差内一致；覆盖由原地图独立计算。',
              '- 新参考按唯一实体外表面积均匀采样，仍使用原可达位置格点、视场、量程与射线容差；每场景固定一次，所有方法共享。',
              '- 改变参考权重同时重新采样，因此该复核包括有限参考采样变化。不能把每个微小数值变化单独归因于去重。',
              '- 这是相同开发轨迹的终点评价敏感性复核，不是新场景、独立种子确认或完整方案优势证明。',
              '- joint-AUC 需要从原保存帧回放检查点重建后单独复核，不能用本终点结果代替。',
              '- 原冻结实验文件保持不变；input_hashes.json 保存读取输入的哈希，references/ 保存共享新 GT 与参考。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    runs = [path.resolve() for path in args.runs]
    if output.exists() or any(output.is_relative_to(run) for run in runs):
        raise ValueError('output must be a new independent directory')
    sources = ('scripts/reevaluate_legacy_surface_endpoints.py', 'scripts/audit_legacy_surface_reference.py',
               'env/__init__.py', 'env/virtual3d.py', 'env/virtual3d_v2.py',
               'env/virtual3d_inspection_v4.py', 'utils/reconstruction_metrics.py',
               'utils/grid_geometry.py', 'utils/rgbd_contract.py', 'requirements-3d.lock.txt')
    source_hashes = {name: sha256(ROOT / name) for name in sources}
    input_hashes = {}

    def remember(path):
        path = path.resolve()
        input_hashes[str(path)] = sha256(path)
        return path

    records = defaultdict(list)
    worlds = {}
    for run in runs:
        metadata = json.loads(remember(run / 'run_metadata.json').read_text())
        config = json.loads(remember(run / 'config.json').read_text())
        if metadata['status'] != 'complete' or config['thresholds_m'] != [.02, .05, .1]:
            raise ValueError('completed runs with shared 2/5/10cm thresholds required')
        with zipfile.ZipFile(remember(run / 'sources.zip')) as archive:
            for name, value in metadata['files_sha256'].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != value:
                    raise AssertionError(f'frozen archive mismatch: {name}')
            for name in ('env/__init__.py', 'env/virtual3d.py', 'env/virtual3d_v2.py',
                         'utils/reconstruction_metrics.py', 'utils/grid_geometry.py', 'utils/rgbd_contract.py'):
                if source_hashes[name] != metadata['files_sha256'][name]:
                    raise AssertionError(f'legacy implementation mismatch: {name}')
        rows = [json.loads(line) for line in remember(run / 'episodes.jsonl').read_text().splitlines()]
        expected = {(scene_key(entry, config), method) for entry in config['scenes'] for method in config['methods']}
        actual = [(row['scene'], row['method']) for row in rows]
        if len(rows) != metadata['expected_episodes'] or len(set(actual)) != len(actual) or set(actual) != expected:
            raise AssertionError('incomplete or duplicate run matrix')
        for entry in config['scenes']:
            key = scene_key(entry, config)
            c = VirtualConfigV2(**(config['environment'] | entry.get('environment', {})))
            if key in worlds and (worlds[key]['config'] != c or worlds[key]['samples'] != config['reference_samples']):
                raise AssertionError('scene key collision with distinct sensor/reference settings')
            worlds.setdefault(key, dict(config=c, samples=config['reference_samples'], entry=entry))
        for row in rows:
            records[row['scene']].append((run, row))
    output.mkdir(parents=True)
    (output / 'references').mkdir()
    started = time.monotonic()
    metadata = dict(status='running', scope='Posthoc endpoint metric sensitivity only', source_runs=list(map(str, runs)),
                    source_sha256=source_hashes, recorded_episode_count=sum(map(len, records.values())),
                    prediction_sampling=dict(count=12000, seed=812), reference_sampling=dict(seed=2026),
                    thresholds_m=[.02, .05, .1], new_navigation_episodes=0, replayed_sensor_frames=0,
                    joint_auc_recomputed=False, frozen_results_modified=False)
    write_json(output / 'metadata.json', metadata)
    results, duplicates, references = [], [], []
    max_old_error = 0.
    for scene, selected in sorted(records.items()):
        info = worlds[scene]
        entry = info['entry']
        world = VirtualWorldV2(info['config'], **{k: v for k, v in entry.items() if k != 'environment'})
        primitives, recovery = recover_box_primitives(world.mesh, world.triangle_classes)
        union, labels, union_audit = union_surface_from_boxes(primitives)
        union_evaluator = ReconstructionEvaluator(SimpleNamespace(
            mesh=union, triangle_classes=labels, config=world.config, reachable=world.reachable, shape=world.shape), info['samples'])
        reference_path = output / 'references' / (scene + '.npz')
        np.savez_compressed(reference_path, points=union_evaluator.reference, classes=union_evaluator.classes,
                            vertices=np.asarray(union.vertices), triangles=np.asarray(union.triangles),
                            triangle_classes=labels, primitives=primitives, reachable=world.reachable)
        references.append(dict(scene=scene, path=str(reference_path.relative_to(output)), sha256=sha256(reference_path),
                               count=len(union_evaluator.reference), candidate_count=info['samples'],
                               reference_array_sha256=array_digest(union_evaluator.reference, union_evaluator.classes),
                               geometry_array_sha256=array_digest(np.asarray(union.vertices), np.asarray(union.triangles)),
                               recovery=recovery, union_surface=union_audit))
        canonical = {}
        saved_reference = None
        legacy_evaluator = ReconstructionEvaluator.__new__(ReconstructionEvaluator)
        legacy_evaluator.truth = ray_scene(world.mesh)
        for run, record in selected:
            path = remember(run / (scene + '_reference.npz'))
            with np.load(path, allow_pickle=False) as reference:
                if not np.array_equal(reference['vertices'], np.asarray(world.mesh.vertices)) or not np.array_equal(reference['triangles'], np.asarray(world.mesh.triangles)):
                    raise AssertionError('frozen geometry mismatch')
                if not np.array_equal(reference['reachable'], world.reachable):
                    raise AssertionError('frozen reachable mismatch')
                current_reference = array_digest(reference['points'], reference['classes'])
                if saved_reference is not None and saved_reference != current_reference:
                    raise AssertionError('methods or runs did not share a frozen reference')
                saved_reference = current_reference
                legacy_evaluator.reference, legacy_evaluator.classes = reference['points'].copy(), reference['classes'].copy()
            episode = run / record['artifact_dir']
            recorded_episode = json.loads(remember(episode / 'episode.json').read_text())
            if recorded_episode != record:
                raise AssertionError('episode record and aggregate differ')
            checkpoints = [json.loads(line) for line in remember(episode / 'metrics.jsonl').read_text().splitlines()]
            steps = [json.loads(line) for line in remember(episode / 'steps.jsonl').read_text().splitlines()]
            with np.load(remember(episode / 'maps.npz'), allow_pickle=False) as maps:
                known = np.unpackbits(maps['known_packed'][-1])[:world.reachable.size].reshape(world.shape).astype(bool)
                if not np.array_equal(known, maps['final_belief'] != -1):
                    raise AssertionError('final packed map and belief disagree')
                map_digest = array_digest(*(maps[key] for key in sorted(maps.files)))
            coverage = float(np.count_nonzero(known & world.reachable) / world.reachable.sum())
            if any(abs(coverage - value) > 1e-12 for value in (record['coverage_2d'], checkpoints[-1]['coverage_2d'], steps[-1]['coverage'])):
                raise AssertionError('final coverage mismatch')
            if record['step'] != checkpoints[-1]['step'] or record['step'] != steps[-1]['step']:
                raise AssertionError('final step mismatch')
            mesh, mesh_digest = load_mesh(remember(episode / 'final_mesh.npz'))
            method = record['method']
            if method in canonical:
                first = canonical[method]
                if mesh_digest != first['prediction_array_sha256'] or map_digest != first['maps_array_sha256'] or coverage != first['coverage_2d']:
                    raise AssertionError('repeated scene-method prediction/map is not identical')
                max_old_error = max(max_old_error, match_metrics(first['legacy'], record), match_metrics(first['legacy'], checkpoints[-1]))
                duplicates.append(dict(scene=scene, method=method, episode_path=str(episode), canonical_path=first['episode_path'],
                                       prediction_and_maps_identical=True, old_metrics_match=True, counted_as_independent=False))
                continue
            old = legacy_evaluator.evaluate(mesh, coverage, [.02, .05, .1])
            error = max(match_metrics(old, record), match_metrics(old, checkpoints[-1]))
            max_old_error = max(max_old_error, error)
            new = union_evaluator.evaluate(mesh, coverage, [.02, .05, .1])
            row = dict(scene=scene, method=method, episode_path=str(episode), prediction_array_sha256=mesh_digest,
                       maps_array_sha256=map_digest, legacy_reference_array_sha256=saved_reference,
                       coverage_2d=coverage, step=record['step'], legacy=old, union=new,
                       legacy_reproduction_max_absolute_error=error)
            canonical[method] = row
            results.append(row)
        print(scene, 'unique endpoints', len(canonical), flush=True)
    if any(sha256(ROOT / name) != value for name, value in source_hashes.items()):
        raise AssertionError('posthoc sources changed during run')
    if any(sha256(path) != value for path, value in input_hashes.items()):
        raise AssertionError('frozen input changed during run')
    metadata.update(status='complete', unique_endpoint_count=len(results), duplicate_endpoint_count=len(duplicates),
                    scene_count=len(records), old_reproduction_max_absolute_error=max_old_error,
                    elapsed_s=time.monotonic() - started)
    summary = summarize(results)
    write_json(output / 'endpoints.json', results)
    write_json(output / 'duplicates.json', duplicates)
    write_json(output / 'references.json', references)
    write_json(output / 'input_hashes.json', input_hashes)
    write_json(output / 'summary.json', summary)
    write_json(output / 'metadata.json', metadata)
    (output / 'results.md').write_text(report(summary, metadata))


if __name__ == '__main__':
    main()
