#!/usr/bin/env python3
"""Read-only analysis of sealed V24 saved-depth development and verification.

No world, backend, mapper, evaluator, sensing, action or TSDF fusion is executed.
The imported V23 analysis helpers check arithmetic in recorded metrics only.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from analyze_facility_shape_v23 import (read, sha, require, close, array_hash,
    validate_outline, validate_surface, completion_reasons)
import numpy as np

DEFAULT_SOURCE = ROOT/'audit_results/observed_shape_v24_development_20260915'
DEFAULT_OUTPUT = ROOT/'audit_results/observed_shape_v24_analysis_20260915'
DEFAULT_REPORT = ROOT/'docs/research/V24_SHARED_SHAPE_DEVELOPMENT_RESULT_20260915.md'
PREFLIGHT = ROOT/'audit_results/observed_shape_v24_preflight_20260915'
STAGES = {'coarse': 23, 'extra': 73, 'returned': 96}
REPRESENTATIONS = {'raw': 'raw_outline', 'observed': 'observed_outline', 'primary': 'primary_outline'}
MESHES = ('observed_mesh', 'inferred_mesh', 'completed_mesh')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n')


def verify_inventory(root, structured=False):
    inventory = read(root/'artifact_hashes.json')
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and p != root/'artifact_hashes.json'}
    require(actual == set(inventory), 'Artifact file set differs: '+str(root))
    for name, value in inventory.items():
        path = root/name
        require(path.resolve().is_relative_to(root.resolve()), 'Inventory escapes its evidence root')
        expected = value['sha256'] if structured else value
        require(sha(path) == expected, 'Artifact content differs: '+str(path))
        if structured: require(path.stat().st_size == value['bytes'], 'Artifact size differs')
    return len(inventory)


def verify_sources(root, sources, archive_hash):
    require(sha(root/'sources.zip') == archive_hash, 'Source archive differs')
    with zipfile.ZipFile(root/'sources.zip') as archive:
        require(set(archive.namelist()) == set(sources), 'Archived source set differs')
        for name, expected in sources.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, 'Archived source content differs')
            require(sha(ROOT/name) == expected, 'Current frozen source differs: '+name)


def verify_inputs(source):
    # Check the completion receipt before touching output or reporting scores.
    manifest = read(source/'manifest.json'); verification = read(source/'verification.json')
    require(manifest['status'] == 'complete' and verification['status'] == 'passed', 'Complete development and passed verification required')
    require(verification['independent_process'] is True, 'Independent verification required')
    timing = read(source/'timing.json')
    require(verification['original_process_id'] == timing['process_id']
        and verification['verification_process_id'] != timing['process_id'], 'Verification must run in a different process')
    counts = {'development_artifact_files': verify_inventory(source),
              'preflight_artifact_files': verify_inventory(PREFLIGHT, structured=True)}
    verify_sources(source, manifest['sources'], manifest['source_archive_sha256'])
    preflight = read(PREFLIGHT/'run.json')
    require(sha(PREFLIGHT/'run.json') == manifest['preflight_result_sha256'], 'Preflight receipt differs')
    require(preflight['status'] == 'passed' and preflight['exit_code'] == 0 and preflight['sources_unchanged'], 'Independent fixture preflight failed')
    require(preflight['source_sha256_before'] == preflight['source_sha256_after'], 'Preflight sources changed during execution')
    for name, expected in preflight['source_sha256_after'].items():
        require(manifest['sources'][name] == expected, 'Development differs from tested source')
    output = (PREFLIGHT/'unittest_output.txt').read_text()
    require('Ran 10 tests' in output and output.rstrip().endswith('OK'), 'Ten passing fixture tests required')
    old_root = Path(manifest['source_root'])
    require(sha(old_root/'artifact_hashes.json') == manifest['old_inventory_sha256'], 'Old evidence identity differs')
    counts['old_artifact_files'] = verify_inventory(old_root)
    old_manifest = read(old_root/'manifest.json')
    require(old_manifest['status'] == 'complete', 'Original physical batch incomplete')
    verify_sources(old_root, old_manifest['source_sha256'], old_manifest['source_archive_sha256'])
    require(manifest['stages'] == STAGES and manifest['primary_representation'] == 'observed_mesh + explicitly inferred_mesh', 'Fixed stage/representation contract changed')
    require(manifest['training'] is False and manifest['new_physical_actions'] == manifest['new_sensor_frames'] == 0, 'New acquisition or training is outside scope')
    require(verification['saved_frames_processed'] == 194 and verification['stage_meshes_verified'] == 18
        and verification['stage_results_verified'] == 6, 'Verification counts differ')
    require(all(verification[key] == 0 for key in ('new_physical_actions', 'new_sensor_frames', 'new_tsdf_fusions')), 'Verification added physical/sensor/fusion work')
    provenance = dict(**counts, frozen_old_sources=len(old_manifest['source_sha256']),
        frozen_new_sources=len(manifest['sources']), preflight_tests_passed=10,
        input_root=str(source), input_inventory_sha256=sha(source/'artifact_hashes.json'),
        input_manifest_sha256=sha(source/'manifest.json'), input_result_sha256=sha(source/'result.json'),
        verification_sha256=sha(source/'verification.json'), preflight_inventory_sha256=sha(PREFLIGHT/'artifact_hashes.json'),
        old_root=str(old_root), old_inventory_sha256=sha(old_root/'artifact_hashes.json'),
        processing_pid=timing['process_id'], verification_pid=verification['verification_process_id'])
    return manifest, old_root, old_manifest, provenance


def load_meshes(source, row):
    arrays = {}
    for kind in MESHES:
        path = source/'meshes'/f'{row["kind"]}_{row["stage"]}_{kind}.npz'
        with np.load(path, allow_pickle=False) as data:
            require(set(data.files) == {'vertices', 'triangles'}, 'Mesh array fields differ')
            vertices, triangles = data['vertices'].copy(), data['triangles'].copy()
        require(vertices.ndim == triangles.ndim == 2 and vertices.shape[1] == triangles.shape[1] == 3, 'Invalid mesh array shape')
        require(np.isfinite(vertices).all() and triangles.dtype.kind in 'iu', 'Invalid mesh coordinates/indices')
        require(not len(triangles) or triangles.min() >= 0 and triangles.max() < len(vertices), 'Mesh index outside vertices')
        require({key: array_hash(value) for key, value in [('vertices', vertices), ('triangles', triangles)]} == row['meshes'][kind], 'Saved geometry SHA differs')
        arrays[kind] = vertices, triangles
    observed, inferred, completed = [arrays[k] for k in MESHES]
    expected_vertices = np.concatenate([observed[0], inferred[0]])
    expected_triangles = np.concatenate([observed[1], inferred[1]+len(observed[0])])
    require(np.array_equal(completed[0], expected_vertices) and np.array_equal(completed[1], expected_triangles), 'Primary mesh is not the declared observed+inferred sum')
    require(row['completion']['accepted'] == bool(len(inferred[1])), 'Inferred source and acceptance receipt differ')
    if not len(inferred[1]): require(row['primary_outline'] == row['observed_outline'], 'Empty inference changed the primary metric')
    return {key: dict(vertices=len(v), triangles=len(t)) for key, (v, t) in arrays.items()}


def analyze(source):
    manifest, old_root, old_manifest, provenance = verify_inputs(source)
    result = read(source/'result.json')
    require(result['status'] == 'complete' and result['version'] == manifest['version'], 'Development result incomplete/version mismatch')
    require(result['saved_frames_processed'] == 194, 'Two saved 97-frame histories required')
    require(all(result[k] == 0 for k in ('new_physical_actions', 'new_sensor_frames', 'new_tsdf_fusions')), 'Unexpected new action/frame/fusion')
    require(result['semantic_policy'] is False and result['full_architecture_efficacy_proven'] is False
        and result['source_data_previously_viewed'] is True, 'Development capability scope changed')
    rows = result['stage_results']
    require([(r['kind'], r['stage'], r['action_id']) for r in rows]
        == [(kind, stage, action) for kind in ('simple', 'complex') for stage, action in STAGES.items()], 'Exactly six declared stages required')
    require(len(result['seed_receipts']) == 2 and {s['kind'] for s in result['seed_receipts']} == {'simple', 'complex'}, 'One seed per shape required')
    seeds = result['seed_receipts']
    require(seeds[0]['action_id'] == seeds[1]['action_id'] and seeds[0]['observed_seed_xyz'] == seeds[1]['observed_seed_xyz'], 'Initial geometry association differs across kinds')
    stages, projections, supports, old_cases = [], [], [], {}
    for case in old_manifest['cases']:
        path = old_root/f'case_{case["index"]:02d}'
        original, receipt = read(path/'result.json'), read(path/'verification.json')
        require(original['status'] == 'complete' and original['paid_actions'] == 96 and original['raw_frames'] == 97, 'Old case counts/status differ')
        require(receipt['status'] == 'passed' and receipt['independent_process'] and receipt['physical_process_id'] != receipt['replay_process_id']
            and receipt['raw_packets_verified'] == 97 and receipt['paid_actions'] == 96, 'Original physical replay receipt differs')
        old_cases[case['kind']] = case, original
    for row in rows:
        case, old = old_cases[row['kind']]
        old_stage = next(r for r in old['checkpoints'] if r['action_id'] == row['action_id'])
        require(old_stage['stage'] == row['stage'] and old_stage['outline'] == row['raw_outline'], 'Old raw metric was changed')
        mesh_counts = load_meshes(source, row)
        returned = old['trace'][row['action_id']]['pose'] == old['trace'][0]['pose']
        coverage = old['trace'][row['action_id']]['coverage_2d']
        surface = old_stage['surface']; validate_surface(surface, coverage, returned, 0, False)
        for label, key in REPRESENTATIONS.items():
            metric = row[key]
            require(metric['reference_signature'] == case['outline_reference_signature'], 'Evaluation reference changed')
            validate_outline(metric, coverage=coverage, returned=returned, collisions=0, failed=False, paid_actions=row['action_id'])
            instance = metric['instances'][0]; dims = instance['dimensions']
            errors = dims['absolute_size_error_xyz_m']
            record = dict(kind=row['kind'], stage=row['stage'], action_id=row['action_id'], representation=label,
                Q=metric['05cm']['outline_macro_quality'], J=metric['05cm']['joint_outline'], C2D=coverage,
                boundary_F1=metric['05cm']['outline_macro_f1'], mean_projection_IoU=sum(p['iou'] for p in instance['projections'].values())/3,
                completed=instance['completed'], completion_failure_reasons=completion_reasons(instance),
                eligible=metric['eligible'], returned=returned, max_size_error_m=max(errors) if errors else None,
                center_error_inf_m=dims['center_error_inf_m'],
                max_hausdorff_upper_m=max(p['hausdorff_upper_m'] for p in instance['projections'].values()) if not instance['missing'] else None)
            for axis, value in zip('xyz', errors or [None]*3): record['size_error_'+axis+'_m'] = value
            if label == 'raw':
                external = surface['instances'][0]
                record['raw_external_F1_05cm'] = external['05cm']['f1']
                for direction in ('accuracy', 'completeness'):
                    for stat in ('mean_m', 'rmse_m', 'p95_m'):
                        record['raw_external_'+direction+'_'+stat] = external[direction][stat]
            stages.append(record)
            for plane, projection in instance['projections'].items():
                for tag in ('02cm', '05cm', '10cm'):
                    projections.append(dict(kind=row['kind'], stage=row['stage'], representation=label, plane=plane, threshold=tag,
                        **projection[tag], iou=projection['iou'], quality=min(projection[tag]['f1'], projection['iou']),
                        boundary_symmetric_mean_m=projection['boundary_symmetric_mean_m'],
                        hausdorff_lower_m=projection['hausdorff_lower_m'], hausdorff_upper_m=projection['hausdorff_upper_m']))
        audit = row['completion']['support_audit']; ground = row['ground']
        require(ground is not None and ground['observation_count'] == row['action_id']+1, 'Ground history count differs')
        normal = np.asarray(ground['normal']); close(float(normal@normal), 1., 'Ground normal unit length')
        require(sum(row['geometry_arrays'][k]['count'] for k in ('ground_points_xyz', 'cleaned_points_xyz', 'unassigned_points_xyz'))
            == row['geometry_arrays']['measured_points_xyz']['count'], 'Evidence partitions do not cover measured points')
        supports.append(dict(kind=row['kind'], stage=row['stage'], accepted=row['completion']['accepted'],
            reason=row['completion']['reason'], free_ray_conflicts=row['completion']['free_ray_conflicts'],
            free_ray_check_reached='free_ray_box_hits' in audit, ground_plane=ground,
            background_bracketed_edge_counts=audit.get('background_bracketed_edge_counts'),
            orthogonal_face_support=audit.get('orthogonal_face_support'), ground_contact_assumed=audit['ground_contact_assumed'],
            geometry_arrays=row['geometry_arrays'], mesh_counts=mesh_counts))
    coarse = [r for r in rows if r['stage'] == 'coarse']
    same = all(coarse[0][key] == coarse[1][key] for key in ('geometry_arrays', 'meshes', 'completion', 'ground'))
    require(same and result['coarse_nonsemantic_geometry_identical'], 'Coarse paired geometry differs')
    by = {(r['kind'], r['stage'], r['representation']): r for r in stages}
    deltas = []
    for kind in ('simple', 'complex'):
        for label in REPRESENTATIONS:
            coarse, extra, returned = [by[kind, s, label] for s in STAGES]
            deltas.append(dict(kind=kind, representation=label, coarse_to_extra_Q=extra['Q']-coarse['Q'],
                extra_to_returned_Q=returned['Q']-extra['Q'], returned_Q=returned['Q'],
                coarse_completed=coarse['completed'], extra_completed=extra['completed'], returned_completed=returned['completed']))
    primary = {r['kind']: r for r in deltas if r['representation'] == 'primary'}
    provenance.update(saved_output_meshes_verified=18, stage_metric_records_verified=18,
        saved_frames_processed=194, saved_frames_reprocessed_for_verification=194,
        new_physical_actions=0, new_sensor_frames=0, new_tsdf_fusions=0,
        original_physical_paid_actions=192, original_physical_replay_paid_actions=192)
    summary = dict(status='complete', scope='previously viewed saved-depth development; no independent new scene or policy experiment',
        primary_representation=manifest['primary_representation'], stage_results=stages, support_audits=supports,
        deltas=deltas, seed_receipts=seeds, coarse_geometry_pair_identical=same,
        inference_accepted_stages=sum(s['accepted'] for s in supports),
        primary_completed_stages=sum(r['completed'] for r in stages if r['representation']=='primary'),
        simple_primary_coarse_completed=primary['simple']['coarse_completed'],
        complex_primary_extra_increment_positive_and_larger=(primary['complex']['coarse_to_extra_Q'] > 0
            and primary['complex']['coarse_to_extra_Q'] > primary['simple']['coarse_to_extra_Q']),
        semantic_value_proven=False, full_architecture_efficacy_proven=False,
        new_physical_actions=0, new_sensor_frames=0, new_tsdf_fusions=0,
        evidence_provenance=provenance)
    return summary, stages, projections, supports, deltas


def csv_rows(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def plot(output, stages):
    os.environ['MPLCONFIGDIR'] = str(output/'.mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [(kind, stage) for kind in ('simple', 'complex') for stage in STAGES]
    by = {(r['kind'], r['stage'], r['representation']): r for r in stages}
    same = all(by[k,s,'observed']['Q'] == by[k,s,'primary']['Q'] for k,s in labels)
    representations = ['raw', 'primary'] if same else ['raw', 'observed', 'primary']
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = np.arange(len(labels)); width = .7/len(representations)
    for index, rep in enumerate(representations):
        offset = (index-(len(representations)-1)/2)*width
        title = 'Raw TSDF' if rep == 'raw' else 'Cleaned observed = primary' if same else rep
        for axis, field in zip(axes, ['Q', 'max_size_error_m']):
            values = [by[k,s,rep][field] for k,s in labels]
            axis.bar(x+offset, values, width=width, label=title)
            axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Outline quality Q'); axes[0].set_ylim(0, 1.)
    axes[0].legend(loc='upper left', fontsize=9)
    axes[1].set_ylabel('Maximum size error (m)')
    axes[1].axhline(.05, linestyle='--', color='black', linewidth=1, label='5 cm size condition')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].set_xticks(x, [f'{k}\n{s} ({STAGES[s]})' for k,s in labels])
    fig.suptitle('V24: fixed saved-depth development stages; no new sensing or policy')
    fig.tight_layout(); fig.savefig(output/'raw_and_common_shape.png', dpi=140); plt.close(fig)
    shutil.rmtree(output/'.mpl-cache', ignore_errors=True)


def report_text(summary, output):
    by = {(r['kind'], r['stage'], r['representation']):r for r in summary['stage_results']}
    lines = ['# V24 保存深度共同外形后端开发结果', '',
        '日期：2026-09-15。结果仅属于已看过的 V23 保存数据开发，不是新场景确认，也没有验证语义策略。', '',
        f"主运行处理194帧；另一进程重新处理194帧并通过18个网格、6个阶段结果的精确复核。新增动作、传感帧、TSDF融合均为0。原V23的192次物理动作与192次物理回放动作仅是旧数据来源，不能重复计作新实验。独立解析夹具10项通过，曾发现的平台当地面及伪三角形质心绕过问题和修正记录仍保留。", '',
        f"本版预定主表示为清理实测网格加显式推断网格。实际{summary['inference_accepted_stages']}/6阶段接受补全，因此本次主表示与清理实测表示完全相同；新分数不来自规则外壳补全。{summary['primary_completed_stages']}/6阶段通过完整外形验收。拒绝、未完成和原始TSDF分数均保留。", '',
        '| 设施 / 阶段（旧付费动作） | 原始Q | 主Q | ΔQ（清理−原始） | 最大尺寸误差 原始→主（米） | 主完成 |',
        '|---|---:|---:|---:|---:|---|']
    for kind in ('simple', 'complex'):
        for stage, action in STAGES.items():
            raw, primary = by[kind,stage,'raw'], by[kind,stage,'primary']
            lines.append(f"| {kind} / {stage} ({action}) | {raw['Q']:.9f} | {primary['Q']:.9f} | {primary['Q']-raw['Q']:+.9f} | {raw['max_size_error_m']:.6f} → {primary['max_size_error_m']:.6f} | {primary['completed']} |")
    lines += ['', '所有阶段均因有效背景射线不足而拒绝规则外壳：顶部端点 `2_max` 的计数为0，粗扫还缺少 `0_min`。这表示当前判据没有获得所需边缘证据，不表示设施没有顶面；不能把尚未执行的自由射线冲突检查解释为已证明无冲突。各阶段具体支撑、地面拟合和失败条目见CSV/JSON。', '',
        '地面分离、实测点连通关联、顶点支撑和短三角形约束同时作用，当前对照只验证这一共同清理组合。尺寸极值和Q改善不能被单独归因为某一种清理步骤；没有进行消融。原始外表面F1及双向连续误差从V23原记录逐项核对并旁列，新清理表示未重新评价完整外表面，不把外形Q称作完整三维精度。', '',
        '两类在粗扫23步的种子、实测点分区、地面、网格和补全决定完全相同，显示本次共同后端没有因类别变化改变几何。评价参考不同，因此Q并不相同；C仍沿用各自V23真可达区域的分母，不能把两类J差全部归因于几何质量。RGB人工标记只提供共同实例关联，尚非自然实例分割或开放词汇识别。', '',
        '73步属于尚未返航的诊断阶段；任务终点为96步。两类Q在73→96步均下降，更多观测并不保证该外形指标逐帧单调上升。简单设施在23步未完成必须保留，但这项严格工程条件不是语义有用的数学必要条件。当前没有两个设施之间的同预算选臂竞争，不能从类型响应差异推出语义收益或四模块优势。', '',
        '下一步应按另行冻结的双对象共同覆盖与选择协议检查几何揭示前的信息、相同预算下的机会成本及强几何对照，保留自然负例；本报告不改变本次算法、门槛或旧结果。', '',
        f"正式结果：[result.json]({output/'result.json'})；阶段与尺寸：[stage_metrics.csv]({output/'stage_metrics.csv'})；逐投影：[projection_metrics.csv]({output/'projection_metrics.csv'})；支撑及拒绝：[support_audits.csv]({output/'support_audits.csv'})。", '',
        f"![固定阶段外形质量与尺寸误差]({output/'raw_and_common_shape.png'})", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(); source, output, report = (p.resolve() for p in (args.source,args.output,args.report))
    require(not output.exists() and not report.exists(), 'New output/report paths required; old evidence is never overwritten')
    dependencies = [Path(__file__).resolve(), ROOT/'scripts/analyze_facility_shape_v23.py']
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in dependencies}
    summary, stages, projections, supports, deltas = analyze(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.v24-analysis-', dir=output.parent) as work:
        temp = Path(work)
        write_json(temp/'result.json', summary)
        for name, rows in [('stage_metrics.csv',stages),('projection_metrics.csv',projections),
                           ('support_audits.csv',supports),('interval_deltas.csv',deltas)]: csv_rows(temp/name, rows)
        plot(temp, stages)
        require(source_hashes == {str(p.relative_to(ROOT)):sha(p) for p in dependencies}, 'Analysis source changed during execution')
        write_json(temp/'source_sha256.json', source_hashes)
        with zipfile.ZipFile(temp/'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in dependencies: archive.write(path,str(path.relative_to(ROOT)))
        text = report_text(summary, output)
        (temp/'report.md').write_text(text)
        write_json(temp/'artifact_hashes.json', {p.name:sha(p) for p in sorted(temp.iterdir()) if p.is_file()})
        os.replace(temp, output)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as stream: stream.write(text)
    require(sha(report) == sha(output/'report.md'), 'Published report differs from archived report')
    print(json.dumps(dict(status='complete', output=str(output), report=str(report),
        accepted_stages=summary['inference_accepted_stages'], completed_stages=summary['primary_completed_stages'],
        paired=summary['coarse_geometry_pair_identical'], result_sha256=sha(output/'result.json')), indent=2))


if __name__ == '__main__': main()
