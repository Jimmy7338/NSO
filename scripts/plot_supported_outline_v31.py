#!/usr/bin/env python3
"""Plot sealed V31 applicability and saved-mesh scores without re-evaluation."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zipfile

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ANALYTIC = ROOT/'audit_results/v31_supported_outline_20260917'
SAVED = ROOT/'audit_results/v31_saved_outline_diagnostic_20260917'
OUTPUT = ROOT/'audit_results/v31_supported_outline_figures_20260917'
FIGURE = ROOT/'docs/research/figures/v31_supported_outline_20260917'
CAP, RESERVE = 1024**2, 64*1024**2
PANELS = [('box_closed', 'Closed box'), ('box_vertical', 'All four vertical faces'),
          ('box_vertical_missing_side', 'One vertical face missing'),
          ('l_vertical', 'Concave L: vertical faces'),
          ('l_convex_shortcut', 'Wrong convex shortcut for L'),
          ('box_extra_open_strip', 'Box + detached open strip')]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pack(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()


def verified_package(folder):
    manifest = read(folder/'manifest.json')
    inventory = read(folder/'artifact_hashes.json')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p.name != 'artifact_hashes.json'}
    require(actual == set(inventory), 'Sealed file set differs: '+str(folder))
    for name, expected in inventory.items():
        require(sha(folder/name) == expected, 'Sealed artifact changed: '+name)
    for name, expected in manifest['source_sha256'].items():
        require(sha(ROOT/name) == expected, 'Frozen source changed: '+name)
    require(sha(folder/'sources.zip') == manifest['source_archive_sha256'], 'Source ZIP changed')
    with zipfile.ZipFile(folder/'sources.zip') as archive:
        require(set(archive.namelist()) == set(manifest['source_sha256']), 'Source ZIP file set differs')
        for name, expected in manifest['source_sha256'].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, 'ZIP source bytes changed')
    for name, expected in manifest.get('input_sha256', {}).items():
        require(sha(ROOT/name) == expected, 'Frozen diagnostic input changed: '+name)
    return manifest, dict(root=str(folder), manifest_sha256=sha(folder/'manifest.json'),
        inventory_sha256=sha(folder/'artifact_hashes.json'), result_sha256=sha(folder/'result.json'),
        scores_sha256=sha(folder/'scores.json'), source_count=len(manifest['source_sha256']),
        artifact_count=len(inventory), input_count=len(manifest.get('input_sha256', {})))


def quality(result):
    q = float(result['05cm']['outline_macro_quality'])
    require(math.isfinite(q) and 0 <= q <= 1, 'Saved Q outside fixed plotting range')
    return q


def load_sealed():
    # Require both terminal result files before constructing any illustration.
    analytic, saved = read(ANALYTIC/'result.json'), read(SAVED/'result.json')
    require(analytic['status'] == 'passed' and analytic['all_gates_passed'], 'Analytic batch not passed/sealed')
    require(saved['status'] == 'complete', 'Wait for sealed saved-mesh diagnostic')
    _, source_a = verified_package(ANALYTIC)
    _, source_s = verified_package(SAVED)
    scores, saved_scores = read(ANALYTIC/'scores.json'), read(SAVED/'scores.json')
    require(len(scores) == 26 and len(saved_scores) == 4, 'Incomplete declared score tables')
    require(analytic['counts']['candidate_evaluations'] == 28
            and analytic['counts']['legacy_evaluations'] == 26
            and analytic['counts']['expected_invalid_rejections'] == 2, 'Analytic call count mismatch')
    require(saved['counts']['candidate_evaluations'] == 8
            and saved['counts']['old_quality_evaluations'] == 0, 'Saved diagnostic call count mismatch')
    require(all(analytic['counts'][key] == saved['counts'][key] == 0
                for key in ('worlds', 'sensor_packets', 'mapper_updates', 'TSDF_integrations', 'new_main_tasks')),
            'Unexpected physical work in offline source audits')
    require(not saved['observed_geometry_changed'] and not saved['semantic_efficacy_proven'], 'Diagnostic scope mismatch')
    analytic_rows = []
    for name, row in scores.items():
        old, new = quality(row['legacy_full_instance']), quality(row['candidate'])
        require(old == analytic['legacy_quality05'][name] and new == analytic['quality05'][name],
                'Analytic summary alias differs')
        analytic_rows.append(dict(fixture=name, legacy_full_Q05=old, candidate_Q05=new, delta_Q05=new-old))
    # Pure frozen fixture arrays, with byte-for-byte provenance. This is display
    # construction only: no rays, projections, score kernels or saved NPZ loads.
    from nso.outline_fixtures_v31 import geometry_fixtures_v31, reference_meshes_v31
    fixtures, references = geometry_fixtures_v31(), reference_meshes_v31()
    for name, _ in PANELS:
        mesh = fixtures[name].prediction
        hashes = {key: hashlib.sha256(getattr(mesh, key).tobytes()).hexdigest()
                  for key in ('vertices', 'triangles')}
        require(scores[name]['input_mesh_hashes'] == [hashes], 'Illustrated fixture differs from evaluated input')
    saved_rows = []
    for index, row in enumerate(saved_scores):
        case = row['case']
        require(case == dict(index=index, hypothesis=index//2, arm='AB'[index % 2]), 'Saved case identity differs')
        for stage in ('prefix', 'final'):
            item = row['stages'][stage]
            old, new = item['original_full_instance'], item['candidate']
            c = float(new['coverage_2d'])
            require(math.isfinite(c) and 0 <= c <= 1 and c == old['coverage_2d'], 'Coverage changed')
            require(item['coverage_and_physical_qualification_inherited'], 'Physical fields not inherited')
            require(new['eligible'] == old['eligible'], 'Qualification changed in representation diagnostic')
            saved_rows.append(dict(case=index, hypothesis=case['hypothesis'], arm=case['arm'], stage=stage,
                C=c, legacy_full_Q05=quality(old), candidate_Q05=quality(new),
                legacy_eligible=old['eligible'], candidate_eligible=new['eligible']))
        final = saved_rows[-1]
        require(final['candidate_Q05'] == saved['final_candidate_Q05'][index]
                and final['legacy_full_Q05'] == saved['final_original_full_Q05'][index]
                and final['candidate_eligible'] == saved['final_candidate_eligible'][index],
                'Saved diagnostic summary alias mismatch')
    return fixtures, references, scores, analytic_rows, saved_rows, dict(
        analytic=source_a, saved_diagnostic=source_s, script_sha256=sha(Path(__file__)),
        fixture_arrays_equal_formal_scoring_inputs=True, analytic_gates_passed=sum(analytic['gates'].values()),
        new_physical_actions=0, new_sensor_frames=0, new_mapper_updates=0, new_tsdf_fusions=0,
        new_metric_evaluations=0, saved_npz_geometry_loaded=False,
        geometry_display='Frozen analytic fixture arrays only, not a new visibility or scoring query',
        standalone_main_tasks_created=0)


def render(fixtures, references, scores, saved_rows):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 8, 'svg.fonttype': 'none', 'svg.hashsalt': 'v31-supported-outline',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(12.2, 11.3), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1., 1., .95])
    blue, orange, gray = '#2874a6', '#b34d20', '#707b7c'
    for index, (name, label) in enumerate(PANELS):
        axis = fig.add_subplot(grid[index//3, index % 3], projection='3d')
        item = fixtures[name]
        mesh, reference = item.prediction, references[item.reference_name]
        negative = name in ('box_vertical_missing_side', 'l_convex_shortcut', 'box_extra_open_strip')
        color = orange if negative else blue
        axis.add_collection3d(Poly3DCollection(mesh.vertices[mesh.triangles],
            facecolors=color, edgecolors=color, linewidths=.65, alpha=.28))
        # Dashed reference wire is contextual ground truth in the illustration,
        # never prediction completion; it is intentionally visually distinct.
        edges = {tuple(sorted((int(a), int(b)))) for triangle in reference.triangles
                 for a, b in zip(triangle, np.roll(triangle, -1))}
        axis.add_collection3d(Line3DCollection([reference.vertices[list(edge)] for edge in sorted(edges)],
            colors='.38', linewidths=.55, linestyles=':', alpha=.4))
        old, new = quality(scores[name]['legacy_full_instance']), quality(scores[name]['candidate'])
        axis.set_title(f'({chr(97+index)}) {label}\nOld Q={old:.3f}   Candidate Q={new:.3f}', fontsize=9, pad=3)
        axis.set(xlim=(-.1, 3.5), ylim=(-.1, 3.3), zlim=(-.1, 1.8),
                 xticks=[0, 1, 2, 3], yticks=[0, 1, 2, 3], zticks=[0, .8, 1.6],
                 xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
        axis.set_box_aspect((3.6, 3.4, 1.9)); axis.view_init(elev=26, azim=42)
        axis.tick_params(labelsize=7, pad=0)
        axis.xaxis.labelpad = axis.yaxis.labelpad = axis.zaxis.labelpad = 0
    sensitive = fig.add_subplot(grid[2, 0])
    names = ['box_vertical', 'one_mm_gap']
    values = [quality(scores[n]['candidate']) for n in names]
    sensitive.bar([0, 1], values, width=.55, color=[blue, orange], alpha=.8)
    old = [quality(scores[n]['legacy_full_instance']) for n in names]
    sensitive.scatter([0, 1], old, marker='_', s=240, linewidths=2.3, color='black', label='Old full-instance Q', zorder=5)
    for x, value in enumerate(values):
        sensitive.text(x, value+.025, f'{value:.6f}', ha='center', fontsize=9)
    sensitive.set(xticks=[0, 1], xticklabels=['Exact closed\nvertical loop', 'Same loop with\n1 mm open gap'],
                  ylim=(0, 1.1), ylabel='Candidate Q5', title='(g) Strict closure is NOT noise robustness')
    sensitive.text(.5, .08, 'No gap repair / tolerance\n1 mm gap: 1 → 2/3', ha='center',
                   transform=sensitive.transAxes, fontsize=8, color=orange,
                   bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
    sensitive.legend(loc='upper right', fontsize=7)
    sensitive.grid(axis='y', alpha=.2)
    qaxis, caxis = fig.add_subplot(grid[2, 1]), fig.add_subplot(grid[2, 2])
    final = [row for row in saved_rows if row['stage'] == 'final']
    labels = [f'H{row["hypothesis"]} / {row["arm"]}' for row in final]
    for index, row in enumerate(final):
        old, new = row['legacy_full_Q05'], row['candidate_Q05']
        qaxis.plot([old, new], [index, index], color=gray, linewidth=1.)
        qaxis.scatter(old, index, marker='o', facecolors='white', edgecolors=gray, s=48, zorder=4)
        qaxis.scatter(new, index, marker='D', color=blue, s=25, zorder=5)
        qaxis.text(.03, index-.23, f'Δ={new-old:+.3g}', color='.3', fontsize=7)
        caxis.barh(index, row['C'], color=orange if not row['candidate_eligible'] else blue, height=.45, alpha=.75)
        caxis.text(.025, index, f'C={row["C"]:.4f}; '+('PASS' if row['candidate_eligible'] else 'FAIL'),
                   va='center', color='black', fontsize=8)
    for axis in (qaxis, caxis):
        axis.set(xlim=(0, 1), ylim=(3.6, -.6), yticks=range(4), yticklabels=labels)
        axis.grid(axis='x', alpha=.2)
    qaxis.set(xlabel='Final Q5 (0 to 1)', title='(h) Same saved meshes: posthoc diagnostic')
    qaxis.legend(handles=[Line2D([], [], marker='o', color=gray, markerfacecolor='white', linestyle='', label='Old full instance'),
        Line2D([], [], marker='D', color=blue, linestyle='', label='V31 candidate')], loc='lower right', fontsize=7)
    caxis.axvline(.8, color='black', linestyle=':', linewidth=1.2, label='C ≥ 0.80 required')
    caxis.set(xlabel='Inherited 2D coverage C (0 to 1)', title='(i) Physical qualification is unchanged')
    caxis.legend(loc='lower right', fontsize=7)
    eligible = sum(bool(row['candidate_eligible']) for row in final)
    fig.suptitle('V31 supported-outline representation: analytic applicability, not mapping or semantic efficacy\n'
        'Colored triangles = submitted fixture; dotted wire = reference. All scores are saved, not recomputed.\n'
        f'V30 fixed-route records: {eligible}/4 final tasks qualified; a new score cannot repair coverage failure.', fontsize=11)
    outputs = {}
    for extension in ('png', 'svg'):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=extension, dpi=145,
            metadata={'Description': 'Sealed analytic fixtures and posthoc saved-mesh diagnostic; no new evaluation'}
            if extension == 'svg' else None)
        outputs[extension] = buffer.getvalue()
    plt.close(fig)
    return outputs


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Only after both source batches are sealed')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return
    require(not OUTPUT.exists() and all(not FIGURE.with_suffix('.'+ext).exists() for ext in ('png', 'svg')),
            'Never overwrite existing figure evidence')
    require(shutil.disk_usage(ROOT).free >= RESERVE+CAP, 'Need 64 MiB reserve plus figure allowance')
    fixtures, references, scores, analytic_rows, saved_rows, receipt = load_sealed()
    rendered = render(fixtures, references, scores, saved_rows)
    receipt.update(status='complete_read_only_figure', illustrated_fixture_names=[n for n, _ in PANELS],
        final_qualified_count=sum(bool(r['candidate_eligible']) for r in saved_rows if r['stage'] == 'final'),
        one_mm_gap_candidate_Q05=quality(scores['one_mm_gap']['candidate']),
        exactly_closed_vertical_candidate_Q05=quality(scores['box_vertical']['candidate']),
        noise_robustness_proven=False, semantic_efficacy_proven=False, score_axis_range=[0, 1],
        prior_metrics_and_geometry_unchanged=True)
    description = ('# V31 解析表示适用性与保存网格诊断图\n\n'
        '前六格显示冻结解析夹具的真实三角形数组（已与正式评分输入逐数组 SHA 匹配）；'
        '灰色点线是参考几何，仅用于图示，不是补全后的预测。顶底缺失、整侧缺失、L 凹口、'
        '凸包捷径及分离薄片明确区分。没有读取实测 NPZ 数组来重建图，也没有调用评价器。\n\n'
        '旧分数为 V30 完整实例 Q5，新分数为 V31 候选表示 Q5，均直接来自已封存 scores.json；'
        '不是窗口 Q，也不是三维表面精度。17 个基本夹具之外的正式解析案例仍全部列在 CSV，'
        '图示只是事先指定的代表性反例，不以图中六个替代完整 26 个合法案例与 2 个非法输入拒绝。\n\n'
        '**严格闭环敏感性：完整四竖面线环的候选 Q=1；仅 1 mm 的真实缺口使 Q=2/3。**'
        '这不是噪声鲁棒性证明。算法没有自动补缝，解析门通过也不能推出稀疏或带噪 TSDF 会形成可靠闭环。'
        '第 g 格固定从 0 展示，1 mm 缺口分数及旧表示分数均保留。\n\n'
        '第 h 格仅用四条旧 V30 固定路线的终点保存网格结果对照旧完整实例与新候选；'
        '没有新增观察、融合、预测修复或原实验任务。它是看过旧结果后的开发诊断，不能当预注册的新性能结果。'
        '第 i 格沿用原 C，并展示 0.80 门；C 不因外形表示改变。所有失败保留，资格还包含返航、碰撞、失败和预算。'
        '两阶段 prefix/final 的全部 C/Q 与资格另存 saved_stages.csv。\n\n'
        'PNG/SVG 位于 docs/research/figures/v31_supported_outline_20260917.*；'
        '本目录 result.json 记录源清单、代码和输入哈希验证及零新增评价/物理调用边界。'
        '图不证明自主 ANS、语义网络、完整架构优势或真实机器人上的建图效果。\n')
    payloads = {FIGURE.with_suffix('.'+ext): value for ext, value in rendered.items()}
    payloads.update({OUTPUT/'result.json': pack(receipt), OUTPUT/'README.md': description.encode(),
                     OUTPUT/'analytic_scores.csv': csv_bytes(analytic_rows),
                     OUTPUT/'saved_stages.csv': csv_bytes(saved_rows)})
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(value).hexdigest() for p, value in payloads.items()}
    payloads[OUTPUT/'artifact_hashes.json'] = pack(hashes)
    size = sum(map(len, payloads.values()))
    require(size < CAP and shutil.disk_usage(ROOT).free-size >= RESERVE, 'Combined figure/audit exceeds allowance')
    for field, folder in (('analytic', ANALYTIC), ('saved_diagnostic', SAVED)):
        for filename, key in (('manifest.json', 'manifest_sha256'), ('artifact_hashes.json', 'inventory_sha256'),
                              ('result.json', 'result_sha256'), ('scores.json', 'scores_sha256')):
            require(sha(folder/filename) == receipt[field][key], 'Source changed during figure creation')
    OUTPUT.mkdir()
    try:
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        for path, value in payloads.items():
            require(shutil.disk_usage(ROOT).free-len(value) >= RESERVE, 'Disk reserve changed')
            temporary = path.with_name(path.name+'.tmp')
            with temporary.open('xb') as stream:
                stream.write(value)
            os.replace(temporary, path)
        for name, expected in hashes.items():
            require(sha(ROOT/name) == expected, 'Saved figure hash differs')
    except BaseException as error:
        failure = pack(dict(status='failed', error=repr(error)))
        if shutil.disk_usage(ROOT).free-len(failure) >= RESERVE:
            (OUTPUT/'failure.json').write_bytes(failure)
        raise
    print(json.dumps(dict(status=receipt['status'], output=str(OUTPUT), combined_bytes=size)))


if __name__ == '__main__':
    main()
