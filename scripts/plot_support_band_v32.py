#!/usr/bin/env python3
"""One figure from sealed V32 numbers; no geometry reconstruction or evaluation."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# These are pure file/hash/serialization helpers; importing this plotting module
# does not import a simulator, mapper or score kernel or execute its main().
from scripts.plot_supported_outline_v31 import verified_package, read, sha, require, pack, csv_bytes

ANALYTIC = ROOT/'audit_results/v32_support_band_20260917'
SAVED = ROOT/'audit_results/v32_saved_support_band_20260917'
OUTPUT = ROOT/'audit_results/v32_support_band_figures_20260917'
FIGURE = ROOT/'docs/research/figures/v32_support_band_20260917'
HELPER = ROOT/'scripts/plot_supported_outline_v31.py'
CAP, RESERVE = 1024**2, 64*1024**2
GAPS = (0, 1, 5, 10, 20, 50, 100, 300)
NEGATIVES = ('one_vertical_side_missing', 'neighbors_gap_200mm_wrong_bridge',
             'anchor_l_notch_filled', 'anchor_l_convex_shortcut')
AMBIGUITY = ('ambiguity_closed_solid_missing_corner', 'ambiguity_true_open_shell_same_observation')


def unit(value):
    value = float(value)
    require(math.isfinite(value) and 0 <= value <= 1, 'Saved quantity outside [0,1]')
    return value


def load_sealed():
    analytic, saved = read(ANALYTIC/'result.json'), read(SAVED/'result.json')
    require(analytic['status'] == 'passed' and analytic['all_gates_passed'], 'Analytic source not passed/sealed')
    require(saved['status'] == 'complete', 'Wait for the complete sealed saved-mesh diagnostic')
    _, source_a = verified_package(ANALYTIC)
    _, source_s = verified_package(SAVED)
    scores, saved_scores = read(ANALYTIC/'scores.json'), read(SAVED/'scores.json')
    require(len(scores) == 56 and len(saved_scores) == 4, 'Incomplete declared tables')
    require(analytic['counts']['candidate_evaluations'] == 58
            and analytic['counts']['expected_invalid_rejections'] == 2
            and saved['counts']['candidate_evaluations'] == 8, 'Frozen evaluation count mismatch')
    for summary in (analytic, saved):
        require(all(summary['counts'][k] == 0 for k in ('standalone_legacy_evaluations', 'worlds',
            'sensor_packets', 'mapper_updates', 'TSDF_integrations', 'new_main_tasks')), 'Wrong diagnostic scope')
        require(not summary['candidate_ready_as_sole_training_target']
                and not summary['semantic_efficacy_proven'] and not summary['full_3d_accuracy_certified'],
                'Unexpected efficacy or completion assertion')
    require(all(analytic['ambiguity'].values()), 'Identical-input demonstration not verified')
    require(saved['physical_qualification_unchanged'] and not saved['observed_geometry_changed']
            and saved['legacy_region_diagnostic_matches_v31'], 'Saved geometry or historical diagnostic changed')
    analytic_rows = []
    for name, row in scores.items():
        candidate = row['candidate']
        require(not candidate['connectivity_certified'] and candidate['completion_fraction'] is None,
                'Support score must not certify physical completion')
        q = candidate['05cm']
        support, region, f1 = map(unit, (q['outline_macro_quality'], q['v31_region_outline_quality'], q['outline_macro_f1']))
        require(support == analytic['support_quality05'][name]
                and region == analytic['v31_region_quality05'][name], 'Analytic aliases differ')
        analytic_rows.append(dict(fixture=name, support_Q05=support, v31_region_Q05=region, boundary_F1_05=f1))
    require(scores[AMBIGUITY[0]]['input_mesh_hashes'] == scores[AMBIGUITY[1]]['input_mesh_hashes'],
            'Ambiguity examples no longer share identical prediction arrays')
    stage_rows = []
    for index, row in enumerate(saved_scores):
        require(row['case'] == dict(index=index, hypothesis=index//2, arm='AB'[index % 2]), 'Case matrix changed')
        for stage in ('prefix', 'final'):
            item, candidate = row['stages'][stage], row['stages'][stage]['candidate']
            c = unit(candidate['coverage_2d'])
            original = item['original_window']
            require(item['physical_qualification_inherited'] and c == original['coverage_2d']
                    and candidate['historical_task_eligible'] == original['eligible'], 'Physical fields changed')
            q = candidate['05cm']
            region = unit(item['original_v31']['05cm']['outline_macro_quality'])
            require(abs(region-q['v31_region_outline_quality']) < 1e-10, 'Old V31 region Q was changed')
            stage_rows.append(dict(case=index, hypothesis=index//2, arm='AB'[index % 2], stage=stage,
                paid_actions=18 if stage == 'prefix' else 48, C=c,
                support_Q05=unit(q['outline_macro_quality']), v31_region_Q05=region,
                support_J05=unit(q['joint_outline']), historical_eligible=original['eligible'],
                candidate_eligible=candidate['eligible']))
        final = stage_rows[-1]
        require(final['support_Q05'] == saved['final_support_Q05'][index]
                and final['v31_region_Q05'] == saved['final_v31_region_Q05'][index]
                and final['candidate_eligible'] == saved['final_candidate_eligible'][index], 'Final aliases differ')
    sensitivity = []
    for tag in ('02cm', '05cm', '10cm'):
        entry = saved['selection'][tag]
        sensitivity.append(dict(tolerance=tag,
            complex_support_Q_relative_to_best_fixed=entry['outline_macro_quality']['complex_relative'],
            complex_support_J_relative_to_best_fixed=entry['joint_outline']['complex_relative'],
            scope='All four physical tasks are retained; posthoc task function, not semantic efficacy'))
    receipt = dict(analytic=source_a, saved_diagnostic=source_s, script_sha256=sha(Path(__file__)),
        file_helper_sha256=sha(HELPER), analytic_passed_gate_count=sum(analytic['gates'].values()),
        source_legal_analytic_cases=56, source_invalid_input_rejections=2,
        new_physical_actions=0, new_sensor_frames=0, new_mapper_updates=0, new_tsdf_fusions=0,
        new_geometry_queries=0, new_metric_evaluations=0, saved_npz_geometry_loaded=False,
        different_task_functions=True, completion_or_topology_certified=False, semantic_efficacy_proven=False,
        candidate_ready_as_sole_training_target=False, no_threshold_selected_for_favorable_result=True)
    return {r['fixture']: r for r in analytic_rows}, analytic_rows, stage_rows, sensitivity, receipt


def render(data, stages, sensitivity):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'svg.fonttype': 'none', 'svg.hashsalt': 'v32-support-band',
                         'axes.spines.top': False, 'axes.spines.right': False})
    blue, orange, purple = '#2471a3', '#c36b2e', '#7d3c98'
    fig, axes = plt.subplots(2, 3, figsize=(14.1, 8.5), constrained_layout=True)
    colors = {'support_Q05': blue, 'v31_region_Q05': orange}
    labels = {'support_Q05': 'V32 support Q', 'v31_region_Q05': 'V31 region Q'}

    def pair(axis, x, names):
        for key, marker in (('v31_region_Q05', 's'), ('support_Q05', 'o')):
            axis.plot(x, [data[n][key] for n in names], marker=marker, markersize=4,
                color=colors[key], linewidth=1.4, label=labels[key])

    gap = axes[0, 0]
    names = [f'corner_gap_{mm:03d}mm' for mm in GAPS]
    pair(gap, range(len(GAPS)), names)
    gap.set(xticks=range(len(GAPS)), xticklabels=GAPS, xlabel='Gap width (mm; discrete tested cases)',
            title='(a) Gap tolerance changes the task function')
    gap.text(.03, .16, '300 mm gap still scores\n'+f'{data[names[-1]]["support_Q05"]:.6f}',
             transform=gap.transAxes, color=blue, fontsize=9)
    gap.legend(loc='lower right', fontsize=7.5)
    jitter = axes[0, 1]
    pair(jitter, [1, 5, 10], [f'independent_face_jitter_{mm:03d}mm' for mm in (1, 5, 10)])
    jitter.set(xticks=[1, 5, 10], xlabel='Per-coordinate displacement bound (mm)',
               title='(b) Independently shifted planar faces')
    jitter.text(.035, .14, 'One shared fixed random realization\nNot three statistical replications',
                transform=jitter.transAxes, fontsize=8)
    jitter.legend(loc='lower right', fontsize=7.5)
    threshold = axes[0, 2]
    names = [f'translate_x_{mm:03d}mm' for mm in (49, 50, 51)]
    pair(threshold, [49, 50, 51], names)
    threshold.plot([49, 50, 51], [data[n]['boundary_F1_05'] for n in names],
        color=purple, marker='^', linestyle=':', linewidth=1.5, label='Boundary F1 at 50 mm')
    threshold.axvline(50, linestyle=':', color='.55', linewidth=.8)
    threshold.set(xticks=[49, 50, 51], xlabel='Whole-mesh x translation (mm)',
                  title='(c) The hard F1 threshold remains')
    threshold.text(.035, .16, 'F1: 1.000 → '+f'{data[names[-1]]["boundary_F1_05"]:.6f}'
                   +'\nGlobal continuity is not claimed', transform=threshold.transAxes, fontsize=8)
    threshold.legend(loc='lower left', fontsize=7.5)
    negative = axes[1, 0]
    x = np.arange(len(NEGATIVES))
    for key, shift in (('v31_region_Q05', -.18), ('support_Q05', .18)):
        values = [data[n][key] for n in NEGATIVES]
        negative.bar(x+shift, values, width=.34, color=colors[key], label=labels[key], alpha=.85)
        if key == 'support_Q05':
            for xi, value in zip(x, values):
                negative.text(xi+.18, value+.025, f'{value:.3f}', ha='center', fontsize=7.5)
    negative.axhline(1., linestyle=':', linewidth=.8, color='.5')
    negative.set(xticks=x, xticklabels=['Missing\nwhole side', 'False bridge\n200 mm',
        'Filled\nL notch', 'Convex\nL shortcut'], title='(d) Penalized errors can still score highly')
    negative.text(.025, .14, 'Missing side: '+f'{data[NEGATIVES[0]]["support_Q05"]:.6f}'
                  +'\nHigh macro score ≠ completion', transform=negative.transAxes, fontsize=8)
    negative.legend(loc='lower right', fontsize=7.5)
    ambiguity = axes[1, 1]
    for key, shift in (('v31_region_Q05', -.18), ('support_Q05', .18)):
        values = [data[n][key] for n in AMBIGUITY]
        ambiguity.bar(np.arange(2)+shift, values, width=.34, color=colors[key], alpha=.85, label=labels[key])
        if key == 'support_Q05':
            for xi, value in enumerate(values):
                ambiguity.text(xi+.18, value+.025, f'{value:.6f}', ha='center', fontsize=8)
    ambiguity.set(xticks=[0, 1], xticklabels=['Closed-solid\nreference', 'Real 8 mm opening\nin thin-shell reference'],
                  title='(e) Same input mesh; different true enclosure')
    ambiguity.text(.03, .25, 'Prediction arrays are byte-identical\n'
                   'Watertight material ≠ sealed enclosure\nNo topology certification',
                   transform=ambiguity.transAxes, fontsize=8,
                   bbox=dict(facecolor='white', edgecolor='none', alpha=.88))
    ambiguity.legend(loc='lower right', fontsize=7.5)
    saved = axes[1, 2]
    finals = [r for r in stages if r['stage'] == 'final']
    x = np.arange(4)
    for key, shift in (('v31_region_Q05', -.18), ('support_Q05', .18)):
        saved.bar(x+shift, [r[key] for r in finals], width=.34, color=colors[key], alpha=.85, label=labels[key])
    saved.scatter(x, [r['C'] for r in finals], color='black', marker='D', s=22, label='Inherited coverage C', zorder=5)
    saved.axhline(.8, color='black', linewidth=.9, linestyle=':', label='C ≥ 0.80 required (C only)')
    saved.set(xticks=x, xticklabels=[f'H{r["hypothesis"]}/{r["arm"]}\nC={r["C"]:.4f}\n'
        + ('FAIL' if not r['historical_eligible'] else 'PASS') for r in finals],
        title='(f) Unchanged saved meshes; all C80 gates fail')
    saved.legend(loc='lower right', fontsize=7)
    for axis in axes.flat:
        axis.set_ylim(0, 1.12)
        axis.set_yticks([0, .2, .4, .6, .8, 1.])
        axis.grid(axis='y', alpha=.18)
        axis.set_axisbelow(True)
        axis.set_ylabel('Saved score (0–1)' if axis is not saved else 'Saved score / coverage (0–1)')
    j = [row['complex_support_J_relative_to_best_fixed']*100 for row in sensitivity]
    fig.suptitle('V32 exterior-support diagnostic: DIFFERENT task function on unchanged geometry\n'
        'No new reconstruction or semantic success. High support agreement does not certify completion or topology.\n'
        f'Saved complex-rule J vs best fixed arm: 2 cm {j[0]:+.3f}%, 5 cm {j[1]:+.3f}%, '
        f'10 cm {j[2]:+.3f}% — all four physical tasks remain ineligible.', fontsize=11)
    rendered = {}
    for extension in ('png', 'svg'):
        stream = io.BytesIO()
        fig.savefig(stream, format=extension, dpi=145,
            metadata={'Description': 'V32 diagnostic; different task functions, no new geometry or scores'}
            if extension == 'svg' else None)
        rendered[extension] = stream.getvalue()
    plt.close(fig)
    return rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Run only after saved diagnostic is sealed')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return
    require(not OUTPUT.exists() and all(not FIGURE.with_suffix('.'+ext).exists() for ext in ('png', 'svg')),
            'Existing figure evidence must not be overwritten')
    require(shutil.disk_usage(ROOT).free >= RESERVE+CAP, 'Need 64 MiB reserve plus figure allowance')
    data, analytic_rows, stages, sensitivity, receipt = load_sealed()
    rendered = render(data, stages, sensitivity)
    receipt.update(status='complete_saved_numbers_figure', full_analytic_csv_rows=len(analytic_rows),
        full_saved_stage_csv_rows=len(stages), score_y_origin=0., all_four_saved_cases_included=True,
        gap_ladder_mm=list(GAPS), random_realizations_in_face_jitter_panel=1,
        final_historical_eligible_count=sum(bool(r['historical_eligible']) for r in stages if r['stage'] == 'final'),
        support_Q05_with_300mm_gap=data['corner_gap_300mm']['support_Q05'],
        support_Q05_with_whole_side_missing=data['one_vertical_side_missing']['support_Q05'],
        score_comparison='V31 supported-region Q versus V32 support-band Q; different task functions')
    description = ('# V32 支持带评分的解析适用性与局限\n\n'
        '图只读取两份封存的 result/scores。橙色是 V31 区域 Q，蓝色是 V32 支持带 Q；'
        '它们是不同任务函数，不能把更高蓝色值叫作更好的重建。紫色/黑色辅助量分别明确标注为 F1 或 C。'
        '绘图没有读取 NPZ 几何数值、重建夹具、计算投影或调用评分器，没有新物理/传感/TSDF。\n\n'
        'a 展示全部预声明 0/1/5/10/20/50/100/300 mm 缺口，横轴为离散测试案例而非等距离连续尺度。'
        'b 的 1/5/10 mm 共用同一随机实现，只改变幅度，不能作三次统计重复。'
        'c 单列 49/50/51 mm 平移，F1 的 5 cm 硬阈值仍会跳变，不声称全局连续或普遍抗噪。'
        '折线仅连接保存点，不代表新增测试或中间值估计。所有数值轴从 0 开始，顶部留标注空间。\n\n'
        '**300 mm 缺口的支持 Q 仍约 0.992778，整侧缺失仍约 0.902764。**'
        '因此高宏平均分不等于设施完成或没有危险缺口；45 个解析门通过只说明冻结夹具中的适用性。'
        'd 保留错误桥和 L 错误外形的非零/较高分数；完整 56 案均见 analytic_scores.csv。\n\n'
        'e 的两预测网格逐字节相同，真参考分别为闭合实心柜和有 8 mm 实际通口的薄壁壳。'
        '两者都高分，候选未认证拓扑。材料表面 watertight 不等于外壳封闭；'
        '这个不可辨性仅限输入网格，不能扩展为完整 RGB-D/自由射线或额外视点也不可区分。\n\n'
        'f 是四条原 V30 固定路线的同一批保存预测，旧区域 Q 保留；黑菱形 C 与 C80 虚线均属覆盖字段。'
        '四案仍全部不合格，不能用新支持分取消旧失败。该面板是看过旧数据后的开发诊断，不是新自主对照。'
        '图顶同时列出 2/5/10 cm 联合 J 的类别规则相对最佳固定臂变化，不能挑阈值宣布语义成功。'
        '阶段和完整阈值表见 saved_stages.csv、threshold_sensitivity.csv。\n\n'
        'PNG/SVG: docs/research/figures/v32_support_band_20260917.*。来源及库存/源码/输入核验见 result.json；'
        '产物全部哈希见 artifact_hashes.json。该支持诊断尚不适合作为唯一训练目标，不认证完整三维精度。\n')
    payloads = {FIGURE.with_suffix('.'+ext): value for ext, value in rendered.items()}
    payloads.update({OUTPUT/'result.json': pack(receipt), OUTPUT/'README.md': description.encode(),
        OUTPUT/'analytic_scores.csv': csv_bytes(analytic_rows), OUTPUT/'saved_stages.csv': csv_bytes(stages),
        OUTPUT/'threshold_sensitivity.csv': csv_bytes(sensitivity)})
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(value).hexdigest() for path, value in payloads.items()}
    payloads[OUTPUT/'artifact_hashes.json'] = pack(hashes)
    size = sum(map(len, payloads.values()))
    require(size < CAP and shutil.disk_usage(ROOT).free-size >= RESERVE, 'Combined figure/audit exceeds allowance')
    require(sha(HELPER) == receipt['file_helper_sha256'], 'Plotting file helper changed')
    for field, folder in (('analytic', ANALYTIC), ('saved_diagnostic', SAVED)):
        for filename, key in (('manifest.json', 'manifest_sha256'), ('artifact_hashes.json', 'inventory_sha256'),
                              ('result.json', 'result_sha256'), ('scores.json', 'scores_sha256')):
            require(sha(folder/filename) == receipt[field][key], 'Source changed during plotting')
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
            require(sha(ROOT/name) == expected, 'Saved artifact changed')
    except BaseException as error:
        failure = pack(dict(status='failed', error=repr(error)))
        if shutil.disk_usage(ROOT).free-len(failure) >= RESERVE:
            (OUTPUT/'failure.json').write_bytes(failure)
        raise
    print(json.dumps(dict(status=receipt['status'], output=str(OUTPUT), combined_bytes=size)))


if __name__ == '__main__':
    main()
