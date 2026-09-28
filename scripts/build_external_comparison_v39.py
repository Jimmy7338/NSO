#!/usr/bin/env python3
"""Build V39 publication artifacts from sealed complete evidence only.

No simulator, planner or mapping import is permitted. Failures remain explicit;
their unobserved metrics are never replaced by zeros. Replays are verification,
not independent samples. All input and output files receive SHA256 provenance.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'audit_results/v39_external_cpu_20260920'
OLD = ROOT/'audit_results/v36_online_confirmation_20260918'
OUT = ROOT/'docs/thesis/figures/v39_external'
REPORT = ROOT/'docs/research/V39_EXTERNAL_CPU_RESULTS_20260920.md'
METHODS = ('G', 'S', 'SWAP', 'VISTA')
LABELS = {'G': 'G', 'S': 'S', 'SWAP': 'SWAP-I', 'VISTA': 'VISTA-I'}
COLORS = {'G': '#737984', 'S': '#0072B2', 'SWAP': '#009E73', 'VISTA': '#D55E00'}
MARKERS = {'G': 'o', 'S': 's', 'SWAP': '^', 'VISTA': 'D'}
CONDITIONS = tuple((p, h) for p in ('P00', 'P01') for h in (0, 1))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def verify_seal(folder, filename, tracked, exclude=None):
    seal = folder/filename
    expected = read(seal)
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()
              and p != seal and not (exclude and p.relative_to(folder).parts[0] == exclude)}
    if actual != set(expected):
        raise ValueError(f'sealed file inventory changed: {seal}')
    for name, digest in expected.items():
        path = folder/name
        path.resolve().relative_to(folder.resolve())
        if sha(path) != digest:
            raise ValueError(f'changed sealed input: {path}')
        tracked[relative(path)] = digest
    tracked[relative(seal)] = sha(seal)


def metric_values(measurement):
    result = dict(C_map=measurement['C_map'], eligible=measurement['eligible'])
    for cm in (2, 5, 10):
        for short, key in (('P', 'precision'), ('R', 'recall'), ('F', 'f1'), ('J', 'joint')):
            result[f'{short}{cm}'] = measurement[f'{cm:02d}cm'][key]
    return result


def semantic_diagnostic(case, evidence):
    plans = [r['planning'] for r in evidence['plans']
             if r['planning']['phase'] == 'external_cpu_mechanism_planning']
    return dict(parent=case['parent'], hypothesis=case['hypothesis'], method=case['method'],
        autonomous_decisions=len(plans),
        registered_class_decisions=sum(p['registered_class'] is not None for p in plans),
        positive_semantic_decisions=sum(bool(p['semantic_term_nonzero']) for p in plans),
        changed_first_action_decisions=sum(bool(p['semantic_changed_first_action']) for p in plans),
        semantic_score_varied_decisions=sum(p['semantic_candidate_range'][1]-p['semantic_candidate_range'][0] > 1e-12 for p in plans),
        measured_direction_varied_decisions=sum(p['measured_direction_candidate_range'][1]-p['measured_direction_candidate_range'][0] > 1e-12 for p in plans),
        inspection_decisions=sum(p['mechanism_phase'] == 'inspection' for p in plans),
        geometric_fallback_decisions=sum(p['mechanism_phase'] == 'geometry_exploration' for p in plans),
        terminal_observed_surfels=max((p['observed_surfel_count'] for p in plans), default=0),
        structural_posterior_used=any(p['posterior_used_to_choose_action'] for p in plans),
        interpretation='same-state candidate-ranking diagnostic, not an executed policy ablation')


def compare_old(case, result, folder, tracked):
    old_cases = read(OLD/'config.json')['physical_cases']
    matching = [c for c in old_cases if c['parent'] == case['parent'] and c['hypothesis'] == case['hypothesis']
                and c.get('mode', c.get('method')) == case['method']]
    if len(matching) != 1:
        raise ValueError(f'old V36 condition not unique: {case}')
    old_folder = OLD/f"case{matching[0]['index']:02d}"
    old = read(old_folder/'result.json')
    for path in (OLD/'config.json', old_folder/'result.json', old_folder/'controller.json'):
        tracked[relative(path)] = sha(path)
    pairs = {}
    for stage in ('prefix', 'final'):
        pairs[f'{stage}_measurement_exact'] = result['stages'][stage]['measurement'] == old['stages'][stage]['measurement']
        for suffix in ('raw.npz', 'extracted.npz', 'maps.npz'):
            a, b = folder/f'{stage}_{suffix}', old_folder/f'{stage}_{suffix}'
            # Compressed NPZ headers can differ; compare named arrays below.
            import numpy as np
            with np.load(a, allow_pickle=False) as aa, np.load(b, allow_pickle=False) as bb:
                equal = set(aa.files) == set(bb.files) and all(np.array_equal(aa[k], bb[k]) for k in aa.files)
            pairs[f'{stage}_{suffix.replace(".npz", "")}_arrays_exact'] = equal
            tracked[relative(b)] = sha(b)
    old_actions = read(old_folder/'controller.json')['actions']
    pairs['actions_exact'] = result['actions'] == old_actions
    pairs['paid_actions_exact'] = result['paid_actions'] == old['paid_actions']
    return dict(parent=case['parent'], hypothesis=case['hypothesis'], method=case['method'],
        old_case=relative(old_folder), checks=pairs, all_exact=all(pairs.values()))


def collect():
    tracked = {}
    verify_seal(SOURCE, 'final_seal.json', tracked)
    aggregate, config, manifest = (read(SOURCE/name) for name in ('result.json', 'config.json', 'manifest.json'))
    if aggregate['status'] not in ('complete', 'complete_with_failures') or not aggregate['all_completed_cases_replayed']:
        raise ValueError('complete sealed main matrix and all successful fresh replays required')
    if config['methods'] != list(METHODS) or len(config['physical_cases']) != 16:
        raise ValueError('exact preregistered 4 conditions x 4 methods required')
    if sha(SOURCE/'sources.zip') != manifest['source_archive_sha256']:
        raise ValueError('frozen source archive changed')
    with zipfile.ZipFile(SOURCE/'sources.zip') as archive:
        for name, digest in manifest['source_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('archive source digest mismatch: '+name)
    rows, semantics, old_parity = [], [], []
    aggregate_rows = {r['index']: r for r in aggregate['rows']}
    for case in config['physical_cases']:
        folder = SOURCE/f"case{case['index']:02d}"
        verify_seal(folder, 'main_seal.json', tracked, 'replay')
        base = {k: case[k] for k in ('index', 'parent', 'hypothesis', 'method')}
        if (folder/'failure.json').exists():
            rows.append(dict(**base, status='failed', eligible=False,
                failure_source=relative(folder/'failure.json')))
            continue
        verify_seal(folder/'replay', 'seal.json', tracked)
        replay, result = read(folder/'replay/result.json'), read(folder/'result.json')
        if not replay['passed'] or replay['main_result_sha256'] != sha(folder/'result.json'):
            raise ValueError('invalid independent replay: '+str(folder))
        m = result['stages']['final']['measurement']
        metrics = metric_values(m)
        a = aggregate_rows[case['index']]
        for key, value in metrics.items():
            aggregate_key = 'Q'+key[1:] if key.startswith('F') else key
            if a[aggregate_key] != value:
                raise ValueError(f'aggregate and main metric differ: case {case["index"]}, {key}')
        if abs(metrics['J5']-metrics['C_map']*metrics['F5']) > 1e-15:
            raise ValueError('joint score is not the declared measured product')
        if result['candidate_world_queries'] or result['counts']['candidate_world_queries']:
            raise ValueError('forbidden candidate-world query')
        timing = read(folder/'timing.json')
        rows.append(dict(**base, status='complete', **metrics,
            paid_actions=result['paid_actions'], forward_m=result['actions'].count('forward'),
            turns=sum(a != 'forward' for a in result['actions']), returned=result['returned'],
            collisions=result['collisions'], planning_s=timing['planning_and_observed_ledger_s'],
            mapping_s=timing['mapping_s'], sensor_s=timing['sensor_s'],
            evaluation_s=timing['evaluation_s'], total_s=timing['total_s'],
            main_source=relative(folder/'result.json'), replay_source=relative(folder/'replay/result.json')))
        if case['method'] in ('SWAP', 'VISTA'):
            semantics.append(semantic_diagnostic(case, read(folder/'controller.json')))
        else:
            old_parity.append(compare_old(case, result, folder, tracked))
    if len(rows) != 16 or len({(r['parent'], r['hypothesis'], r['method']) for r in rows}) != 16:
        raise ValueError('missing or duplicate declared conditions')
    if not all(p['equal'] for p in aggregate['prefix_comparisons']):
        raise ValueError('shared paid prefix mismatch')
    return config, aggregate, rows, semantics, old_parity, tracked


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def mean_rows(rows):
    result = []
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method and r['status'] == 'complete']
        out = dict(method=method, completed=len(selected), declared=4,
            eligible=sum(r['eligible'] for r in selected))
        for key in ('C_map', 'F2', 'F5', 'F10', 'J2', 'J5', 'J10', 'paid_actions', 'forward_m', 'turns', 'planning_s', 'total_s'):
            out[key] = sum(r[key] for r in selected)/len(selected) if selected else None
        result.append(out)
    return result


def figures(rows, output):
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso_v39_mpl')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8.5,
        'axes.labelsize': 8.5, 'axes.titlesize': 9, 'legend.fontsize': 8,
        'axes.linewidth': .65, 'lines.linewidth': 1., 'xtick.major.width': .65,
        'ytick.major.width': .65, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'svg.hashsalt': 'nso-v39-external-comparison',
        'savefig.facecolor': 'white', 'mathtext.fontset': 'dejavusans'})
    lookup = {(r['parent'], r['hypothesis'], r['method']): r for r in rows}

    def panel(ax, key, label):
        offsets = np.array([-.18, -.06, .06, .18])
        for i, condition in enumerate(CONDITIONS):
            points = [(i+offsets[j], lookup[(*condition, m)].get(key)) for j, m in enumerate(METHODS)]
            if all(y is not None for _, y in points):
                ax.plot([x for x, _ in points], [y for _, y in points], color='#BFC3C9', lw=.7, zorder=1)
        for j, method in enumerate(METHODS):
            for i, condition in enumerate(CONDITIONS):
                row = lookup[(*condition, method)]
                if row['status'] == 'complete':
                    ax.plot(i+offsets[j], row[key], marker=MARKERS[method], color=COLORS[method],
                        markerfacecolor=COLORS[method] if row['eligible'] else 'white',
                        markersize=5, linestyle='none', label=LABELS[method] if i == 0 else None, zorder=3)
                else:
                    ax.text(i+offsets[j], .025, '×', color=COLORS[method], ha='center',
                        va='bottom', transform=ax.get_xaxis_transform(), fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E6E8EB', linewidth=.5)
        ax.set_xticks(range(4), [f'{p}\nh{h}' for p, h in CONDITIONS])
        ax.set(xlim=(-.5, 3.5), ylim=(0, 1.015), ylabel=label)

    def save(fig, name):
        for ext in ('pdf', 'svg', 'png'):
            meta = {'Creator': 'NSO V39 sealed-evidence figure builder', 'CreationDate': None, 'ModDate': None} if ext == 'pdf' else ({'Date': None} if ext == 'svg' else None)
            fig.savefig(output/f'{name}.{ext}', dpi=300, metadata=meta)
        plt.close(fig)

    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle='none',
                      markersize=5, label=LABELS[m]) for m in METHODS]

    fig, ax = plt.subplots(figsize=(7.05, 3.1))
    fig.subplots_adjust(left=.085, right=.985, bottom=.23, top=.80)
    panel(ax, 'J5', r'Measured $J_5=C_{map}F_{1,5cm}$')
    ax.set_title('Four paired conditions under the same action budget', loc='left', pad=12)
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.33), ncol=4, frameon=False)
    fig.text(.085, .025, 'CPU mechanism bridge; 2 previously used layouts. SWAP-I and VISTA-I are inspired adaptations.', fontsize=7.4)
    save(fig, 'fig01_paired_joint')
    fig, axs = plt.subplots(1, 2, figsize=(7.05, 3.1))
    fig.subplots_adjust(left=.075, right=.985, bottom=.24, top=.79, wspace=.30)
    panel(axs[0], 'C_map', r'Measured coverage $C_{map}$')
    panel(axs[1], 'F5', r'Surface $F_{1,5cm}$')
    for ax, title in zip(axs, ('a  Coverage', 'b  Reconstruction surface quality')):
        ax.set_title(title, loc='left', pad=11)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, 1.01), ncol=4, frameon=False)
    fig.text(.075, .025, 'Same sensors, CPU TSDF and evaluator; exact simulated poses. Four conditions are not four independent layouts.', fontsize=7.0)
    save(fig, 'fig02_coverage_surface')


def report(rows, means, semantics, parity, aggregate):
    finished = sum(r['status'] == 'complete' for r in rows)
    lines = ['# V39 共同 CPU 外部机制比较结果独立审查', '',
        '日期：2026-09-20。所有数字均从已封存主轨迹、独立重放和聚合结果读取；本审查与制图不运行模拟器或重新选择实验路线。', '',
        f'固定矩阵共 16 次主轨迹，完成 {finished} 次，失败 {16-finished} 次；已完成轨迹的独立新进程完整重放均通过。统计独立单位为 **2 个已在开发中使用的布局**，每布局含 h0/h1 两个条件；重放不增加样本数。', '',
        '## 数值结果', '',
        '|方法|完成/计划|合格|平均 C_map|平均 F1@5cm|平均 J5|平均动作数|平均规划及账本耗时/s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in means:
        fmt = lambda k: 'NA' if r[k] is None else f'{r[k]:.6f}'
        lines.append(f"|{LABELS[r['method']]}|{r['completed']}/4|{r['eligible']}|{fmt('C_map')}|{fmt('F5')}|{fmt('J5')}|{fmt('paid_actions')}|{fmt('planning_s')}|")
    lookup = {(r['parent'], r['hypothesis'], r['method']): r for r in rows}
    lines += ['', '|配对比较|完整配对数|平均 J5 差（S 减对照）|S 胜/平/负|', '|---|---:|---:|---:|']
    for method in ('G', 'SWAP', 'VISTA'):
        pairs = [(lookup[p, h, 'S'], lookup[p, h, method]) for p, h in CONDITIONS]
        deltas = [a['J5']-b['J5'] for a, b in pairs if a['status'] == b['status'] == 'complete']
        effect = 'NA' if not deltas else f'{sum(deltas)/len(deltas):+.6f}'
        counts = [sum(test(d) for d in deltas) for test in (lambda d: d > 1e-12, lambda d: abs(d) <= 1e-12, lambda d: d < -1e-12)]
        lines.append(f"|S − {LABELS[method]}|{len(deltas)}|{effect}|{'/'.join(map(str, counts))}|")
    by_method = {r['method']: r for r in means}
    if all(by_method[m]['completed'] == 4 and by_method[m]['J5'] > 0 for m in METHODS):
        effect = {m: 100*(by_method['S']['J5']/by_method[m]['J5']-1) for m in ('G', 'SWAP', 'VISTA')}
        lines += ['', f"S 相对 G 的平均 J5 增幅为 **{effect['G']:.2f}%**，该对照保留共同规划器并改变类别信息使用，延续原 V36 的受控语义增量证据。S 相对 SWAP-I / VISTA-I 的平均差距分别为 {effect['SWAP']:.2f}% / {effect['VISTA']:.2f}%；后两项同时改变规划机制、目标函数与适配细节，**不能把全部差距归因于语义模块，也不能写作击败原作者完整系统**。"]
    lines += ['', '均值仅对已完成任务计算，分母明确列出；失败未填零，未从计划矩阵中删除。结果仅作描述，不进行以轨迹数伪装独立样本的显著性检验。', '',
        '## 与原 V36 的逐条件一致性', '',
        f"G/S 共核查 {len(parity)} 个条件，{sum(p['all_exact'] for p in parity)} 个条件在动作序列、前缀/终点全部评价字段以及原始网格、裁剪网格、地图数组上完全相同。", '',
        '|布局|构型|方法|全部检查严格相等|', '|---|---:|---|---|']
    for p in parity:
        lines.append(f"|{p['parent']}|h{p['hypothesis']}|{p['method']}|{p['all_exact']}|")
    if any(not p['all_exact'] for p in parity):
        lines += ['', '存在不一致；不能称作原 V36 的精确复现。具体差异见 `v36_parity.json`，不得以数值接近代替严格一致。']
    lines += ['', '## 外部机制的实际语义激活', '',
        '|布局|构型|方法|自主决策|语义项非零|去语义后首动作改变|巡检决策|几何回退|',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for s in semantics:
        lines.append(f"|{s['parent']}|h{s['hypothesis']}|{LABELS[s['method']]}|{s['autonomous_decisions']}|{s['positive_semantic_decisions']}|{s['changed_first_action_decisions']}|{s['inspection_decisions']}|{s['geometric_fallback_decisions']}|")
    for method in ('SWAP', 'VISTA'):
        selected = [s for s in semantics if s['method'] == method]
        n = sum(s['autonomous_decisions'] for s in selected)
        active = sum(s['positive_semantic_decisions'] for s in selected)
        changed = sum(s['changed_first_action_decisions'] for s in selected)
        lines += ['', f"{LABELS[method]} 在 {active}/{n} 次自主决策有正语义项，在同状态去语义排序诊断中有 {changed}/{n} 次首动作不同。因此本次外部对照并非全零语义占位策略。"]
    lines += ['', '“首动作改变”是在同一实际状态与同一候选集合上移除语义项后的排序诊断，不是另执行一条去语义策略轨迹，也不等价于该模块的因果性能增益。非零评分、排名变化与最终重建收益是三个不同证据层次。若没有动作变化或 SWAP-I 主要退回几何探索，应如实限定解释。', '',
        '## 信息边界、评价与适用范围', '',
        '- 源代码封存、完整矩阵封存和每次主轨迹/重放封存逐文件 SHA256 校验通过；聚合表逐项与主任务结果一致。归档源码逐条核验其文件哈希。',
        '- 决策只读取公共安全图、两个公开构型的预测支持、已付 RGB-D/扫描、公共标定和设施 ROI。结构真值、实际未选视角和最终网格评分没有进入策略；候选 World 查询为 0。',
        '- 同一 CPU TSDF 使用 4 cm 体素与 12 cm 截断；实测 J5 为 C_map×F1@5cm。GT 评价位于完整预测和决策封存之后。误差只在公共 ROI 的设施外立面评价，姿态为精确模拟里程计。',
        '- SWAP-I 为巡检机制适配，未实现原始三角网格孔洞补全；VISTA-I 为稀疏已测方向与目标相关性适配，并显式加入双模板等权公共支持探索。二者均不是原作者完整系统。',
        '- 这组数据可以比较当前受控资源和先验条件下的策略行为；不能建立自然语义网络优势、完整神经 ANS 优势、真实 SLAM 位姿精度或主流系统 SOTA 排名。TARE 完整规划器适配须单独报告。', '',
        '## 可复核产物', '',
        '- `docs/thesis/figures/v39_external/fig01_paired_joint.{pdf,svg,png}`：四条件四方法 J5 配对点图。',
        '- `fig02_coverage_surface.{pdf,svg,png}`：覆盖率和 F1 分开展示；矢量 PDF/SVG 与 300 dpi PNG。',
        '- `measurements.csv`、`method_means.csv`、`semantic_activation.csv`：LF 换行，保留完整源精度。',
        '- `table_method_rows.tex`、`table_condition_rows.tex`、`v36_parity.json`、`manifest.json`：论文表格、一致性检查及每个主任务/重放/聚合输入和输出的 SHA256。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    config, aggregate, rows, semantics, parity, tracked = collect()
    out = args.output.resolve()
    out.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    means = mean_rows(rows)
    keys = ['index', 'parent', 'hypothesis', 'method', 'status', 'eligible', 'C_map']
    keys += [f'{metric}{cm}' for cm in (2, 5, 10) for metric in ('P', 'R', 'F', 'J')]
    keys += ['paid_actions', 'forward_m', 'turns', 'returned', 'collisions', 'planning_s', 'mapping_s', 'sensor_s', 'evaluation_s', 'total_s', 'main_source', 'replay_source', 'failure_source']
    write_csv(out/'measurements.csv', rows, keys)
    write_csv(out/'method_means.csv', means, list(means[0]))
    if semantics:
        write_csv(out/'semantic_activation.csv', semantics, list(semantics[0]))
    (out/'v36_parity.json').write_text(json.dumps(parity, indent=2)+'\n')
    def number(value):
        return '--' if value is None else f'{value:.4f}'
    table = [f"{LABELS[r['method']]} & {number(r['C_map'])} & {number(r['F5'])} & {number(r['J5'])} & {number(r['paid_actions'])} & {r['eligible']}/{r['declared']} \\\\" for r in means]
    (out/'table_method_rows.tex').write_text('\n'.join(table)+'\n')
    table = [f"{r['parent']} / h{r['hypothesis']} & {LABELS[r['method']]} & {number(r.get('C_map'))} & {number(r.get('F5'))} & {number(r.get('J5'))} & {r.get('paid_actions', '--')} & {'yes' if r['eligible'] else 'no'} \\\\" for r in rows]
    (out/'table_condition_rows.tex').write_text('\n'.join(table)+'\n')
    (out/'captions.md').write_text(
        '# Figure captions\n\n'
        '**Paired joint metric.** Measured terminal J5 under a common 42-action budget, including an 18-action paid prefix. Each group contains the same layout and hidden configuration for G, S, SWAP-inspection-inspired/CPU (SWAP-I), and VISTA-view-inspired/CPU (VISTA-I). Grey lines link condition-matched measurements. These are four conditions from two previously used layouts; independent replay verifies execution but does not enlarge the sample size. No significance or original-system ranking is claimed.\n\n'
        '**Coverage and reconstruction quality.** Terminal measured coverage and surface F1 at 5 cm are shown separately. All methods use the same sensor model, public safe graph, prior-input permissions, CPU TSDF backend, evaluator, and action accounting. The exact simulated poses and public facility ROI constrain interpretation. Outlined markers denote ineligible completed cases; a cross on the bottom margin would denote a failed case without assigning a numerical score.\n')
    figures(rows, out)
    REPORT.write_text(report(rows, means, semantics, parity, aggregate))
    tracked[relative(__file__)] = sha(__file__)
    manifest = dict(scope='saved-evidence descriptive L1-inspired CPU comparison',
        source=relative(SOURCE/'result.json'), source_sha256=tracked,
        independent_layouts=2, conditions=4, declared_methods=4, main_starts=16,
        failed_cases=sum(r['status'] == 'failed' for r in rows), replay_is_independent_sample=False,
        confidence_intervals_not_computed=True, full_original_system_comparison=False,
        all_old_V36_checks_exact=all(p['all_exact'] for p in parity),
        outputs_sha256={relative(p): sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'},
        report_sha256={relative(REPORT): sha(REPORT)})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(complete=True, rows=len(rows), old_V36_all_exact=manifest['all_old_V36_checks_exact'],
        failed_cases=manifest['failed_cases'], output=relative(out))))


if __name__ == '__main__':
    main()
