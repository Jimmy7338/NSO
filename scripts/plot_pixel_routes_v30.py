#!/usr/bin/env python3
"""Plot the sealed V30 fixed-route batch; no world, sensor, mapper or evaluation."""
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

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'audit_results/v30_pixel_routes_20260917'
OUTPUT = ROOT / 'audit_results/v30_pixel_routes_figures_20260917'
FIGURE = ROOT / 'docs/research/figures/v30_pixel_routes_20260917'
SCENE = ROOT / 'configs/virtual3d/v29_information_scene_20260917.json'
CONFIG = ROOT / 'configs/virtual3d/v30_pixel_routes_20260917.json'
SPEC = [(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B')]
COLORS = {'A': '#2471a3', 'B': '#ca6f1e'}
CAP, RESERVE = 1024**2, 64 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def inventory(folder):
    saved = read(folder / 'artifact_hashes.json')
    # The collector deliberately excludes all nested inventory filenames.
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p.name != 'artifact_hashes.json'}
    require(actual == set(saved), 'Artifact set changed: ' + str(folder))
    for name, expected in saved.items():
        require(sha(folder / name) == expected, 'Artifact changed: ' + name)
    return len(saved)


def bounded(value, name):
    value = float(value)
    require(math.isfinite(value) and 0 <= value <= 1, 'Invalid saved ' + name)
    return value


def load_complete(source):
    manifest = read(source / 'manifest.json')
    require(manifest['status'] == 'complete', 'Wait for all four cases, replays and final analysis')
    require([(c['hypothesis'], c['arm']) for c in manifest['cases']] == SPEC,
            'Unexpected fixed four-case declaration')
    require(manifest['new_main_started'] == manifest['replay_started'] == 4,
            'Unexpected counted attempt totals; review before plotting')
    source_hashes = manifest['source_sha256']
    for name, expected in source_hashes.items():
        require(sha(ROOT / name) == expected, 'Frozen source changed: ' + name)
    require(sha(source / 'sources.zip') == manifest['source_zip_sha256'], 'Source ZIP changed')
    checked = inventory(source)
    scene, config = read(SCENE), read(CONFIG)
    require(source_hashes[str(SCENE.relative_to(ROOT))] == sha(SCENE)
            == config['scene_sha256'], 'Scene is not the frozen input')
    require(source_hashes[str(CONFIG.relative_to(ROOT))] == sha(CONFIG), 'Unbound config')
    require(config['budget'] == 48 and len(config['prefix']) == 18
            and all(len(config['suffixes'][a]) == 30 for a in ('A', 'B')), 'Action contract changed')
    aggregate = read(source / 'result.json')
    require(aggregate['all_replays_passed'] and aggregate['new_main_tasks'] == 4
            and not aggregate['full_architecture_advantage_proven']
            and not aggregate['semantic_runtime_executed'], 'Unexpected aggregate scope')
    require(len(aggregate['table']) == 4, 'The four outcomes must all be retained')
    routes = read(source / 'static_routes.json')
    require(len(routes) == 4, 'Missing frozen routes')
    rows, traces, provenance = [], [], []
    for index, (hypothesis, arm) in enumerate(SPEC):
        folder = source / f'case{index:02d}'
        checked += inventory(folder)
        result, replay = read(folder / 'result.json'), read(folder / 'replay.json')
        trace, table = read(folder / 'trace.json'), aggregate['table'][index]
        identity = dict(index=index, hypothesis=hypothesis, arm=arm)
        require(result['case'] == routes[index]['case'] == identity
                and all(table[k] == v for k, v in identity.items()), 'Case/route identity mismatch')
        require(replay['passed'] and replay['all49_packets_byte_equal'] and replay['full_result_equal']
                and replay['result_sha256'] == sha(folder / 'result.json'), 'Unverified replay')
        require(replay['main_pid'] != replay['replay_pid']
                and replay['main_pid'] == read(folder / 'started.json')['pid']
                and replay['replay_pid'] == read(folder / 'replay_started.json')['pid'],
                'Replay must have its own recorded process')
        require(result['steps'] == 48 and result['measured_only']
                and not result['full_architecture_executed'], 'Wrong experiment type')
        require(len(trace) == len(routes[index]['poses']) == 49
                and [r['paid'] for r in trace] == list(range(49)), 'Incomplete paid trajectory')
        require([r['action'] for r in trace] == [None] + config['prefix'] + config['suffixes'][arm],
                'Actual actions differ from the frozen fixed route')
        row0, col0 = trace[0]['position']
        # Public V30 contract: .2 m cells and identical translated anchor.
        xyh = [[(r['position'][1] - col0) * .2, (row0 - r['position'][0]) * .2, r['heading']]
               for r in trace]
        require(all(all(abs(float(a) - float(b)) < 1e-9 for a, b in zip(p, q))
                    for p, q in zip(xyh, routes[index]['poses'])), 'Actual pose departed from static route')
        require(xyh[0] == xyh[18] == xyh[48] == [0., 0., 0], 'Exact stage return mismatch')
        traces.append(xyh)
        for stage, paid in (('prefix', 18), ('final', 48)):
            saved = result['stages'][stage]
            primary, full = saved['window']['main_observed'], saved['full_instance']
            alias = table[stage]
            c = bounded(primary['coverage_2d'], 'C')
            q = bounded(primary['05cm']['outline_macro_quality'], 'window Q')
            j = bounded(primary['05cm']['joint_outline'], 'window J')
            fq = bounded(full['05cm']['outline_macro_quality'], 'full-instance Q')
            fj = bounded(full['05cm']['joint_outline'], 'full-instance J')
            require((alias['C'], alias['Q'], alias['J'], alias['eligible'])
                    == (c, q, j, primary['eligible']) and alias['full_instance'] == full,
                    'Aggregate/case metric mismatch')
            require(abs(j - c*q) < 1e-10 and abs(fj - c*fq) < 1e-10
                    and full['coverage_2d'] == c, 'Saved J/C aliases disagree')
            rows.append(dict(case=index, hypothesis=hypothesis, arm=arm, stage=stage, paid=paid,
                C=c, window_Q=q, window_J=j, full_instance_Q=fq, full_instance_J=fj,
                eligible=primary['eligible'], instance_gate=alias['instance_gate'],
                marker_gate=alias['marker_gate'], collisions=result['collisions'], returned=result['returned']))
        provenance.append(dict(index=index, result_sha256=sha(folder / 'result.json'),
            trace_sha256=sha(folder / 'trace.json'), replay_sha256=sha(folder / 'replay.json'),
            case_inventory_sha256=sha(folder / 'artifact_hashes.json'),
            main_pid=replay['main_pid'], replay_pid=replay['replay_pid']))
    require(all(traces[i] == traces[j] for i, j in ((0, 2), (1, 3))), 'Cross-assignment paths differ')
    require(all(t[:19] == traces[0][:19] for t in traces), 'Shared paid prefix differs')
    return scene, rows, traces, aggregate, dict(
        source_root=str(source), source_manifest_sha256=sha(source / 'manifest.json'),
        source_result_sha256=sha(source / 'result.json'), source_inventory_sha256=sha(source / 'artifact_hashes.json'),
        scene_sha256=sha(SCENE), config_sha256=sha(CONFIG), static_routes_sha256=sha(source / 'static_routes.json'),
        source_zip_sha256=manifest['source_zip_sha256'], verified_artifact_hash_entries=checked,
        case_sources=provenance, script_sha256=sha(Path(__file__)),
        actual_fixed_paths_equal_declared_paths=True, shared_19_frame_pose_prefix=True,
        checked_successful_main_records=4, checked_successful_independent_replays=4,
        attempt_scope='Four counted main/replay claims in this sealed V30 manifest, not earlier project experiments',
        new_actions=0, new_sensor_frames=0, new_tsdf_fusions=0, new_evaluations=0)


def render(scene, rows, traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none', 'svg.hashsalt': 'v30-fixed-routes',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.1), constrained_layout=True,
                             gridspec_kw={'height_ratios': [1.55, 1]})
    wall_bounds = scene['class_independent_wall_bounds']
    xlim = (min(b[0] for b in wall_bounds)-.3, max(b[1] for b in wall_bounds)+.3)
    ylim = (min(b[2] for b in wall_bounds)-.3, max(b[3] for b in wall_bounds)+.3)
    for h, axis in enumerate(axes[0]):
        for b in wall_bounds:
            axis.add_patch(Rectangle((b[0], b[2]), b[1]-b[0], b[3]-b[2], facecolor='.3', zorder=1))
        for i, station in enumerate(scene['stations']):
            center, kind = station['center_x'], scene['hypotheses'][h][i]
            b = scene['body_relative_bounds']
            axis.add_patch(Rectangle((center+b[0], b[2]), b[1]-b[0], b[3]-b[2],
                facecolor='#dce5e9', edgecolor='.35', linewidth=.7, zorder=1))
            if kind == 'complex':
                a = scene['complex_attachment_relative_bounds']
                axis.add_patch(Rectangle((center+a[0], a[2]), a[1]-a[0], a[3]-a[2],
                    facecolor='#96b1bd', edgecolor='.3', linewidth=.7, hatch='///', zorder=1))
            axis.text(center, (b[2]+b[3])/2, station['name']+'\n'+kind,
                      ha='center', va='center', fontsize=8)
        prefix = traces[2*h][:19]
        axis.plot([p[0] for p in prefix], [p[1] for p in prefix], color='.65',
                  linewidth=4.2, alpha=.7, label='Shared prefix (18 actions)', zorder=2)
        for index in (2*h, 2*h+1):
            arm, points = SPEC[index][1], traces[index][18:]
            axis.plot([p[0] for p in points], [p[1] for p in points], color=COLORS[arm],
                      linewidth=1.6, linestyle='-' if arm == 'A' else '--',
                      label='Fixed '+arm+' suffix (30 actions)', zorder=3)
        axis.scatter([0], [0], marker='*', s=90, color='black', zorder=5)
        axis.annotate('Start = both stage ends', (0, 0), xytext=(0, -.7),
                      ha='center', fontsize=7.5)
        axis.set(xlim=xlim, ylim=ylim, xlabel='x relative to common anchor (m)', ylabel='y (m)',
                 title=f'({chr(97+h)}) H{h}: frozen scene and actual paths')
        axis.set_aspect('equal'); axis.legend(loc='upper center', fontsize=7, framealpha=.92)
    quality, coverage = axes[1]
    for index, (_, arm) in enumerate(SPEC):
        before, after = rows[2*index:2*index+2]
        color = COLORS[arm]
        for key, offset, marker, style in (('window_Q', .11, 'o', '-'),
                                           ('full_instance_Q', -.11, 's', '--')):
            y = index + offset
            quality.plot([before[key], after[key]], [y, y], color=color, linestyle=style, linewidth=1.3)
            quality.scatter(before[key], y, marker=marker, facecolors='white', edgecolors=color, s=38, zorder=4)
            quality.scatter(after[key], y, marker=marker, color=color, s=38, zorder=5)
        coverage.plot([before['C'], after['C']], [index, index], color=color, linewidth=1.3)
        coverage.scatter(before['C'], index, facecolors='white', edgecolors=color, s=38, zorder=4)
        coverage.scatter(after['C'], index, color=color, s=38, zorder=5)
        coverage.text(.025, index-.24, 'Final eligibility: '+str(after['eligible']), fontsize=7, color='.3')
    labels = [f'H{h} / fixed {arm}' for h, arm in SPEC]
    for axis in (quality, coverage):
        axis.set(xlim=(0, 1), ylim=(3.5, -.65), yticks=range(4), yticklabels=labels)
        axis.grid(axis='x', alpha=.22)
    quality.set(title='(c) Two saved shape evaluations', xlabel='Three-projection outline quality Q (0 to 1)')
    quality.legend(handles=[Line2D([], [], color='.3', marker='o', label='Window Q (primary)'),
        Line2D([], [], color='.3', marker='s', linestyle='--', label='Full-instance Q (companion)')],
        loc='lower right', fontsize=7.2)
    coverage.axvline(.8, color='.25', linestyle=':', linewidth=1, label='C = 0.80 gate')
    coverage.set(title='(d) Two saved coverage evaluations', xlabel='Evaluated 2D coverage C (0 to 1)')
    coverage.legend(loc='lower right', fontsize=7.2)
    qualified = sum(bool(row['eligible']) for row in rows if row['stage'] == 'final')
    fig.suptitle('V30 fixed-route pixel experiment — not autonomous ANS\n'
                 f'Final qualification: {qualified}/4. Hollow: prefix (18 actions); filled: final (48).\n'
                 'Segments join two saved stages only; unqualified scores are diagnostic.', fontsize=11)
    rendered = {}
    for extension in ('png', 'svg'):
        stream = io.BytesIO()
        fig.savefig(stream, format=extension, dpi=160,
                    metadata={'Description': 'Four fixed routes; source hashes in independent audit receipt'}
                    if extension == 'svg' else None)
        rendered[extension] = stream.getvalue()
    plt.close(fig)
    return rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    require(not output.exists() and all(not FIGURE.with_suffix('.'+x).exists() for x in ('png', 'svg')),
            'Do not overwrite prior figure evidence')
    scene, rows, traces, aggregate, receipt = load_complete(source)
    require(shutil.disk_usage(ROOT).free >= RESERVE+CAP, 'Need 64 MiB reserve plus 1 MiB allowance')
    rendered = render(scene, rows, traces)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    markdown = ['| 案例 | 阶段 / 已付动作 | C | 窗口 Q | 完整实例 Q | 窗口 J | 资格 |',
                '|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        markdown.append(f'| H{row["hypothesis"]}/{row["arm"]} | {row["stage"]}/{row["paid"]} '
            f'| {row["C"]:.6f} | {row["window_Q"]:.6f} | {row["full_instance_Q"]:.6f} '
            f'| {row["window_J"]:.6f} | {row["eligible"]} |')
    description = ('# V30 固定路线像素实验图\n\n'
        '两排列各执行固定 A、B，一共四条主任务；四条独立回放用于复现，不增加独立样本。'
        '路线来自事先冻结的真值几何目录，不是自主 G/S 或 ANS 四模块端到端实验。'
        '场景平面图含评价侧真值实体及附件，仅用于解释实验。轨迹读取实际已付动作记录，'
        '逐姿态核对冻结 static_routes；两排列复用同一路线，共享 18 动作前缀，终点均为 48 动作。'
        '原地转向、同路往返在 XY 图中重叠，但都计费；相同起终位置还核对了朝向。\n\n'
        '空心点为 prefix，实心点为 final；连线只配对两个已保存评价点，不是中间时刻的质量/覆盖曲线。'
        '四案完整保留，两个数值轴固定 0–1，无平滑、插值或伪独立误差条。'
        '窗口 Q 是原主指标；方形虚线为完整预测实例伴随指标，未用它替换主指标。'
        '两者均为两设施、三正交投影等权的 min(边界 F1@5cm, 投影 IoU)，不是完整三维精度。'
        '窗口指标仅评价侧裁剪，完整实例指标不裁预测。C 是评价侧 known∩reachable/reachable。'
        'C≥0.8 仅是资格的一部分；还须返航、无碰撞、未失败、预算合规；图中另列实际资格。\n\n'
        + '\n'.join(markdown) + '\n\n'
        f'本批已保存推进门：{aggregate["progress_to_learning_gate_passed"]}；'
        '本图不把该门或人工颜色标签解释为自然语义网络、强几何基线或完整架构优势。'
        '固定臂、类别规则及 oracle 的准确数值见原始封存 result.json，本图不重新选臂。\n\n'
        '绘图未调用 world、sensor、mapper、TSDF 或评价器；新增物理动作、传感、建图和评价均为 0。'
        '来源和哈希见 result.json；CSV 保留全部两阶段 C/Q/J 与门。'
        '图像位于 docs/research/figures/v30_pixel_routes_20260917.png 和同名 SVG。\n')
    receipt.update(status='complete_saved_record_figure', all_four_cases_included=True,
        figure_scope='fixed routes, not autonomous ANS', score_axes=[0, 1],
        saved_paid_stages=[18, 48], no_intermediate_quality_or_coverage_inferred=True,
        progress_to_learning_gate_passed=aggregate['progress_to_learning_gate_passed'])
    payloads = {FIGURE.with_suffix('.'+ext): data for ext, data in rendered.items()}
    payloads.update({output/'stages.csv': stream.getvalue().encode(),
                     output/'README.md': description.encode(), output/'result.json': json_bytes(receipt)})
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(data).hexdigest() for path, data in payloads.items()}
    payloads[output/'artifact_hashes.json'] = json_bytes(hashes)
    used = sum(map(len, payloads.values()))
    require(used < CAP and shutil.disk_usage(ROOT).free-used >= RESERVE, 'Combined figure/audit allowance exceeded')
    require(sha(source/'manifest.json') == receipt['source_manifest_sha256']
            and sha(source/'artifact_hashes.json') == receipt['source_inventory_sha256']
            and sha(source/'result.json') == receipt['source_result_sha256'], 'Source batch changed while plotting')
    output.mkdir(parents=True)
    try:
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        for path, data in payloads.items():
            require(shutil.disk_usage(ROOT).free-len(data) >= RESERVE, 'Disk reserve changed')
            temporary = path.with_name(path.name+'.tmp')
            with temporary.open('xb') as stream:
                stream.write(data)
            os.replace(temporary, path)
        for name, expected in hashes.items():
            require(sha(ROOT/name) == expected, 'Saved figure/audit hash mismatch')
    except BaseException as error:
        failure = json_bytes(dict(status='failed', error=repr(error), source_root=str(source)))
        if shutil.disk_usage(ROOT).free-len(failure) >= RESERVE:
            (output/'failure.json').write_bytes(failure)
        raise
    print(json.dumps(dict(status=receipt['status'], output=str(output), combined_bytes=used)))


if __name__ == '__main__':
    main()
