#!/usr/bin/env python3
"""One 2x2 V26 figure from four sealed autonomous records; no new evaluation."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT/'audit_results/observed_autonomous_v26_20260916'
DEFAULT_OUTPUT = ROOT/'audit_results/observed_autonomous_v26_figures_20260916'
SPEC = [('A_complex_B_simple', 'G'), ('A_complex_B_simple', 'S'),
        ('A_simple_B_complex', 'G'), ('A_simple_B_complex', 'S')]
COLORS = {'G': '#1764ab', 'S': '#d46716'}
RESERVE = 64*1024**2
OUTPUT_CAP = 5*1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_inventory(folder):
    inventory = read(folder/'artifact_hashes.json')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p != folder/'artifact_hashes.json'}
    require(actual == set(inventory), 'Frozen artifact file set changed')
    for name, expected in inventory.items():
        require(sha(folder/name) == expected, 'Frozen artifact changed: '+name)
    return len(inventory)


def load_complete(source):
    manifest = read(source/'manifest.json')
    require(manifest['status'] == 'complete', 'Wait for the complete sealed four-case batch')
    require(len(manifest['cases']) == 4 and [(c['assignment'], c['mode'])
            for c in manifest['cases']] == SPEC, 'Unexpected declared case matrix')
    require(manifest['config']['budget'] == 400, 'Frozen 400-action budget changed')
    require(all(c['physical_status'] == c['replay_status'] == 'complete'
                for c in manifest['cases']), 'All four independent replays must be complete')
    for field in ('source_sha256', 'input_sha256'):
        for name, expected in manifest[field].items():
            require(sha(ROOT/name) == expected, 'Frozen source/input changed: '+name)
    require(sha(source/'sources.zip') == manifest['source_archive_sha256'], 'Frozen source archive changed')
    checked = verify_inventory(source)
    aggregate = read(source/'result.json')
    require(aggregate['status'] == 'complete_four_autonomous_tasks_and_independent_replays',
            'Complete autonomous aggregate required')
    cases, provenance, pids = [], [], []
    for index, (assignment, mode) in enumerate(SPEC):
        folder = source/f'case_{index:02d}'
        case, verification = read(folder/'result.json'), read(folder/'verification.json')
        require(case['status'] == 'complete' and (case['index'], case['assignment'], case['mode'])
                == (index, assignment, mode), 'Case identity/status mismatch')
        require(verification['status'] == 'passed' and verification['independent_process']
                and verification['fresh_world'] and verification['fresh_autonomous_runtime']
                and not verification['saved_actions_used_to_drive_policy'], 'Autonomous replay is unverified')
        require(all(verification[k] for k in ('all_decisions_equal', 'all_packets_equal',
                'all_meshes_and_metrics_equal')), 'Independent replay mismatch')
        pids += [verification['main_process_id'], verification['replay_process_id']]
        trace = case['trace']
        require(case['autonomous_policy'] and case['imposed_prefix_actions'] == 0
                and not case['gt_service_catalogue_used'], 'Unexpected fixed-route experiment')
        require(len(trace) == case['raw_frames'] == case['paid_actions']+1
                and [r['action_id'] for r in trace] == list(range(len(trace))), 'Discontinuous paid trajectory')
        require(verification['independent_paid_actions'] == case['paid_actions']
                and verification['saved_packets_verified'] == len(trace), 'Replay action/frame count mismatch')
        require(0 <= case['paid_actions'] <= 400 and all(0 <= r['coverage_2d'] <= 1 for r in trace),
                'Trajectory outside frozen axes/metric domain')
        require(trace[-1]['coverage_2d'] == case['final_coverage_2d'], 'Terminal coverage alias mismatch')
        c = case['final_coverage_2d']; q = case['final_main_observed']['05cm']['outline_macro_quality']
        j = case['final_main_observed']['05cm']['joint_outline']
        require(abs(j-c*q) < 1e-10, 'Saved terminal J does not equal C times Q')
        require(case['final_main_observed']['mission_asset_count'] == 2, 'Both fixed facilities must remain scored')
        require(sha(folder/'result.json') == aggregate['cases'][index]['result_sha256'], 'Aggregate case hash mismatch')
        cases.append(case)
        provenance.append(dict(index=index, result_sha256=sha(folder/'result.json'),
            verification_sha256=sha(folder/'verification.json'),
            physical_pid=verification['main_process_id'], replay_pid=verification['replay_process_id'],
            trace_points=len(trace), trajectory_sha256=case['trajectory_sha256']))
    require(len(set(pids)) == 8, 'The four main records and four successful replay receipts need distinct PIDs')
    base = cases[0]
    require(all(c['shape'] == base['shape'] and c['public_config'] == base['public_config']
                and c['trace'][0]['pose'] == base['trace'][0]['pose'] for c in cases),
            'Shared public map/sensor/start contract differs')
    return cases, dict(source_root=str(source), manifest_sha256=sha(source/'manifest.json'),
        root_result_sha256=sha(source/'result.json'), root_inventory_sha256=sha(source/'artifact_hashes.json'),
        verified_artifacts=checked, case_sources=provenance, plot_script_sha256=sha(Path(__file__)),
        successful_receipt_process_ids=pids, successful_receipt_process_count=len(set(pids)),
        pid_check_scope='Four main records and four successful replay receipts only; not all execution attempts',
        total_replay_attempts=None, total_replay_attempts_not_inferred_from_successful_receipts=True,
        capacity_recovery_evidence=str(ROOT/'audit_results/v26_capacity_recovery_20260916'))


def figure(cases, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    rows, cols = cases[0]['shape']; resolution = cases[0]['public_config']['resolution_m']
    width, height = cols*resolution, rows*resolution
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), constrained_layout=True)
    labels = ('A complex / B simple', 'A simple / B complex')
    for column, first in enumerate((0, 2)):
        top, bottom = axes[0, column], axes[1, column]
        for case in cases[first:first+2]:
            mode, trace = case['mode'], case['trace']
            x = [(row['pose'][1]+.5)*resolution for row in trace]
            y = [(rows-row['pose'][0]-.5)*resolution for row in trace]
            top.plot(x, y, color=COLORS[mode], linewidth=1.2, alpha=.8, label=mode,
                     linestyle='-' if mode == 'G' else '--')
            top.scatter(x[-1], y[-1], marker='o' if mode == 'G' else '^',
                s=100 if mode == 'G' else 180, facecolors='none', edgecolors=COLORS[mode],
                linewidths=1.5, zorder=5, label=mode+' end')
            bottom.step([r['action_id'] for r in trace], [100*r['coverage_2d'] for r in trace],
                where='post', color=COLORS[mode], linewidth=1.5,
                linestyle='-' if mode == 'G' else '--', label=f'{mode} ({case["paid_actions"]} paid actions)')
            bottom.scatter(trace[-1]['action_id'], 100*trace[-1]['coverage_2d'],
                color=COLORS[mode], s=22, zorder=4)
        start = cases[first]['trace'][0]['pose']
        top.scatter((start[1]+.5)*resolution, (rows-start[0]-.5)*resolution,
            marker='*', s=65, color='black', zorder=6, label='shared start')
        top.set(xlim=(0, width), ylim=(0, height), xlabel='World x (m)', ylabel='World y (m)',
                title=labels[column]+' | actual paths')
        top.set_aspect('equal', adjustable='box'); top.grid(alpha=.2)
        top.legend(fontsize=7.5, ncol=2, loc='best')
        bottom.axhline(80, color='.35', linestyle=':', linewidth=1., label='80% coverage gate')
        bottom.set(xlim=(0, 400), ylim=(0, 100), xlabel='Paid actions (turns included)',
                   ylabel='Evaluated 2D coverage C (%)', title=labels[column]+' | observed trajectory')
        bottom.grid(alpha=.2); bottom.legend(fontsize=8, loc='lower right')
    fig.suptitle('V26 autonomous G/S development test: trajectories and measured coverage', fontsize=12)
    fig.savefig(output/'trajectories_and_coverage.png', dpi=170)
    fig.savefig(output/'trajectories_and_coverage.pdf', metadata={'Title': 'V26 autonomous G/S development test'})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); source, output = args.source.resolve(), args.output.resolve()
    require(not output.exists(), 'Existing figures must not be overwritten')
    cases, provenance = load_complete(source)
    require(shutil.disk_usage(output.parent).free >= RESERVE+OUTPUT_CAP, 'Need 64 MiB reserve plus figure allowance')
    output.mkdir(parents=True)
    try:
        figure(cases, output)
        table = ['| 排列 | 方法 | 动作 | C | Q | J | 返航 | 碰撞 | 合格 |',
                 '|---|---|---:|---:|---:|---:|---|---:|---|']
        for case in cases:
            q = case['final_main_observed']['05cm']
            table.append(f'| {case["assignment"]} | {case["mode"]} | {case["paid_actions"]} '
                f'| {case["final_coverage_2d"]:.6f} | {q["outline_macro_quality"]:.6f} '
                f'| {q["joint_outline"]:.6f} | {case["returned_to_anchor"]} '
                f'| {case["collisions"]} | {case["eligible"]} |')
        description = ('# V26 自主 G/S 实际轨迹与覆盖\n\n'
            '两列对应两个类别排列，上排为实际 XY 轨迹，下排为逐付费动作的评价覆盖率。'
            'G 为蓝色实线，S 为橙色虚线；黑星为共同起点，空心圆/三角为各自终点。'
            '重合终点可能叠在起点；平面轨迹不显示原地转向，但动作轴计入转向。\n\n'
            '所有四条记录均保留，采用共同完整任务坐标、0–400 动作和 0–100% 覆盖范围。'
            '折线/阶梯仅连接已保存记录，不平滑、不续画提前终止后的成绩。'
            '未叠加 GT 障碍或隐藏设施。主 Q 只有终点评价，因此不绘制或插值质量曲线。\n\n'
            +'\n'.join(table)+'\n\n'
            'Q 是全部两设施等权的三投影 min(边界 F1@5cm, IoU) 外形符合度，J=C×Q；'
            '不是完整三维重建精度。C 是评价侧 known∩reachable/reachable，未作为规划输入。'
            '返航字段包含位置和朝向，二维路径图本身不能验证朝向。'
            '这是单父布局、单噪声种子的开发批次；回放不增加独立样本数，图片不证明语义创新或整体优势。\n\n'
            '每条主轨迹仅画一次。8 个互异 PID 的检查只覆盖 4 个主记录和 4 份成功回放回执，'
            '不代表总共只执行了 8 次。本批 case_01 曾完成 400 个回放动作后因空间门未能保存验证回执；'
            '该失败尝试与恢复重试不能被成功回执计数抹去，也不能算成新增独立主样本。'
            '恢复证据目录为 audit_results/v26_capacity_recovery_20260916；本图不推算所有回放尝试数。\n\n'
            '绘图只读取封存数据：新增动作、传感、TSDF 和评价调用均为 0。'
            '完整来源与产物摘要见 sources.json 和 artifact_hashes.json。\n')
        (output/'README.md').write_text(description)
        require(sha(source/'manifest.json') == provenance['manifest_sha256']
                and sha(source/'artifact_hashes.json') == provenance['root_inventory_sha256'],
                'Input batch changed during plotting')
        provenance.update(layout='columns=assignments; rows=XY trajectory/actual step coverage',
            all_four_cases_included=True, fixed_paid_axis=[0, 400], fixed_coverage_axis=[0, 1],
            no_quality_interpolation=True, new_physical_actions=0, new_sensor_frames=0,
            new_tsdf_fusions=0, new_evaluations=0)
        (output/'sources.json').write_text(json.dumps(provenance, indent=2)+'\n')
        used = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        require(used < OUTPUT_CAP and shutil.disk_usage(output).free >= RESERVE, 'Figure storage allowance exceeded')
        inventory = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        (output/'artifact_hashes.json').write_text(json.dumps(inventory, indent=2)+'\n')
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'status': 'failed', 'error': repr(error)})+'\n')
        raise
    print(json.dumps({'status': 'complete_saved_record_figure', 'output': str(output), 'bytes': used}))


if __name__ == '__main__':
    main()
