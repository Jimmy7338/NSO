#!/usr/bin/env python3
"""Scientific figure from the complete V9.1 analysis; no simulation or tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(analysis, destination):
    summary_path = analysis / 'summary.json'; rows_path = analysis / 'selected_branches.json'
    summary = json.loads(summary_path.read_text()); rows = json.loads(rows_path.read_text())
    if summary['available_branches'] != 44 or summary['pending_gates']:
        raise ValueError('complete reviewed batch required, including all failed gates')
    methods = ('G', 'O', 'N', 'S', 'X', 'M')
    parents = ('Q0', 'Q1', 'Q2', 'Q3')
    colors = ('#4C78A8', '#F58518', '#54A24B', '#B279A2')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for ax, field, label in zip(axes, ('new_area_m2', 'final_f1_05cm'), ('New visible surface (m²)', 'Terminal global F1 @ 5 cm')):
        vals = np.array([[np.mean([r[field] for r in rows if r['family'] == 'total' and r['method'] == m and r['context'] == p])
                          for m in methods] for p in parents])
        ax.scatter(np.arange(6), vals.mean(0), marker='_', s=600, linewidths=2.5,
                   color=['#222222' if m != 'S' else '#2B6F7D' for m in methods], zorder=4)
        for i, (p, color) in enumerate(zip(parents, colors)):
            ax.scatter(np.arange(6) + (i - 1.5) * .11, vals[i], s=30, label=p, color=color, zorder=3)
        ax.set_xticks(np.arange(6), methods); ax.set_ylabel(label)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        if field.startswith('final'):
            ax.set_ylim(max(0., np.floor((vals.min() - .004) * 100) / 100), 1.)
        else:
            ax.set_ylim(0, vals.max() * 1.18)
    axes[0].legend(title='Fixed parent blocks', ncol=4, fontsize=8, loc='upper left')
    fig.suptitle('V9.1: pre-outcome total score, four geometry/noise blocks\n48-action window; artificial category markers; exact input poses', fontsize=11)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix('.png'), dpi=180)
    fig.savefig(destination.with_suffix('.pdf'))
    plt.close(fig)
    record = {'analysis_summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
              'selected_rows_sha256': hashlib.sha256(rows_path.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'status': summary['status'], 'means': summary['method_averages']['total'],
              'points': 'each dot averages two arrangements within one fixed parent; no confidence interval',
              'failed_gates': summary['failed_gates'], 'whole_system_superiority': False}
    destination.with_suffix('.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    destination.with_suffix('.caption.md').write_text('V9.1 固定总收益评分的四父场景结果。短横线为八历史等权均值；彩色点为每父场景两种安排的均值，未绘制置信区间。G/O/N/S/X/M 为几何边缘先验、对象性、已测覆盖质量、正确类别、交换类别和缺失类别。全部44条实际路线与失败门槛保留；人工标记、精确输入位姿和48动作局部分支限制了结论范围。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args(); main(args.analysis, args.destination)
