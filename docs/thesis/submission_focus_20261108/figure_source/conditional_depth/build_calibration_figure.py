"""Plot all declared validation/held-out conditions from frozen B8 metrics."""
from pathlib import Path
import hashlib
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'audit/summary.json'
data = json.loads(SOURCE.read_text())
conditions = ['front', 'front_side', 'around', 'occluded']
labels = ['Front', 'Front + side', 'Around object', 'Occluded']
arms = ['fixed_0.65', 'prior_only', 'posterior']
names = ['Fixed 0.65 m', 'Conditioned prior', 'Observation posterior']
colors = ['#8b9298', '#d4872f', '#007c91']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none'})
fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.5), constrained_layout=True)
x = np.arange(4)
rows = []
upper = 0
for ax, split, title in zip(axes[0], ['validation', 'heldout'],
                           ['(a) Validation: six objects', '(b) Held out: six objects']):
    by = data['splits'][split]['by_condition']
    for j, (arm, name, color) in enumerate(zip(arms, names, colors)):
        means, deviations = [], []
        for cond in conditions:
            r = by[arm][cond]
            values = [v['crps_m'] for v in r['per_object'].values()]
            assert len(values) == 6
            mean = float(np.mean(values))
            assert np.isclose(mean, r['crps_m'], atol=1e-12)
            sd = float(np.std(values, ddof=1))
            means.append(mean)
            deviations.append(sd)
            rows.append(dict(split=split, condition=cond, arm=arm,
                             objects=6, mean_crps_m=mean, sd_crps_m=sd,
                             interval95_covered=r['intervals']['0.95']['covered_cases'],
                             interval95_cases=r['intervals']['0.95']['case_denominator'],
                             interval95_mean_width_m=r['intervals']['0.95']['equal_object_mean_width_m']))
        upper = max(upper, max(np.array(means) + deviations))
        ax.bar(x+(j-1)*.24, means, width=.22, color=color, label=name,
               yerr=deviations, capsize=2, error_kw={'lw': .8, 'ecolor': '#444444'})
    ax.set(title=title, ylabel='CRPS (m; lower is better)', xticks=x, xticklabels=labels)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
for ax in axes[0]:
    ax.set_ylim(0, upper*1.08)
axes[0, 0].legend(loc='upper right', frameon=False, fontsize=9)

for j, (split, color) in enumerate(zip(['validation', 'heldout'], ['#76aeb5', '#006d80'])):
    values = [data['splits'][split]['by_condition']['posterior'][c]['intervals']['0.95']
              for c in conditions]
    coverage = [100*v['covered_cases']/v['case_denominator'] for v in values]
    widths = [v['equal_object_mean_width_m'] for v in values]
    xpos = x + (j-.5)*.34
    axes[1, 0].bar(xpos, coverage, width=.31, color=color, label=split.capitalize())
    for xx, yy, v in zip(xpos, coverage, values):
        axes[1, 0].text(xx, yy+2, f"{v['covered_cases']}/{v['case_denominator']}",
                        ha='center', va='bottom', fontsize=9)
    axes[1, 1].bar(xpos, widths, width=.31, color=color, label=split.capitalize())
axes[1, 0].axhline(95, color='#a74a3e', ls='--', lw=1)
axes[1, 0].set(title='(c) Posterior 95% interval: empirical coverage',
               ylabel='Truth covered (%)', ylim=(0, 116))
axes[1, 1].set(title='(d) Posterior 95% interval: mean width',
               ylabel='Interval width (m)', ylim=(0, None))
axes[1, 1].legend(frameon=False)
for ax in axes[1]:
    ax.set(xticks=x, xticklabels=labels)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
for suffix in ('pdf', 'svg', 'png'):
    fig.savefig(HERE/f'calibration_distribution.{suffix}', dpi=160)
plt.close(fig)
with (HERE/'calibration_plot_data.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(HERE/'CAPTION.md').write_text('''Independent conditional-depth evaluation. Panels (a,b)
report mean CRPS with sample standard deviation across six objects in each
split, for every declared observation condition. Views and conditions of the
same object are dependent. One common scale was selected from eight candidates
on 12 separate training objects; neither shown split selected the scale.
Declared failed/unusable cases use the public-prior evaluation fallback and
remain in the denominator. Panels (c,d) show empirical truth coverage counts
and mean widths of the posterior's nominal 95% central intervals; the dashed
line is 95%, not an achieved guarantee. In particular, directly measured
opposing-face point estimates can under-cover despite small extent error.
The results concern depth conditional on B6 orientation, width and part
assumptions, ideal visible instance masks and exact simulated poses. They do
not establish physical accuracy, nominal probability calibration or G/S
exploration efficiency. Prior-only means the same acquired-span-conditioned
geometric prior used by the posterior, not an unconditional distribution.
''')
files = [p for p in HERE.iterdir() if p.is_file() and p.name != 'validation.json']
(HERE/'validation.json').write_text(json.dumps({
    'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'rows': len(rows), 'six_objects_per_split': True, 'all_four_conditions': True,
    'error_bars': 'sample standard deviation across six objects, not confidence intervals',
    'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
}, indent=2)+'\n')
print(json.dumps({'rows': len(rows), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest()}))
