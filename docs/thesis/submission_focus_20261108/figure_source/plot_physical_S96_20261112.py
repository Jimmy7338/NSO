"""Reproduce the single interrupted real-robot trajectory from saved-scan evaluation."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/ouc/nso_robot/analyses/resume_20261112/laser_coverage.json')
d = json.loads(SOURCE.read_text())
rows = d['curve']
with (HERE/'Physical_S96_20261112.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['paid_actions', 'observed_union_m2'])
    writer.writerows((r['paid_actions'], r['observed_union_m2']) for r in rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig, ax = plt.subplots(figsize=(5.7,2.65), layout='constrained')
x=[r['paid_actions'] for r in rows];y=[r['observed_union_m2'] for r in rows]
ax.step(x,y,where='post',color='#23618c',linewidth=1.7)
ax.scatter([x[-1]],[y[-1]],marker='x',color='#a63f36',s=38,zorder=3)
for paid in [29,56]:
    ax.axvline(paid,color='#777777',linestyle=':',linewidth=.8)
ax.annotate('61.72',xy=(0,y[0]),xytext=(3,65),fontsize=8)
ax.annotate('117.92; interrupted after action 60',xy=(60,y[-1]),xytext=(20,125),fontsize=8,arrowprops={'arrowstyle':'-','color':'#777777','lw':.7})
ax.set(xlim=(0,63),ylim=(55,132),xlabel='Paid primitive actions',ylabel=r'Observed laser union (m$^2$)')
ax.set_xticks([0,12,24,36,48,60]);ax.grid(axis='y',alpha=.2,linewidth=.6)
fig.savefig(HERE.parent/'Physical_S96_20261112.pdf')
(HERE/'Physical_S96_20261112_provenance.json').write_text(json.dumps(dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),samples=len(rows),unit='one robot trajectory; dependent checkpoints',metric=d['area_semantics'],sampling=d['observation_sampling'],roi=d['ROI'],sensor=d['sensor'],endpoint='last accepted observation at paid60; episode interrupted later on goal61',dotted_lines='recorded replans at paid29 and56, next_footprint_not_known_safe'),indent=2))
