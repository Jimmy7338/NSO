"""Reproducible curves from saved, fixed Q1 comparison; no policy execution."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;rows=json.load(open(D/'execution/results.json'));by={r['condition']:r for r in rows}
fig,axes=plt.subplots(1,2,figsize=(7.2,2.6),layout='constrained');data=[]
for mode,color,style in [('G','#0072B2','-'),('S','#D55E00','-'),('X','#009E73','--')]:
 r=by[mode];steps=json.load(open(D/f'execution/Q1_shelf_west_96_{mode}_steps.json'));area=r['after_metrics']['covered_area_m2']/r['after_metrics']['coverage_2d']
 x=[0]+[s['step'] for s in steps];cov=[r['before_metrics']['coverage_2d']*100]+[100*s['covered_area_m2']/area for s in steps];surf=[0]+[s['new_visible_surface_area_m2'] for s in steps]
 axes[0].plot(x,cov,label=mode,color=color,ls=style,lw=1.6);axes[1].plot(x,surf,label=mode,color=color,ls=style,lw=1.6)
 data.extend(dict(mode=mode,paid_action=i,coverage_percent=a,new_reference_visible_surface_m2=b) for i,a,b in zip(x,cov,surf))
for ax in axes:
 ax.set_xlabel('Paid actions');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_xlim(0,96)
axes[0].set_ylabel('Observed area (%)');axes[1].set_ylabel('New visible surface (m²)');axes[1].legend(frameon=False,ncol=3,loc='lower right')
fig.savefig(D/'Q1_curves.pdf');fig.savefig(D/'Q1_curves.png',dpi=220)
with open(D/'Q1_curves.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
