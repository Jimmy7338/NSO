"""Reproducible measured B11 figures; no generated experimental imagery."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
COLORS={'chain_a':'#335f8c','cycle_a':'#177e89','chain_b':'#ca7b37','cycle_b':'#8b658b'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def export(fig,out,stem):
 for suffix in ('pdf','svg','png'):fig.savefig(out/(stem+'.'+suffix),bbox_inches='tight',dpi=220)
 plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);summary=read(a.summary);assert summary['state_results']==8 and summary['unique_actual_branches']==24 and summary['complete_layouts']==4
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 group=summary['layout_rows'];labels=[x['layout'].replace('_',' ')for x in group];x=np.arange(4);w=.24;fig,ax=plt.subplots(figsize=(6.8,3.3))
 for dx,key,label,color in [(-w,'old_selected_area_m2','Visibility-selected','#3d6d98'),(0,'new_selected_area_m2','Acquisition-selected','#16938b'),(w,'candidate_best_observed_area_m2','Best of three','#d4a56e')]:
  vals=[g[key]for g in group];ax.bar(x+dx,vals,w,label=label,color=color,edgecolor='white',linewidth=.5)
 ymax=max(g['candidate_best_observed_area_m2']for g in group)
 for i,g in enumerate(group):ax.text(i,g['candidate_best_observed_area_m2']+ymax*.04,f"same route: {g['unchanged_states']}/2",ha='center',va='bottom',fontsize=8)
 ax.set_xticks(x,labels);ax.set_ylabel('Actual gross-new object area (m²)');ax.set_ylim(0,ymax*1.24);ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.18));ax.grid(axis='y',color='#e9ecef',lw=.6);ax.set_axisbelow(True);fig.tight_layout();export(fig,a.output,'selected_actual_area')
 allpoints=[];inputs={str(a.summary.resolve()):sha(a.summary)}
 for ref in summary['inputs']:
  f=Path(ref['path']);assert sha(f)==ref['sha256'];s=read(f);inputs[str(f)]=sha(f);pred=Path(s['prediction_binding']['path']);assert sha(pred)==s['prediction_binding']['sha256'];d=read(pred);inputs[str(pred)]=sha(pred)
  for mode in ('visibility','acquisition'):
   for row in d['conditions'][mode+'_S']['rows']:
    cid=str(row['candidate_id']);actual=s['actual_by_candidate'][cid]['object_gross_new_m2'];allpoints.append(dict(layout=s['layout'],prefix_id=s['prefix_id'],candidate_id=int(cid),mode=mode,actual=actual,predicted=row['predicted_gain_m2']))
 finite=[p for p in allpoints if p['actual']is not None];maximum=max(max(p['actual'],p['predicted'])for p in finite)*1.04;fig,axes=plt.subplots(1,2,figsize=(6.8,3.4),sharex=True,sharey=True)
 for ax,mode,title in zip(axes,('visibility','acquisition'),('Visibility forecast','Acquisition forecast')):
  pts=[p for p in finite if p['mode']==mode];mae=np.mean([abs(p['predicted']-p['actual'])for p in pts]);ax.plot([0,maximum],[0,maximum],color='#9da7b0',lw=.8,zorder=0)
  for layout,color in COLORS.items():
   pp=[p for p in pts if p['layout']==layout];ax.scatter([p['actual']for p in pp],[p['predicted']for p in pp],s=22,color=color,alpha=.85,edgecolors='white',linewidths=.4,label=layout.replace('_',' '))
  ax.set_title(f'{title}\nMAE = {mae:.3f} m²',fontsize=10);ax.set_xlabel('Actual gross-new object area (m²)');ax.set_xlim(0,maximum);ax.set_ylim(0,maximum);ax.set_aspect('equal',adjustable='box');ax.grid(color='#e9ecef',lw=.5);ax.set_axisbelow(True)
 axes[0].set_ylabel('Predicted object area (m²)');axes[1].legend(frameon=False,fontsize=7,loc='lower right');fig.tight_layout();export(fig,a.output,'forecast_vs_actual')
 receipt=dict(status='measured_short_figures_built',inputs_sha256=inputs,source_sha256=sha(__file__),independent_layouts=4,dependent_states=8,unique_actual_branches=24,points=allpoints,source_referenced_tolerance_m=.05,areas='gross_new_open31_plus_closed118_m2',mean_unit='two prespecified states per layout',confidence_intervals='none; single recorded branch per candidate and dependent states',files_sha256={p.name:sha(p)for p in a.output.iterdir()if p.is_file()})
 (a.output/'FIGURE_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=receipt['status'],output=str(a.output))))
if __name__=='__main__':main()
