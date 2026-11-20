"""Plot every paid action in the frozen B8/B9 full unknown-exploration comparisons."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; O=Path(__file__).resolve().parent
B=R.parent/'dual_template_uncertainty_20261120/rollouts/joint_depth/layouts'
N=R/'structure/rollouts/temperature/layouts'
rows=[];sources=[];d={}
for layout in ['shelf_left','shelf_right']:
 for name,arm,base in [('G, T=1','G_uncertainty',B),('S, T=1','S_uncertainty',B),('G, T=16','G_temperature',N),('S, T=16','S_temperature',N)]:
  p=base/layout/'runs'/arm/'attempt_001';steps=json.loads((p/'steps.json').read_text());result=json.loads((p/'result.json').read_text())
  assert len(steps)==256 and result['status']=='completed_execution'
  vals=[result['initial_coverage_fraction']]+[x['free_correct_fraction'] for x in steps]
  assert vals[-1]==1.0
  d[layout,name]=vals
  for i,v in enumerate(vals):rows.append({'layout':layout,'condition':name,'paid_action':i,'correct_free_coverage_fraction':v})
  for fn in ['steps.json','result.json']:sources.append({'path':str(p/fn),'sha256':hashlib.sha256((p/fn).read_bytes()).hexdigest()})
 assert d[layout,'G, T=1']==d[layout,'S, T=1']
with (O/'full_exploration_coverage.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(8.2,3.2),sharey=True,constrained_layout=True)
for ax,layout,title in zip(axes,['shelf_left','shelf_right'],['(a) Left layout','(b) Right layout']):
 for name,label,color,style,z in [('G, T=1','T=1 (G and S)','#30343b','-',1),('G, T=16','T=16, G','#276eaf','--',2),('S, T=16','T=16, S','#d66b24',':',3)]:
  vals=np.asarray(d[layout,name])*100
  ax.plot(np.arange(257),vals,label=label,color=color,linestyle=style,lw=1.8,zorder=z)
  first=next(i for i,v in enumerate(vals) if v==100)
  ax.scatter(first,100,marker={'G, T=1':'o','G, T=16':'s','S, T=16':'^'}[name],s=35,facecolors='none',edgecolors=color,zorder=4)
 ax.set_title(title,loc='left',fontweight='bold');ax.set_xlim(0,256);ax.set_ylim(40,103)
 ax.set_xticks([0,64,128,192,256]);ax.set_xlabel('Paid primitive actions');ax.grid(axis='y',alpha=.18)
 ax.text(.98,.10,'Initial coverage: 42.61%',transform=ax.transAxes,ha='right',fontsize=8,color='#555555')
axes[0].set_ylabel('Correctly mapped free space (%)')
axes[1].legend(loc='center right',frameon=False,fontsize=8)
for ext in ['pdf','svg','png']:fig.savefig(O/f'full_exploration_coverage.{ext}',dpi=180)
(O/'FULL_EXPLORATION_CAPTION.txt').write_text('Complete unknown-exploration coverage curves for the two previously inspected layout permutations, using all 256 paid primitive actions. The four new T=16 executions are compared with the frozen B8 T=1 executions; the latter are reused baselines, not new trials. G and S at T=1 have identical curves in each layout. Markers denote first full correct-free coverage. The environment initially contains 42.61% observed free space; the denominator is evaluator-only reference free cells. The two layouts share a floorplan and do not establish cross-scene generalization. The common action/return contract is unchanged.\n')
(O/'coverage_validation.json').write_text(json.dumps({'status':'complete_all256_actions_eight_curves','new_runs':4,'reused_baseline_runs':4,'layouts':2,'rows':len(rows),'sources':sources},indent=2)+'\n')
print(json.dumps({'rows':len(rows),'first_full':{layout:{name:next(i for i,v in enumerate(d[layout,name]) if v==1.0) for name in ['G, T=1','S, T=1','G, T=16','S, T=16']} for layout in ['shelf_left','shelf_right']}}))
