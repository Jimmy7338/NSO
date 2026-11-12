import json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).parent;R=Path(__file__).resolve().parents[2]/'evidence';P=json.loads((D/'protocol.json').read_text())
plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
styles={'Frontier+guard':('#0072B2','-',2),'Original G':('#333333',':',1.7),'Joint G':('#D55E00','--',1.8),'Joint S':('#009E73','-.',1.7),'Joint C':('#CC79A7',(0,(3,1,1,1)),1.5)}
rows=[];data={};fig,axes=plt.subplots(2,2,figsize=(7.3,5.3),sharey=True,layout='constrained')
for ax,(arr,budget) in zip(axes.flat,P['conditions']):
 items=[('Frontier+guard',D/f'{arr}_{budget}'/'result.json',D/f'{arr}_{budget}'/'steps.json'),('Original G',R/'fresh_matrix'/f'{arr}_{budget}_G_result.json',R/'fresh_matrix'/f'{arr}_{budget}_G_steps.json')]
 for mode in 'GSC':
  folder='coverage_ablation_matrix' if mode=='C' else 'deployed_policy_matrix';items.append((f'Joint {mode}',R/folder/f'{arr}_{budget}_{mode}_result.json',R/folder/f'{arr}_{budget}_{mode}_steps.json'))
 data[(arr,budget)]={}
 for name,rp,sp in items:
  r=json.loads(rp.read_text());s=json.loads(sp.read_text());initial=r['before_metrics']['coverage_2d'];base_area=r['before_metrics']['covered_area_m2']; assert base_area>0 and initial>0;reachable_area=base_area/initial
  xs=[0]+[v['step'] for v in s];ys=[100*initial]+[100*v['covered_area_m2']/reachable_area for v in s];surfaces=[0]+[v['new_visible_surface_area_m2'] for v in s]
  assert len(s)==r['steps'] and xs==list(range(len(xs))) and abs(ys[-1]-100*r['after_metrics']['coverage_2d'])<1e-8
  row=dict(layout=arr,budget=budget,method=name,initial_percent=100*initial,action_90=next((x for x,y in zip(xs,ys) if y>=90-1e-10),None),action_95=next((x for x,y in zip(xs,ys) if y>=95-1e-10),None),action_99=next((x for x,y in zip(xs,ys) if y>=99-1e-10),None),used=len(s),final_percent=ys[-1],new_surface_m2=surfaces[-1]);rows.append(row);data[(arr,budget)][name]=(xs,ys,surfaces)
  color,ls,lw=styles[name];ax.step(xs,ys,where='post',color=color,linestyle=ls,linewidth=lw,label=name);ax.plot(xs[-1],ys[-1],marker='o',markersize=2.5,color=color)
 ax.axhline(90,color='#bcbcbc',lw=.6);ax.axhline(95,color='#bcbcbc',lw=.6);ax.axhline(99,color='#bcbcbc',lw=.6);ax.set_title(f'{arr.replace("shelf_", "").title()} · {budget} actions');ax.set_xlim(0,budget);ax.set_ylim(58,101);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12)
for ax in axes[-1,:]:ax.set_xlabel('Paid actions')
for ax in axes[:,0]:ax.set_ylabel('Reachable coverage (%)')
h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=5,frameon=False)
fig.savefig(D/'coverage_progress.pdf');fig.savefig(D/'coverage_progress.svg');fig.savefig(D/'coverage_progress.png',dpi=300);plt.close(fig)
with (D/'progress_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows([{k:("null" if v is None else v) for k,v in r.items()} for r in rows])
# At common paid action indices, compare joint S minus G coverage and visible surface.
notes=[]
for (arr,budget),group in data.items():
 gx,gy,gs=group['Joint G'];sx,sy,ss=group['Joint S'];n=min(len(gx),len(sx));different=[i for i in range(n) if abs(gs[i]-ss[i])>1e-8];coverage_diff=[i for i in range(n) if abs(gy[i]-sy[i])>1e-8]
 entry=dict(layout=arr,budget=budget,first_surface_difference_action=(different[0] if different else None),first_coverage_difference_action=(coverage_diff[0] if coverage_diff else None),coverage_at_first_surface_difference_G_percent=(gy[different[0]] if different else None),coverage_at_first_surface_difference_S_percent=(sy[different[0]] if different else None),final_surface_difference_m2=ss[-1]-gs[-1],final_coverage_difference_percentage_points=sy[-1]-gy[-1],first_90_G=next((i for i,v in enumerate(gy) if v>=90-1e-10),None),first_95_G=next((i for i,v in enumerate(gy) if v>=95-1e-10),None))
 notes.append(entry)
(D/'semantic_timing.json').write_text(json.dumps(notes,indent=2));print(json.dumps(notes,indent=2));print('\n'.join(str(r) for r in rows))
