#!/usr/bin/env python3
import json,hashlib,csv
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent; R=P.parent
sources=[R/'evidence/fresh_matrix',R/'evidence/deployed_policy_matrix']; names=['Original v10.3.1','Exploration joint v1']; sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protos=[json.loads((d/'protocol.json').read_text()) for d in sources]; assert protos[0]['source_sha256']==protos[1]['source_sha256']
rows=[]; diagnostics=[]
for d,name in zip(sources,names):
 results=json.loads((d/'results.json').read_text()); assert len(results)==8
 for r in results:
  k=f"{r['arrangement']}_{r['budget']}_{r['condition']}"; steps=json.loads((d/(k+'_steps.json')).read_text()); assert len(steps)==r['steps']
  rows.append(dict(policy=name,arrangement=r['arrangement'],budget=r['budget'],mode=r['condition'],steps=r['steps'],path_m=sum(x['action']=='forward' for x in steps)*.2,coverage_gain_m2=r['coverage_gain_m2'],visible_new_m2=r['new_surface_m2'],initial_f1_5cm=r['before_metrics']['f1_05cm'],final_f1_5cm=r['after_metrics']['f1_05cm'],collision_count=r['collisions'],anchor=r['anchor'],initial_packet_sha256=r['initial_packet_sha256'],termination=r['termination']['reason']))
 for arr in ('shelf_west','shelf_east'):
  for budget in (48,96):
   g=json.loads((d/f'{arr}_{budget}_G_steps.json').read_text());s=json.loads((d/f'{arr}_{budget}_S_steps.json').read_text()); first=next((i+1 for i,(x,y) in enumerate(zip(g,s)) if x['action']!=y['action']),None)
   diagnostics.append(dict(policy=name,arrangement=arr,budget=budget,actions_identical=first is None and len(g)==len(s),first_action_divergence=first))
for arr in ('shelf_west','shelf_east'):
 subset=[x for x in rows if x['arrangement']==arr];assert len(set(x['initial_packet_sha256'] for x in subset))==1;assert len({tuple(x['anchor']) for x in subset})==1
for x in rows: x['anchor']=' '.join(map(str,x['anchor']))
with (P/'results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(P/'action_comparison.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
cols=['policy','arrangement','budget','mode','path_m','coverage_gain_m2','visible_new_m2','final_f1_5cm']
with (P/'table.md').open('w') as f:
 f.write('| Policy | Arrangement | Budget | Mode | Path (m) | Coverage gain (m²) | New visible surface (m²) | Final F1 @ 5 cm |\n|---|---|---:|:---:|---:|---:|---:|---:|\n')
 for x in rows:f.write('| '+ ' | '.join([x['policy'],x['arrangement'].replace('shelf_',''),str(x['budget']),x['mode']]+[f"{x[k]:.3f}" for k in cols[4:]])+' |\n')
 f.write('\nEach row is one deterministic trial; the two arrangements share Q0 and are not independent scenes. Surface is newly truth-visible area after the initial packet. F1 uses a 0.05 m distance threshold. Path is forward actions × 0.2 m. Physical adapters and real sensors were not used.\n')
plt.rcParams.update({'font.size': 9.5, 'axes.labelsize': 9.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9})
fig,grid=plt.subplots(2,2,figsize=(6.5,4.3),layout='constrained'); axes=grid.flatten(); metrics=[('coverage_gain_m2','Coverage gain\nS − G (m²)'),('visible_new_m2','New visible surface\nS − G (m²)'),('final_f1_5cm','Final F1 @ 5 cm\nS − G'),('path_m','Path length\nS − G (m)')]; groups=[('shelf_west',48),('shelf_west',96),('shelf_east',48),('shelf_east',96)]
for ax,(key,label) in zip(axes,metrics):
 for i,(name,marker,color) in enumerate(zip(names,['o','s'],['#0072B2','#D55E00'])):
  vals=[]
  for arr,b in groups:
   sub=[x for x in rows if x['policy']==name and x['arrangement']==arr and x['budget']==b];g=next(x for x in sub if x['mode']=='G');s=next(x for x in sub if x['mode']=='S');vals.append(s[key]-g[key])
  ax.scatter(vals,np.arange(4)+(i-.5)*.15,label=name,marker=marker,color=color,s=31,zorder=3)
 ax.axvline(0,color='0.6',lw=.8);ax.set_xlabel(label);ax.set_yticks(range(4),['West / 48','West / 96','East / 48','East / 96'] if ax in (axes[0],axes[2]) else []);ax.invert_yaxis();ax.grid(axis='x',color='0.9');ax.spines[['top','right']].set_visible(False)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
fig.savefig(P/'paired_differences.pdf',bbox_inches='tight');fig.savefig(P/'paired_differences.svg',bbox_inches='tight');plt.close(fig)
(P/'caption.txt').write_text('Paired S minus G outcomes for the original v10.3.1 policy and exploration_joint_v1. Points show all four arrangement–budget conditions for each policy; zero denotes a tie. Each condition is one deterministic Q0 simulator trial, and the two arrangements are not independent scenes. Coverage is the increase in observed area; new visible surface is truth-referenced area newly visible after the initial packet. F1 uses a 5 cm distance threshold. Path length counts 0.2 m forward moves. Real-sensor, range, extent, and natural-semantic adapters were disabled; no uncertainty is estimated.\n')
files=[p for d in sources for p in sorted(d.glob('*.json'))]+[R/'matrix.py',R/'deployed_policy_matrix_runner.py',R/'evidence/deployed_variant_probe_receipt.json',Path(__file__)]
manifest={'input_sha256':{str(p):sha(p) for p in files},'outputs_sha256':{p.name:sha(p) for p in P.iterdir() if p.is_file() and p.name!='manifest.json'},'limitations':['16 deterministic trajectories, only Q0; two layouts are not independent scenes','No real sensors or physical adapters; no statistical uncertainty','F1 key f1_05cm means 5 cm (0.05 m)'],'source_protocol_hashes_equal':True}
(P/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Saved trajectory records only: no simulation is run here.
fig,axs=plt.subplots(1,2,figsize=(6.5,3.0),layout='constrained',sharex=True)
for d,name,color in zip(sources,names,['#0072B2','#D55E00']):
 for mode,style in [('G','-'),('S','--')]:
  st=json.loads((d/f'shelf_west_96_{mode}_steps.json').read_text())
  x=[0]+[r['step'] for r in st]
  res=next(r for r in json.loads((d/'results.json').read_text()) if r['arrangement']=='shelf_west' and r['budget']==96 and r['condition']==mode)
  for ax,key,start in zip(axs,['covered_area_m2','new_visible_surface_area_m2'],[res['before_metrics']['covered_area_m2'],0.]):
   ax.plot(x,[start]+[r[key] for r in st],linestyle=style,color=color,lw=1.6,label=f'{name} / {mode}')
for ax,label in zip(axs,['Observed coverage (m²)','New truth-visible surface (m²)']):
 ax.set_xlabel('Paid actions');ax.set_ylabel(label);ax.grid(color='0.9');ax.spines[['top','right']].set_visible(False);ax.set_xlim(0,96)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
fig.savefig(P/'west96_trajectories.pdf',bbox_inches='tight');fig.savefig(P/'west96_trajectories.svg',bbox_inches='tight');fig.savefig(P/'west96_trajectories.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Raster preview of existing vector plot.
import subprocess
subprocess.run(['pdftoppm','-f','1','-singlefile','-png','-r','180',str(P/'paired_differences.pdf'),str(P/'paired_differences')],check=True)
manifest=json.loads((P/'manifest.json').read_text())
manifest['input_sha256'][str(Path(__file__))]=sha(Path(__file__))
manifest['outputs_sha256']={p.name:sha(p) for p in P.iterdir() if p.is_file() and p.name!='manifest.json'}
(P/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
