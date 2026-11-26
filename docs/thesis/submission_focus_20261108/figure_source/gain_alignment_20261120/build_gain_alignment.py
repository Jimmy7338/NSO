from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent
p=R/'policy_audit/point_support_summary.json';q=R/'replay_attempt_001/summary.json'
a=json.loads(p.read_text());b=json.loads(q.read_text());rows=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
f,(ax,bx)=plt.subplots(1,2,figsize=(9,3.4),gridspec_kw={'width_ratios':[1,1.25]},constrained_layout=True)
keys=['unsupported_object_gain_fraction','supported_object_already_before_fraction','supported_object_new_unrealized_fraction','supported_object_new_realized_fraction']
labels=['Unsupported by target geometry','Already reconstructed target surface','Unreconstructed target surface','Newly reconstructed target surface']
colors=['#b85a63','#d5a854','#779bb4','#3e928e']
for i,arm in enumerate(['G','S']):
 x=a['fixed_equal_state_mean'][arm];left=0
 for k,label,c in zip(keys,labels,colors):
  v=100*x[k];ax.barh(i,v,left=left,color=c,height=.55,label=label if i==0 else None);ax.text(left+v/2,i,f'{v:.1f}%',ha='center',va='center',fontsize=8,color='white' if k!=keys[1] else '#202020');left+=v
  rows.append({'panel':'partition','arm':arm,'metric':k,'prefix':'all_equal_state_area','condition':'original','value':v})
ax.set_yticks([0,1],['G','S']);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Predicted template-area mass (%)');ax.set_title('(a) Where predicted reward goes',loc='left',fontweight='bold')
ax.legend(loc='upper left',bbox_to_anchor=(-.03,-.19),frameon=False,fontsize=7.8,ncol=1)
groups=[('G','gross'),('S','gross'),('G','net'),('S','net')]
for j,(arm,metric)in enumerate(groups):
 old=np.array([s['arms'][arm][f'old_MAE_object_{metric}_m2']for s in b['states']]);new=np.array([s['arms'][arm][f'new_MAE_object_{metric}_m2']for s in b['states']])
 for k,st in enumerate(b['states']):
  bx.plot([j-.13,j+.13],[old[k],new[k]],color='#b9bfc5',lw=.7,zorder=1)
  for cond,val in [('original',old[k]),('native_novelty',new[k])]:rows.append({'panel':'MAE','arm':arm,'metric':metric,'prefix':f"{st['layout']}_{st['paid']}",'condition':cond,'value':val})
 bx.scatter(np.full(6,j-.13),old,color='#69727a',s=16,zorder=2,label='Original' if j==0 else None)
 bx.scatter(np.full(6,j+.13),new,color='#138d92',marker='s',s=16,zorder=2,label='+ native-mesh consumption' if j==0 else None)
 bx.plot([j-.21,j-.05],[old.mean()]*2,color='#20252a',lw=2.1);bx.plot([j+.05,j+.21],[new.mean()]*2,color='#20252a',lw=2.1)
bx.set_xticks(range(4),['G gross','S gross','G net','S net']);bx.set_ylabel('Object-area prediction MAE (m²)');bx.set_ylim(bottom=0);bx.grid(axis='y',alpha=.18);bx.set_title('(b) Paired acquired-prefix errors',loc='left',fontweight='bold');bx.legend(frameon=False,loc='upper left',bbox_to_anchor=(-.05,-.19),fontsize=8)
for ext in ['pdf','svg','png']:f.savefig(O/f'gain_alignment.{ext}',dpi=180,bbox_inches='tight')
with (O/'gain_alignment.csv').open('w')as fp:w=csv.DictWriter(fp,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(O/'CAPTION.txt').write_text('Reward diagnosis and the isolated native-mesh novelty correction on all66 previously acquired candidate branches from six inspected prefixes. (a) The ratio of equal-prefix mean predicted areas under four disjoint target-geometry/measurement categories; this is probability-weighted template mass, not unique physical area. Native consumption also removes some unsupported template points near other measured scene surfaces. (b) Each dot is one prefix MAE over every candidate; black bars are equal-prefix means. Original and corrected predictions reuse the same actual endpoints. Gross and net recovered source area remain separate. No candidate winner changes in these fixed states, and these development diagnostics alone do not establish closed-loop exploration benefit.\n')
(O/'validation.json').write_text(json.dumps({'status':'complete_all_sources_all_prefixes','independent_states':False,'reused_branches':66,'prefixes':6,'sources':[{'path':str(s),'sha256':hashlib.sha256(s.read_bytes()).hexdigest()}for s in [p,q]],'rows':len(rows)},indent=2)+'\n')
print('generated gain_alignment figure and64 data rows')
