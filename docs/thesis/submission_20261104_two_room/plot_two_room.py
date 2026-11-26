from pathlib import Path
import json, shutil
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
paper=Path(__file__).resolve().parent; root=paper.parent; thesis=root/'graduation_integrated_20261104_two_room'; ev=paper; m=json.loads((ev/'mechanism_audit.json').read_text()); a=m['selected']; fig,axes=plt.subplots(1,2,figsize=(8.4,3.35),sharey=True); colors=['#8da9bb','#dc9246'];
for ax,mode in zip(axes,['G','S']):
 for j,v in enumerate(a):
  ax.bar(j,v['common_proxy'],color=colors[0],width=.55,label='Common coverage + quality' if j==0 else None); ax.bar(j,v['prior_'+mode],bottom=v['common_proxy'],color=colors[1],width=.55,label='G/S object prior' if j==0 else None); total=v['total_'+mode];ax.text(j,total+.35,f'{total:.3f}',ha='center',fontsize=10);ax.text(j,1.8,f"{v['common_proxy']:.3f}",ha='center',fontsize=9);ax.text(j,v['common_proxy']+v['prior_'+mode]/2,f"{v['prior_'+mode]:.3f}",ha='center',fontsize=9)
 ax.set_xticks([0,1],['Candidate 3','Candidate 7']);ax.set_title(f'({"a" if mode=="G" else "b"}) {mode}: selects candidate {3 if mode=="G" else 7}',fontsize=11);ax.set_ylim(0,25);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Policy score (proxy units)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.02),fontsize=9);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(paper/'Fig13.pdf',bbox_inches='tight');fig.savefig(paper/'Fig13.svg',bbox_inches='tight');fig.savefig(paper/'Fig13.png',dpi=180,bbox_inches='tight');shutil.copy2(paper/'Fig13.png',thesis/'assets/two_room_scores.png');shutil.copy2(paper/'Fig13.svg',thesis/'assets/two_room_scores.svg');shutil.copy2(paper/'Fig13.pdf',thesis/'assets/two_room_scores.pdf')
