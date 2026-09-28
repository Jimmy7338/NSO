#!/usr/bin/env python3
"""Plot every declared option; reference ranges are not confidence intervals."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=json.loads((a.source/'result.json').read_text())
    if result['status']!='complete':raise ValueError('complete analysis required')
    with (a.source/'outcomes.csv').open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=36:raise ValueError('all12 outcomes and3 references required')
    a.output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    options=['continue_coverage','observe_asset_A','observe_asset_B']
    labels=['Coverage','Observe A','Observe B'];colors=['#596E79','#318C87','#DC9254']
    for ax,parent in zip(axes,['D18-P00','D18-P01']):
        for i,assignment in enumerate(['A_open_B_closed','A_closed_B_open']):
            for j,option in enumerate(options):
                selected=[r for r in rows if r['parent']==parent and r['assignment']==assignment and r['option']==option]
                ref=next(r for r in selected if r['reference']=='2026');value=float(ref['joint'])
                values=[float(r['joint']) for r in selected];x=i+(j-1)*.24
                ax.bar(x,value,.22,color=colors[j],label=labels[j] if i==0 else None,
                    hatch='' if ref['eligible']=='True' else '//')
                ax.errorbar(x,value,yerr=[[value-min(values)],[max(values)-value]],fmt='none',ecolor='#222222',capsize=3)
                ax.text(x,value+.018,f'{value:.3f}',ha='center',va='bottom',fontsize=9)
        ax.set_title(parent);ax.set_xticks([0,1],['A open / B closed','A closed / B open'])
        ax.set_ylim(0,1.08);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('2D coverage × mean asset F1 @ 5 cm')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.925),ncol=3,frameon=False)
    fig.suptitle('Declared first options with a shared geometric continuation',y=.985,fontsize=12)
    fig.text(.5,.015,'12 physical outcomes; bars: reference 2026; error bars: range over 3 reference samples, not confidence intervals.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.84]);fig.savefig(a.output/'option_outcomes.png',dpi=160,bbox_inches='tight')
    fig.savefig(a.output/'option_outcomes.svg',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
