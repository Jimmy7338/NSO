#!/usr/bin/env python3
"""Plot only sealed analytic geometry and policy witnesses; never solve again."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/virtual3d/v29_information_scene_20260917.json'
SOURCE=ROOT/'audit_results/v29_scene_information_20260917'


def run():
    c=json.loads(CONFIG.read_text())
    rows=json.loads((SOURCE/'main_policy_witnesses.json').read_text())['records']
    fig,axes=plt.subplots(1,2,figsize=(11,5.7),sharex=True,sharey=True)
    for h,ax in enumerate(axes):
        for wall in c['class_independent_wall_bounds']:
            ax.add_patch(Rectangle((wall[0],wall[2]),wall[1]-wall[0],wall[3]-wall[2],
                                  facecolor='#57616b',edgecolor='none',zorder=1))
        for i,station in enumerate(c['stations']):
            x=station['center_x'];body=c['body_relative_bounds']
            ax.add_patch(Rectangle((x+body[0],body[2]),body[1]-body[0],body[3]-body[2],
                facecolor='#d5dce3',edgecolor='#3a4149',lw=1.2,zorder=2))
            kind=c['hypotheses'][h][i]
            if kind=='complex':
                b=c['complex_attachment_relative_bounds']
                ax.add_patch(Rectangle((x+b[0],b[2]),b[1]-b[0],b[3]-b[2],
                    facecolor='#b76f3e',edgecolor='#633612',lw=1.2,zorder=2))
            ax.text(x,3.6,f'{station["name"]}\n{kind}',ha='center',va='center',fontsize=11,zorder=5)
        for policy,color,style,lw,label in (
            ('G','#1766a8','-',3.4,'Bayes-optimal geometry'),
            ('correct_class_oracle','#ee9827','--',2.3,'Correct-class oracle')):
            row=next(r for r in rows if r['actual_hypothesis']==h and r['policy']==policy)
            poses=row['suffix_states'];xs=[p[0] for p in poses];ys=[p[1] for p in poses]
            ax.plot(xs,ys,style,color=color,lw=lw,label=label,zorder=3)
        ax.plot(0,0,'*',color='#222222',markersize=13,zorder=6)
        ax.text(.2,-.12,'start / exact return',fontsize=9,va='top')
        g=next(r['potential_macro'] for r in rows if r['actual_hypothesis']==h and r['policy']=='G')
        s=next(r['potential_macro'] for r in rows if r['actual_hypothesis']==h and r['policy']=='correct_class_oracle')
        ax.set_title(f'Assignment {h}\nG = {g:.6f}; class oracle = {s:.6f}',fontsize=12)
        ax.set_aspect('equal');ax.set_xlim(-6,6);ax.set_ylim(-1,8)
        ax.set_xlabel('x (m)');ax.grid(alpha=.15,zorder=0)
    axes[0].set_ylabel('y (m)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.055),frameon=False)
    fig.suptitle('V29 finite information model: 48 paid actions, including an 18-action shared prefix',fontsize=12)
    fig.text(.5,.018,'Displayed paths are the 30-action suffixes. Score: potential visible reference area, not reconstruction Q.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.94))
    destination=ROOT/'docs/research/figures';destination.mkdir(exist_ok=True)
    for extension in ('png','svg'):
        path=destination/f'v29_information_scene_20260917.{extension}'
        fig.savefig(path,dpi=160,bbox_inches='tight')
        print(path)


if __name__=='__main__':run()
