#!/usr/bin/env python3
"""Make editable vector scientific figures from actual recorded trajectories."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

from study import ROOT, EXECUTION, freeze, read


STYLE={'original_G':('#414D65','--','Original + G'), 'region_G':('#0072B2','-','Region + G'),
       'original_S':('#C3561B','--','Original + S'), 'region_S':('#128668','-','Region + S'),
       'F_strict':('#8C4C8B',':','Nearest frontier')}


def to_xy(positions,shape,res):
    b=np.asarray(positions)
    return ((b[:,1]+.5)*res,(shape[0]-.5-b[:,0])*res)


def write_figure(figure, path):
    figure.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    figure.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    plt.close(figure)


def paths(phase, spec, receipts, output):
    layouts=tuple(dict.fromkeys(j['layout'] for j in spec['jobs']))
    arms=tuple(dict.fromkeys(j['arm'] for j in spec['jobs']))
    fig,axes=plt.subplots(len(layouts),len(arms),figsize=(3.0*len(arms),3.05*len(layouts)+.25),
                         squeeze=False,layout='constrained')
    for i,layout in enumerate(layouts):
        scene=Path(spec['scene_root'])
        initial=read(scene/'prepared'/layout/'initial_state.json')
        with np.load(scene/'evaluator_references'/layout/'evaluator_truth.npz',allow_pickle=False) as record:
            occupancy=record['occupancy']
        res=initial['grid_resolution_m']
        bounds=(0,occupancy.shape[1]*res,0,occupancy.shape[0]*res)
        for j,arm in enumerate(arms):
            ax=axes[i,j]
            ax.imshow(occupancy,cmap=ListedColormap(['#FFFFFF','#CED5DB']),interpolation='nearest',
                      origin='upper',extent=bounds,zorder=0)
            ax.set(xticks=[],yticks=[],aspect='equal')
            ax.spines[:].set_visible(False)
            if i==0:
                ax.set_title(STYLE[arm][2],weight='semibold')
            if j==0:
                ax.set_ylabel(layout.removeprefix('floor26_').replace('_',' '),weight='semibold')
            if (layout,arm) not in receipts:
                ax.text(.5,.5,'Endpoint unavailable',transform=ax.transAxes,ha='center',fontsize=8)
                continue
            item=receipts[layout,arm]
            if item['exit_code']!=0:
                ax.text(.5,.5,'Technical stop',transform=ax.transAxes,ha='center',fontsize=8)
                continue
            root=Path(item['run_path'])
            episode=read(root/'manifest.json')
            trace=read(root/'steps.json')
            x,y=to_xy([initial['anchor'][:2],*(row['position'] for row in trace)],occupancy.shape,res)
            ax.plot(x,y,color=STYLE[arm][0],lw=1.48,alpha=.90)
            ax.scatter(x[0],y[0],marker='D',c='white',edgecolors='#172636',s=23,zorder=4,lw=.9)
            ax.scatter(x[-1],y[-1],marker='+',c='#172636',s=28,zorder=5,lw=1)
            top=next((row['selected_target_pose'] for row in trace if row.get('selected_target_pose')),None)
            if top:
                x0,y0=to_xy([top[:2]],occupancy.shape,res)
                ax.scatter(x0,y0,marker='o',facecolors='white',edgecolors=STYLE[arm][0],s=20,lw=1,zorder=4)
            ax.text(.97,.03,f"{episode['paid_actions']} actions · {episode['path_length_m']:.1f} m",
                    transform=ax.transAxes,ha='right',fontsize=7,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
    fig.suptitle('Actual first-entry trajectories',weight='semibold')
    fig.supxlabel(f"One floor per row · {spec['budget']}-action cap including return · ◇ start   + stop   ○ first dispatched goal\n"
                  f"Source: execution/{phase}/phase_receipt.json and each run's steps.json",fontsize=8)
    write_figure(fig,output/'executed_trajectories')


def common_clock(phase,spec,output):
    source=output/'checkpoints.csv'
    with source.open(newline='') as stream:
        records=list(csv.DictReader(stream))
    layouts=tuple(dict.fromkeys(j['layout'] for j in spec['jobs']))
    columns=min(2,len(layouts))
    rows=(len(layouts)+columns-1)//columns
    fig, axes=plt.subplots(rows,columns,figsize=(4.7*columns,3.0*rows+.35),squeeze=False,
                           layout='constrained',sharex=True,sharey=True)
    arms=tuple(dict.fromkeys(j['arm'] for j in spec['jobs']))
    grid=spec['capture_paid']
    for axis,layout in zip(axes.flat,layouts):
        axis.set(xticks=grid,xlim=(-3,grid[-1]+3),ylim=(-2,104))
        axis.set_title(layout.removeprefix('floor26_').replace('_',' '),loc='left',weight='semibold')
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',alpha=.20)
        for arm in arms:
            by_clock={int(row['actual_paid']):float(row['J'])*100 for row in records
                      if row['layout']==layout and row['arm']==arm and row.get('checkpoint')!='terminal'
                      and row.get('J') not in ('',None) and row.get('actual_paid') not in ('',None)}
            axis.plot(grid,[by_clock.get(stamp,np.nan) for stamp in grid],marker='o',ms=3,lw=1.2,
                      color=STYLE[arm][0],ls=STYLE[arm][1])
    handles=[Line2D([],[],color=STYLE[a][0],ls=STYLE[a][1],label=STYLE[a][2]) for a in arms]
    fig.legend(handles=handles,ncol=len(arms),loc='outside lower center',frameon=False)
    fig.supylabel('Joint map quality J (%)')
    fig.supxlabel(f"Actual paid actions · {len(layouts)} floor(s), cap {spec['budget']} including return\n"
                  f"Source: results/{phase}/checkpoints.csv · no post-stop padding",fontsize=8)
    write_figure(fig,output/'equal_paid_quality')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=('development','formal'))
    phase=cli.parse_args().phase
    spec=read(freeze(phase))
    items=read(EXECUTION/phase/'phase_receipt.json')['episodes']
    receipts={(i['layout'],i['arm']):i for i in items}
    output=ROOT/'results'/phase
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    paths(phase,spec,receipts,output)
    common_clock(phase,spec,output)
