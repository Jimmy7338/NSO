#!/usr/bin/env python3
"""Draw only saved public navigation assets; no GT geometry or trajectory."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source,output):
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(9.,6.2))
    fig.subplots_adjust(left=.075,right=.97,bottom=.17,top=.83,wspace=.4,hspace=.55)
    fig.suptitle('Shared coarse navigation prior across six development scenes',y=.96,fontsize=12)
    fig.text(.5,.91,'Static public assets · identical navigation information for all methods',ha='center',color='#555555')
    names=['Cabinet room','Shelves','Mixed facilities','Planar control','Coverage pressure','Prior shift']
    inputs={}
    for index,(family,ax,title) in enumerate(zip('ABCDEF',axes.flat,names)):
        path=source/f'DEV_{family}_00'/'graph.json'; graph=json.loads(path.read_text())
        inputs[str(path.relative_to(ROOT))]=sha(path)
        nodes=graph['nodes']; xy=np.asarray(list(nodes.values()))
        lines=[[nodes[a],nodes[b]] for a,b in graph['edges']]
        ax.add_collection(LineCollection(lines,colors='#aeb7c0',linewidths=.7))
        ax.scatter(xy[:,0],xy[:,1],s=6,color='#0072B2',zorder=2)
        ax.scatter(*nodes['home'],s=58,marker='*',color='#D55E00',zorder=3)
        ax.autoscale();ax.set_aspect('equal',adjustable='box');ax.margins(.09)
        ax.set_title(f'({chr(97+index)}) {family}: {title}',loc='left',pad=8)
        ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)')
        ax.text(.03,.97,f'{len(nodes)} nodes / {len(lines)} edges',transform=ax.transAxes,
                va='top',fontsize=6.5,bbox=dict(facecolor='white',edgecolor='none',alpha=.88))
        ax.tick_params(length=3,width=.6)
    fig.text(.075,.075,'Blue: shared coarse nodes. Orange star: initial position. Edges expand into paid 0.25 m primitives.\n'
        'No robot trajectory, semantic prediction, measured coverage or reconstruction result is shown.',fontsize=7,color='#444444',linespacing=1.6)
    for extension in ('pdf','svg','png'):
        fig.savefig(output/f'public_navigation.{extension}',dpi=220,facecolor='white')
    plt.close(fig)
    receipt=dict(scope='static public prior visualization; no research trajectory',input_sha256=inputs,
        renderer_sha256=sha(Path(__file__)),private_geometry_read=False,new_worlds=0,
        outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.iterdir())})
    (output/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='rendered_public_prior',bytes=sum(p.stat().st_size for p in output.iterdir()))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'audit_results/v43_public_navigation_r1_20260921')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/thesis/figures/v43_public_navigation')
    args=parser.parse_args();render(args.source.resolve(),args.output.resolve())
