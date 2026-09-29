#!/usr/bin/env python3
"""Plot all frozen final-layout trajectories and measured meshes, without simulation."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso-final-validation-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'audit_results/final_local_layout_validation_20260929'
OUT=ROOT/'docs/thesis/figures/final_validation_20260929'
COLORS={'G':'#0072B2','S':'#D55E00'}
NAMES={'L02_side_column':'Side-column layout','L03_rear_partition':'Rear-partition layout'}
SOURCES={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source(path):
    path=Path(path);SOURCES[str(path.relative_to(ROOT))]=sha(path);return path

def read(path):return json.loads(source(path).read_text())

def save(fig,name):
    for ext in ('pdf','svg','png'):
        meta={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
        fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white',metadata=meta)
    plt.close(fig)

def main():
    from plot_thesis_scene_details_20260928 import prepare_facets, render
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=read(BASE/'protocol.json');summary=read(BASE/'analysis/summary.json')
    if summary['completed']!=8:raise ValueError('This final figure requires all 8 measured tasks; failures need explicit rendering.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.3,'axes.titlesize':9,
        'axes.labelsize':8.3,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'nso-final-20260929'})
    records={};parents={}
    for case in cfg['cases']:
        folder=BASE/'cases'/case['id'];result=read(folder/'result.json');trace=read(folder/'trace.json')
        for rel,expected in read(folder/'seal.json').items():
            if sha(folder/rel)!=expected:raise ValueError('Sealed input changed: '+str(folder/rel))
        meta=read(BASE/'public_templates'/case['scene']/'metadata.json')
        parent=read(BASE/'public_templates'/case['scene']/'geometry.json')['parent'];parents[case['scene']]=parent
        with np.load(source(folder/'final_mesh.npz'),allow_pickle=False) as d:
            vertices=d['vertices']-np.asarray(meta['shift']);triangles=d['triangles'].copy()
        records[case['scene'],case['hypothesis'],case['mode']]=dict(
            result=result,trace=trace,path=np.asarray([r['pose'][:2] for r in trace]),
            vertices=vertices,triangles=triangles)
    conditions=[(s['id'],h) for s in cfg['layouts'] for h in (0,1)]
    fig,axes=plt.subplots(2,2,figsize=(7.1,7.0))
    fig.subplots_adjust(left=.075,right=.975,bottom=.19,top=.84,wspace=.23,hspace=.50)
    for ax,(name,h) in zip(axes.flat,conditions):
        parent=parents[name]
        for b in parent['hypotheses'][h]['assets'][0]['boxes']:
            ax.add_patch(Rectangle((b[0],b[2]),b[1]-b[0],b[3]-b[2],facecolor='#DDE5E9',edgecolor='#8C9AA3',lw=.6))
        for b in parent['background_boxes']:
            ax.add_patch(Rectangle((b[0],b[2]),b[1]-b[0],b[3]-b[2],facecolor='#A69C88',edgecolor='#645C4D',lw=.8,hatch='///'))
        cells=np.asarray(parent['nav_cells']);ax.scatter(cells[:,0],cells[:,1],s=4,c='#D7DEE3',zorder=2)
        for mode in ('G','S'):
            r=records[name,h,mode];path=r['path']
            ax.plot(path[18:,0],path[18:,1],color=COLORS[mode],lw=1.4,
                ls='-' if mode=='G' else (0,(3,2)),alpha=.9,zorder=4 if mode=='S' else 3)
            # Headings are actual recorded orientations at fixed paid indices.
            for step in (19,26,33,40):
                if step>=len(r['trace']):continue
                x,y,heading=r['trace'][step]['pose'];dx,dy=((0,1),(1,0),(0,-1),(-1,0))[heading]
                ax.annotate('',xy=(x+.22*dx,y+.22*dy),xytext=(x,y),
                    arrowprops={'arrowstyle':'-|>','lw':.8,'color':COLORS[mode],'mutation_scale':7},zorder=5)
        p=records[name,h,'G']['path'][:19]
        ax.plot(p[:,0],p[:,1],color='#8D99A3',lw=3,alpha=.35,zorder=2)
        ax.scatter([0],[0],marker='*',s=65,c='#213742',zorder=7)
        mg=records[name,h,'G']['result']['measurements']['final'];ms=records[name,h,'S']['result']['measurements']['final']
        delta=ms['05cm']['joint']-mg['05cm']['joint']
        ax.set_title(f'{NAMES[name]} / h{h}\n$J_5$: G {mg["05cm"]["joint"]:.3f}, S {ms["05cm"]["joint"]:.3f}  ($\\Delta$ {delta:+.3f})',loc='left',pad=7)
        ax.set(xlim=(-3.6,3.6),ylim=(-.45,4.6),aspect='equal',xlabel='x (m)',ylabel='y (m)')
        ax.set_xticks([-3,0,3]);ax.set_yticks([0,2,4]);ax.spines[['top','right']].set_visible(False)
    fig.text(.075,.965,'Frozen policy in two new background layouts',fontsize=11,fontweight='bold')
    fig.text(.075,.927,'All four G/S pairs • full paid trajectories • common metric scale',fontsize=8.5,color='#536572')
    fig.legend(handles=[Line2D([],[],color=COLORS['G'],lw=1.8,label='Active geometry G'),
        Line2D([],[],color=COLORS['S'],lw=1.8,ls='--',label='Category + geometry S'),
        Line2D([],[],color='#8D99A3',lw=3,alpha=.5,label='Common prefix'),
        Patch(facecolor='#A69C88',hatch='///',label='New background')],
        loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False,fontsize=8.2)
    fig.text(.075,.014,'Geometry is offline context; public equipment templates remain unchanged.',fontsize=8,color='#536572')
    save(fig,'transfer_paths')

    fig,axes=plt.subplots(4,2,figsize=(7.1,8.3))
    fig.subplots_adjust(left=.035,right=.98,bottom=.05,top=.93,hspace=.3,wspace=.05)
    mesh_counts={}
    for row,(name,h) in enumerate(conditions):
        for col,mode in enumerate(('G','S')):
            ax=axes[row,col];rec=records[name,h,mode]
            render(ax,prepare_facets(rec,h),((-3.9,3.9),(-2.1,2.15)))
            m=rec['result']['measurements']['final']['05cm']
            short='Column' if name.startswith('L02') else 'Partition'
            ax.set_title(f'{short} / h{h}   {mode}    $F_1$ {m["f1"]:.3f}; $J_5$ {m["joint"]:.3f}',loc='left',fontsize=8.2,color=COLORS[mode])
            ax.plot([-3.7,-2.7],[-1.8,-1.8],color='#465C68',lw=1.4)
            ax.text(-3.2,-1.6,'1 m',ha='center',fontsize=8)
            mesh_counts[f'{name}/h{h}/{mode}']=len(rec['triangles'])
    fig.text(.035,.975,'Measured facility surfaces after the complete task',fontsize=11,fontweight='bold')
    fig.text(.035,.01,'Same view within pairs; every stored triangle rendered; shading denotes illumination.',fontsize=8,color='#536572')
    save(fig,'transfer_meshes')

    fig,axes=plt.subplots(1,3,figsize=(7.1,2.8));fig.subplots_adjust(left=.09,right=.98,bottom=.33,top=.86,wspace=.4)
    labels=['C / h0','C / h1','P / h0','P / h1'];x=np.arange(4)
    for ax,metric,label in zip(axes,('C_map','f1','joint'),('Planar coverage','Surface F1 @ 5 cm','Joint quality J5')):
        for mode,off in [('G',-.16),('S',.16)]:
            vals=[]
            for name,h in conditions:
                m=records[name,h,mode]['result']['measurements']['final']
                vals.append(m['C_map'] if metric=='C_map' else m['05cm'][metric])
            ax.bar(x+off,vals,width=.29,color=COLORS[mode],label=mode)
        ax.set(ylim=(0,1),xticks=x,xticklabels=labels,title=label)
        ax.tick_params(axis='x',labelrotation=45);ax.set_yticks([0,.5,1]);ax.spines[['top','right']].set_visible(False)
    fig.legend(*axes[-1].get_legend_handles_labels(),frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.53,-.005),fontsize=8)
    save(fig,'transfer_quality')
    source(Path(__file__));source(ROOT/'scripts/plot_thesis_scene_details_20260928.py')
    for name,expected in SOURCES.items():
        if sha(ROOT/name)!=expected:raise ValueError('Source changed while plotting: '+name)
    (OUT/'captions.md').write_text('# Final layout validation figures\n\n'
        '**Paths.** All four paired trajectories in two frozen background layouts. A side column and a rear partition change access around unchanged public equipment templates. Lines show recorded XY motion; arrows show recorded headings at fixed steps. All methods share the initial 18 paid actions and the 42-action budget. Geometry supplies offline context.\n\n'
        '**Meshes.** Actual stored TSDF facility meshes, with a common camera and scale within each pair. All triangles are rendered without smoothing, completion, or error-based selection. The existing evaluation crop remains unchanged; background structures are outside that facility region.\n\n'
        '**Quality.** Coverage, surface F1 and joint quality of all four pairs. C/P denote column/partition. These are paired conditions from two new same-family layouts, not eight independent layouts.\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(source_sha256=SOURCES,all_eight_runs_rendered=True,
        new_sensor_queries=0,new_trajectories=0,new_surface_evaluations=0,mesh_triangles=mesh_counts,
        output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}),indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'figures':3,'trajectories':8,'new_runs':0}))

if __name__=='__main__':main()
