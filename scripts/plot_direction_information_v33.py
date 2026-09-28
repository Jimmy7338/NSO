#!/usr/bin/env python3
"""Plot sealed V33 geometry and finite-policy results; never run a model."""
import io
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nso.research_evidence_v31 import ROOT, read, sha, write, write_bytes, freeze, verify_sources, verify_inventory, seal


def main():
    packages = [ROOT/f'audit_results/v33_direction_information_{r}_20260917' for r in ('r0','r1')]
    inputs = {}
    for package in packages:
        verify_sources(package); verify_inventory(package)
        for p in package.iterdir():
            if p.is_file(): inputs[str(p.relative_to(ROOT))] = sha(p)
    cfg_path = ROOT/'configs/virtual3d/v33_direction_scene_r1_20260917.json'
    cfg = read(cfg_path); inputs[str(cfg_path.relative_to(ROOT))] = sha(cfg_path)
    latest = packages[-1]
    result = read(latest/('result.json' if (latest/'result.json').exists() else 'failure.json'))
    out = ROOT/'audit_results/v33_direction_information_figures_20260917'
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    freeze(out, [Path(__file__)], input_sha256=inputs, geometry_or_solver_executions=0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.size':9, 'svg.fonttype':'none', 'svg.hashsalt':'v33-direction-information'})
    fig, axes = plt.subplots(1,3,figsize=(13.6,4.5),constrained_layout=True)
    for ax,parent in zip(axes[:2],cfg['parents']):
        for index,box in enumerate(parent['hypotheses'][0]['assets'][0]['boxes']):
            x0,x1,y0,y1,_,_ = box
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#c8c8c8' if index<3 else '#2e86ab',
                alpha=.35 if index==1 else .8,edgecolor='white',linewidth=.6))
        for box in parent['hypotheses'][1]['assets'][0]['boxes'][3:]:
            x0,x1,y0,y1,_,_ = box
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='none',edgecolor='#c86828',linestyle='--',linewidth=1.6))
        cells = parent['nav_cells']
        ax.scatter([p[0] for p in cells],[p[1] for p in cells],s=8,c='#555555',label='All safe grid cells')
        ax.scatter(*parent['anchor'][:2],marker='*',s=110,c='#bd3535',label='Start / exact return',zorder=5)
        table = read(latest/(parent['id']+'_geometry.json'))
        first = table['pair']['first_information_action_layer']
        missing = [len(t['unobservable_targets']) for t in table['tables']]
        ax.set(title=f"{parent['id']} r1 | first geometry difference: {first} actions\nUnobservable target patches: {missing}",
            xlabel='World x (m)',ylabel='World y (m)',aspect='equal')
        ax.set_xlim(min(p[0] for p in cells)-.5,max(p[0] for p in cells)+.5)
        ax.set_ylim(min(p[1] for p in cells)-.5,max(p[1] for p in cells)+.5)
        ax.grid(alpha=.15)
    ax = axes[2]
    rows = result.get('policy_summaries',[])
    if len(rows)==2 and all(r['feasible'] for r in rows):
        for i,r in enumerate(rows):
            ax.bar(i-.17,r['G_optimal'],width=.3,color='#666666',label='Optimal adaptive G' if i==0 else None)
            ax.bar(i+.17,r['class_optimal_mean'],width=.3,color='#2e86ab',label='Known-class oracle' if i==0 else None)
            ax.text(i,max(r['G_optimal'],r['class_optimal_mean'])+.035,
                f"{100*r['information_relative']:+.3f}%",ha='center')
        ax.set(xticks=range(2),xticklabels=[r['parent_id'] for r in rows],ylim=(0,1.07),
            ylabel='Expected ideal C_grid x visible surface fraction',title='Finite information value (r1)')
        ax.legend(loc='lower right',fontsize=8)
    else:
        ax.axis('off')
        ax.text(.03,.7,'No certified two-parent policy comparison\nStatus: '+result['status'],transform=ax.transAxes)
    axes[0].legend(loc='lower left',fontsize=7)
    fig.suptitle('V33: service-side information screen\nBlue solid = type A fins; orange dashed = alternative type B (not simultaneous)',fontsize=12)
    fig.supxlabel('Forced 18-action prefix + 24-action suffix. Ideal finite geometry; no measured map accuracy or semantic efficacy.',fontsize=9)
    paths = []
    for extension in ('png','svg'):
        target=ROOT/f'docs/research/figures/v33_direction_information_20260917.{extension}'
        buffer=io.BytesIO(); fig.savefig(buffer,format=extension,dpi=160)
        write_bytes(out,target,buffer.getvalue()); paths.append(target)
    plt.close(fig)
    verify_sources(out)
    for rel,expected in inputs.items():
        if sha(ROOT/rel)!=expected: raise ValueError('input changed: '+rel)
    write(out,out/'result.json',dict(status='complete',source_status=result['status'],
        figures={str(p.relative_to(ROOT)):sha(p) for p in paths},
        inputs_verified=len(inputs),geometry_or_solver_executions=0,new_physical_tasks=0))
    seal(out)
    print(read(out/'result.json'))


if __name__=='__main__': main()
