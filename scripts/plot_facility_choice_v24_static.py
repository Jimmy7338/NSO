#!/usr/bin/env python3
"""Explain frozen V24 static geometry and proposed paid routes; no sensors."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/v24-static-mpl')
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from env.facility_choice_v24 import FacilityChoiceWorldV24


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = ROOT/'audit_results/facility_choice_v24_static_backstops_20260915'
    result = json.loads((audit/'result.json').read_text())
    for name, expected in result['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    output = ROOT/'audit_results/facility_choice_v24_figures_20260915'
    output.mkdir(exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    for ax, parent in zip(axes, result['parents']):
        world = FacilityChoiceWorldV24(parent['parent'])
        ax.imshow(world.reachable, extent=[0,world.width,0,world.height],
                  cmap='Greys', vmin=0, vmax=5, origin='upper', alpha=.5)
        for x,y,z,sx,sy,sz,owner in world._solid_primitives:
            if z+sz <= .001:
                continue
            color = '#6a7279' if owner == 1 else '#f3ab57' if sz < 1.5 else '#5b9eb8'
            ax.add_patch(Rectangle((x,y),sx,sy,facecolor=color,edgecolor='white',linewidth=.4))
        for item in world.objects:
            cx,front,_ = item['front_center']
            ax.text(cx,front+.38,item['name'],ha='center',va='center',fontweight='bold',color='white')
            ax.plot([cx-.36,cx+.36],[front,front],color='#98204a',lw=3)
        for region in world.service_regions:
            (x0,y0),(x1,y1) = region['xy_bounds_m']
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='none',edgecolor='#286c39',lw=1.5))
        states = np.asarray(world.prefix_proposal['states'])
        xy = np.column_stack(((states[:,1]+.5)*.2,(world.shape[0]-states[:,0]-.5)*.2))
        ax.plot(xy[:,0],xy[:,1],'--',color='#2b5686',lw=1.2,alpha=.8)
        ax.scatter(*xy[0],marker='*',s=130,color='#c22932',zorder=5)
        ax.text(xy[0,0],xy[0,1]-.5,'start / decision',ha='center',fontsize=8)
        ax.set(xlim=(0,world.width),ylim=(0,world.height),aspect='equal',xlabel='x (m)',ylabel='y (m)',
            title=f"{world.parent}: {world.width:g} × {world.height:g} m\nProposed prefix {parent['prefix_actions']} actions; candidate budget {parent['proposed_total_budget']}")
        assert world.step_count == 0
    fig.suptitle('V24 two-facility static task: A complex / B simple (fixed illustration)',fontsize=13)
    fig.text(.5,-.018,'Gray: walls/backstops and safe floor · blue/orange: external cabinet geometry · green: offline declared views\nDashed: proposed paid prefix, NOT executed trajectory · 5 m sensors · budgets not frozen · GT used only to explain geometry',
             ha='center',fontsize=9)
    fig.savefig(output/'layouts.png',dpi=145,bbox_inches='tight'); plt.close(fig)
    meta = dict(status='static_illustration',actual_physical_actions=0,
        world_assignment='A_complex_B_simple',source_sha256={
            str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
            'env/facility_choice_v24.py':sha(ROOT/'env/facility_choice_v24.py'),
            str((audit/'result.json').relative_to(ROOT)):sha(audit/'result.json')},
        no_quality_or_semantic_gain_proven=True)
    (output/'result.json').write_text(json.dumps(meta,indent=2)+'\n')
    (output/'artifact_hashes.json').write_text(json.dumps({p.name:sha(p) for p in output.iterdir() if p.is_file()},indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
