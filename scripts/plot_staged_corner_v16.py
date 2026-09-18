#!/usr/bin/env python3
"""Observed-map figure; no hidden world construction or policy execution."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from env.virtual3d_inspection_v4 import InspectionConfigV4
from nso.decision_replay_v13 import load_packet, array_hash
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True)
    p.add_argument('--history', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.input / 'manifest.json').read_text())
    if manifest['status'] != 'complete': raise ValueError('complete preflight required')
    for name, expected in json.loads((args.input / 'artifact_hashes.json').read_text()).items():
        if hashlib.sha256((args.input/name).read_bytes()).hexdigest() != expected: raise ValueError('artifact changed')
    h = json.loads((args.history/'manifest.json').read_text())
    doc = json.loads((ROOT/h['protocol']['scene_protocol']).read_text())
    c = next(x for x in doc['contexts'] if x['id'] == h['protocol']['context'])
    config = InspectionConfigV4(**{**doc['shared_conditions'], **{k:v for k,v in c.items() if k not in ('id','seed')}})
    seed = h['protocol']['structure_seeds'][0]
    old = next(x for x in json.loads((args.history/f'structure_{seed}/all_decisions.json').read_text()) if x['action_id'] == 26)
    row = next(x for x in json.loads((args.input/f'structure_{seed}_decisions.json').read_text()) if x['action_id'] == 26)
    bounds = old['bounds']; mapper = ObservedRuntimeMapperV10((bounds[1], bounds[3]), config)
    for path in sorted((args.history/f'structure_{seed}/packets').glob('*.npz'))[:27]:
        packet = load_packet(path); mapper.update(packet.frame, packet.scan)
    if array_hash(mapper.belief) != old['state_before']['evidence']['map_arrays']['belief']:
        raise ValueError('observed map changed')
    resolution = config.resolution_m
    def xy(states):
        x = np.asarray(states)
        return (x[:,1]+.5)*resolution, (mapper.shape[0]-x[:,0]-.5)*resolution
    fig, axes = plt.subplots(1,2,figsize=(11,6),sharex=True,sharey=True,layout='constrained')
    for ax, title, stages in zip(axes, ('Corrected axes: existing targets', 'Shared front-corner approach targets'), (False,True)):
        ax.imshow(mapper.belief, cmap=ListedColormap(['#cdd2d7','#fafaf7','#4b5057']), vmin=-1,vmax=1,
            extent=[0,mapper.shape[1]*resolution,0,mapper.shape[0]*resolution])
        for asset in row['canonical_observed_assets']:
            low, high = np.array(asset['observed_low']), np.array(asset['observed_high'])
            center=np.array(asset['aabb_center']); front=np.array(asset['front_axis'])
            ax.plot([low[0],high[0]],[low[1],high[1]],color='#197f95',lw=3)
            ax.arrow(*center[:2],*(front*.45),width=.025,color='#197f95',length_includes_head=True)
        for route in row['candidates']:
            staged='_corner_' in route['group']
            if staged and not stages: continue
            x,y=xy([route['pose']])
            ax.scatter(x,y,c='#de7418' if staged else '#7061a8',s=65 if staged else 28,marker='^' if staged else 'o',zorder=4)
            ax.annotate(str(route['candidate_id']),(x[0]+.13,y[0]+(.20 if route['candidate_id'] % 2 == 0 else -.30)),fontsize=9)
            if staged:
                xx,yy=xy(route['outbound_states']); ax.plot(xx,yy,color='#de7418',lw=1.2,alpha=.65)
        ax.scatter(*xy([[*packet.position,packet.heading]]),marker='*',s=140,c='#222222',zorder=5,label='Current camera')
        ax.set_title(title,fontsize=12); ax.set_xlabel('x (m)'); ax.set_aspect('equal')
    known = np.argwhere(mapper.belief != -1)
    max_x = min(mapper.shape[1]*resolution, (known[:,1].max()+1)*resolution+1.)
    for ax in axes: ax.set_xlim(0,max_x); ax.set_ylim(0,mapper.shape[0]*resolution)
    axes[0].set_ylabel('y (m)')
    fig.suptitle(f'Observed evidence only | D12-00 / {seed} / action 26\nOrange: proposed approach, not executed; grey: unknown',fontsize=12)
    args.output.mkdir(parents=True,exist_ok=False)
    fig.savefig(args.output/'observed_staging.png',dpi=170,bbox_inches='tight'); fig.savefig(args.output/'observed_staging.svg',bbox_inches='tight')
    sources={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        str(args.input/'artifact_hashes.json'):hashlib.sha256((args.input/'artifact_hashes.json').read_bytes()).hexdigest()}
    (args.output/'figure_provenance.json').write_text(json.dumps(dict(source_sha256=sources,seed=seed,action_id=26,
        map_exact=True,executed_approaches=False,hidden_world_read=False),indent=2)+'\n')


if __name__ == '__main__': main()
