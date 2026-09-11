#!/usr/bin/env python3
"""Archive-based development diagnostics, no policy tuning or new rewards."""
import json
from collections import defaultdict, Counter
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nso.frontier_policy import shortest_paths,trace_path
from utils.grid_geometry import inflated_obstacles


def diagnose(source, output):
    source=Path(source)
    episodes=[json.loads(s) for s in (source/'episodes.jsonl').read_text().splitlines()]
    ranking={}
    for method in sorted({e['method'] for e in episodes}):
        changed=[]
        for e in episodes:
            if e['method']!=method:continue
            path=source/e['artifact_dir']/'candidates.jsonl'
            if not path.exists():continue
            choices=defaultdict(list)
            for s in path.read_text().splitlines():
                c=json.loads(s)
                if not c['excluded']:choices[c['step']].append(c)
            for rows in choices.values():
                chosen=next((c for c in rows if c['selected']),None)
                base=min(rows,key=lambda c:(-c['gain_per_cost'],c['features'][0],c['goal']))
                if chosen is not None and chosen['goal']!=base['goal']:
                    changed.append(dict(gain_per_cost_ratio=chosen['gain_per_cost']/base['gain_per_cost'],
                        estimated_action_delta=chosen['estimated_actions']-base['estimated_actions']))
        ranking[method]=dict(changed_selections=len(changed),
             mean_selected_to_geometric_score_ratio=float(np.mean([c['gain_per_cost_ratio'] for c in changed])) if changed else None,
             mean_estimated_action_delta=float(np.mean([c['estimated_action_delta'] for c in changed])) if changed else None)
    # Inspect the pre-existing loop failure case, selected before formal testing.
    e=next(e for e in episodes if e['layout']=='loop' and e['method']=='gain_semantic_structure')
    path=source/e['artifact_dir'];config=json.loads((source/'config.json').read_text())
    radius=config['environment']['robot_radius_m']/config['environment']['resolution_m']
    attempts=[json.loads(s) for s in (path/'goal_attempts.jsonl').read_text().splitlines()]
    reasons=Counter()
    with np.load(path/'observations.npz') as d:
        world=d['occupancy'];h,w=world.shape
        masks=np.unpackbits(d['visible_packed'],axis=1)[:,:h*w].reshape(-1,h,w)
        known=np.maximum.accumulate(masks,axis=0)
        for a in attempts:
            if a['status']!='path_invalidated_censored':continue
            t,u=a['start_step'],a['end_step'];start=tuple(d['poses'][t,:2])
            blocked=inflated_obstacles(world & known[t],radius)
            free=known[t] & ~world & ~blocked;free[start]=True
            _,parent=shortest_paths(free,start)
            cells=trace_path(parent,start,tuple(a['goal']))
            end=tuple(d['poses'][u,:2])
            if end in cells:cells=cells[cells.index(end)+1:]
            idx=tuple(np.asarray(cells,dtype=int).T) if cells else ([],[])
            newly_blocked=inflated_obstacles(world & known[u],radius)
            if np.any((world & known[u])[idx]):reasons['new_obstacle_on_path']+=1
            elif np.any(newly_blocked[idx]):reasons['new_obstacle_footprint_margin']+=1
            else:reasons['not_explained_by_reconstruction']+=1
    report=dict(source=str(source),ranking=ranking,
                path_case=dict(episode_key=e['episode_key'],reasons=dict(reasons)),
                interpretation='ranking ratios measure geometric heuristic sacrifice, not causal coverage loss; path case is one preselected development episode')
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();diagnose(a.source,a.output)
