#!/usr/bin/env python3
"""Freeze V18 world, observable references and a geometry-only budget witness.

The perimeter sweep uses truth solely as an evaluator feasibility witness.
It is not a planning baseline, initial history, or semantic outcome.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,sys,zipfile
from collections import deque
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from env.facility_documentation_v18 import FacilityWorldV18
from utils.facility_metrics_v18 import FacilityEvaluatorV18
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.observed_asset_axes_v16 import measured_assets_v16
from nso.axis_history_view_v16 import AxisHistoryViewV16
from nso.cpu_sensor_contract_v10 import digest,json_value
from scripts.collect_semantic_gain_v13_history import sha,write


def sweep(world):
    mapper=ObservedRuntimeMapperV10(world.shape,world.config)
    mapper.update(world.sense(),world.scan());actions=[]
    targets=[world._cell(.9,1.4),world._cell(.9,world.height-.9),
             world._cell(world.width-.9,world.height-.9),world._cell(world.width-.9,1.4),world.start]
    for target in targets:
        initial=(*world.position,world.heading);queue=deque([initial]);previous={initial:None}
        while queue:
            state=queue.popleft()
            if state[:2]==target:break
            r,col,h=state;dr,dc=((-1,0),(0,1),(1,0),(0,-1))[h]
            for action,nxt in [('forward',(r+dr,col+dc,h)),('right',(r,col,(h+1)%4)),('left',(r,col,(h-1)%4))]:
                nr,nc,_=nxt
                if not (0<=nr<world.shape[0] and 0<=nc<world.shape[1]) or world._blocked[nr,nc] or nxt in previous:continue
                previous[nxt]=(state,action);queue.append(nxt)
        else:raise ValueError('prescribed witness waypoint unreachable')
        route=[]
        while previous[state] is not None:
            state,action=previous[state];route.append(action)
        for action in reversed(route):
            frame,collision,done=world.step(action);actions.append(action)
            if collision or done:raise ValueError('safe budget witness execution infeasible')
            mapper.update(frame,world.scan())
    while world.heading!=0:
        frame,collision,done=world.step('right');actions.append('right');mapper.update(frame,world.scan())
        if collision or done:raise ValueError('return-heading witness infeasible')
    coverage=float(np.count_nonzero((mapper.belief!=-1)&world.reachable)/world.reachable.sum())
    if coverage<.8:raise ValueError('fixed sweep cannot establish minimum coverage feasibility')
    return dict(actions=actions,paid_actions=len(actions),coverage_2d=coverage,returned=True,
        purpose='GT-safe BFS between fixed perimeter anchors; feasibility only, not policy/prefix',
        proposed_task_budget=len(actions)+32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('env','nso','utils') for p in sorted((ROOT/folder).glob('*.py'))}
    for name in ('scripts/preflight_facility_v18_1.py','scripts/collect_semantic_gain_v13_history.py','tests/virtual3d/test_facility_v18.py','configs/virtual3d/facility_documentation_v18_design.json'):
        sources[name]=sha(ROOT/name)
    manifest=dict(status='running',source_sha256=sources,semantic_outcomes=0,training_allowed=False)
    write(a.output/'manifest.json',manifest)
    with zipfile.ZipFile(a.output/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in sources:archive.write(ROOT/name,name)
    try:
        rows=[]
        for parent in ('D18-P00','D18-P01'):
            variants=[FacilityWorldV18(parent,name) for name in ('A_open_B_closed','A_closed_B_open')]
            first=[w.sense() for w in variants];scans=[w.scan() for w in variants]
            for key in ('depth_m','intrinsic','world_from_camera'):
                if not np.array_equal(getattr(first[0],key),getattr(first[1],key)):raise ValueError('initial geometry pairing failed')
            if not np.array_equal(scans[0].ranges_m,scans[1].ranges_m):raise ValueError('initial scan pairing failed')
            marker=first[0].semantic>0
            if not np.array_equal(marker,first[1].semantic>0) or not np.array_equal(first[0].color_rgb[~marker],first[1].color_rgb[~marker]):raise ValueError('RGB masking contract')
            records=[]
            for world,frame,scan in zip(variants,first,scans):
                mapper=ObservedRuntimeMapperV10(world.shape,world.config);mapper.update(frame,scan)
                assets=measured_assets_v16(AxisHistoryViewV16(mapper,[frame]))
                observed=[{k:v for k,v in x.items() if k not in ('class_vote','marked_points')} for x in assets]
                e=FacilityEvaluatorV18(world)
                refs={str(x['id']):dict(count=x['observable_samples'],area_m2=x['observable_area_m2'],
                    reference_sha256=digest(x['reference']),bounds=x['bounds'].tolist()) for x in e.instances}
                np.savez_compressed(a.output/f'{parent}_{world.assignment}_references.npz',
                    **{f'asset_{x["id"]}':x['reference'] for x in e.instances})
                witness=sweep(world)
                records.append(dict(assignment=world.assignment,config=asdict(world.config),
                    geometry_asset_sha256=digest(observed),observed_assets=len(assets),references=refs,witness=witness))
            if records[0]['geometry_asset_sha256']!=records[1]['geometry_asset_sha256']:raise ValueError('measured asset geometry differs')
            budget=max(r['witness']['proposed_task_budget'] for r in records)
            if budget>variants[0].config.max_steps:raise ValueError('budget exceeds world limit')
            row=dict(parent=parent,initial_nonsemantic_pairing=True,asset_geometry_pairing=True,
                budget=budget,variants=records);rows.append(row);write(a.output/f'{parent}.json',row)
            print(parent,'reference and geometry feasibility passed; budget',budget,flush=True)
        for name,wanted in sources.items():
            if sha(ROOT/name)!=wanted:raise ValueError('source changed during preflight')
        write(a.output/'result.json',dict(status='passed',parents=rows,new_semantic_outcomes=0,
            initial_history_actions=0,policy_efficacy_proven=False,training_allowed=False))
        manifest['status']='complete'
    except Exception as error:
        manifest.update(status='failed',error=repr(error));raise
    finally:
        write(a.output/'manifest.json',manifest)
        write(a.output/'artifact_hashes.json',{p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':main()
