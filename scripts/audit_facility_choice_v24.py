#!/usr/bin/env python3
"""Offline V24 feasibility only: no world.step, fusion or strategy quality.

The exact BFS cost applies to the declared rear+inward-side observation-region
tasks. It is not a universal lower bound on arbitrary reconstruction quality.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from env.facility_choice_v24 import (FacilityChoiceWorldV24, PARENTS_V24,
    ASSIGNMENTS_V24, VERSION_V24, LAYOUTS_V24)
from env.canonical_rgbd_v15 import render_axial_depth
from env.virtual3d import camera_pose
from utils.grid_geometry import DIRECTIONS

CONFIG=ROOT/'configs/virtual3d/facility_choice_v24_static.json'
DEFAULT_OUTPUT=ROOT/'audit_results/facility_choice_v24_static_backstops_20260915'
ACTIONS=('forward','left','right')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    x=np.ascontiguousarray(array)
    return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()


def write(path,data):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


class StaticWorld(FacilityChoiceWorldV24):
    def step(self,action):
        raise AssertionError('No physical actions authorized by static audit')


def shortest_service_tours(world):
    rows,cols=world.shape
    poses=rows*cols*4
    transitions=np.full((poses,3),-1,np.int32)
    marks=np.zeros(poses,np.uint8)
    def encode(r,c,h):return (r*cols+c)*4+h
    def decode(p):
        cell,h=divmod(int(p),4);r,c=divmod(cell,cols)
        return [r,c,h]
    for r,c in np.argwhere(world.reachable):
        for h,(dr,dc) in enumerate(DIRECTIONS):
            p=encode(int(r),int(c),h)
            transitions[p,1]=encode(int(r),int(c),(h-1)%4)
            transitions[p,2]=encode(int(r),int(c),(h+1)%4)
            rr,cc=int(r)+dr,int(c)+dc
            if 0<=rr<rows and 0<=cc<cols and world.reachable[rr,cc]:
                transitions[p,0]=encode(rr,cc,h)
    for region in world.service_regions:
        for r,c,h in region['states']:
            marks[encode(r,c,h)] |= 1<<region['bit']
    anchor=encode(*world.start,world.start_heading)
    assert marks[anchor]==0
    prefix_mask=0
    for pose in world.prefix_proposal['states']:
        prefix_mask |= int(marks[encode(*pose)])
    assert prefix_mask==0, 'Basic prefix already visits a declared supplementary region'
    source=anchor*16
    previous=np.full(poses*16,-1,np.int32)
    action_taken=np.full(poses*16,-1,np.int8)
    distance=np.full(poses*16,-1,np.int32)
    previous[source]=-2;distance[source]=0
    queue=deque([source]); terminals={}; required={'A':3,'B':12,'both':15}
    expanded=0
    while queue:
        node=queue.popleft();expanded+=1
        pose,mask=divmod(node,16)
        if pose==anchor:
            for name,target in required.items():
                if name not in terminals and mask&target==target:
                    terminals[name]=node
            if len(terminals)==3:
                break
        for action,next_pose in enumerate(transitions[pose]):
            if next_pose < 0:
                continue
            next_mask=mask|int(marks[next_pose])
            following=int(next_pose)*16+next_mask
            if previous[following]!=-1:
                continue
            previous[following]=node;action_taken[following]=action
            distance[following]=distance[node]+1;queue.append(following)
    if len(terminals)!=3:
        raise ValueError('Declared service task is unreachable')
    outputs={}
    for name,end in terminals.items():
        nodes=[];actions=[];node=end
        while node!=source:
            nodes.append(node);actions.append(ACTIONS[int(action_taken[node])]);node=int(previous[node])
        nodes.append(source);nodes.reverse();actions.reverse()
        states=[decode(node//16) for node in nodes]
        # Independently validate reconstructed witness and exact cost.
        state=states[0];mask=0;visits=[]
        for index,(action,expected) in enumerate(zip(actions,states[1:]),start=1):
            r,c,h=state
            if action=='forward':
                dr,dc=DIRECTIONS[h];state=[r+dr,c+dc,h]
            else:
                state=[r,c,(h+(1 if action=='right' else -1))%4]
            assert state==expected and world.reachable[state[0],state[1]]
            observed=int(marks[encode(*state)])
            for bit in range(4):
                if observed&(1<<bit) and not mask&(1<<bit):
                    visits.append(dict(bit=bit,paid_action_id=index,state=state.copy()))
            mask|=observed
        assert state==decode(anchor) and mask&required[name]==required[name]
        assert len(actions)==int(distance[end])
        outputs[name]=dict(minimum_paid_actions=len(actions),translation_actions=actions.count('forward'),
            turn_actions=len(actions)-actions.count('forward'),required_mask=required[name],final_mask=mask,
            first_service_visits=visits,actions=actions,states=states,
            lower_bound_and_attaining_static_witness=True,
            executed_world_actions=0,all_transitions_use_physical_safe_floor=True)
    return dict(tours=outputs,expanded_states=expanded,graph_pose_count=int(world.reachable.sum())*4,
        graph_mask_count=16,decision_anchor=decode(anchor),prefix_service_mask=prefix_mask,
        proof='Breadth-first search over every safe (row,col,heading,visited-region-mask), unit paid actions; earliest return state for each task',
        scope='exact for declared rear+inward-side region/heading visits, not arbitrary quality achievement',
        sensor_frames_paid_on_arrival_or_turn=True,unknown_space_not_an_online_graph=True)


def initial_readout(world):
    first=world.sense();scan=world.scan()
    repeat=world.sense()
    assert np.array_equal(first.depth_m,repeat.depth_m)
    assert world.step_count==0
    return dict(nonsemantic={k:array_sha(v) for k,v in dict(depth=first.depth_m,
        intrinsic=first.intrinsic,pose=first.world_from_camera,scan=scan.ranges_m,
        laser_pose=scan.world_from_laser).items()},
        observed_semantic_values=np.unique(first.semantic).astype(int).tolist(),
        initial_heading=world.heading,initial_frame_only=True,
        full_prefix_pairing_verified=False,same_step_depth_repeatable=True)


def static_prefix_projection_check(worlds):
    """Compare planned pixel rays only; no packets, noise, fusion or motion.

    Static camera projection at declared poses is an offline visibility check,
    not a performed trajectory. Depth arrays are discarded and never exported
    to an online mapper. Real paid history still requires collector validation.
    """
    states=worlds[0].prefix_proposal['states']
    assert states==worlds[1].prefix_proposal['states']
    boxes=[np.asarray(w._solid_primitives,float)[:,:6] for w in worlds]
    changes=[]
    for action_id,(r,col,h) in enumerate(states):
        images=[]
        for w,primitives in zip(worlds,boxes):
            pose=camera_pose((r,col),h,w.config,w.shape[0])
            depth,_=render_axial_depth(primitives,w.intrinsic,pose,w.config.height_px,
                                       w.config.width_px,w.config.max_depth_m)
            images.append(depth)
        changed=int(np.count_nonzero(images[0]!=images[1]))
        if changed:
            changes.append(dict(proposed_action_id=action_id,state=[r,col,h],changed_axial_depth_pixels=changed))
    assert all(w.step_count==0 for w in worlds)
    return dict(static_planned_pose_count=len(states),all_planned_raster_depths_equal=not changes,
        first_difference_action=changes[0]['proposed_action_id'] if changes else None,differences=changes,
        camera_geometry_only=True,sensor_packets_produced=0,raw_depth_arrays_saved=0,
        no_noise_or_mapper_used=True,actual_paid_prefix_history_verified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists():
        raise ValueError('Retain existing audit; choose a fresh output directory')
    config=json.loads(CONFIG.read_text())
    assert config['parents']==list(PARENTS_V24) and config['assignments']==list(ASSIGNMENTS_V24)
    source_paths=[Path(__file__).resolve(),CONFIG,ROOT/'env/facility_choice_v24.py',
        ROOT/'env/facility_documentation_v19.py',ROOT/'env/canonical_rgbd_v15.py',
        ROOT/'env/canonical_box_scan_v14.py',ROOT/'env/virtual3d_inspection_v4.py',
        ROOT/'env/virtual3d.py',ROOT/'env/virtual3d_v2.py',ROOT/'utils/grid_geometry.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    cases=[];parents=[]
    for parent in PARENTS_V24:
        current=[];worlds=[]
        for assignment in ASSIGNMENTS_V24:
            world=StaticWorld(parent,assignment,config['sensor_model'],config['noise_seed'])
            initial=initial_readout(world)
            proof=shortest_service_tours(world)
            case=dict(parent=parent,assignment=assignment,initial=initial,safe_floor_cells=int(world.reachable.sum()),
                world_geometry_sha256={k:array_sha(np.asarray(getattr(world.mesh,k))) for k in ('vertices','triangles')},
                service_regions=world.service_regions,prefix_proposal=world.prefix_proposal,proof=proof,
                source_world_version=VERSION_V24,paid_physical_actions=world.step_count,
                online_coverage_or_shape_quality_verified=False)
            assert world.step_count==0
            cases.append(case);current.append(case);worlds.append(world)
            print(parent,assignment,{k:v['minimum_paid_actions'] for k,v in proof['tours'].items()},flush=True)
        pair=current[0]['initial']['nonsemantic']==current[1]['initial']['nonsemantic']
        assert pair, 'Initial nonsemantic comparison failed; retain condition for diagnosis'
        assert current[0]['prefix_proposal']==current[1]['prefix_proposal']
        prefix=current[0]['prefix_proposal']['paid_actions']
        lo=max(c['proof']['tours'][name]['minimum_paid_actions'] for c in current for name in ('A','B'))
        hi=min(c['proof']['tours']['both']['minimum_paid_actions'] for c in current)-1
        recommended=math.ceil((lo+hi)/2) if lo<=hi else None
        prefix_projection=static_prefix_projection_check(worlds)
        parents.append(dict(parent=parent,layout=LAYOUTS_V24[parent],initial_nonsemantic_pair_identical=pair,
            prefix_actions=prefix,supplementary_remaining_budget_interval=[lo,hi] if lo<=hi else None,
            proposed_remaining_budget=recommended,proposed_total_budget=prefix+recommended if recommended is not None else None,
            both_service_tours_excluded_by_conditional_budget=lo<=hi,
            static_prefix_projection=prefix_projection,
            budget_is_conditional_on_same_fully_paid_prefix=True,budget_frozen_for_physical_experiment=False,
            full_prefix_sensor_pairing_and_online_coverage_pending=True))
    assert all(sha(ROOT/name)==expected for name,expected in hashes.items())
    result=dict(status='complete_static_feasibility_only',world_version=VERSION_V24,config=config,
        parents=parents,cases=cases,source_sha256=hashes,
        physical_actions_executed=0,quality_scores_computed=0,
        new_tsdf_or_reference_created=False,initial_sense_pairs=2,
        online_candidates_must_not_receive_truth_regions_or_paths=True,
        semantic_or_full_architecture_efficacy_proven=False)
    output.mkdir(parents=True)
    write(output/'result.json',result)
    write(output/'source_sha256.json',hashes)
    write(output/'artifact_hashes.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps(dict(status=result['status'],budgets=parents,output=str(output)),indent=2),flush=True)


if __name__=='__main__':
    main()
