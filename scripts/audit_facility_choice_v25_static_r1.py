#!/usr/bin/env python3
"""One bounded V25 static candidate check; no robot, packets, mapping or Q."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys
from time import perf_counter
import traceback
import zipfile

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from env.facility_choice_v25_r1 import (FacilityChoiceWorldV25,StaticGeometryErrorV25,
    PARENTS_V25,ASSIGNMENTS_V25,LAYOUTS_V25,VERSION_V25)
from env.canonical_rgbd_v15 import render_axial_depth
from env.virtual3d import camera_pose
from utils.grid_geometry import DIRECTIONS

OUTPUT=ROOT/'audit_results/facility_choice_v25_static_r1_20260915'
PROTOCOL=ROOT/'docs/research/V25_R1_PARTITION_STATIC_PROTOCOL_20260915.md'
PROTECTED=ROOT/'audit_results/facility_choice_v24_shape_prefix_20260915/manifest.json'
EXPECTED_PROTECTED_SHA='22777ba8918161cf5a963fd837d15c24e6ce43f0b86d3a073faac2c65015a658'
R0_MANIFEST=ROOT/'audit_results/facility_choice_v25_static_r0_20260915/manifest.json'
EXPECTED_R0_MANIFEST_SHA='2dd04d0bb54d0704027787641cf359b98cc99a8492c3a6641bbbb266b78fa032'
REPAIR_COUNT=1
REPAIR_SCOPE='inner_long x[1.8,2.0]; inner_front_return x[.8,2.0]; all other geometry unchanged'
OUTPUT_CAP=2*1024**2
DISK_RESERVE=64*1024**2
FAILURE_RESERVE=32*1024
LAYER_LIMIT=72
MARKER_MIN_PIXELS=16
ACTIONS=('forward','left','right')


class StaticLimitReached(BaseException): pass


def interrupt(number,frame): raise StaticLimitReached('signal '+str(number))


def require(condition,message):
    if not condition: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def digest_array(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def capacity(output,amount=0,emergency=False):
    ancestor=output
    while not ancestor.exists(): ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free-amount < DISK_RESERVE:
        raise OSError('64 MiB free-space reserve would be crossed')
    used=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) if output.exists() else 0
    limit=OUTPUT_CAP if emergency else OUTPUT_CAP-FAILURE_RESERVE
    if used+amount+4096>limit: raise OSError('2 MiB static output cap would be crossed')


def write(output,name,value,emergency=False):
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    capacity(output,len(data),emergency)
    path=output/name; temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream: stream.write(data)
    os.replace(temporary,path)


def budget_time(deadline):
    if perf_counter()>=deadline: raise StaticLimitReached('600-second process budget reached')


class StaticWorld(FacilityChoiceWorldV25):
    def step(self,*args,**kwargs): raise AssertionError('No physical motion in static check')
    def sense(self,*args,**kwargs): raise AssertionError('No actual sensor packets in static check')
    def scan(self,*args,**kwargs): raise AssertionError('No actual sensor packets in static check')


def projected(world,state,boxes=None):
    pose=camera_pose(state[:2],state[2],world.config,world.shape[0])
    primitives=np.asarray(world._solid_primitives,float)[:,:6] if boxes is None else boxes
    return render_axial_depth(primitives,world.intrinsic,pose,world.config.height_px,
                              world.config.width_px,world.config.max_depth_m)


def prefix_actions_valid(world):
    route=world.prefix_proposal; states=route['states']; actions=route['actions']; errors=[]
    if len(states)!=len(actions)+1 or len(actions)!=route['paid_actions']: errors.append('action_count')
    for i,state in enumerate(states):
        if not world.reachable[tuple(state[:2])]: errors.append('unsafe_pose_'+str(i))
        if i:
            before=states[i-1]; action=actions[i-1]
            if action not in ACTIONS: errors.append('invalid_action_'+str(i)); continue
            dr,dc=DIRECTIONS[before[2]] if action=='forward' else (0,0)
            expected=[before[0]+int(dr),before[1]+int(dc),
                      (before[2]+(1 if action=='right' else -1 if action=='left' else 0))%4]
            if state!=expected: errors.append('transition_'+str(i))
    if states[0]!=states[-1] or states[-1]!=route['decision_anchor']: errors.append('anchor_return')
    return dict(passed=not errors,errors=errors,proposed_paid_actions=len(actions),
                proposed_pose_count=len(states),actual_actions_executed=0)


def marker_count(depth,points,marker):
    mask=((depth>0)&(np.abs(points[...,1]-marker['y'])<1e-5)
          &(points[...,0]>marker['x0'])&(points[...,0]<marker['x1'])
          &(points[...,2]>marker['z0'])&(points[...,2]<marker['z1']))
    return int(mask.sum())


def check_prefix(worlds,deadline,progress):
    states=worlds[0].prefix_proposal['states']
    require(states==worlds[1].prefix_proposal['states'],'Paired proposed paths differ')
    differences=[]; hashes=[hashlib.sha256(),hashlib.sha256()]
    markers=[[dict(asset_id=i,maximum_valid_clean_pixels=0,first_ge_16_action=None,
                   maximum_action_id=None,physical_class=w._physical_markers[i]['physical_class'])
              for i in range(2)] for w in worlds]
    for action_id,state in enumerate(states):
        budget_time(deadline); images=[]
        for j,w in enumerate(worlds):
            depth,points=projected(w,state); images.append(depth)
            hashes[j].update(np.ascontiguousarray(depth).tobytes())
            for i,marker in enumerate(w._physical_markers):
                count=marker_count(depth,points,marker); row=markers[j][i]
                if count>row['maximum_valid_clean_pixels']:
                    row.update(maximum_valid_clean_pixels=count,maximum_action_id=action_id)
                if count>=MARKER_MIN_PIXELS and row['first_ge_16_action'] is None:
                    row['first_ge_16_action']=action_id
        changed=int(np.count_nonzero(images[0]!=images[1]))
        if changed: differences.append(dict(action_id=action_id,state=state,differing_pixels=changed))
        progress['prefix_poses_checked']=action_id+1
    return dict(pose_count=len(states),all_clean_depths_identical=not differences,
        depth_sequence_sha256=[h.hexdigest() for h in hashes],first_difference=differences[0] if differences else None,
        differences=differences,markers=markers,marker_min_pixels=MARKER_MIN_PIXELS,
        both_actual_marker_patches_ge_16=all(r['first_ge_16_action'] is not None for rows in markers for r in rows),
        marker_threshold_source='existing MarkerTracks effective-pixel seed contract',
        actual_paid_sensor_pairing_verified=False,noisy_marker_seed_verified=False)


def boundary_support(world,asset_id,kind):
    cx,front=LAYOUTS_V25[world.parent]['fronts'][asset_id]; side=1 if asset_id==0 else -1
    stage='AB'[asset_id]+('_front' if kind=='front_top' else '_left' if asset_id==0 else '_right')
    action=world.prefix_proposal['stage_indices'][stage]; state=world.prefix_proposal['states'][action]
    if kind=='front_top': points=np.asarray([[cx-.4,front,1.6],[cx+.4,front,1.6]])
    elif kind=='outer_top': points=np.asarray([[cx-side*.8,front+.2,1.6],[cx-side*.8,front+.6,1.6]])
    else: points=np.asarray([[cx-side*.8,front+.8,.4],[cx-side*.8,front+.8,1.2]])
    pose=camera_pose(state[:2],state[2],world.config,world.shape[0]); camera=(points-pose[:3,3])@pose[:3,:3]
    uvw=camera@world.intrinsic.T; uv=uvw[:,:2]/uvw[:,2,None]
    depth,_=projected(world,state); expected=float(camera[:,2].mean())
    if kind.endswith('top'):
        coordinate=int(np.floor(uv[:,1].min())); varying=np.arange(int(np.ceil(uv[:,0].min())),int(np.floor(uv[:,0].max()))+1)
        background=[(coordinate,int(x)) for x in varying]; foreground=[(coordinate+1,int(x)) for x in varying]
    else:
        coordinate=int(np.floor(uv[:,0].min())) if asset_id==0 else int(np.ceil(uv[:,0].max()))
        varying=np.arange(int(np.ceil(uv[:,1].min())),int(np.floor(uv[:,1].max()))+1)
        background=[(int(x),coordinate) for x in varying]
        foreground=[(int(x),coordinate+(1 if asset_id==0 else -1)) for x in varying]
    inside=bool(background) and all(0<=r<depth.shape[0] and 0<=c<depth.shape[1] for r,c in background+foreground)
    if inside:
        bg=np.asarray([depth[p] for p in background]); fg=np.asarray([depth[p] for p in foreground])
        hit=np.abs(fg-expected)<1e-4; farther=bg>expected+.2; valid=hit&farther
    else: bg=fg=np.array([]); hit=farther=valid=np.array([],bool)
    return dict(asset_id=asset_id,kind=kind,stage=stage,proposed_action_id=action,pose=state,
        true_segment_xyz=points.tolist(),projected_uv=uv.tolist(),in_image=inside,tested_pixel_pairs=len(background),
        foreground_body_plane_hits=int(hit.sum()),actual_farther_background_hits=int(farther.sum()),
        valid_bracket_pairs=int(valid.sum()),finite_contract_passed=bool(valid.any()),
        all_segment_pairs_passed=bool(len(valid) and valid.all()),expected_body_axial_depth_m=expected,
        actual_background_depth_range_m=[float(bg.min()),float(bg.max())] if len(bg) else None,
        observed_whole_shape_completion_proven=False)


def attachment_visibility(world,deadline,progress):
    complex_asset=next(o for o in world.objects if o['category']==3)
    ai=complex_asset['id']; cx,front=LAYOUTS_V25[world.parent]['fronts'][ai]
    primitives=np.asarray(world._solid_primitives,float)
    ids=np.flatnonzero(primitives[:,6]==complex_asset['owner']).tolist()[1:]
    require(len(ids)==2,'Original two complex attachments must remain unchanged')
    poses=[]
    for r,c in np.argwhere(world.reachable):
        x=(int(c)+.5)*world.config.resolution_m; y=(world.shape[0]-int(r)-.5)*world.config.resolution_m
        if cx-1.8-1e-9<=x<=cx+1.8+1e-9 and front+1.8-1e-9<=y<=front+2.9+1e-9:
            for heading in range(4): poses.append([int(r),int(c),heading])
    rows=[]
    for primitive_id in ids:
        witness=None; checked=0; reduced=np.delete(primitives[:,:6],primitive_id,axis=0)
        for state in poses:
            budget_time(deadline); full,_=projected(world,state); removed,_=projected(world,state,reduced)
            checked+=1; changed=int(np.count_nonzero(full!=removed))
            progress['attachment_static_poses_checked']=progress.get('attachment_static_poses_checked',0)+1
            if changed:
                witness=dict(state=state,differing_depth_pixels=changed,current_true_safe=True,
                             complete_depth_sha256=digest_array(full),removed_component_depth_sha256=digest_array(removed))
                break
        rows.append(dict(asset_id=ai,primitive_id=primitive_id,physical_primitive=primitives[primitive_id,:6].tolist(),
            checked_view_count=checked,declared_candidate_count=len(poses),visible_external_contribution=witness is not None,
            witness=witness,unseen_does_not_prove_global_invisibility=True,not_an_online_view_template=True))
    return rows


def pose_graph(world):
    """Complete unit-action GT safe graph; all in-place turns remain legal."""
    nrows,ncols=world.shape; count=nrows*ncols*4
    distance=np.full(count,-1,np.int32); previous=np.full(count,-1,np.int32); action=np.full(count,-1,np.int8)
    def encode(r,c,h): return (r*ncols+c)*4+h
    anchor=encode(*world.start,world.start_heading); distance[anchor]=0; previous[anchor]=-2
    queue=deque([anchor])
    while queue:
        current=queue.popleft(); cell,h=divmod(current,4); r,c=divmod(cell,ncols)
        dr,dc=DIRECTIONS[h]; rr,cc=r+int(dr),c+int(dc)
        nexts=[encode(rr,cc,h) if 0<=rr<nrows and 0<=cc<ncols and world.reachable[rr,cc] else -1,
               encode(r,c,(h-1)%4),encode(r,c,(h+1)%4)]
        for ai,target in enumerate(nexts):
            if target<0 or distance[target]>=0: continue
            distance[target]=distance[current]+1; previous[target]=current; action[target]=ai; queue.append(target)
    require(int(np.count_nonzero(distance>=0))==int(world.reachable.sum())*4,'Incomplete legal posture graph')
    return dict(distance=distance,previous=previous,action=action,anchor=anchor)


def decode_pose(index,shape):
    cell,heading=divmod(int(index),4); r,c=divmod(cell,shape[1]); return [r,c,heading]


def graph_path(graph,target,world):
    node=int(target); states=[]; actions=[]
    while node!=graph['anchor']:
        states.append(decode_pose(node,world.shape)); actions.append(ACTIONS[int(graph['action'][node])]); node=int(graph['previous'][node])
        require(node>=0,'Broken BFS predecessor')
    states.append(decode_pose(node,world.shape)); states.reverse(); actions.reverse()
    for before,act,after in zip(states,actions,states[1:]):
        dr,dc=DIRECTIONS[before[2]] if act=='forward' else (0,0)
        expected=[before[0]+int(dr),before[1]+int(dc),(before[2]+(1 if act=='right' else -1 if act=='left' else 0))%4]
        require(after==expected and world.reachable[tuple(after[:2])],'Invalid attaining path')
    require(len(actions)==int(graph['distance'][target]),'BFS distance/path mismatch')
    return dict(actions=actions,states=states,paid_action_cost=len(actions),executed=False,
                GT_graph_only=True,unknown_map_policy_cost_proven=False)


def outside_changed_frusta(world,state,boxes):
    """Conservative box-corner projection exclusion; no hidden ray masking."""
    pose=camera_pose(state[:2],state[2],world.config,world.shape[0]); c=world.config
    for box in boxes:
        corners=np.asarray([[box[0]+i*box[3],box[1]+j*box[4],box[2]+k*box[5]]
                            for i in (0,1) for j in (0,1) for k in (0,1)])
        camera=(corners-pose[:3,3])@pose[:3,:3]; z=camera[:,2]
        if z.max()<=.15 or z.min()>c.max_depth_m: continue
        if z.min()<=0: return False
        uvw=camera@world.intrinsic.T; uv=uvw[:,:2]/uvw[:,2,None]
        if uv[:,0].max()<0 or uv[:,0].min()>c.width_px-1 or uv[:,1].max()<0 or uv[:,1].min()>c.height_px-1: continue
        return False
    return True


def first_information(worlds,deadline,output,progress):
    graphs=[pose_graph(w) for w in worlds]
    distances=np.stack([g['distance'] for g in graphs]); available=np.where(distances<0,np.iinfo(np.int32).max,distances)
    closest=available.min(axis=0)
    primitives=[np.asarray(w._solid_primitives,float)[:,:6] for w in worlds]
    sets=[{tuple(row) for row in b} for b in primitives]; changed_boxes=np.asarray(sorted(sets[0]^sets[1]),float)
    require(len(changed_boxes)==4,'Expected only four assignment-dependent attachment boxes')
    layers=[]; first=None; censored=[]; total_rendered=total_excluded=0
    for layer in range(LAYER_LIMIT+1):
        budget_time(deadline); ids=np.flatnonzero(closest==layer); witnesses=[]; exclusions=rendered=0
        progress['information_current_incomplete_layer']=layer
        for index in ids:
            budget_time(deadline); state=decode_pose(index,worlds[0].shape)
            origin=camera_pose(state[:2],state[2],worlds[0].config,worlds[0].shape[0])[:3,3]
            inside=[bool(np.any(np.all((origin>np.round(b[:,:3],9))&(origin<np.round(b[:,:3]+b[:,3:],9)),axis=1))) for b in primitives]
            if any(inside):
                censored.append(dict(distance=layer,state=state,origin_inside_world=inside)); continue
            if outside_changed_frusta(worlds[0],state,changed_boxes): exclusions+=1; continue
            images=[projected(w,state)[0] for w in worlds]; rendered+=1
            changed=int(np.count_nonzero(images[0]!=images[1]))
            if changed:
                valid=[bool(w.reachable[tuple(state[:2])]) for w in worlds]
                graph_index=int(np.argmin(available[:,index])); where=np.argwhere(images[0]!=images[1])[0]
                witness=dict(state=state,differing_pixels=changed,safe_by_assignment=valid,both_assignments_safe=all(valid),
                    distance_by_assignment=[int(d[index]) if d[index]>=0 else None for d in distances],
                    optimistic_distance=layer,attaining_assignment=worlds[graph_index].assignment,
                    first_differing_pixel_rc=where.tolist(),first_pixel_depths=[float(x[tuple(where)]) for x in images],
                    depth_sha256=[digest_array(x) for x in images])
                if not witnesses: witness['attaining_GT_path']=graph_path(graphs[graph_index],index,worlds[graph_index])
                witnesses.append(witness)
        total_rendered+=rendered; total_excluded+=exclusions
        rows=dict(distance=layer,all_layer_poses_checked=True,pose_count=len(ids),rendered_pose_pairs=rendered,
            conservative_frustum_exclusions=exclusions,differing_pose_count=len(witnesses),
            censored_pose_count=sum(r['distance']==layer for r in censored))
        layers.append(rows); progress['information_completed_layers']=layers
        progress['information_current_incomplete_layer']=None
        write(output,'progress.json',progress)
        if witnesses:
            first=dict(distance=layer,witnesses=witnesses); break
    first_censor=min((r['distance'] for r in censored),default=None)
    found=None if first is None else first['distance']
    exact=found is not None and (first_censor is None or first_censor>=found)
    excluded_through=(found-1 if found is not None else layers[-1]['distance'])
    if first_censor is not None: excluded_through=min(excluded_through,first_censor-1)
    return dict(status='first_difference_layer_complete' if first else 'bounded_distance_search_complete',
        layer_limit=LAYER_LIMIT,complete_layers=layers,optimistic_first_clean_difference_distance=found if exact else None,
        witnessed_difference=first,censored_poses=censored,first_censored_distance=first_censor,
        certified_no_difference_through_distance=excluded_through,all_prior_layers_comparable=first_censor is None or (found is not None and first_censor>=found),
        rendered_pose_pairs=total_rendered,conservative_frustum_exclusions=total_excluded,
        cost_scope='minimum of two full GT-safe graphs with ideal single-pixel recognition; optimistic geometric bound only',
        exact_executable_unknown_map_policy_cost=False,continuous_pose_claim=False,no_noisy_recognition_claim=True,
        all_legal_turns_and_moves_included=True,full_information_absence_proven=False,switching_cost_not_checked=True)


def execute(output,deadline,progress):
    require(sha(PROTECTED)==EXPECTED_PROTECTED_SHA,'Prior147 source manifest changed')
    protected=read(PROTECTED)['source_sha256']; require(len(protected)==147,'Expected explicit old147 list')
    require(sha(R0_MANIFEST)==EXPECTED_R0_MANIFEST_SHA,'Frozen r0 manifest changed')
    r0_sources=read(R0_MANIFEST)['source_sha256']; require(len(r0_sources)==3,'Expected explicit r0 three-source list')
    require(not set(protected)&set(r0_sources),'Unexpected overlap in protected source lists')
    protected.update(r0_sources)
    require(len(protected)==150,'Expected old147 plus r0 three frozen sources')
    require(all(sha(ROOT/k)==v for k,v in protected.items()),'Old147 or r0 source changed')
    paths=[Path(__file__).resolve(),PROTOCOL,ROOT/'env/facility_choice_v25_r1.py']
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    capacity(output,OUTPUT_CAP-65536); output.mkdir(parents=True); progress['created_output']=True
    write(output,'manifest.json',dict(status='running',world_version=VERSION_V25,source_sha256=sources,
        protected_source_sha256=protected,protected_manifest_sha256=sha(PROTECTED),protocol=str(PROTOCOL),
        protected_r0_manifest_sha256=sha(R0_MANIFEST),repair_count=REPAIR_COUNT,repair_scope=REPAIR_SCOPE,
        maximum_seconds=600,output_cap_bytes=OUTPUT_CAP,disk_reserve_bytes=DISK_RESERVE,
        parents=list(PARENTS_V25),assignments=list(ASSIGNMENTS_V25),no_physical_or_sensor_calls=True))
    capacity(output,sum(p.stat().st_size for p in paths)+8192)
    with zipfile.ZipFile(output/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in paths: archive.write(path,str(path.relative_to(ROOT)))
    cases=[]; parents=[]
    for parent in PARENTS_V25:
        progress.update(parent=parent,phase='geometry',prefix_poses_checked=0,information_completed_layers=[])
        worlds=[]; parent_cases=[]
        for assignment in ASSIGNMENTS_V25:
            budget_time(deadline)
            try:
                world=StaticWorld(parent,assignment)
                row=dict(status='constructed',parent=parent,assignment=assignment,
                    geometry=world.static_geometry_audit,prefix_actions=prefix_actions_valid(world),
                    original_sensor_config=asdict(world.config),task_asset_count=len(world.objects),
                    legacy_service_regions=world.legacy_service_region_audit,service_contract=world.service_contract,
                    mesh_sha256={k:digest_array(np.asarray(getattr(world.mesh,k))) for k in ('vertices','triangles')},
                    physical_actions_executed=world.step_count)
                worlds.append(world)
            except Exception as error:
                row=dict(status='geometry_failed',parent=parent,assignment=assignment,error=repr(error),
                         audit=getattr(error,'audit',None),traceback=traceback.format_exc())
                worlds.append(None)
            parent_cases.append(row); cases.append(row)
        if not all(worlds):
            parents.append(dict(parent=parent,status='geometry_failed',finite_static_contract_passed=False))
        else:
            progress['phase']='full_prefix_and_marker_projection'
            pairing=check_prefix(worlds,deadline,progress)
            for world,row in zip(worlds,parent_cases):
                progress.update(phase='body_edge_and_attachment_visibility',assignment=world.assignment)
                edges=[boundary_support(world,i,kind) for i in range(2) for kind in ('front_top','outer_top','outer_rear')]
                visibility=attachment_visibility(world,deadline,progress)
                row.update(status='static_checks_complete',body_edge_support=edges,attachment_visibility=visibility,
                    all_body_edge_finite_contracts_passed=all(e['finite_contract_passed'] for e in edges),
                    all_complex_attachments_have_reachable_witness=all(v['visible_external_contribution'] for v in visibility))
            checks=dict(prefix_actions_safe=all(r['prefix_actions']['passed'] for r in parent_cases),
                clean_prefix_paired=pairing['all_clean_depths_identical'],both_markers_meet_existing_16px_contract=pairing['both_actual_marker_patches_ge_16'],
                body_top_and_rear_support=all(r['all_body_edge_finite_contracts_passed'] for r in parent_cases),
                key_attachments_reachable_visible=all(r['all_complex_attachments_have_reachable_witness'] for r in parent_cases))
            parent_row=dict(parent=parent,status='limited_observability_checked',prefix_projection=pairing,
                            finite_static_checks=checks,finite_static_contract_passed=all(checks.values()))
            parents.append(parent_row)
            # Retain all finite-contract outcomes even if the following bounded
            # information search is interrupted or reaches its time cap.
            write(output,'partial_result.json',dict(status='partial_before_information_search',cases=cases,parents=parents))
            progress.update(phase='first_information_BFS',assignment=None)
            information=first_information(worlds,deadline,output,progress)
            parent_row.update(status='static_checks_complete',information=information)
            require(all(w.step_count==0 and w.collisions==0 for w in worlds),'Unexpected actual world action')
        write(output,'partial_result.json',dict(status='partial',cases=cases,parents=parents))
        print(parent,parents[-1]['status'],'finite_static_gate',parents[-1]['finite_static_contract_passed'],flush=True)
    require(all(sha(ROOT/k)==v for k,v in protected.items()),'Old source changed during static process')
    require(all(sha(ROOT/k)==v for k,v in sources.items()),'New source changed during static process')
    result=dict(status='complete_static_only',world_version=VERSION_V25,parents=parents,cases=cases,
        finite_static_contract_passed=all(r['finite_static_contract_passed'] for r in parents),
        new_physical_actions=0,new_sensor_packets=0,new_tsdf_fusions=0,quality_scores_computed=0,
        GT_information_graphs_are_evaluator_only=True,old_service_success_claims_reused=False,
        original_P01_coverage_failure_preserved=True,actual_common_coverage_verified=False,
        noisy_instance_separation_verified=False,semantic_efficacy_proven=False,
        automatic_followup_trajectories_authorized=False,repair_count=REPAIR_COUNT,
        repair_scope=REPAIR_SCOPE,r0_failure_preserved=True,second_repair_authorized=False)
    write(output,'result.json',result)
    manifest=read(output/'manifest.json'); manifest['status']='complete'; write(output,'manifest.json',manifest)
    print(json.dumps(dict(output=str(output),finite_static_contract_passed=result['finite_static_contract_passed'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--run',action='store_true',help='Explicit separately authorized single static process')
    args=parser.parse_args()
    if not args.run: parser.error('Explicit --run required; no automatic world construction')
    output=args.output.resolve(); require(not output.exists(),'Existing static evidence must not be replaced')
    progress=dict(created_output=False,phase='verify_sources'); started=perf_counter(); code=0
    for number in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM): signal.signal(number,interrupt)
    signal.setitimer(signal.ITIMER_REAL,600)
    try:
        execute(output,started+600,progress)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0); code=124 if isinstance(error,StaticLimitReached) else 1
        failure=dict(status='bounded_incomplete' if code==124 else 'failed',error=repr(error),
                     traceback=traceback.format_exc(),progress=progress,process_id=os.getpid(),elapsed_s=perf_counter()-started)
        print(json.dumps(failure,ensure_ascii=False),file=sys.stderr)
        if progress['created_output']:
            try:
                temporary=output/'manifest.json.tmp'
                if temporary.exists(): temporary.replace(output/'interrupted_manifest.json.tmp')
                write(output,'failure.json',failure,emergency=True)
                if (output/'manifest.json').exists():
                    value=read(output/'manifest.json'); value['status']=failure['status']; write(output,'manifest.json',value,emergency=True)
            except Exception as receipt_error: print('Failure receipt could not preserve storage contract: '+repr(receipt_error),file=sys.stderr)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if progress['created_output']:
            try:
                write(output,'timing.json',dict(process_id=os.getpid(),elapsed_s=perf_counter()-started,exit_code=code),emergency=code!=0)
                inventory={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
                write(output,'artifact_hashes.json',inventory,emergency=code!=0)
            except Exception as final_error: print('Final receipt error: '+repr(final_error),file=sys.stderr); code=code or 1
    raise SystemExit(code)


if __name__=='__main__': main()
