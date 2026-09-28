#!/usr/bin/env python3
"""Four fixed r1 P00 paid acquisitions and independent physical/packet replay.

Development actions are frozen GT-designed routes shared across assignments;
this does not run an autonomous ANS policy. Only V24 measured meshes are scored.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from dataclasses import asdict,fields
import io
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
import zipfile

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import open3d as o3d
from env.facility_choice_v25_r1 import FacilityChoiceWorldV25,VERSION_V25,prefix_spec_v25
from env.virtual3d_inspection_v4 import MARKER_COLORS
from nso.cpu_sensor_contract_v10 import GridTransform,json_value,digest
from nso.decision_replay_v13 import array_hash,load_packet
from nso.mapping3d_v2 import QualityMapperV2
from nso.observed_shape_v24 import ObservedShapeBackendV24,empty_mesh
from scripts.probe_facility_choice_v24_prefix import measured,MarkerTracks,nonsemantic,geometry_evidence_sha,compressed_arrays
from scripts.probe_facility_shape_v23 import sha,read,versions,mesh_hashes
from scripts.replay_facility_choice_shape_v24 import component_seed_hits,mesh_hash,fixed_seed_association

PROTOCOL=ROOT/'docs/research/V25_R1_P00_PAID_TASK_PROTOCOL_20260915.md'
ANALYSIS=ROOT/'docs/research/V25_P00_FIXED_OPTION_ANALYSIS_PROTOCOL_20260915.md'
COST=ROOT/'audit_results/facility_choice_v25_service_cost_20260915'
DEFAULT=ROOT/'audit_results/facility_choice_v25r1_paid_p00_20260915'
ACQUISITION='two_views_each_attachment_16px_2m_separation_0p4m_bearing_15deg'
CONFIG=dict(version='v25r1-p00-four-scripted-paid-tasks-1',parent='D25-P00',total_budget=400,
    paid_prefix_actions=234,noise_seed=1901,sensor_model='iid_025px',task_asset_count=2,
    task_output_limit_bytes=40*1024**2,shared_metadata_limit_bytes=2*1024**2,free_reserve_bytes=64*1024**2,
    packet_limit_bytes=65536,metadata_reserve_bytes=65536,
    scheduled_primary_tasks=4,prior_main_attempts=0,two_week_main_attempt_limit=36,
    marker_minimum_valid_pixels_per_component=16,marker_track_association_distance_m=.75,
    minimum_supported_distinct_poses_per_track=2,
    representation='V24 measured observed_mesh only; enclosure disabled',
    outline=dict(boundary_spacing_m=.01,thresholds=[.02,.05,.10],
        completion_tolerance_m=.05,completion_minimum_iou=.9),
    stages=['prefix','final'],raw_quality_is_secondary=True,autonomous_planner=False)
PROGRESS=dict(case_index=None,replay=False,phase='unstarted',last_attempted_action=None,last_saved_packet=None,write_owned=False)


class MeasuredOnlyBackend(ObservedShapeBackendV24):
    """Preserve actual ground/component/mesh processing, explicitly disable inference."""
    def _enclosure(self,points,normals,ground):
        return empty_mesh(),dict(accepted=False,reason='disabled_by_frozen_measured_only_contract',
            unseen_surfaces_measured=False,free_ray_conflicts=None)


def total_bytes(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())

def reserve(root,size,receipt=False):
    extra=0 if receipt else CONFIG['metadata_reserve_bytes']
    is_task=root.name.startswith('case_')
    used=total_bytes(root) if is_task else sum(p.stat().st_size for p in root.iterdir() if p.is_file())
    limit=CONFIG['task_output_limit_bytes'] if is_task else CONFIG['shared_metadata_limit_bytes']
    if used+size>limit-extra:
        raise OSError(('task40MiB' if is_task else 'shared metadata2MiB')+' cap would be exceeded; evidence must not be discarded')
    allocated=((size+4095)//4096)*4096+4096+extra
    if shutil.disk_usage(root).free-allocated<CONFIG['free_reserve_bytes']:
        raise OSError('actual64MiB disk reserve would be crossed')

def protected_bytes(root,path,payload,receipt=False):
    reserve(root,len(payload),receipt)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:stream.write(payload)
    os.replace(temporary,path)

def write(root,path,value,receipt=False):
    protected_bytes(root,path,(json.dumps(json_value(value),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode(),receipt)

def seal(root,receipt=False):
    write(root,root/'artifact_hashes.json',{str(p.relative_to(root)):sha(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'artifact_hashes.json'},receipt)

def verify_inventory(root):
    inventory=read(root/'artifact_hashes.json')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'artifact_hashes.json'}
    if actual!=set(inventory):raise ValueError('sealed file set changed: '+str(root))
    for name,h in inventory.items():
        if sha(root/name)!=h:raise ValueError('sealed artifact changed: '+str(root/name))

def packet_payload(packet):
    arrays={};metadata={}
    for field in fields(packet):
        value=getattr(packet,field.name)
        if field.name in ('frame','scan'):
            for inner in fields(value):arrays[field.name+'__'+inner.name]=np.asarray(getattr(value,inner.name))
        else:metadata[field.name]=json_value(value)
    arrays['metadata']=np.asarray(json.dumps(metadata,sort_keys=True))
    payload=compressed_arrays(arrays)
    if len(payload)>CONFIG['packet_limit_bytes']:
        raise ValueError('actual packet exceeds frozen64KiB serialization cap')
    return payload

def prior_packet_pool(root):
    pool={}
    for folder in sorted(root.glob('case_*')):
        if not (folder/'result.json').exists():continue
        for row in read(folder/'result.json')['trace']:
            p=folder/'packets'/f"{row['action_id']:04d}.npz"
            pool.setdefault(row['packet_sha256'],p)
    return pool

def save_packet(folder,action_id,packet,pool):
    payload=packet_payload(packet);path=folder/'packets'/f'{action_id:04d}.npz'
    reserve(folder,len(payload))
    previous=pool.get(packet.sha256())
    if previous is not None:
        import hashlib
        if sha(previous)==hashlib.sha256(payload).hexdigest():
            os.link(previous,path);return 'exact_hardlink_after_actual_sensor_generation'
    with path.open('xb') as stream:stream.write(payload)
    return 'new_compressed_packet'

def mesh_bytes(mesh):return compressed_arrays({key:np.asarray(getattr(mesh,key)) for key in ('vertices','triangles','vertex_colors')})

def save_observed_mesh(folder,path,mesh):
    protected_bytes(folder,path,compressed_arrays({key:np.asarray(getattr(mesh,key)) for key in ('vertices','triangles')}))

def cost_inputs():
    m=read(COST/'manifest.json');r=read(COST/'result.json')
    if m['status']!='complete' or r['status']!='complete_static_cost_only':
        raise ValueError('completed reviewed static cost/acquisition contract required')
    verify_inventory(COST)
    for group in ('source_sha256','protected_source_sha256'):
        for name,h in m[group].items():
            if sha(ROOT/name)!=h:raise ValueError('cost source changed: '+name)
    parent=next(row for row in r['parents'] if row['parent']=='D25-P00')
    if parent['total_budget']!=400 or parent['paid_prefix_actions']!=234:
        raise ValueError('fixed P00 budget/prefix contract differs')
    catalog=parent['route_catalog']
    for arm in ('A','B'):
        option=catalog[arm]
        if (option['acquisition_contract']!=ACQUISITION or not option['reachable']
                or not option['budget_fits'] or option['paid_total']>400
                or not option['scripted_development_arm'] or option['not_for_collection'] or option['cost_bound_only']):
            raise ValueError('no complete budget-compliant reviewed acquisition for '+arm)
        actions=option['actions'];states=option['states']
        if len(actions)!=option['paid_total'] or len(states)!=len(actions)+1 or set(actions)-{'forward','left','right'}:
            raise ValueError('action/state catalog malformed')
        prefix=prefix_spec_v25('D25-P00')
        if actions[:234]!=prefix['actions'] or states[:235]!=prefix['states']:
            raise ValueError('entire actual public prefix must be retained')
        schedule=option['frame_schedule']
        if [row['frame_index'] for row in schedule]!=list(range(1,len(actions)+1)) or [row['action'] for row in schedule]!=actions:
            raise ValueError('frame schedule must describe every paid action')
        if states[-1]!=parent['anchor'] or states[0]!=parent['anchor']:
            raise ValueError('full option must return to original cell AND heading')
    return m,parent

def check_frozen(root):
    m=read(root/'manifest.json')
    if m['config']!=CONFIG:raise ValueError('task configuration changed')
    for name,h in m['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('frozen source changed: '+name)
    for name,h in m['input_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('frozen cost input changed: '+name)
    if sha(root/'sources.zip')!=m['source_archive_sha256'] or versions()!=m['versions']:
        raise ValueError('new archive/dependency versions changed')
    _,parent=cost_inputs()
    if m['route_catalog']!={arm:parent['route_catalog'][arm] for arm in ('A','B')}:
        raise ValueError('frozen A/B route catalog changed')
    return m

def prepare(root):
    cost,parent=cost_inputs()
    if shutil.disk_usage(ROOT).free<CONFIG['task_output_limit_bytes']+CONFIG['free_reserve_bytes']:
        raise OSError('prepare requires40MiB task capacity plus64MiB actual reserve')
    sources={**cost['protected_source_sha256'],**cost['source_sha256']}
    new=[Path(__file__).resolve(),PROTOCOL,ANALYSIS]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in new})
    root.mkdir(parents=True,exist_ok=False)
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in new:archive.writestr(str(p.relative_to(ROOT)),p.read_bytes())
    protected_bytes(root,root/'sources.zip',stream.getvalue())
    cases=[]
    for assignment in ('A_complex_B_simple','A_simple_B_complex'):
        for arm in ('A','B'):
            cases.append(dict(index=len(cases),parent='D25-P00',assignment=assignment,arm=arm,
                budget=400,route_sha256=digest(parent['route_catalog'][arm]),
                physical_status='unstarted',replay_status='unstarted'))
    inputs=[COST/name for name in ('manifest.json','result.json','artifact_hashes.json','sources.zip') if (COST/name).exists()]
    m=dict(status='prepared',config=CONFIG,world_version=VERSION_V25,source_sha256=sources,
        source_archive_sha256=sha(root/'sources.zip'),source_archive_scope='new collector and protocols only; protected source archives referenced through sealed cost provenance',input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        versions=versions(),route_catalog={arm:parent['route_catalog'][arm] for arm in ('A','B')},
        anchor=parent['anchor'],cases=cases,main_attempts_started=0,quota_start=0,quota_limit=36,
        preparation_sensor_frames=0,preparation_world_actions=0,preparation_snapshots=0,
        scripted_gt_designed_shared_routes=True,autonomous_ans_policy=False,
        raw_shared_only_after_actual_exact_generation=True)
    write(root,root/'manifest.json',m);check_frozen(root)
    print('PREPARED4 P00 tasks;0/36 attempted;no world constructed or sensed',flush=True)

def make_stage(folder,name,action_id,mapper,backends,seeds,tracks,replay,old_stage=None):
    raw=mapper.mesh();rows=[];meshes=[]
    for slot,backend in enumerate(backends):
        snapshot=backend.snapshot(raw_mesh=raw)
        if len(snapshot['inferred_mesh'].triangles) or mesh_hash(snapshot['observed_mesh'])!=mesh_hash(snapshot['completed_mesh']):
            raise ValueError('measured-only adapter produced inference')
        geometry={key:dict(count=len(snapshot[key]),sha256=array_hash(snapshot[key])) for key in
            ('measured_points_xyz','ground_points_xyz','cleaned_points_xyz','unassigned_points_xyz')}
        mesh=snapshot['observed_mesh'];path=folder/'meshes'/f'{name}_slot_{slot}_observed_mesh.npz'
        hashes=mesh_hash(mesh)
        if replay:
            with np.load(path,allow_pickle=False) as data:
                if {k:array_hash(data[k]) for k in ('vertices','triangles')}!=hashes:
                    raise ValueError('saved instance mesh differs from independent replay')
        else:save_observed_mesh(folder,path,mesh)
        rows.append(dict(observed_slot=slot,seed=seeds[slot],observed_frames=snapshot['observed_frames'],
            ground=snapshot['ground_plane'],geometry_arrays=geometry,observed_mesh_sha256=hashes,
            observed_mesh_file=str(path.relative_to(folder)),observed_mesh_size=dict(vertices=len(mesh.vertices),triangles=len(mesh.triangles)),
            inferred_mesh_disabled=True,unseen_surfaces_measured=False))
        meshes.append(mesh)
    duplicate=rows[0]['geometry_arrays']['cleaned_points_xyz']['count']>0 and rows[0]['geometry_arrays']['cleaned_points_xyz']==rows[1]['geometry_arrays']['cleaned_points_xyz']
    nonempty=all(row['geometry_arrays']['cleaned_points_xyz']['count']>0 and row['observed_mesh_size']['triangles']>0 for row in rows)
    stage=dict(name=name,action_id=action_id,instances=rows,raw_mesh_sha256=mesh_hashes(raw),
        class_stripped_mapper_geometry_sha256=geometry_evidence_sha(mapper),observed_track_summary=tracks.summary(),
        duplicate_seeded_components=duplicate,all_observed_instances_nonempty=nonempty,
        minimum_instance_separation_gate_passed=nonempty and not duplicate,
        minimum_separation_scope='nonempty and not exactly the same observed component; not verified semantic segmentation')
    stage=json_value(stage)
    if replay:
        for key in stage:
            if stage[key]!=old_stage[key]:raise ValueError('independent stage representation mismatch: '+name+'/'+key)
    else:write(folder,folder/f'{name}_snapshot.json',stage)
    return stage,raw,meshes

def evaluate_stages(world,case,stages,stage_geometry,trace,execution_failed):
    # The reference is instantiated only after all physical actions and all
    # available prefix/final measured outputs, never handed to a controller.
    from utils.facility_outline_v23 import OutlineEvaluatorV23
    from env.facility_choice_v25_r1 import FacilityChoiceWorldV25 as ReferenceBase
    class ReferenceWorld(ReferenceBase):
        def sense(self,*a,**kw):raise AssertionError('evaluation-only reference cannot sense')
        def scan(self,*a,**kw):raise AssertionError('evaluation-only reference cannot scan')
        def step(self,*a,**kw):raise AssertionError('evaluation-only reference cannot step')
    reference=ReferenceWorld(case['parent'],case['assignment'],CONFIG['sensor_model'],CONFIG['noise_seed'])
    evaluator=OutlineEvaluatorV23.from_world(reference,**CONFIG['outline'])
    for name,stage in stages.items():
        raw,meshes=stage_geometry[name];row=trace[stage['action_id']]
        arguments=dict(coverage=row['coverage_2d'],returned=row['returned_to_anchor'],collisions=row['cumulative_collisions'],
            failed=execution_failed if name=='final' else False,paid_actions=stage['action_id'],budget=400)
        association=fixed_seed_association([x['seed'] for x in stage['instances']],reference.objects,len(stage['observed_track_summary']))
        joined=o3d.geometry.TriangleMesh()
        for mesh in meshes:joined+=mesh
        stage['main_observed']=evaluator.evaluate(joined,**arguments)
        stage['raw_secondary']=evaluator.evaluate(raw,**arguments)
        stage['union_observed_mesh_sha256']=mesh_hash(joined)
        stage['fixed_seed_association']=association
        foreign=[]
        for slot,item in enumerate(stage['instances']):
            scores=evaluator.evaluate(meshes[slot],**arguments)['instances']
            item['per_reference_window']=scores;item['fixed_seed_association']=association['rows'][slot]
            match=association['rows'][slot]['reference_id']
            item['foreign_window_nonempty_reference_ids']=[x['id'] for x in scores if not x['missing'] and x['id']!=match]
            foreign+=item['foreign_window_nonempty_reference_ids']
        stage['foreign_window_nonempty_reference_ids']=foreign
        stage['evaluation_association_and_minimum_separation_passed']=association['seed_association_gate_passed'] and stage['minimum_instance_separation_gate_passed'] and not foreign
        stage['instance_association_gate_scope']='fixed seed mapping, nonempty/different component, no other task-window mesh; no GT output correction'
    return evaluator.reference_signature

def append_trace(folder,row):
    payload=(json.dumps(json_value(row),separators=(',',':'),allow_nan=False)+'\n').encode()
    reserve(folder,len(payload))
    with (folder/'trace.jsonl').open('ab') as stream:stream.write(payload)

def run_case(root,index,replay):
    m=check_frozen(root);case=m['cases'][index];route=m['route_catalog'][case['arm']]
    folder=root/f'case_{index:02d}';expected=None
    if replay:
        verify_inventory(folder);expected=read(folder/'result.json')
        if read(folder/'timing.json')['physical_process_id']==os.getpid():raise ValueError('independent replay PID required')
        reserve(folder,CONFIG['metadata_reserve_bytes'])
        PROGRESS['write_owned']=True
        case['replay_status']='running';case['replay_process_id']=os.getpid()
        write(root,root/'manifest.json',m)
    else:
        if shutil.disk_usage(ROOT).free<CONFIG['task_output_limit_bytes']+CONFIG['free_reserve_bytes']:
            raise OSError('unstarted task needs40MiB plus64MiB;no attempt claimed')
        if m['quota_start']+m['main_attempts_started']>=m['quota_limit']:
            raise ValueError('36 main-attempt limit exhausted')
        folder.mkdir(exist_ok=False);PROGRESS['write_owned']=True
        (folder/'packets').mkdir();(folder/'meshes').mkdir()
        m['main_attempts_started']+=1;m['status']='running';case['physical_status']='running'
        case['main_attempt_number']=m['quota_start']+m['main_attempts_started'];case['physical_process_id']=os.getpid()
        write(root,root/'manifest.json',m)
    PROGRESS.update(case_index=index,replay=replay,phase='initializing',last_attempted_action=None,last_saved_packet=None)
    started=perf_counter();pool={} if replay else prior_packet_pool(root)
    world=FacilityChoiceWorldV25(case['parent'],case['assignment'],CONFIG['sensor_model'],CONFIG['noise_seed'])
    public_world_config=asdict(world.config);public_shape=list(world.shape)
    mapper=QualityMapperV2(world.shape,world.config,truncation_m=.12)
    backends=[MeasuredOnlyBackend(),MeasuredOnlyBackend()];tracks=MarkerTracks(CONFIG)
    seeds=[None,None];extras=[];trace=[];stages={};stage_geometry={};storage=dict(new_packets=0,exact_hardlinks=0)
    anchor=tuple(m['anchor']);execution_failed=False;termination='complete_fixed_acquisition_and_return'
    for action_id in range(len(route['actions'])+1):
        PROGRESS.update(phase='paid_acquisition' if not replay else 'independent_physical_and_saved_replay',last_attempted_action=action_id)
        if not replay:
            # All remaining packets can occupy the full declared64KiB each;
            # reserve them before a new paid action. Terminal meshes may fail
            # the remaining40MiB cap, but complete raw history stays preserved.
            remaining=(CONFIG['total_budget']+1-action_id)*CONFIG['packet_limit_bytes']
            reserve(folder,remaining)
        action=None if action_id==0 else route['actions'][action_id-1]
        if action_id:
            frame,collision,done=world.step(action);actual=measured(world,frame,action,collision,done)
        else:actual=measured(world)
        path=folder/'packets'/f'{action_id:04d}.npz'
        if replay:
            packet=load_packet(path)
            packet.validate(GridTransform(world.shape,world.config.resolution_m),world.config)
            if actual.sha256()!=packet.sha256():raise ValueError(f'fresh physical sensor mismatch at{action_id}')
        else:
            packet=actual;mode=save_packet(folder,action_id,packet,pool)
            storage['exact_hardlinks' if mode.startswith('exact') else 'new_packets']+=1
        PROGRESS['last_saved_packet']=action_id
        mapper.update(packet.frame,packet.scan);components=tracks.consume(packet)
        hits=component_seed_hits(packet.frame,CONFIG['marker_minimum_valid_pixels_per_component'])
        if len(components)!=len(hits):raise ValueError('actual marker component ordering differs')
        current={}
        for component,hit in zip(components,hits):
            slot=component['observed_track_index']
            if slot>=2:extras.append(dict(action_id=action_id,observed_track_index=slot,**hit));continue
            if seeds[slot] is None:
                rr,cc=hit['pixel'];color=packet.frame.color_rgb[rr,cc]
                codes=[int(k) for k,v in MARKER_COLORS.items() if np.array_equal(color,v)]
                if len(codes)!=1:raise ValueError('seed lacks an actual physical marker code')
                seeds[slot]=dict(action_id=action_id,observation_id=action_id,observed_slot=slot,
                    packet_sha256=packet.sha256(),actual_marker_code=codes[0],**hit)
                current[slot]=hit['observed_seed_xyz']
        for slot,backend in enumerate(backends):
            f=packet.frame
            if not backend.observe(f.depth_m,f.intrinsic,f.world_from_camera,
                observation_id=action_id,observed_seed_xyz=current.get(slot)):
                raise ValueError('duplicate paid frame in shape backend')
        known=int(np.count_nonzero((mapper.belief!=-1)&world.reachable));reachable=int(world.reachable.sum())
        pose=[*packet.position,packet.heading];state_matches=pose==route['states'][action_id]
        row=dict(action_id=action_id,action=action,phase='initial' if action_id==0 else route['frame_schedule'][action_id-1]['phase'],
            pose=pose,expected_state_matches=state_matches,collision=packet.collision,world_done=packet.done,
            cumulative_collisions=world.collisions,returned_to_anchor=tuple(pose)==anchor,
            packet_sha256=packet.sha256(),raw_file_sha256=sha(path),nonsemantic=nonsemantic(packet),
            coverage_2d=known/reachable,known_reachable_cells=known,reachable_cells=reachable,
            belief_sha256=array_hash(mapper.belief),visible_sha256=array_hash(mapper.visible),marker_components=components)
        trace.append(row)
        if replay:
            if row!=expected['trace'][action_id]:raise ValueError(f'independent per-step metadata mismatch at{action_id}')
        else:append_trace(folder,row)
        if action_id==234:
            stage,raw,meshes=make_stage(folder,'prefix',action_id,mapper,backends,seeds,tracks,replay,
                read(folder/'prefix_snapshot.json') if replay else None)
            stages['prefix']=stage;stage_geometry['prefix']=(raw,meshes)
        if packet.collision or packet.done or not state_matches:
            execution_failed=True;termination='collision' if packet.collision else 'premature_world_done' if packet.done else 'declared_state_mismatch'
            break
        if action_id%50==0 or action_id==len(route['actions']):
            print(f"case={index} replay={replay} step={action_id}/{len(route['actions'])} C={known/reachable:.4f}",flush=True)
    final_id=trace[-1]['action_id']
    stage,raw,meshes=make_stage(folder,'final',final_id,mapper,backends,seeds,tracks,replay,
        read(folder/'final_snapshot.json') if replay else None)
    stages['final']=stage;stage_geometry['final']=(raw,meshes)
    raw_path=folder/'final_mesh.npz'
    if replay:
        with np.load(raw_path,allow_pickle=False) as data:
            if {k:array_hash(data[k]) for k in ('vertices','triangles','vertex_colors')}!=stage['raw_mesh_sha256']:
                raise ValueError('saved raw terminal mesh arrays differ')
    else:protected_bytes(folder,raw_path,mesh_bytes(raw))
    # Keep all representation receipts before any GT evaluator is constructed.
    if not replay:write(folder,folder/'representation_checkpoint.json',dict(stages=stages,all_available_snapshots_complete=True))
    PROGRESS['phase']='independent_terminal_evaluation' if replay else 'terminal_evaluation'
    reference_signature=evaluate_stages(world,case,stages,stage_geometry,trace,execution_failed)
    final=stages['final'];returned=trace[-1]['returned_to_anchor']
    severe_instance_failure=not final['evaluation_association_and_minimum_separation_passed']
    fatal=execution_failed or not returned or severe_instance_failure
    arrival_receipts=[dict(view_index=i,planned_frame=frame,actual_frame_present=frame<len(trace),
        actual_pose=None if frame>=len(trace) else trace[frame]['pose'],
        matches_declared_view=frame<len(trace) and trace[frame]['pose']==route['selected_view_states'][i],
        actual_packet_sha256=None if frame>=len(trace) else trace[frame]['packet_sha256'])
        for i,frame in enumerate(route['arrival_frame_indices'])]
    result=dict(status='complete',index=index,parent=case['parent'],assignment=case['assignment'],arm=case['arm'],
        world_config=public_world_config,shape=public_shape,return_anchor=list(anchor),
        acquisition_contract=ACQUISITION,paid_view_arrival_receipts=arrival_receipts,
        all_declared_acquisition_views_physically_reached=all(x['matches_declared_view'] for x in arrival_receipts),
        view_pixel_support_scope='predeclared evaluation-only clean-ray visibility from frozen cost catalog; actual stored noisy frames retained without GT masking',
        total_budget=400,paid_actions=world.step_count,remaining_budget=400-world.step_count,
        raw_frames=mapper.frames,returned_to_anchor=returned,collisions=world.collisions,
        termination=termination,execution_failed=execution_failed,fatal_for_unstarted_tasks=fatal,
        final_coverage_2d=trace[-1]['coverage_2d'],eligible=final['main_observed']['eligible'],
        final_main_observed=final['main_observed'],final_raw_secondary=final['raw_secondary'],
        main_quality_05cm=final['main_observed']['05cm'],stages=stages,prefix_snapshot_available='prefix' in stages,
        trace=trace,trajectory_sha256=digest([(x['action_id'],x['action'],x['pose']) for x in trace]),
        final_mesh_sha256=mesh_hashes(raw),final_geometry_evidence_sha256=geometry_evidence_sha(mapper),
        observed_marker_tracks=tracks.summary(),extra_observed_track_receipts=extras,
        route_sha256=case['route_sha256'],reference_signature=reference_signature,
        backend_observe_calls=2*mapper.frames,backend_snapshots=2*len(stages),
        scripted_gt_designed_shared_acquisition=True,autonomous_planner=False,ans_four_module_execution=False,
        semantic_policy_executed=False,inference_disabled=True,all_snapshots_before_gt_evaluator=True,
        geometry_inputs_exclude_reference_and_hidden_assignment=True)
    result=json_value(result)
    if replay:
        if result!=expected:raise ValueError('independent complete task/shape/metric replay differs')
        write(folder,folder/'verification.json',dict(status='passed',independent_process=True,
            physical_process_id=read(folder/'timing.json')['physical_process_id'],replay_process_id=os.getpid(),
            actual_physical_replay_actions=world.step_count,saved_packets_verified=mapper.frames,
            all_per_step_metadata_equal=True,all_stage_geometry_arrays_equal=True,
            all_raw_and_observed_mesh_arrays_equal=True,all_frozen_metrics_equal=True,
            repeated_raw_or_mesh_files_written=0,elapsed_s=perf_counter()-started))
    else:
        write(folder,folder/'result.json',result)
        write(folder,folder/'timing.json',dict(physical_process_id=os.getpid(),elapsed_s=perf_counter()-started,storage=storage))
    seal(folder);check_frozen(root)
    m=read(root/'manifest.json');case=m['cases'][index]
    case['replay_status' if replay else 'physical_status']='complete'
    if fatal:m['status']='stopped_structural_safety_or_instance_failure'
    write(root,root/'manifest.json',m)
    print('TASK COMPLETE',dict(index=index,replay=replay,paid=world.step_count,C=trace[-1]['coverage_2d'],
        Q=result['main_quality_05cm']['outline_macro_quality'],eligible=result['eligible'],fatal=fatal,
        task_output_bytes=total_bytes(folder),free_bytes=shutil.disk_usage(ROOT).free),flush=True)
    if all(c['replay_status']=='complete' for c in m['cases']):aggregate(root)


def aggregate(root):
    m=check_frozen(root);cases=[]
    for item in m['cases']:
        folder=root/f"case_{item['index']:02d}";verify_inventory(folder)
        if read(folder/'verification.json')['status']!='passed':raise ValueError('all four independent replays required')
        cases.append(read(folder/'result.json'))
    prefix_checks=[]
    for i,j in ((0,1),(2,3),(0,2),(1,3)):
        left,right=cases[i],cases[j]
        def shared_prefix(case):
            return [{k:row[k] for k in ('action_id','action','pose','nonsemantic','belief_sha256','visible_sha256','marker_components')}
                    for row in case['trace'][:235]]
        a,b=shared_prefix(left),shared_prefix(right)
        prefix_checks.append(dict(left=i,right=j,full_nonsemantic_prefix_identical=a==b,
            left_sha256=digest(a),right_sha256=digest(b)))
    result=dict(status='complete',cases=[{k:r[k] for k in ('index','parent','assignment','arm','paid_actions','total_budget',
        'final_coverage_2d','returned_to_anchor','collisions','eligible','main_quality_05cm','fatal_for_unstarted_tasks')} for r in cases],
        primary_tasks=4,independent_replays=4,main_attempts_started=m['main_attempts_started'],
        two_week_main_attempts_used=CONFIG['prior_main_attempts']+m['main_attempts_started'],two_week_main_limit=36,
        physical_paid_actions=sum(r['paid_actions'] for r in cases),replay_paid_actions=sum(r['paid_actions'] for r in cases),
        saved_raw_packets=sum(r['raw_frames'] for r in cases),all_eligible=all(r['eligible'] for r in cases),
        prefix_checks=prefix_checks,all_nonsemantic_prefix_checks_passed=all(p['full_nonsemantic_prefix_identical'] for p in prefix_checks),
        inference_disabled=True,not_adaptive_G_or_full_ANS=True,
        interpretation='fixed scripted A/B acquisition matrix; information analysis follows separately frozen protocol')
    write(root,root/'result.json',result);m['status']='complete';write(root,root/'manifest.json',m);seal(root)


def failure(root,index,replay,error):
    folder=root/f'case_{index:02d}'
    if folder.exists():
        receipt=folder/('failure_replay.json' if replay else 'failure_physical.json')
        if not receipt.exists():write(folder,receipt,dict(status='failed',error=repr(error),
            traceback=traceback.format_exc()[-8000:],process_id=os.getpid(),progress=PROGRESS),receipt=True)
        seal(folder,receipt=True)
    m=read(root/'manifest.json');m['status']='stopped_io_or_implementation_failure'
    case=m['cases'][index]
    if not replay and PROGRESS['write_owned'] and 'main_attempt_number' not in case:
        # The directory claim already consumed an attempt, even when the first
        # manifest replacement failed. Never lose or double-count that claim.
        m['main_attempts_started']+=1
        case['main_attempt_number']=m['quota_start']+m['main_attempts_started']
        case['physical_process_id']=os.getpid()
    case['replay_status' if replay else 'physical_status']='failed'
    write(root,root/'manifest.json',m,receipt=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true');modes.add_argument('--case',type=int)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    if args.replay and args.case is None:parser.error('--replay requires --case')
    if args.prepare:
        if root.exists():raise ValueError('new root required; existing evidence untouched')
        try:prepare(root)
        except Exception as error:
            if root.exists() and not (root/'failure_prepare.json').exists():
                write(root,root/'failure_prepare.json',dict(status='failed',error=repr(error),process_id=os.getpid(),
                    preparation_sensor_frames=0,preparation_world_actions=0),receipt=True)
            raise
        return
    m=check_frozen(root)
    if args.case not in range(4):raise ValueError('only four frozen P00 cases exist')
    folder=root/f'case_{args.case:02d}'
    if args.replay:
        if (m['cases'][args.case]['physical_status']!='complete' or m['cases'][args.case]['replay_status']!='unstarted'
                or not (folder/'result.json').exists() or (folder/'verification.json').exists() or (folder/'failure_replay.json').exists()):
            raise ValueError('replay requires completed unverified physical task')
    else:
        if m['status'] not in ('prepared','running') or folder.exists():raise ValueError('unstarted task required; prior evidence retained')
        if any(m['cases'][i]['replay_status']!='complete' for i in range(args.case)):
            raise ValueError('earlier tasks must complete independent replay before next physical task')
        # Capacity refusal happens before ownership/attempt acquisition.
        if shutil.disk_usage(ROOT).free<CONFIG['task_output_limit_bytes']+CONFIG['free_reserve_bytes']:
            raise OSError('need40MiB+64MiB before this unstarted attempt;no task claimed')
    try:run_case(root,args.case,args.replay)
    except Exception as error:
        if PROGRESS['write_owned']:failure(root,args.case,args.replay,error)
        raise

if __name__=='__main__':main()
