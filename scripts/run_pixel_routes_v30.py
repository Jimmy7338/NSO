#!/usr/bin/env python3
"""Frozen four-case acquisition, separately invoked fresh replay and analysis."""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import zipfile
from dataclasses import fields

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import open3d as o3d
from env.information_pixel_v30 import InformationPixelWorldV30, scene_boxes, swept_clear, SHIFT
from nso.cpu_sensor_contract_v10 import json_value, digest
from nso.decision_replay_v13 import array_hash, load_packet
from nso.facility_measurement_v26 import FacilityMeasurementV26
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10

OUTPUT = ROOT / 'audit_results/v30_pixel_routes_20260917'
CONFIG = ROOT / 'configs/virtual3d/v30_pixel_routes_20260917.json'
PROTOCOL = ROOT / 'docs/research/V30_PIXEL_ROUTE_PROTOCOL_20260917.md'
WITNESS = ROOT / 'audit_results/v29_scene_information_20260917/main_policy_witnesses.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def guard_write(path, size):
    path=Path(path)
    if shutil.disk_usage(ROOT).free-size < 64*1024**2:
        raise RuntimeError('write would violate disk reserve')
    try:
        relative=path.relative_to(OUTPUT)
    except ValueError:
        return
    part=relative.parts[0]
    folder=OUTPUT/part if part.startswith('case') else OUTPUT
    paths=folder.rglob('*') if part.startswith('case') else folder.glob('*')
    current=sum(p.stat().st_size for p in paths if p.is_file()) if folder.exists() else 0
    cap=(40 if part.startswith('case') else 2)*1024**2
    if current+size>cap:
        raise RuntimeError('write would violate output cap')


def write(path, value):
    path = Path(path)
    payload = (json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    guard_write(path,len(payload))
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
    os.replace(temporary, path)


def quota_guard(folder):
    if shutil.disk_usage(ROOT).free < 64*1024**2:
        raise RuntimeError('disk reserve violated')
    if folder.exists() and sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) > 40*1024**2:
        raise RuntimeError('case storage cap violated')


def closure(paths):
    visited = set()
    pending = list(paths)
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        if path.suffix != '.py':
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
            for name in names:
                rel = Path(*name.split('.'))
                choices = [ROOT / rel.with_suffix('.py'), ROOT / rel / '__init__.py']
                choices.extend(ROOT/Path(*name.split('.')[:i])/'__init__.py' for i in range(1,len(name.split('.'))))
                for candidate in choices:
                    if candidate.is_file() and candidate.resolve() not in visited:
                        pending.append(candidate)
    return sorted(visited)


def seal(folder):
    write(folder/'artifact_hashes.json', {str(p.relative_to(folder)):sha(p)
        for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})


def verify_inventory(folder):
    for rel,expected in read(folder/'artifact_hashes.json').items():
        if sha(folder/rel)!=expected:
            raise ValueError('saved artifact changed: '+str(folder/rel))


def frozen():
    manifest = read(OUTPUT/'manifest.json')
    for rel, expected in manifest['source_sha256'].items():
        if sha(ROOT/rel) != expected:
            raise RuntimeError('frozen source changed: '+rel)
    if sha(OUTPUT/'sources.zip')!=manifest['source_zip_sha256']:
        raise RuntimeError('source archive changed')
    return manifest


def prepare():
    if OUTPUT.exists() or CONFIG.exists():
        raise RuntimeError('refuse to overwrite an existing experiment/config')
    scene = read(ROOT/'configs/virtual3d/v29_information_scene_20260917.json')
    verify_inventory(WITNESS.parent)
    old_manifest=read(WITNESS.parent/'manifest.json')
    # V29's own source hashes, not a new approval of changed old inputs.
    for rel,expected in old_manifest['source_sha256'].items():
        if sha(ROOT/rel)!=expected:
            raise ValueError('V29 frozen input changed: '+rel)
    witness = next(x for x in read(WITNESS)['complete_options'] if x['hypothesis']==0 and x['station']==0)
    a = witness['witness']['suffix_actions']
    b = [{'left':'right', 'right':'left', 'forward':'forward'}[x] for x in a]
    config = dict(version='v30-pixel-routes-r0', main_tasks_already_used=12, main_task_limit=36,
        prefix=scene['prefix_actions'], suffixes={'A':a, 'B':b}, budget=48,
        cases=[dict(index=i, hypothesis=h, arm=arm) for i,(h,arm) in enumerate(((0,'A'),(0,'B'),(1,'A'),(1,'B')))],
        task_scope='fixed GT-designed route mechanism; not autonomous G/S',
        scene_sha256=sha(ROOT/'configs/virtual3d/v29_information_scene_20260917.json'),
        witness_sha256=sha(WITNESS), protocol_sha256=sha(PROTOCOL))
    # No World/Sensor/mapper creation: validate all continuous swept paths.
    traces = []
    for case in config['cases']:
        _, boxes, _ = scene_boxes(case['hypothesis'])
        x=y=h=0
        trace = [[x,y,h]]
        for action in config['prefix']+config['suffixes'][case['arm']]:
            if action not in ('forward','left','right'):
                raise ValueError('unknown action in frozen witness')
            if action=='forward':
                dx,dy=scene['heading_vectors'][h]
                if not swept_clear(np.array([x,y])+SHIFT[:2], np.array([x+dx,y+dy])+SHIFT[:2], boxes):
                    raise ValueError('unsafe full swept path')
                x+=dx; y+=dy
            else:
                h=(h+(1 if action=='right' else -1))%4
            trace.append([x,y,h])
        if len(trace)!=49 or trace[-1]!=[0,0,0] or trace[18]!=[0,0,0]:
            raise ValueError('paid/return/prefix contract failed')
        traces.append(dict(case=case, poses=trace))
    # Audit artifacts must already be complete; no retuning if they fail.
    gates = [ROOT/'audit_results/v30_continuous_geometry_20260917/result.json',
             ROOT/'audit_results/v30_outline_applicability_20260917/result.json']
    for gate in gates:
        if not gate.is_file():
            raise ValueError('missing preregistered audit: '+str(gate))
        verify_inventory(gate.parent)
        for rel,expected in read(gate.parent/'manifest.json')['source_sha256'].items():
            if sha(ROOT/rel)!=expected:
                raise ValueError('audit source changed: '+rel)
    geometry, applicability = [read(p) for p in gates]
    if not (geometry['exact_area_and_closed_boundary_passed']
            and geometry['prefix_finite_ray_pairing_passed']
            and geometry['source_and_inputs_unchanged']
            and applicability['closed_self_and_missing_gates']
            and applicability['at_least_two_projections_evaluable']):
        raise ValueError('geometry or metric applicability prerequisite failed')
    OUTPUT.mkdir()
    write(CONFIG, config)
    files = closure([Path(__file__), ROOT/'nso/facility_outline_v30.py',
        ROOT/'tests/virtual3d/test_information_pixel_v30.py', PROTOCOL, CONFIG,
        ROOT/'configs/virtual3d/v29_information_scene_20260917.json', WITNESS,
        ROOT/'docs/research/V30_MEASUREMENT_PROTOCOL_REVIEW_20260917.md', *gates])
    archive_bytes=io.BytesIO()
    with zipfile.ZipFile(archive_bytes, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(ROOT)))
    payload=archive_bytes.getvalue();guard_write(OUTPUT/'sources.zip',len(payload))
    with (OUTPUT/'sources.zip').open('xb') as stream:stream.write(payload)
    write(OUTPUT/'manifest.json', dict(status='prepared', source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        source_zip_sha256=sha(OUTPUT/'sources.zip'), new_main_started=0, historical_main_used=12,
        maximum_main_used=16, replay_started=0, versions=dict(python=sys.version,numpy=np.__version__,open3d=o3d.__version__),
        cases=config['cases'], script_routes_not_ANS=True))
    write(OUTPUT/'static_routes.json', traces)
    print(json.dumps({'prepared':True,'source_count':len(files),'archive_bytes':(OUTPUT/'sources.zip').stat().st_size}),flush=True)


def save_arrays(path, **arrays):
    buffer=io.BytesIO()
    np.savez_compressed(buffer,**arrays)
    payload=buffer.getvalue();guard_write(path,len(payload))
    with path.open('xb') as stream:
        stream.write(payload)


def save_mesh(path, mesh):
    save_arrays(path,**{key:np.asarray(getattr(mesh,key)) for key in ('vertices','triangles','vertex_colors')})


def save_packet(path,packet):
    arrays={};metadata={}
    for field in fields(packet):
        value=getattr(packet,field.name)
        if field.name in ('frame','scan'):
            for inner in fields(value):
                arrays[field.name+'__'+inner.name]=np.asarray(getattr(value,inner.name))
        else:metadata[field.name]=json_value(value)
    arrays['metadata']=np.asarray(json.dumps(metadata,sort_keys=True))
    save_arrays(path,**arrays)


def verify_snapshot(folder,stage,snapshot):
    if digest(read(folder/f'{stage}_metadata.json'))!=snapshot['metadata_sha256']:
        raise AssertionError('saved snapshot metadata differs from fresh replay')
    for name,mesh in [('raw',snapshot['raw_mesh']),*[(f'slot{i}',m) for i,m in enumerate(snapshot['observed_meshes'])]]:
        with np.load(folder/f'{stage}_{name}.npz',allow_pickle=False) as saved:
            for key in ('vertices','triangles','vertex_colors'):
                if array_hash(saved[key])!=array_hash(np.asarray(getattr(mesh,key))):
                    raise AssertionError('saved prediction array differs from fresh replay')


def run_case(index, replay):
    from nso.facility_outline_v30 import FullInstanceOutlineEvaluatorV30
    manifest = frozen()
    cfg = read(CONFIG)
    case = cfg['cases'][index]
    folder = OUTPUT/f'case{index:02d}'
    if replay:
        verify_inventory(folder)
        expected = read(folder/'result.json')
        if any((folder/name).exists() for name in ('replay.json','replay_started.json','replay_failure.json')):
            raise ValueError('only one replay is declared')
        if read(folder/'started.json')['pid']==os.getpid():
            raise ValueError('replay must run in a fresh process')
        write(folder/'replay_started.json',dict(time=time.time(),pid=os.getpid()))
        manifest['replay_started']+=1
    else:
        if index!=len(list(OUTPUT.glob('case[0-9][0-9]'))):
            raise ValueError('fixed acquisition order or counted-case quota violated')
        folder.mkdir()
        manifest['new_main_started']+=1
    started=time.monotonic()
    progress=dict(worlds=0, sensor_packets=0, paid_actions=0, mapper_updates=0,
        mapper_mesh_extractions=0, measurement_snapshots=0, backend_observe_calls=0,
        evaluation_stages=0)
    progress_path=folder/('replay_progress.json' if replay else 'progress.json')
    trace=[]
    try:
        write(OUTPUT/'manifest.json', manifest)
        quota_guard(folder)
        if not replay:
            write(folder/'started.json',dict(pid=os.getpid(),time=time.time(),cumulative_main_used=12+manifest['new_main_started']))
            (folder/'packets').mkdir()
        write(progress_path, progress)
        def deadline(signum,frame):
            raise TimeoutError('declared 600 second whole-case deadline')
        signal.signal(signal.SIGALRM,deadline);signal.alarm(600)
        world=InformationPixelWorldV30(case['hypothesis'], f'v30-case{index:02d}')
        progress['worlds']+=1
        mapper=ObservedRuntimeMapperV10(world.shape, world.config, truncation_m=.12)
        prefix_measurement, final_measurement = FacilityMeasurementV26(), FacilityMeasurementV26()
        snapshots={}; beliefs={}
        actions=[None]+cfg['prefix']+cfg['suffixes'][case['arm']]
        for paid,action in enumerate(actions):
            if time.monotonic()-started>600:
                raise TimeoutError('declared case time cap')
            packet=world.packet() if action is None else world.step(action)
            progress['sensor_packets']+=1
            progress['paid_actions']+=int(action is not None)
            if replay:
                stored=load_packet(folder/'packets'/f'{paid:03d}.npz')
                if packet.sha256()!=stored.sha256():
                    raise AssertionError(f'fresh sensor bytes differ at action {paid}')
            else:
                save_packet(folder/'packets'/f'{paid:03d}.npz', packet)
            mapper.update(packet.frame, packet.scan)
            progress['mapper_updates']+=1
            receipt=final_measurement.observe(packet)
            progress['backend_observe_calls']+=2
            if paid<=18:
                prefix_measurement.observe(packet)
                progress['backend_observe_calls']+=2
            marker=np.any(packet.frame.color_rgb!=packet.frame.color_rgb[...,0,None],axis=-1)
            stripped=packet.frame.color_rgb.copy(); stripped[marker]=[153,153,153]
            trace.append(dict(paid=paid, action=action, position=packet.position, heading=packet.heading,
                packet_sha256=packet.sha256(), clean_depth_sha256=array_hash(world.last_clean_depth),
                nonsemantic=dict(depth=array_hash(packet.frame.depth_m),scan=array_hash(packet.scan.ranges_m),
                    rgb_without_class=array_hash(stripped),binary_marker=array_hash(marker),
                    pose=array_hash(packet.frame.world_from_camera)),
                marker_receipt=receipt, collision=packet.collision,
                mapper_footprint_conflict=mapper.current_footprint_conflict))
            if paid in (18,48):
                stage='prefix' if paid==18 else 'final'
                progress['mapper_mesh_extractions']+=1
                progress['measurement_snapshots']+=2
                write(progress_path,progress)
                print(f'case={index} replay={replay} stage={stage} snapshot',flush=True)
                snapshots[stage]=(prefix_measurement if paid==18 else final_measurement).snapshot(mapper)
                beliefs[stage]=mapper.belief.copy()
                if not replay:
                    save_mesh(folder/f'{stage}_raw.npz', snapshots[stage]['raw_mesh'])
                    for slot,mesh in enumerate(snapshots[stage]['observed_meshes']):
                        save_mesh(folder/f'{stage}_slot{slot}.npz',mesh)
                    write(folder/f'{stage}_metadata.json',snapshots[stage]['metadata'])
                else:
                    verify_snapshot(folder,stage,snapshots[stage])
                quota_guard(folder)
            if paid%10==0:
                write(progress_path,progress)
        # Reference and coverage truth are accessed only after final freeze.
        reference,reachable=world.evaluation_reference()
        assets=[dict(id=obj['id'],mesh=reference.instance_mesh(obj['id']),bounds=np.asarray(obj['evaluation_bounds']))
            for obj in reference.objects]
        evaluator=FullInstanceOutlineEvaluatorV30(assets)
        results={}
        for stage,paid in (('prefix',18),('final',48)):
            coverage=float(np.mean(beliefs[stage][reachable]!=-1))
            args=dict(coverage=coverage,returned=True,collisions=world.collisions,
                failed=False,paid_actions=paid,budget=48)
            snapshot=snapshots[stage]
            # The trace independently establishes exact return at both stages.
            if trace[paid]['position']!=world.start or trace[paid]['heading']!=0:
                raise AssertionError('stage did not return to original pose')
            window=FacilityMeasurementV26.evaluate(snapshot,reference,**args)
            companion=evaluator.evaluate(snapshot['observed_meshes'],
                [row['seed'] for row in snapshot['metadata']['instances']],
                len(snapshot['metadata']['observed_track_summary']),**args)
            progress['evaluation_stages']+=1
            results[stage]=dict(window=window,full_instance=companion,
                measurement_metadata_sha256=snapshot['metadata_sha256'],
                belief_sha256=array_hash(beliefs[stage]),safe_floor_cells=int(reachable.sum()))
        result=dict(case=case,steps=48,collisions=world.collisions,returned=world.position==world.start and world.heading==0,
            trajectory_sha256=digest([{k:row[k] for k in ('paid','action','position','heading')} for row in trace]),
            trace_sha256=digest(trace),stages=results,counts=progress,
            mapper_footprint_conflict_count=mapper.current_footprint_conflict_count,
            full_architecture_executed=False,measured_only=True)
        frozen()
        if replay:
            if digest(read(folder/'trace.json'))!=digest(trace):
                raise AssertionError('saved trace differs from fresh replay')
            with np.load(folder/'maps.npz',allow_pickle=False) as saved:
                for name,value in dict(**beliefs,reachable=reachable).items():
                    if array_hash(saved[name])!=array_hash(value):
                        raise AssertionError('saved map differs from fresh replay')
            if digest(result)!=digest(expected):
                write(folder/'replay_mismatch_result.json',result)
                raise AssertionError('fresh replay prediction or evaluation differs')
            write(folder/'replay.json',dict(passed=True,all49_packets_byte_equal=True,
                full_result_equal=True,result_sha256=sha(folder/'result.json'),counts=progress,
                main_pid=read(folder/'started.json')['pid'],replay_pid=os.getpid(),
                elapsed_seconds=time.monotonic()-started))
        else:
            write(folder/'trace.json',trace)
            write(folder/'result.json',result)
            save_arrays(folder/'maps.npz',**beliefs,reachable=reachable)
        write(progress_path,progress)
        quota_guard(folder)
        seal(folder)
        signal.alarm(0)
        print(json.dumps(dict(case=index,replay=replay,complete=True,seconds=time.monotonic()-started)),flush=True)
    except BaseException:
        signal.alarm(0)
        write(folder/('replay_failure.json' if replay else 'failure.json'),dict(error=traceback.format_exc(),counts=progress,
            pid=os.getpid(),counted_main_case_directories=len(list(OUTPUT.glob('case[0-9][0-9]')))))
        write(folder/('replay_partial_trace.json' if replay else 'partial_trace.json'),trace)
        write(progress_path,progress)
        seal(folder)
        raise


def analyze():
    frozen()
    for i in range(4):
        folder=OUTPUT/f'case{i:02d}'
        verify_inventory(folder)
        receipt=read(folder/'replay.json')
        if not receipt['passed'] or receipt['result_sha256']!=sha(folder/'result.json'):
            raise ValueError('invalid or failed independent replay')
    results=[read(OUTPUT/f'case{i:02d}'/'result.json') for i in range(4)]
    traces=[read(OUTPUT/f'case{i:02d}'/'trace.json') for i in range(4)]
    pairings=[]
    for left,right in ((0,2),(1,3)):
        first={key:None for key in traces[left][0]['nonsemantic']}
        clean_first=None
        for i,(a,b) in enumerate(zip(traces[left],traces[right])):
            for key in first:
                if first[key] is None and a['nonsemantic'][key]!=b['nonsemantic'][key]: first[key]=i
            if clean_first is None and a['clean_depth_sha256']!=b['clean_depth_sha256']: clean_first=i
        prefix_equal=all(v is None or v>18 for v in first.values())
        pairings.append(dict(cases=[left,right],first_difference_action=first,
            clean_depth_first_difference_action=clean_first,prefix_nonsemantic_equal=prefix_equal))
    table=[]
    for result in results:
        stages={}
        for name,stage in result['stages'].items():
            main=stage['window']['main_observed']
            stages[name]=dict(C=main['coverage_2d'],Q=main['05cm']['outline_macro_quality'],
                J=main['05cm']['joint_outline'],eligible=main['eligible'],
                Q_per_instance=[x['05cm']['outline_quality'] for x in main['instances']],
                instance_gate=stage['window']['evaluation_association_and_minimum_separation_passed'],
                marker_gate=stage['window']['observed_marker_support_gate_passed'],
                full_instance=stage['full_instance'])
        table.append(dict(**result['case'],**stages))
    observed_rules=[]
    for pair in ((0,1),(2,3)):
        rules=[]
        for index in pair:
            metadata=read(OUTPUT/f'case{index:02d}'/'prefix_metadata.json')
            cues={}
            for row in metadata['instances']:
                seed=row['seed']
                if seed is not None:
                    arm='A' if seed['observed_seed_xyz'][0]<SHIFT[0] else 'B'
                    cues.setdefault(seed['actual_marker_code'],[]).append(arm)
            valid=(metadata['observed_marker_support_gate_passed'] and set(cues)=={2,3}
                and all(len(v)==1 for v in cues.values()) and cues[2]!=cues[3])
            rules.append(dict(complex=cues[3][0],swapped=cues[2][0]) if valid else None)
        observed_rules.append(rules[0] if rules[0]==rules[1] else None)
    rules_available=all(rule is not None for rule in observed_rules)
    selected={name:[2*h+(observed_rules[h][name]=='B') for h in range(2)]
        for name in ('complex','swapped')} if rules_available else {}
    selection={}
    for metric in ('Q','J'):
        y=[r['final'][metric] for r in table]
        mean=lambda ids:float(np.mean([y[i] for i in ids]))
        fixedA,fixedB=mean([0,2]),mean([1,3])
        best=max(fixedA,fixedB)
        oracle=(max(y[0:2])+max(y[2:4]))/2
        selection[metric]=dict(always_A=fixedA,always_B=fixedB,best_fixed=max(fixedA,fixedB),
            uniform_random=float(np.mean(y)),complex_rule=mean(selected['complex']) if rules_available else None,
            swapped_rule=mean(selected['swapped']) if rules_available else None,
            two_option_oracle=oracle,oracle_relative_to_best_fixed=oracle/best-1 if best else None,
            complex_relative_to_best_fixed=mean(selected['complex'])/best-1 if best and rules_available else None,
            complex_minus_simple_per_assignment=[y[a]-y[b] for a,b in zip(selected['complex'],selected['swapped'])]
                if rules_available else None,
            difference_of_differences=(y[0]-y[1])-(y[2]-y[3]),
            scope='posthoc four-cell fixed-route selection table; not autonomous G/S')
    all_eligible=all(row['final']['eligible'] for row in table)
    replay=[read(OUTPUT/f'case{i:02d}'/'replay.json')['passed'] for i in range(4)]
    gate=bool(all_eligible and all(replay) and rules_available and all(x['prefix_nonsemantic_equal'] for x in pairings)
        and all(row['prefix']['marker_gate'] and row['final']['instance_gate'] for row in table)
        and selection['J']['oracle_relative_to_best_fixed'] is not None
        and selection['J']['oracle_relative_to_best_fixed']>.05
        and min(selection['J']['complex_minus_simple_per_assignment'])>0)
    analysis=dict(table=table,pairings=pairings,fixed_route_selection=selection,
        observed_class_rules=observed_rules,observed_rule_case_indices=selected,
        all_main_eligible=all_eligible,all_replays_passed=all(replay),progress_to_learning_gate_passed=gate,
        full_architecture_advantage_proven=False,semantic_runtime_executed=False,
        new_main_tasks=4,total_main_used=16,main_limit=36,
        counts_with_replays={k:sum(r['counts'][k] for r in results)*2 for k in results[0]['counts']})
    write(OUTPUT/'result.json',analysis)
    manifest=read(OUTPUT/'manifest.json');manifest['status']='complete'
    write(OUTPUT/'manifest.json',manifest)
    shared=sum(p.stat().st_size for p in OUTPUT.iterdir() if p.is_file())
    if shared>2*1024**2: raise RuntimeError('shared metadata cap exceeded')
    seal(OUTPUT)
    print(json.dumps(dict(all_eligible=all_eligible,learning_gate=gate,selection=selection,shared_bytes=shared)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','case','analyze'))
    parser.add_argument('--index',type=int,choices=range(4))
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        existed=OUTPUT.exists()
        try:
            prepare()
        except BaseException:
            if not existed and OUTPUT.exists():
                write(OUTPUT/'prepare_failure.json',dict(error=traceback.format_exc(),
                    new_main_started=0,worlds=0,owned_partial_prepare_preserved=True))
            raise
    elif args.command=='case':run_case(args.index,args.replay)
    else:analyze()
