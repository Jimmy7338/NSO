#!/usr/bin/env python3
"""Frozen two-shape paid observation bench, with independent saved-packet replay."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from time import perf_counter
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import open3d as o3d
import scipy
import shapely
from env.facility_shape_probe_v23 import ShapeProbeWorldV23, route_spec_v23
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, json_value
from nso.decision_replay_v13 import array_hash, load_packet, save_packet
from nso.mapping3d_v2 import QualityMapperV2
from utils.facility_metrics_v19 import FacilityEvaluatorV19
from utils.facility_outline_v23 import OutlineEvaluatorV23

CONFIG = ROOT / 'configs/virtual3d/facility_shape_v23_probe.json'
PROTOCOL = ROOT / 'docs/research/V23_OUTLINE_AND_SHAPE_PROBE_PROTOCOL_20260915.md'
DEFAULT_OUTPUT = ROOT / 'audit_results/facility_shape_v23_probe_20260915'
PROGRESS = {'last_attempted_action_id': None, 'last_verified_sensor_action_id': None}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    data = json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temp = path.with_name(path.name + '.tmp')
    with temp.open('x') as stream:
        stream.write(data)
    os.replace(temp, path)


def seal(path):
    write(path / 'artifact_hashes.json', {str(p.relative_to(path)): sha(p)
        for p in sorted(path.rglob('*')) if p.is_file() and p != path / 'artifact_hashes.json'})


def verify_inventory(path):
    for name, expected in read(path / 'artifact_hashes.json').items():
        if sha(path / name) != expected:
            raise ValueError('artifact changed: ' + str(path / name))


def versions():
    return dict(python=sys.version, numpy=np.__version__, open3d=o3d.__version__,
                scipy=scipy.__version__, shapely=shapely.__version__, geos=shapely.geos_version_string)


def check_frozen(root, manifest):
    for name, expected in manifest['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('frozen source changed: ' + name)
    if sha(root / 'sources.zip') != manifest['source_archive_sha256'] or versions() != manifest['versions']:
        raise ValueError('frozen sources archive or dependency versions changed')
    for case in manifest['cases']:
        if sha(root / case['surface_reference']) != case['surface_reference_sha256']:
            raise ValueError('reference cache changed')


def measured(world, frame=None, action=None, collision=False, done=False):
    frame = world.sense() if frame is None else frame
    packet = SensorPacket(scene_id='D23-SHAPE', episode_id=world.kind,
        frame_id=f'frame-{world.step_count:04d}', action_id=int(world.step_count),
        frame=frame, scan=world.scan(), position=tuple(map(int, world.position)),
        heading=int(world.heading), sensor_source='canonical_rgbd_v19_and_planar_scan',
        pose_source='exact_discrete_simulator_pose', action=action,
        collision=bool(collision), done=bool(done))
    packet.validate(GridTransform(world.shape, world.config.resolution_m), world.config)
    return packet


def nonsemantic_hashes(packet):
    result = {name: array_hash(np.asarray(getattr(packet.frame, name)))
              for name in ('depth_m', 'intrinsic', 'world_from_camera')}
    result.update({name: array_hash(np.asarray(getattr(packet.scan, name)))
        for name in ('ranges_m', 'world_from_laser', 'angle_min_rad', 'angle_increment_rad', 'range_max_m', 'timestamp_s')})
    return result


def prepare(root):
    config = read(CONFIG)
    if config['kinds'] != ['simple', 'complex'] or config['sensor_model'] != 'iid_025px' or config['noise_seed'] != 1901:
        raise ValueError('unexpected fixed two-shape protocol')
    if config['paid_actions_per_case'] != 96 or config['stages'] != route_spec_v23()['stage_indices']:
        raise ValueError('route and stage contract differ')
    if shutil.disk_usage(ROOT).free < 64 * 1024**2:
        raise OSError('prepare requires 64MiB available for this small bench')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'references').mkdir()
    paths = sorted({p for directory in ('env', 'nso', 'utils') for p in (ROOT / directory).rglob('*.py')}
        | {Path(__file__).resolve(), CONFIG, PROTOCOL, ROOT/'requirements-3d-v23.txt',
           ROOT/'tests/virtual3d/test_facility_outline_v23.py'})
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    with zipfile.ZipFile(root/'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(ROOT)))
    manifest = dict(status='preparing', config=config, route=route_spec_v23(), versions=versions(),
        source_sha256=hashes, source_archive_sha256=sha(root/'sources.zip'), cases=[],
        preparation_paid_actions=0, new_policy_training=False, full_architecture_efficacy_proven=False)
    write(root/'manifest.json', manifest)
    for index, kind in enumerate(config['kinds']):
        world = ShapeProbeWorldV23(kind, sensor_model=config['sensor_model'], noise_seed=config['noise_seed'])
        first = measured(world)
        reference = f'references/{kind}.npz'
        surface = FacilityEvaluatorV19(world, reference_cache=root/reference, **config['surface'])
        outline = OutlineEvaluatorV23.from_world(world, **config['outline'])
        manifest['cases'].append(dict(index=index, kind=kind, initial_packet_sha256=first.sha256(),
            nonsemantic_initial=nonsemantic_hashes(first), world_config=asdict(world.config),
            static_route_audit=world.static_route_audit, outline_reference_signature=outline.reference_signature,
            surface_reference=reference, surface_reference_sha256=sha(root/reference),
            surface_reference_signature=surface.reference_signature,
            world_geometry_sha256={key:array_hash(np.asarray(getattr(world.mesh,key))) for key in ('vertices','triangles')}))
        if world.step_count != 0:
            raise ValueError('preparation executed a paid action')
    if manifest['cases'][0]['nonsemantic_initial'] != manifest['cases'][1]['nonsemantic_initial']:
        raise ValueError('initial nonsemantic pair is not identical')
    manifest.update(status='prepared', initial_nonsemantic_pair_identical=True)
    write(root/'manifest.json', manifest)
    check_frozen(root, manifest)
    print('Prepared two shapes, zero paid actions; sources and references frozen.', flush=True)


def mesh_hashes(mesh):
    return {key: array_hash(np.asarray(getattr(mesh, key))) for key in ('vertices','triangles','vertex_colors')}


def run_case(root, index, replay):
    manifest = read(root/'manifest.json'); config = manifest['config']
    check_frozen(root, manifest)
    if index not in range(2):
        raise ValueError('case must be 0 or 1')
    if shutil.disk_usage(root).free < config['minimum_free_space_mib'] * 1024**2:
        raise OSError('free space below declared reserve')
    case = manifest['cases'][index]; folder = root/f'case_{index:02d}'
    expected = None
    if replay:
        verify_inventory(folder)
        expected = read(folder/'result.json')
        if (folder/'verification.json').exists() or read(folder/'timing.json')['process_id'] == os.getpid():
            raise ValueError('a fresh, not-yet-recorded replay process required')
    else:
        folder.mkdir(); (folder/'packets').mkdir(); (folder/'meshes').mkdir()
    world = ShapeProbeWorldV23(case['kind'], sensor_model=config['sensor_model'], noise_seed=config['noise_seed'])
    mapper = QualityMapperV2(world.shape, world.config, truncation_m=world.config.truncation_m)
    surface = FacilityEvaluatorV19(world, reference_cache=root/case['surface_reference'], **config['surface'])
    outline = OutlineEvaluatorV23.from_world(world, **config['outline'])
    if outline.reference_signature != case['outline_reference_signature']:
        raise ValueError('outline reference changed')
    checkpoints, trace = [], []
    stage_by_action = {value:key for key,value in config['stages'].items()}
    anchor = (*world.start,0)
    t0 = perf_counter()
    for action_id in range(97):
        PROGRESS['last_attempted_action_id'] = action_id
        action, collision, done = None, False, False
        if action_id:
            action = manifest['route']['actions'][action_id-1]
            frame, collision, done = world.step(action)
            actual = measured(world, frame, action, collision, done)
        else:
            actual = measured(world)
            if actual.sha256() != case['initial_packet_sha256']:
                raise ValueError('initial packet changed')
        path = folder/'packets'/f'{action_id:04d}.npz'
        if replay:
            packet = load_packet(path)
            packet.validate(GridTransform(world.shape, world.config.resolution_m),world.config)
            if actual.sha256() != packet.sha256():
                raise ValueError(f'fresh sensor replay differs at {action_id}')
        else:
            packet = actual; save_packet(path,packet)
        PROGRESS['last_verified_sensor_action_id'] = action_id
        # Replay fuses saved bytes after separately verifying fresh physical sensors.
        mapper.update(packet.frame,packet.scan)
        coverage = float(np.count_nonzero((mapper.belief != -1) & world.reachable)/world.reachable.sum())
        trace.append(dict(action_id=action_id,action=action,pose=[*packet.position,packet.heading],
            collision=packet.collision,world_done=packet.done,packet_sha256=packet.sha256(),
            nonsemantic=nonsemantic_hashes(packet),coverage_2d=coverage))
        if action_id in stage_by_action:
            stage=stage_by_action[action_id]; mesh=mapper.mesh()
            returned=(*packet.position,packet.heading)==anchor
            row=dict(stage=stage,action_id=action_id,mesh_sha256=mesh_hashes(mesh),
                outline=outline.evaluate(mesh,coverage,returned=returned,collisions=world.collisions,
                    failed=bool(collision),paid_actions=action_id,budget=96),
                surface=surface.evaluate(mesh,coverage,returned=returned,collisions=world.collisions,failed=bool(collision)))
            checkpoints.append(row)
            if not replay:
                with (folder/'meshes'/f'{stage}.npz').open('xb') as stream:
                    np.savez_compressed(stream,**{key:np.asarray(getattr(mesh,key)) for key in ('vertices','triangles','vertex_colors')})
            print(f"case={index} replay={replay} stage={stage} action={action_id} Q={row['outline']['05cm']['outline_macro_quality']:.6f}",flush=True)
        if collision or done:
            raise ValueError(f'collision or premature world termination at action {action_id}')
    result=dict(status='complete',index=index,kind=case['kind'],paid_actions=world.step_count,
        returned_to_anchor=(*world.position,world.heading)==anchor,collisions=world.collisions,
        termination='declared_paid_route_complete',world_done_at_endpoint=False,
        translation_actions=world.moves,path_distance_m=world.moves*.2,
        checkpoints=checkpoints,trace=trace,raw_frames=mapper.frames,
        shared_scripted_route=True,semantic_policy=False,autonomous_planner=False,
        four_module_closed_loop=False,full_architecture_efficacy_proven=False)
    if replay:
        if result != expected:
            raise ValueError('saved-packet fusion, stages, or actual route replay differs')
        write(folder/'verification.json',dict(status='passed',independent_process=True,
            physical_process_id=read(folder/'timing.json')['process_id'],replay_process_id=os.getpid(),
            raw_packets_verified=97,paid_actions=96,stage_mesh_and_metric_checks=6,
            fresh_world_sensor_replay=True,saved_packet_tsdf_replay=True))
    else:
        write(folder/'result.json',result)
        write(folder/'timing.json',dict(process_id=os.getpid(),elapsed_s=perf_counter()-t0,
            includes_sensor_mapper_stage_metrics_and_packet_saving=True,reference_initialization_excluded=True))
    seal(folder); check_frozen(root,manifest)


def aggregate(root):
    manifest=read(root/'manifest.json'); check_frozen(root,manifest)
    cases=[]
    for index in range(2):
        folder=root/f'case_{index:02d}'; verify_inventory(folder)
        verification=read(folder/'verification.json')
        if verification['status']!='passed' or verification['physical_process_id']==verification['replay_process_id']:
            raise ValueError('two independent replay checks required')
        cases.append(read(folder/'result.json'))
    stages=[{x['stage']:x for x in case['checkpoints']} for case in cases]
    deltas=[s['extra']['outline']['05cm']['outline_macro_quality']-s['coarse']['outline']['05cm']['outline_macro_quality'] for s in stages]
    differences=[dict(action_id=a['action_id'],fields=[key for key in a['nonsemantic'] if a['nonsemantic'][key]!=b['nonsemantic'][key]])
        for a,b in zip(cases[0]['trace'],cases[1]['trace']) if a['nonsemantic']!=b['nonsemantic']]
    pairing=dict(initial_identical=not differences or differences[0]['action_id']>0,
        first_difference_action= differences[0]['action_id'] if differences else None,
        first_difference_fields= differences[0]['fields'] if differences else [],
        coarse_prefix_identical=not differences or differences[0]['action_id']>23,
        differences=differences,
        same_paid_actions_and_poses=all((a['action'],a['pose'])==(b['action'],b['pose']) for a,b in zip(cases[0]['trace'],cases[1]['trace'])))
    gates=dict(simple_coarse_completed=stages[0]['coarse']['outline']['instances'][0]['completed'],
        complex_positive_and_larger_increment=deltas[1]>0 and deltas[1]>deltas[0],
        all_returned_without_collision=all(c['returned_to_anchor'] and c['collisions']==0 for c in cases),
        semantic_information_or_efficacy_proven=False)
    result=dict(status='complete',physical_trajectories=2,independent_process_replays=2,
        physical_paid_actions=192,replay_paid_actions=192,paired_history=pairing,
        coarse_to_extra_quality_delta=dict(zip(('simple','complex'),deltas)),gates=gates,
        observation_demand_gate_passed=gates['simple_coarse_completed'] and gates['complex_positive_and_larger_increment'],
        cases=[dict(kind=c['kind'],checkpoints=c['checkpoints'],path_distance_m=c['path_distance_m'],
                    paid_actions=c['paid_actions'],returned=c['returned_to_anchor']) for c in cases],
        full_architecture_efficacy_proven=False,semantic_policy=False,training=False,
        independent_parent_layouts=1,noise_seeds=1,
        interpretation='Paid scripted observation response only. Initial pairing does not imply paired geometry after coarse survey. Negative gates prohibit automatic six-asset matrix or training expansion.')
    write(root/'result.json',result)
    manifest['status']='complete'; write(root/'manifest.json',manifest); seal(root)
    print(json.dumps(dict(status='complete',gates=gates,quality_deltas=result['coarse_to_extra_quality_delta'],pairing={k:v for k,v in pairing.items() if k!='differences'}),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--case',type=int)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    if args.prepare or args.case is not None:
        try:
            if args.prepare:
                prepare(root)
            else:
                run_case(root,args.case,args.replay)
        except Exception as error:
            if root.is_dir():
                tag = 'preparation' if args.prepare else f'case_{args.case:02d}_' + ('replay' if args.replay else 'physical')
                receipt = root / f'failure_{tag}.json'
                if not receipt.exists():
                    write(receipt, dict(status='failed', phase=tag, process_id=os.getpid(),
                        error_type=type(error).__name__, error=repr(error),
                        traceback=traceback.format_exc(), progress=PROGRESS))
                if args.prepare and (root/'manifest.json').exists():
                    m=read(root/'manifest.json');m.update(status='failed',error=repr(error))
                    write(root/'manifest.json',m)
            raise
        return
    if not args.all:
        parser.error('choose --prepare, --all, or --case')
    manifest=read(root/'manifest.json')
    if manifest['status']!='prepared':
        raise ValueError('only a prepared, unstarted batch may be launched')
    manifest['status']='running';write(root/'manifest.json',manifest)
    try:
        for index in range(2):
            for replay in (False,True):
                command=[sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(root),'--case',str(index)]
                if replay:command.append('--replay')
                subprocess.run(command,check=True,cwd=ROOT,env=os.environ.copy())
        aggregate(root)
    except Exception as error:
        manifest=read(root/'manifest.json');manifest.update(status='failed',error=str(error))
        write(root/'manifest.json',manifest)
        raise


if __name__=='__main__':
    main()
