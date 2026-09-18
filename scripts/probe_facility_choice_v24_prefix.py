#!/usr/bin/env python3
"""Four fixed paid V24 prefixes plus different-process exact packet replays."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from dataclasses import asdict, fields
import io
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
from scipy.ndimage import label
from env.facility_choice_v24_1 import (FacilityChoiceWorldV24_1 as FacilityChoiceWorld,
    PARENTS_V24, ASSIGNMENTS_V24, VERSION_V24_1 as WORLD_VERSION, prefix_spec_v24)
from env.virtual3d_inspection_v4 import MARKER_COLORS
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, digest, json_value
from nso.decision_replay_v13 import array_hash, load_packet
from nso.mapping3d_v2 import QualityMapperV2
# Explicitly reuse frozen pure IO/hash helpers, not the V23 runner/evaluators.
from scripts.probe_facility_shape_v23 import (sha, read, write,
    verify_inventory, versions, nonsemantic_hashes, mesh_hashes)

CONFIG = ROOT/'configs/virtual3d/facility_choice_v24_prefix.json'
PROTOCOL = ROOT/'docs/research/V24_PAID_COMMON_PREFIX_PROTOCOL_20260915.md'
DEFAULT_OUTPUT = ROOT/'audit_results/facility_choice_v24_paid_prefix_20260915'
PROGRESS = {'last_attempted_action_id': None, 'last_saved_or_verified_action_id': None}


def measured(world, frame=None, action=None, collision=False, done=False):
    frame = world.sense() if frame is None else frame
    packet = SensorPacket(scene_id=world.parent, episode_id=world.assignment,
        frame_id=f'frame-{world.step_count:04d}', action_id=int(world.step_count),
        frame=frame, scan=world.scan(), position=tuple(map(int,world.position)),
        heading=int(world.heading), sensor_source='canonical_rgbd_v19_and_planar_scan',
        pose_source='exact_discrete_simulator_pose', action=action,
        collision=bool(collision), done=bool(done))
    return packet.validate(GridTransform(world.shape,world.config.resolution_m),world.config)


def marker_mask(frame):
    return np.logical_or.reduce([np.all(frame.color_rgb == color,axis=2)
                                  for color in MARKER_COLORS.values()])


def nonsemantic(packet):
    values = nonsemantic_hashes(packet)
    mask = marker_mask(packet.frame)
    neutral = packet.frame.color_rgb.copy(); neutral[mask] = 153
    values.update(binary_rgb_marker=array_hash(mask),
                  rgb_without_marker_class=array_hash(neutral),
                  camera_timestamp=array_hash(np.asarray(packet.frame.timestamp_s)))
    return values


class MarkerTracks:
    """Observed binary-marker centroids only; no truth identity or class input."""
    def __init__(self, config):
        self.config = config
        self.tracks = []

    def consume(self, packet):
        frame = packet.frame
        mask = marker_mask(frame) & (frame.depth_m > .15) & np.isfinite(frame.depth_m)
        groups, count = label(mask,np.ones((3,3),bool))
        rows = []
        for number in range(1,count+1):
            rr,cc = np.nonzero(groups == number)
            if len(rr) < self.config['marker_minimum_valid_pixels_per_component']:
                continue
            depth = frame.depth_m[rr,cc]
            xyz = np.column_stack(((cc-frame.intrinsic[0,2])*depth/frame.intrinsic[0,0],
                (rr-frame.intrinsic[1,2])*depth/frame.intrinsic[1,1],depth))
            points = xyz@frame.world_from_camera[:3,:3].T+frame.world_from_camera[:3,3]
            centre = np.median(points,axis=0)
            distances = [np.linalg.norm(centre-t['centre']) for t in self.tracks]
            if distances and min(distances) <= self.config['marker_track_association_distance_m']:
                index = int(np.argmin(distances)); track = self.tracks[index]
            else:
                index = len(self.tracks)
                track = dict(centre=centre.copy(),frames=0,total_valid_pixels=0,max_valid_pixels=0,
                    first_action=packet.action_id,last_action=packet.action_id,poses=set())
                self.tracks.append(track)
            track['frames'] += 1
            track['centre'] += (centre-track['centre'])/track['frames']
            track['total_valid_pixels'] += len(rr)
            track['max_valid_pixels'] = max(track['max_valid_pixels'],len(rr))
            track['last_action'] = packet.action_id
            if packet.action_id > 0:
                track['poses'].add((*packet.position,packet.heading))
            rows.append(dict(observed_track_index=index,valid_rgb_marker_depth_pixels=len(rr),
                observed_world_centroid=centre.tolist()))
        return rows

    def summary(self):
        rows = []
        for index,track in enumerate(self.tracks):
            rows.append(dict(observed_track_index=index,centre=track['centre'].tolist(),
                frames=track['frames'],total_valid_pixels=track['total_valid_pixels'],
                max_valid_pixels=track['max_valid_pixels'],first_action=track['first_action'],
                last_action=track['last_action'],distinct_paid_poses=len(track['poses']),
                paid_poses=[list(x) for x in sorted(track['poses'])]))
        return sorted(rows,key=lambda x:x['centre'][0])


def geometry_evidence_sha(mapper):
    surface = [(list(key),value[0].tolist(),int(value[1]))
               for key,value in sorted(mapper.surface.items())]
    quality = [(list(key),{k:v for k,v in value.items() if k != 'label'})
               for key,value in sorted(mapper.quality.items())]
    return dict(surface_without_class=digest(surface),quality_without_class=digest(quality))


def check_frozen(root, manifest):
    for name,expected in manifest['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('frozen source changed: '+name)
    if sha(root/'sources.zip') != manifest['source_archive_sha256'] or versions() != manifest['versions']:
        raise ValueError('source archive or dependency version changed')


def raw_bytes(root):
    return sum(p.stat().st_size for p in root.glob('case_*/packets/*.npz'))


def capacity(root, config):
    if shutil.disk_usage(root).free < config['minimum_free_space_mib']*1024**2:
        raise OSError('32 MiB preserved-free-space boundary reached')
    if raw_bytes(root) > config['raw_limit_mib']*1024**2:
        raise OSError('50 MiB raw packet boundary exceeded')


def reserve_for_write(path, size, config):
    allocated = ((size+4095)//4096)*4096+65536
    if shutil.disk_usage(path.parent).free-allocated < config['minimum_free_space_mib']*1024**2:
        raise OSError('write would cross preserved-free-space boundary')


def protected_json(path, value, config):
    payload = json.dumps(json_value(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    reserve_for_write(path,len(payload.encode()),config)
    write(path,value)


def protected_seal(path, config):
    protected_json(path/'artifact_hashes.json',{str(p.relative_to(path)):sha(p)
        for p in sorted(path.rglob('*')) if p.is_file() and p != path/'artifact_hashes.json'},config)


def compressed_arrays(arrays):
    stream = io.BytesIO(); np.savez_compressed(stream,**arrays)
    return stream.getvalue()


def save_paid_packet(path, packet, root, config):
    # Same NPZ schema as decision_replay_v13; compress before disk-cap checks.
    arrays = {}; metadata = {}
    for field in fields(packet):
        value = getattr(packet,field.name)
        if field.name in ('frame','scan'):
            for inner in fields(value):
                arrays[field.name+'__'+inner.name] = np.asarray(getattr(value,inner.name))
        else:
            metadata[field.name] = json_value(value)
    arrays['metadata'] = np.asarray(json.dumps(metadata,sort_keys=True))
    payload = compressed_arrays(arrays)
    if raw_bytes(root)+len(payload) > config['raw_limit_mib']*1024**2:
        raise OSError('compressed packet would cross 50 MiB raw cap')
    reserve_for_write(path,len(payload),config)
    with path.open('xb') as stream:
        stream.write(payload)


def prepare(root):
    config = read(CONFIG)
    if config['parents'] != list(PARENTS_V24) or config['assignments'] != list(ASSIGNMENTS_V24):
        raise ValueError('undeclared parent/assignment matrix')
    if config['world_version'] != WORLD_VERSION or sha(ROOT/config['world_source']) != config['world_sha256']:
        raise ValueError('fixed V24.1 world changed')
    if config['sensor_model'] != 'iid_025px' or config['noise_seed'] != 1901:
        raise ValueError('fixed sensor configuration changed')
    if shutil.disk_usage(ROOT).free < config['prepare_minimum_free_space_mib']*1024**2:
        raise OSError('insufficient disk for prefix and preserved reserve')
    root.mkdir(parents=True,exist_ok=False)
    paths = sorted({p for name in ('env','nso','utils') for p in (ROOT/name).rglob('*.py')}
        | {Path(__file__).resolve(),CONFIG,PROTOCOL,
           ROOT/'scripts/probe_facility_shape_v23.py', ROOT/'scripts/audit_facility_choice_v24.py',
           ROOT/'scripts/audit_facility_choice_v24_1.py',
           ROOT/'configs/virtual3d/facility_choice_v24_static.json', ROOT/'requirements-3d-v23.txt'})
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    with zipfile.ZipFile(root/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path,str(path.relative_to(ROOT)))
    manifest = dict(status='preparing',config=config,versions=versions(),source_sha256=hashes,
        source_archive_sha256=sha(root/'sources.zip'),cases=[],preparation_paid_actions=0,
        no_reference_or_shape_quality_created=True,semantic_policy=False)
    write(root/'manifest.json',manifest)
    for parent in PARENTS_V24:
        route = prefix_spec_v24(parent)
        if route['paid_actions'] != config['paid_actions_by_parent'][parent]:
            raise ValueError('declared fixed prefix length changed')
        for assignment in ASSIGNMENTS_V24:
            world = FacilityChoiceWorld(parent,assignment,config['sensor_model'],config['noise_seed'])
            packet = measured(world)
            manifest['cases'].append(dict(index=len(manifest['cases']),parent=parent,assignment=assignment,
                initial_packet_sha256=packet.sha256(),initial_nonsemantic=nonsemantic(packet),
                world_config=asdict(world.config),shape=list(world.shape),
                decision_anchor=[*packet.position,packet.heading],route=route,
                world_geometry_sha256={k:array_hash(np.asarray(getattr(world.mesh,k))) for k in ('vertices','triangles')}))
            assert world.step_count == 0
    if sum(c['route']['paid_actions'] for c in manifest['cases']) != config['expected_physical_actions']:
        raise ValueError('total action contract changed')
    for offset in (0,2):
        if manifest['cases'][offset]['initial_nonsemantic'] != manifest['cases'][offset+1]['initial_nonsemantic']:
            raise ValueError('zero-action nonsemantic mismatch')
    manifest['status'] = 'prepared'; write(root/'manifest.json',manifest)
    check_frozen(root,manifest)
    print(f"Prepared four prefixes; {len(hashes)} frozen sources; zero paid actions",flush=True)


def run_case(root, index, replay):
    manifest = read(root/'manifest.json'); config = manifest['config']; check_frozen(root,manifest)
    if index not in range(4):
        raise ValueError('case must be 0..3')
    capacity(root,config)
    case = manifest['cases'][index]; folder = root/f'case_{index:02d}'
    if replay:
        verify_inventory(folder)
        expected = read(folder/'result.json')
        if (folder/'verification.json').exists() or read(folder/'timing.json')['process_id'] == os.getpid():
            raise ValueError('new independent replay PID and empty verification required')
    else:
        folder.mkdir(); (folder/'packets').mkdir()
    world = FacilityChoiceWorld(case['parent'],case['assignment'],config['sensor_model'],config['noise_seed'])
    if asdict(world.config) != case['world_config']:
        raise ValueError('world public config changed')
    mapper = QualityMapperV2(world.shape,world.config,truncation_m=config['tsdf_truncation_m'])
    tracks = MarkerTracks(config); trace = []; t0 = perf_counter()
    route = case['route']; anchor = tuple(route['decision_anchor'])
    for action_id in range(route['paid_actions']+1):
        PROGRESS['last_attempted_action_id'] = action_id
        capacity(root,config)
        action = None; collision = done = False
        if action_id:
            action = route['actions'][action_id-1]
            frame,collision,done = world.step(action)
            actual = measured(world,frame,action,collision,done)
        else:
            actual = measured(world)
            if actual.sha256() != case['initial_packet_sha256']:
                raise ValueError('initial sensor packet changed')
        path = folder/'packets'/f'{action_id:04d}.npz'
        if replay:
            packet = load_packet(path)
            packet.validate(GridTransform(world.shape,world.config.resolution_m),world.config)
            if actual.sha256() != packet.sha256():
                raise ValueError(f'fresh sensor packet mismatch at {action_id}')
        else:
            packet = actual; save_paid_packet(path,packet,root,config)
        PROGRESS['last_saved_or_verified_action_id'] = action_id
        if [*packet.position,packet.heading] != route['states'][action_id]:
            raise ValueError(f'paid pose differs from declared route at {action_id}')
        mapper.update(packet.frame,packet.scan)
        observed = tracks.consume(packet)
        known = int(np.count_nonzero((mapper.belief != -1)&world.reachable))
        count = int(world.reachable.sum())
        trace.append(dict(action_id=action_id,action=action,pose=[*packet.position,packet.heading],
            collision=packet.collision,world_done=packet.done,packet_sha256=packet.sha256(),
            nonsemantic=nonsemantic(packet),coverage_2d=known/count,known_reachable_cells=known,
            reachable_cells=count,belief_sha256=array_hash(mapper.belief),visible_sha256=array_hash(mapper.visible),
            marker_components=observed))
        capacity(root,config)
        if collision or done:
            raise ValueError(f'collision or premature termination at action {action_id}')
        if action_id % 50 == 0 or action_id == route['paid_actions']:
            print(f"case={index} replay={replay} step={action_id}/{route['paid_actions']} C={known/count:.4f} marker_tracks={len(tracks.tracks)}",flush=True)
    mesh = mapper.mesh(); marker_tracks = tracks.summary()
    support = len(marker_tracks) == config['expected_observed_marker_tracks'] and all(
        row['distinct_paid_poses'] >= config['minimum_supported_distinct_poses_per_track'] for row in marker_tracks)
    result = dict(status='complete',index=index,parent=case['parent'],assignment=case['assignment'],
        paid_actions=world.step_count,raw_frames=mapper.frames,returned_to_anchor=(*world.position,world.heading)==anchor,
        collisions=world.collisions,translation_actions=world.moves,path_distance_m=world.moves*.2,
        termination='declared_common_prefix_complete',world_done_at_endpoint=False,
        final_coverage_2d=trace[-1]['coverage_2d'],coverage_at_least_80=trace[-1]['coverage_2d']>=config['report_coverage_threshold'],
        marker_tracks=marker_tracks,two_observed_marker_tracks_supported=support,
        final_mesh_sha256=mesh_hashes(mesh),final_geometry_evidence_sha256=geometry_evidence_sha(mapper),
        trace=trace,trajectory_sha256=digest([(x['action_id'],x['action'],x['pose']) for x in trace]),
        shared_scripted_prefix=True,autonomous_planner=False,semantic_policy=False,
        shape_quality_evaluated=False,full_architecture_efficacy_proven=False)
    if replay:
        with np.load(folder/'final_mesh.npz',allow_pickle=False) as saved_mesh:
            saved_hashes = {key:array_hash(saved_mesh[key]) for key in ('vertices','triangles','vertex_colors')}
        if saved_hashes != expected['final_mesh_sha256']:
            raise ValueError('saved NPZ mesh arrays differ from declared hashes')
        if result != expected:
            raise ValueError('saved-packet mapper/geometry/trace/marker replay differs')
        protected_json(folder/'verification.json',dict(status='passed',independent_process=True,
            physical_process_id=read(folder/'timing.json')['process_id'],replay_process_id=os.getpid(),
            raw_packets_verified=len(trace),paid_actions=world.step_count,fresh_sensor_replay=True,
            saved_packet_mapper_replay=True,all_three_mesh_arrays_equal=True,
            saved_mesh_npz_arrays_verified=True),config)
    else:
        payload = compressed_arrays({key:np.asarray(getattr(mesh,key)) for key in ('vertices','triangles','vertex_colors')})
        reserve_for_write(folder/'final_mesh.npz',len(payload),config)
        with (folder/'final_mesh.npz').open('xb') as stream:
            stream.write(payload)
        protected_json(folder/'result.json',result,config)
        protected_json(folder/'timing.json',dict(process_id=os.getpid(),elapsed_s=perf_counter()-t0),config)
    protected_seal(folder,config); check_frozen(root,manifest); capacity(root,config)


def aggregate(root):
    manifest = read(root/'manifest.json'); check_frozen(root,manifest); cases = []
    for index in range(4):
        folder = root/f'case_{index:02d}'; verify_inventory(folder)
        verify = read(folder/'verification.json')
        if verify['status'] != 'passed' or verify['physical_process_id'] == verify['replay_process_id']:
            raise ValueError('four different-process replay verifications required')
        cases.append(read(folder/'result.json'))
    parents = []
    for offset in (0,2):
        a,b = cases[offset:offset+2]; differences = []
        if len(a['trace']) != len(b['trace']):
            raise ValueError('paired trace lengths differ')
        for x,y in zip(a['trace'],b['trace']):
            fields = [name for name in ('action_id','action','pose','nonsemantic','belief_sha256','visible_sha256','marker_components') if x[name] != y[name]]
            if fields:
                differences.append(dict(action_id=x['action_id'],fields=fields,
                    nonsemantic_fields=[k for k in x['nonsemantic'] if x['nonsemantic'][k]!=y['nonsemantic'][k]]))
        mesh_equal = all(a['final_mesh_sha256'][key] == b['final_mesh_sha256'][key] for key in ('vertices','triangles'))
        geom_equal = a['final_geometry_evidence_sha256'] == b['final_geometry_evidence_sha256']
        parents.append(dict(parent=a['parent'],full_nonsemantic_input_and_belief_pair_identical=not differences,
            first_difference_action=differences[0]['action_id'] if differences else None,differences=differences,
            final_tsdf_vertices_triangles_identical=mesh_equal,final_class_stripped_mapper_geometry_identical=geom_equal,
            full_pair_gate_passed=not differences and mesh_equal and geom_equal,
            raw_color_may_differ_only_at_class_markers=True,paid_actions_per_case=a['paid_actions']))
    safe = all(c['returned_to_anchor'] and c['collisions']==0 and
        c['paid_actions']==manifest['config']['paid_actions_by_parent'][c['parent']] for c in cases)
    supported = all(c['two_observed_marker_tracks_supported'] for c in cases)
    result = dict(status='complete',physical_trajectories=4,independent_process_replays=4,
        physical_paid_actions=sum(c['paid_actions'] for c in cases),replay_paid_actions=sum(c['paid_actions'] for c in cases),
        saved_raw_packets=sum(c['raw_frames'] for c in cases),raw_packet_bytes=raw_bytes(root),parents=parents,
        all_returned_without_collision=safe,all_two_observed_marker_tracks_supported=supported,
        all_prefix_coverage_at_least_80=all(c['coverage_at_least_80'] for c in cases),
        prefix_sensor_feasibility_passed=safe and supported and all(p['full_pair_gate_passed'] for p in parents),
        cases=[{key:c[key] for key in ('index','parent','assignment','paid_actions','returned_to_anchor',
            'collisions','final_coverage_2d','coverage_at_least_80','two_observed_marker_tracks_supported',
            'marker_tracks','final_mesh_sha256','final_geometry_evidence_sha256','trajectory_sha256')} for c in cases],
        semantic_or_shape_efficacy_proven=False,autonomous_policy=False,new_reference_created=False,
        interpretation='Actual scripted common-prefix sensor/mapper feasibility only; no A/B policy quality or completed task claim.')
    if result['saved_raw_packets'] != manifest['config']['expected_raw_packets']:
        raise ValueError('unexpected final packet count')
    capacity(root,manifest['config']); protected_json(root/'result.json',result,manifest['config'])
    manifest['status'] = 'complete'
    protected_json(root/'manifest.json',manifest,manifest['config'])
    protected_seal(root,manifest['config']); capacity(root,manifest['config'])
    print({k:result[k] for k in ('status','physical_paid_actions','saved_raw_packets','prefix_sensor_feasibility_passed','all_prefix_coverage_at_least_80')},flush=True)


def failure(root, phase, error):
    if root.is_dir():
        receipt = root/f'failure_{phase}.json'
        if not receipt.exists():
            write(receipt,dict(status='failed',phase=phase,process_id=os.getpid(),error_type=type(error).__name__,
                error=repr(error),traceback=traceback.format_exc(),progress=PROGRESS))
        if (root/'manifest.json').exists():
            manifest = read(root/'manifest.json'); manifest.update(status='failed',error=repr(error))
            write(root/'manifest.json',manifest)


def validate_entry(root, args):
    """Reject stale/repeated entry before obtaining any writable experiment."""
    if args.prepare:
        if root.exists():
            raise ValueError('prepare requires a new directory; existing evidence untouched')
        return
    manifest = read(root/'manifest.json')
    expected = 'prepared' if args.all else 'running'
    if manifest['status'] != expected:
        raise ValueError(f'entry requires {expected}; existing evidence untouched')
    check_frozen(root,manifest)
    if args.all:
        if any(root.glob('case_*')) or any(root.glob('failure_*.json')):
            raise ValueError('batch requires unstarted prepared evidence')
    else:
        if args.case not in range(4):
            raise ValueError('case must be 0..3')
        folder = root/f'case_{args.case:02d}'
        if args.replay:
            if not (folder/'result.json').is_file() or not (folder/'timing.json').is_file() or (folder/'verification.json').exists():
                raise ValueError('replay requires completed unverified case')
            verify_inventory(folder)
        elif folder.exists():
            raise ValueError('physical case already exists; evidence untouched')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true'); modes.add_argument('--all',action='store_true')
    modes.add_argument('--case',type=int)
    parser.add_argument('--replay',action='store_true')
    args = parser.parse_args(); root = args.output.resolve()
    if args.replay and args.case is None:
        parser.error('--replay requires --case')
    validate_entry(root,args)
    phase = 'preparation' if args.prepare else 'batch' if args.all else f'case_{args.case:02d}_'+('replay' if args.replay else 'physical')
    try:
        if args.prepare:
            prepare(root)
        elif args.case is not None:
            run_case(root,args.case,args.replay)
        else:
            manifest = read(root/'manifest.json')
            if manifest['status'] != 'prepared':
                raise ValueError('only a prepared, unstarted batch may run')
            manifest['status'] = 'running'; write(root/'manifest.json',manifest)
            for index in range(4):
                for replay in (False,True):
                    command = [sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(root),'--case',str(index)]
                    if replay:
                        command.append('--replay')
                    subprocess.run(command,check=True,cwd=ROOT,env=os.environ.copy())
            aggregate(root)
    except Exception as error:
        failure(root,phase,error)
        raise


if __name__ == '__main__':
    main()
