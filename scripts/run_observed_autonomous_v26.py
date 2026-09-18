#!/usr/bin/env python3
"""Frozen four-case autonomous V26 acquisition and independent policy replay.

The driver owns simulator truth; runtime receives only a public sensor whitelist
and actual packets. Evaluation starts after the last autonomous decision and
the measured meshes have been frozen. No service catalogue is imported/read.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from dataclasses import asdict
import fcntl
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
from types import SimpleNamespace
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from env.facility_choice_v25_r1 import FacilityChoiceWorldV25, VERSION_V25
from nso.cpu_sensor_contract_v10 import digest, json_value
from nso.decision_replay_v13 import load_packet, array_hash
from nso.observed_runtime_v26 import ObservedANSRuntimeV26
from nso.facility_measurement_v26 import FacilityMeasurementV26
from scripts.probe_facility_choice_v24_prefix import measured, nonsemantic, compressed_arrays
from scripts.probe_facility_choice_v25r1 import packet_payload
from scripts.probe_facility_shape_v23 import versions

DEFAULT = ROOT/'audit_results/observed_autonomous_v26_20260916'
INTERFACE = ROOT/'audit_results/observed_decisions_v26_20260916'
OLD = ROOT/'audit_results/facility_choice_v25r1_paid_p00_20260915'
PROTOCOL = ROOT/'docs/research/V26_AUTONOMOUS_COMPARISON_PROTOCOL_20260916.md'
ANALYSIS = ROOT/'docs/research/V26_AUTONOMOUS_ANALYSIS_PROTOCOL_20260916.md'
PUBLIC_FIELDS = ('resolution_m', 'robot_radius_m', 'camera_height_m', 'width_px', 'height_px',
    'fov_deg', 'max_depth_m', 'depth_sigma_m', 'dropout', 'action_duration_s', 'voxel_m',
    'laser_height_m', 'laser_rays', 'width_m', 'height_m', 'truncation_m', 'pose_noise_m',
    'stereo_model', 'stereo_reference_fx_px', 'stereo_baseline_m')
CONFIG = dict(version='observed-autonomous-v26-1', parent='D25-P00', budget=400,
    sensor_model='iid_025px', noise_seed=1901, replan_interval=5, candidate_limit=12,
    quota_start=4, quota_limit=36, scheduled_tasks=4,
    task_cap_bytes=40*1024**2, shared_cap_bytes=2*1024**2,
    free_reserve_bytes=64*1024**2, receipt_reserve_bytes=65536, packet_cap_bytes=65536,
    shape_representation='unchanged V24 measured-only; enclosure disabled',
    independent_replay='fresh process/world/runtime computes its own actions then verifies saved packets and decisions')
PROGRESS = dict(case=None, replay=False, phase='unstarted', last_attempted_action=None,
                last_saved_packet=None, owned=False, prepare_owned=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def public_config(config):
    """Only public calibration/noise/grid constants cross the runtime boundary."""
    return SimpleNamespace(**{name: getattr(config, name) for name in PUBLIC_FIELDS})


def stable(value):
    """Exclude only wall-clock profiling, retaining all semantic/geometry data."""
    value = json_value(value)
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k != 'planning_seconds'}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def bytes_used(folder):
    paths = folder.rglob('*') if folder.name.startswith('case_') else folder.iterdir()
    return sum(p.stat().st_size for p in paths if p.is_file())


def reserve(folder, amount=0, receipt=False):
    cap = CONFIG['task_cap_bytes'] if folder.name.startswith('case_') else CONFIG['shared_cap_bytes']
    padding = 0 if receipt else CONFIG['receipt_reserve_bytes']
    if bytes_used(folder) + amount > cap - padding:
        raise OSError('frozen task/shared output cap would be exceeded')
    allocation = ((amount + 4095)//4096)*4096 + 4096 + padding
    if shutil.disk_usage(folder).free - allocation < CONFIG['free_reserve_bytes']:
        raise OSError('64 MiB real free-space reserve would be crossed')


def save_bytes(folder, path, payload, receipt=False):
    reserve(folder, len(payload), receipt)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write(folder, path, value, receipt=False):
    payload = (json.dumps(json_value(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode()
    save_bytes(folder, path, payload, receipt)


def write_gzip(folder, path, value):
    payload = json.dumps(json_value(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    save_bytes(folder, path, gzip.compress(payload, mtime=0))


def read_gzip(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def seal(folder, receipt=False):
    paths = folder.rglob('*')
    inventory = {str(p.relative_to(folder)): sha(p) for p in sorted(paths)
                 if p.is_file() and p != folder/'artifact_hashes.json'}
    write(folder, folder/'artifact_hashes.json', inventory, receipt)


def verify_inventory(folder):
    inventory = read(folder/'artifact_hashes.json')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p != folder/'artifact_hashes.json'}
    if actual != set(inventory):
        raise ValueError('sealed artifact set changed: '+str(folder))
    for name, expected in inventory.items():
        if sha(folder/name) != expected:
            raise ValueError('sealed artifact changed: '+str(folder/name))


def verify_mapping(mapping):
    for name, expected in mapping.items():
        if sha(ROOT/name) != expected:
            raise ValueError('frozen source/input changed: '+name)


def source_files():
    paths = {Path(__file__).resolve(), PROTOCOL, ANALYSIS}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            p = Path(filename).resolve()
            if p.suffix == '.py' and p.is_relative_to(ROOT) and not any(
                    part.startswith('.venv') for part in p.relative_to(ROOT).parts):
                paths.add(p)
    for name in ('test_observed_state_v26.py', 'test_observed_planner_v26.py',
                 'test_observed_runtime_v26.py', 'test_facility_measurement_v26.py',
                 'test_autonomous_collector_v26.py'):
        p = ROOT/'tests/virtual3d'/name
        if p.exists():
            paths.add(p)
    return paths


def prepare(root):
    if shutil.disk_usage(ROOT).free < CONFIG['task_cap_bytes'] + CONFIG['free_reserve_bytes']:
        raise OSError('prepare requires 40 MiB task capacity plus 64 MiB free reserve')
    verify_inventory(INTERFACE)
    if read(INTERFACE/'result.json')['status'] != 'complete_offline_interface_check':
        raise ValueError('completed observation interface check required')
    old = read(OLD/'manifest.json')
    if old['status'] != 'complete' or old['main_attempts_started'] != 4:
        raise ValueError('original four attempts must remain accounted for')
    previous_sources = old['source_sha256']
    verify_mapping(previous_sources)
    verify_mapping(read(INTERFACE/'manifest.json')['source_sha256'])
    sources = {**previous_sources, **{str(p.relative_to(ROOT)): sha(p) for p in source_files()}}
    cases = [dict(index=i, parent=CONFIG['parent'], assignment=assignment, mode=mode,
                  physical_status='unstarted', replay_status='unstarted')
             for i, (assignment, mode) in enumerate((
                 ('A_complex_B_simple', 'G'), ('A_complex_B_simple', 'S'),
                 ('A_simple_B_complex', 'G'), ('A_simple_B_complex', 'S')))]
    root.mkdir(parents=True, exist_ok=False)
    PROGRESS.update(prepare_owned=True, phase='preparing_source_archive')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.writestr(name, (ROOT/name).read_bytes())
    save_bytes(root, root/'sources.zip', stream.getvalue())
    inputs = [INTERFACE/'manifest.json', INTERFACE/'result.json', INTERFACE/'artifact_hashes.json', OLD/'manifest.json']
    manifest = dict(status='prepared', config=CONFIG, cases=cases, source_sha256=sources,
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        source_archive_sha256=sha(root/'sources.zip'), versions=versions(), world_version=VERSION_V25,
        main_attempts_started=0, quota_start=4, quota_limit=36,
        preparation_worlds=0, preparation_sensor_calls=0, preparation_actions=0)
    write(root, root/'manifest.json', manifest)
    check_frozen(root)
    PROGRESS['prepare_owned'] = False
    print('PREPARED: 4 autonomous tasks; prior attempts4/36; no world constructed', flush=True)


def check_frozen(root):
    manifest = read(root/'manifest.json')
    if manifest['config'] != CONFIG or manifest['quota_start'] != 4:
        raise ValueError('frozen configuration/quota changed')
    verify_mapping(manifest['source_sha256'])
    verify_mapping(manifest['input_sha256'])
    if sha(root/'sources.zip') != manifest['source_archive_sha256'] or versions() != manifest['versions']:
        raise ValueError('source archive/dependency version changed')
    return manifest


def claim_case(root, index):
    manifest = check_frozen(root)
    if manifest['status'].startswith('stopped') or manifest['status'] == 'complete':
        raise ValueError('batch is stopped or complete')
    if index not in range(4) or manifest['cases'][index]['physical_status'] != 'unstarted':
        raise ValueError('case already claimed or invalid')
    if any(c['replay_status'] != 'complete' for c in manifest['cases'][:index]):
        raise ValueError('fixed order requires earlier independent replays first')
    owned = len(list(root.glob('case_*')))
    if owned != manifest['main_attempts_started'] or owned != index or 4 + owned >= 36:
        raise ValueError('case ownership and main-attempt ledger disagree')
    if shutil.disk_usage(ROOT).free < CONFIG['task_cap_bytes'] + CONFIG['free_reserve_bytes']:
        raise OSError('unstarted task requires40MiB plus64MiB; no directory claimed')
    folder = root/f'case_{index:02d}'
    folder.mkdir(exist_ok=False)
    PROGRESS['owned'] = True
    manifest['main_attempts_started'] += 1
    manifest['status'] = 'running'
    case = manifest['cases'][index]
    case.update(physical_status='running', physical_process_id=os.getpid(),
                main_attempt_number=4+manifest['main_attempts_started'])
    write(root, root/'manifest.json', manifest)
    for name in ('packets', 'audit', 'meshes'):
        (folder/name).mkdir()
    return manifest, folder


def require_replay_equal(actual, expected, label):
    if stable(actual) != stable(expected):
        raise ValueError('independent autonomous replay mismatch: '+label)


def save_or_compare_mesh(folder, name, mesh, replay):
    arrays = {k: np.asarray(getattr(mesh, k)) for k in ('vertices', 'triangles', 'vertex_colors')}
    hashes = {k: array_hash(v) for k, v in arrays.items()}
    path = folder/'meshes'/name
    if replay:
        with np.load(path, allow_pickle=False) as stored:
            require_replay_equal(hashes, {k: array_hash(stored[k]) for k in arrays}, name)
    else:
        save_bytes(folder, path, compressed_arrays(arrays))
    return hashes


def run(root, index, replay=False):
    PROGRESS.update(case=index, replay=replay)
    if replay:
        manifest = check_frozen(root)
        if manifest['cases'][index]['physical_status'] != 'complete' or manifest['cases'][index]['replay_status'] != 'unstarted':
            raise ValueError('one independent replay required for a completed main task')
        folder = root/f'case_{index:02d}'
        verify_inventory(folder)
        expected = read(folder/'result.json')
        if read(folder/'timing.json')['process_id'] == os.getpid():
            raise ValueError('independent process required')
        manifest['cases'][index]['replay_status'] = 'running'
        PROGRESS['owned'] = True
        write(root, root/'manifest.json', manifest)
    else:
        manifest, folder = claim_case(root, index)
        expected = None
    started = perf_counter()
    case = manifest['cases'][index]
    PROGRESS['phase'] = 'initializing_world_and_observation_only_runtime'
    world = FacilityChoiceWorldV25(case['parent'], case['assignment'], CONFIG['sensor_model'], CONFIG['noise_seed'])
    config = public_config(world.config)
    runtime = ObservedANSRuntimeV26(tuple(world.shape), config, CONFIG['budget'], case['mode'], CONFIG['replan_interval'])
    measurement = FacilityMeasurementV26()
    trace, plan_timings = [], []
    calls_seen = plans_seen = 0
    actual = measured(world)
    while True:
        action_id = actual.action_id
        PROGRESS.update(phase='replay' if replay else 'autonomous_acquisition', last_attempted_action=action_id)
        path = folder/'packets'/f'{action_id:04d}.npz'
        if replay:
            packet = load_packet(path)
            if packet.sha256() != actual.sha256():
                raise ValueError('fresh physical packet mismatch at '+str(action_id))
        else:
            packet = actual
            payload = packet_payload(packet)
            if len(payload) > CONFIG['packet_cap_bytes']:
                raise ValueError('sensor packet exceeds declared 64 KiB serialization bound')
            save_bytes(folder, path, payload)
        PROGRESS['last_saved_packet'] = action_id
        runtime.accept(packet)
        measurement.observe(packet)
        # This command is computed BEFORE collector-only truth diagnostics.
        next_action = runtime.next_action()
        new_plans = runtime.plans[plans_seen:]
        plan_timings.extend(p['audit']['planning_seconds'] for p in new_plans)
        event = stable(dict(action_id=action_id, next_action=next_action,
            geometry_sha256=runtime.state.geometry_sha256,
            plans=new_plans, module_calls=runtime.planner.calls[calls_seen:],
            runtime_summary=runtime.summary(), cues=[asdict(c) for c in runtime.cues]))
        calls_seen, plans_seen = len(runtime.planner.calls), len(runtime.plans)
        event_path = folder/'audit'/f'{action_id:04d}.json.gz'
        if replay:
            require_replay_equal(event, read_gzip(event_path), 'decision at '+str(action_id))
        else:
            write_gzip(folder, event_path, event)
        # Evaluation-only counts never return to runtime or the next-action API.
        reachable = int(world.reachable.sum())
        coverage = int(np.count_nonzero((runtime.mapper.belief != -1) & world.reachable))/reachable
        row = dict(action_id=action_id, action=packet.action, pose=[*packet.position, packet.heading],
            next_action=next_action, collision=packet.collision, done=packet.done,
            packet_sha256=packet.sha256(), file_sha256=sha(path), nonsemantic=nonsemantic(packet),
            geometry_sha256=runtime.state.geometry_sha256, decision_sha256=digest(event),
            coverage_2d=coverage, cumulative_collisions=world.collisions,
            known_public_roi_cells=int(np.count_nonzero(runtime.mapper.belief != -1)),
            returned_to_anchor=(*packet.position, packet.heading) == runtime.anchor,
            observed_semantic_cues=len(runtime.cues), plans_so_far=plans_seen,
            audit_file=str(event_path.relative_to(folder)))
        trace.append(row)
        if replay:
            require_replay_equal(row, expected['trace'][action_id], 'trace at '+str(action_id))
        else:
            write(folder, folder/'progress.json', dict(status='running', phase=PROGRESS['phase'],
                last_saved_packet=action_id, paid_actions=action_id, coverage_2d=coverage,
                next_action=next_action, runtime=runtime.summary()))
        if action_id % 50 == 0 or next_action is None:
            print(f'case={index} replay={replay} action={action_id} C={coverage:.4f} plans={plans_seen} next={next_action}', flush=True)
        if next_action is None:
            break
        if replay:
            require_replay_equal(next_action, expected['trace'][action_id]['next_action'], 'issued action')
        else:
            # Reserve worst-case remaining raw packets before another action.
            reserve(folder, (CONFIG['budget']-action_id)*CONFIG['packet_cap_bytes'])
        PROGRESS['last_attempted_action'] = action_id + 1
        frame, collision, done = world.step(next_action)
        actual = measured(world, frame, next_action, collision, done)
    PROGRESS['phase'] = 'freeze_measured_representation_before_truth_evaluation'
    if runtime.pending is not None:
        raise ValueError('cannot evaluate while an autonomous action is pending')
    snapshot = measurement.snapshot(runtime.mapper)
    mesh_hashes = dict(raw=save_or_compare_mesh(folder, 'raw.npz', snapshot['raw_mesh'], replay),
        observed=[save_or_compare_mesh(folder, f'instance_{i}.npz', mesh, replay)
                  for i, mesh in enumerate(snapshot['observed_meshes'])])
    if replay:
        require_replay_equal(snapshot['metadata'], read(folder/'measured_snapshot.json'), 'measured snapshot before evaluation')
    else:
        write(folder, folder/'measured_snapshot.json', snapshot['metadata'])
    PROGRESS['phase'] = 'terminal_evaluation_only'
    returned = trace[-1]['returned_to_anchor']
    failed = not returned or bool(world.collisions)
    evaluation = measurement.evaluate(snapshot, world, coverage, returned, world.collisions, failed,
                                      packet.action_id, CONFIG['budget'])
    result = stable(dict(status='complete', index=index, parent=case['parent'], assignment=case['assignment'],
        mode=case['mode'], public_config=vars(config), shape=list(world.shape),
        summary=runtime.summary(), trace=trace, plans_file='plans.json.gz',
        calls_file='module_calls.json.gz', final_main_observed=evaluation['main_observed'],
        final_raw_secondary=evaluation['raw_secondary'], evaluation_audit=evaluation,
        measured_snapshot_file='measured_snapshot.json', mesh_hashes=mesh_hashes,
        trajectory_sha256=digest([(r['action_id'], r['action'], r['pose']) for r in trace]),
        observed_marker_tracks=measurement.tracks.summary(),
        autonomous_policy=True, imposed_prefix_actions=0, gt_service_catalogue_used=False,
        raw_frames=len(trace), paid_actions=packet.action_id, collisions=world.collisions,
        returned_to_anchor=returned, final_coverage_2d=coverage,
        eligible=evaluation['main_observed']['eligible'], evaluation_after_last_decision=True))
    if replay:
        require_replay_equal(result, expected, 'complete result')
        require_replay_equal(runtime.plans, read_gzip(folder/'plans.json.gz'), 'all global decisions')
        require_replay_equal(runtime.planner.calls, read_gzip(folder/'module_calls.json.gz'), 'four module calls')
        write(folder, folder/'verification.json', dict(status='passed', independent_process=True,
            main_process_id=read(folder/'timing.json')['process_id'], replay_process_id=os.getpid(),
            fresh_world=True, fresh_autonomous_runtime=True, saved_actions_used_to_drive_policy=False,
            independent_paid_actions=packet.action_id, saved_packets_verified=len(trace),
            all_decisions_equal=True, all_packets_equal=True, all_meshes_and_metrics_equal=True,
            elapsed_s=perf_counter()-started))
    else:
        write_gzip(folder, folder/'plans.json.gz', stable(runtime.plans))
        write_gzip(folder, folder/'module_calls.json.gz', stable(runtime.planner.calls))
        write(folder, folder/'result.json', result)
        write(folder, folder/'timing.json', dict(process_id=os.getpid(), elapsed_s=perf_counter()-started,
            planning_seconds=plan_timings, total_planning_seconds=sum(plan_timings)))
        write(folder, folder/'progress.json', dict(status='complete', paid_actions=packet.action_id,
            termination=runtime.terminal_reason))
    seal(folder)
    manifest = check_frozen(root)
    manifest['cases'][index]['replay_status' if replay else 'physical_status'] = 'complete'
    write(root, root/'manifest.json', manifest)
    print('COMPLETE', dict(index=index, replay=replay, paid=packet.action_id, C=coverage,
        Q=result['final_main_observed']['05cm']['outline_macro_quality'], eligible=result['eligible'],
        returned=returned, termination=runtime.terminal_reason, bytes=bytes_used(folder)), flush=True)
    if all(c['replay_status'] == 'complete' for c in manifest['cases']):
        aggregate(root)


def aggregate(root):
    manifest = check_frozen(root)
    rows = []
    for case in manifest['cases']:
        folder = root/f"case_{case['index']:02d}"
        verify_inventory(folder)
        result = read(folder/'result.json')
        verification = read(folder/'verification.json')
        if verification['status'] != 'passed':
            raise ValueError('all four independent autonomous replays required')
        rows.append(dict(index=case['index'], assignment=case['assignment'], mode=case['mode'],
            paid_actions=result['paid_actions'], returned=result['returned_to_anchor'],
            collisions=result['collisions'], eligible=result['eligible'],
            C=result['final_coverage_2d'], Q=result['final_main_observed']['05cm']['outline_macro_quality'],
            J=result['final_main_observed']['05cm']['joint_outline'],
            missing=result['final_main_observed']['missing_asset_count'],
            termination=result['summary']['terminal_reason'], result_sha256=sha(folder/'result.json')))
    manifest['status'] = 'complete'
    write(root, root/'manifest.json', manifest)
    write(root, root/'result.json', dict(status='complete_four_autonomous_tasks_and_independent_replays',
        cases=rows, new_main_attempts=4, cumulative_main_attempts=8, quota_limit=36,
        main_paid_actions=sum(r['paid_actions'] for r in rows),
        independent_replay_paid_actions=sum(r['paid_actions'] for r in rows),
        full_semantic_efficacy_claimed=False))
    seal(root)


def failure(root, error):
    if PROGRESS['prepare_owned'] and not PROGRESS['owned']:
        write(root, root/'failure_prepare.json', dict(status='failed_before_any_main_attempt',
            error=repr(error), traceback=traceback.format_exc(), progress=PROGRESS,
            new_main_attempts=0, prior_main_attempts=4), True)
        return
    if not PROGRESS['owned']:
        print('UNSTARTED/UNOWNED ERROR', repr(error), flush=True)
        return
    folder = root/f"case_{PROGRESS['case']:02d}"
    receipt = dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
                   progress=PROGRESS, process_id=os.getpid(), quota_refunded=False)
    write(folder, folder/('replay_failure.json' if PROGRESS['replay'] else 'failure.json'), receipt, True)
    seal(folder, True)
    manifest = read(root/'manifest.json')
    manifest['main_attempts_started'] = len(list(root.glob('case_*')))
    manifest['status'] = 'stopped_input_source_io_or_interface_failure'
    manifest['cases'][PROGRESS['case']]['replay_status' if PROGRESS['replay'] else 'physical_status'] = 'failed'
    write(root, root/'manifest.json', manifest, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'replay', 'aggregate'))
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--case', type=int, choices=range(4))
    args = parser.parse_args()
    # Process-wide exclusive lease prevents overlapping physical/replay runs.
    with (ROOT/'audit_results/.observed_autonomous_v26.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.command == 'prepare':
                prepare(args.output)
            elif args.command == 'aggregate':
                aggregate(args.output)
            else:
                if args.case is None:
                    raise ValueError('--case required')
                run(args.output, args.case, args.command == 'replay')
        except BaseException as error:
            failure(args.output, error)
            raise


if __name__ == '__main__':
    main()
