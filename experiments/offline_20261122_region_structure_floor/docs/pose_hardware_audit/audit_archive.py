"""Join existing NSO hardware evidence. This program only reads robot archives."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_ARCHIVE = Path('/home/ouc/nso_robot/reports/nso-hybrid-20261115/NSO_primitive_optimized_S_20261115_01')
STATIC_REPORT = Path('/home/ouc/nso_robot/analyses/postfall_sensor_check_20261118/result.json')
WORKSPACE = Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def journal(path):
    with path.open() as file:
        return [(line, json.loads(row)) for line, row in enumerate(file, 1) if row.strip()]


def utc(unix_timestamp):
    return datetime.fromtimestamp(unix_timestamp, timezone.utc).isoformat()


def read_frame(path):
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(source['joint_metadata_json'].item())
        receipt = json.loads(source['source_receipt_json'].item())
        graph = metadata.get('rtabmap_graph_snapshot') or {}
        return {
            'path': str(path), 'sha256': sha256(path),
            'frame_id': str(source['frame_id'].item()),
            'rgbd_timestamp_s': float(source['rgbd_timestamp_s']),
            'scan_timestamp_s': float(source['scan_timestamp_s']),
            'rgbd_receipt_monotonic': receipt['rgbd_receipt_monotonic'],
            'scan_receipt_monotonic': receipt['scan_receipt_monotonic'],
            'source_topics': receipt['selected_topics'],
            'sensor_timestamps_independent': metadata['source_stamps_are_independent'],
            'raw_depth_rgb_scan_preserved': {'depth_m', 'rgb_raw', 'scan_ranges_raw'} <= set(source.files),
            'no_missing_modality_substitution': receipt['no_missing_modality_substitution'],
            'physical_extrinsic_calibration_verified': receipt['physical_extrinsic_calibration_verified'],
            'truth_map_used': metadata['truth_map_used'],
            'pose_source': metadata['base_odom_pose_source'],
            'geometry_pose_backend': metadata.get('geometry_pose_backend'),
            'rtabmap_revision_sha256': graph.get('graph_revision_sha256'),
            'rtabmap_graph_source_stamp_ns': graph.get('graph_source_stamp_ns'),
            'rtabmap_pose_association': metadata.get('rtabmap_pose_association'),
        }


def fusion_evidence(archive, observation, rpc, worker):
    observation_line, observation = observation
    rpc_line, rpc = rpc
    worker_line, worker = worker
    assert worker['response']['ok'] and worker['request']['params'] == rpc['params']
    assert worker['response']['result'] == rpc['response']
    assert observation['path'] == rpc['params']['path']
    response = rpc['response']
    frame = read_frame(Path(observation['path']))
    assert str(observation['state']['frame_id']) == frame['frame_id']
    assert str(response['state']['frame_id']) == frame['frame_id']
    assert not observation['synthetic_observations']
    gain = (response.get('primitive_outcome') or {}).get('actual_gain_event') or {}
    rebase = response.get('geometry_rebase') or {}
    generation = response['summary'].get('geometry_rebase_generations')
    boundary = None
    if generation:
        generation_path = archive / 'joint/primitive_feedback/optimized_geometry' / f'generation_{generation:06d}'
        boundary_path = generation_path / 'boundary_receipt.json'
        assert json.loads(boundary_path.read_text()) == rebase
        records_path = generation_path / 'records.json'
        records = json.loads(records_path.read_text()) if records_path.is_file() else []
        boundary = {'generation': generation, 'path': str(boundary_path), 'sha256': sha256(boundary_path),
                    'next_actual_action_written_into_rebased_generation': any(
                        record['action_id'] == generation + 1 for record in records),
                    'state_before_rebase': response['execution_epoch_state'],
                    'state_after_rebase': response['planning_state_after_rebase'],
                    'metric_state_changed': response['execution_epoch_state'] != response['planning_state_after_rebase']}
    derived = (response.get('primitive_outcome') or {}).get('source_path')
    saved_digest = (response.get('primitive_outcome') or {}).get('source_sha256')
    derived_matches = sha256(Path(derived)) == saved_digest if derived else None
    if derived:
        assert derived_matches
    return {
        'observation_event_line': observation_line, 'supervisor_rpc_line': rpc_line,
        'worker_rpc_line': worker_line, 'worker_accepted': response['accepted'],
        'worker_duplicate': response['duplicate'], 'worker_actual_paid_actions': response['summary']['actual_paid_actions'],
        'worker_arrival_verified': (response.get('primitive_outcome') or {}).get('primitive_arrival_verified'),
        'worker_feedback_used_for_planning': gain.get('feedback_consumed_for_planning'),
        'worker_gain': {key: (response.get('parts') or {}).get(key) for key in (
            'new_observed_2d_area_m2', 'new_camera_observed_2d_area_m2', 'measured_support_proxy_m2')},
        'actual_gain_event': gain, 'frame': frame, 'derived_worker_input_path': derived,
        'derived_worker_input_sha256': saved_digest, 'derived_worker_input_hash_matches': derived_matches,
        'rtab_rebase_status': rebase.get('status'), 'rtab_historical_sensor_refusions': rebase.get('actual_sensor_fusions'),
        'rtab_boundary_generation': boundary,
    }


def audit(archive):
    files = {relative: archive / relative for relative in (
        'result.json', 'source_manifest.json', 'events.jsonl', 'joint/decisions.jsonl', 'adapter/events.jsonl')}
    result = json.loads(files['result.json'].read_text())
    events = journal(files['events.jsonl'])
    workers = journal(files['joint/decisions.jsonl'])
    adapter = journal(files['adapter/events.jsonl'])

    def by_event(name, identity):
        return {identity(row): (line, row) for line, row in events if row['event'] == name}

    goals = by_event('goal', lambda row: row['id'])
    terminals = by_event('goal_result', lambda row: row['id'])
    arrivals = by_event('arrival_evidence', lambda row: row['goal']['id'])
    dispatched = by_event('NSO_primitive_feedback_goal_sent', lambda row: row['action_id'])
    outcomes = by_event('NSO_primitive_feedback_goal_result', lambda row: row['action_id'])
    observations = by_event('NSO_primitive_feedback_actual_observation', lambda row: row['action_id'])
    rpcs = {row['params']['action_id']: (line, row) for line, row in events
            if row.get('method') == 'observe_primitive_result'}
    worker_rpcs = {row['request']['params']['action_id']: (line, row) for line, row in workers
                   if row['request']['method'] == 'observe_primitive_result'}
    feedback = [(line, row) for line, row in adapter
                if row['event'] == 'CAN_feedback' and row.get('id') == '0x221' and 'moving' in row]
    records = []
    for action_id, (sent_line, sent) in sorted(dispatched.items()):
        goal_id = sent['goal']['id']
        goal_line, goal = goals[goal_id]
        terminal_line, terminal = terminals[goal_id]
        arrival_line, arrival = arrivals[goal_id]
        assert arrival['goal'] == goal and arrival['terminal'] == terminal
        assert goal['receipt_monotonic'] <= terminal['receipt_monotonic']
        completed = outcomes.get(action_id)
        if completed:
            terminal_excerpt = completed[1]['terminal']
            assert terminal_excerpt == {key: terminal[key] for key in ('id', 'status')}
        actual = fusion_evidence(archive, observations[action_id], rpcs[action_id], worker_rpcs[action_id]) if action_id in rpcs else None
        if actual:
            assert actual['worker_arrival_verified'] == arrival['verified']
            assert completed and actual['worker_actual_paid_actions'] == action_id
        can_in_window = [(line, row) for line, row in feedback
                         if goal['receipt_monotonic'] <= row['source_monotonic'] <= terminal['receipt_monotonic']]
        moving = [(line, row) for line, row in can_in_window if row['moving']]
        sensor_after_goal = None
        sensor_receipt_after_terminal = None
        if actual:
            frame = actual['frame']
            sensor_after_goal = min(frame['rgbd_timestamp_s'], frame['scan_timestamp_s']) >= goal['stamp']
            sensor_receipt_after_terminal = min(frame['rgbd_receipt_monotonic'], frame['scan_receipt_monotonic']) >= terminal['receipt_monotonic']
        records.append({
            'action_id': action_id, 'action': sent['action'], 'goal_id': goal_id,
            'goal_frame': goal['frame'], 'goal_xyyaw': [goal[k] for k in ('x', 'y', 'yaw')],
            'goal_event_line': goal_line, 'nso_dispatch_event_line': sent_line,
            'goal_stamp_unix_s': goal['stamp'], 'goal_stamp_utc': utc(goal['stamp']),
            'goal_receipt_monotonic': goal['receipt_monotonic'], 'terminal_event_line': terminal_line,
            'terminal_receipt_monotonic': terminal['receipt_monotonic'],
            'ros_terminal_status': terminal['status'], 'ros_terminal_text': terminal['text'],
            'arrival_event_line': arrival_line, 'nso_goal_result_event_line': completed[0] if completed else None,
            'arrival_verified': arrival['verified'], 'arrival_reason': arrival['reason'],
            'target_xy_error_m': arrival.get('target_error_m'), 'target_yaw_error_rad': arrival.get('target_yaw_error_rad'),
            'can_distance_delta_m': arrival.get('CAN_distance_delta_m'),
            'can_absolute_yaw_delta_rad': arrival.get('CAN_absolute_yaw_delta_rad'),
            'can_velocity_samples_in_goal_interval': len(can_in_window), 'can_moving_samples': len(moving),
            'can_first_moving_event_line': moving[0][0] if moving else None,
            'can_first_moving_stamp_unix_s': moving[0][1]['kernel_unix_s'] if moving else None,
            'recorded_after_pose_source_stamp_s': (arrival.get('after') or {}).get('source_stamp_s'),
            'sensor_capture_after_goal_by_sensor_unix_stamps': sensor_after_goal,
            'sensor_receipt_after_terminal_by_monotonic_clock': sensor_receipt_after_terminal,
            'actual_observation': actual,
        })
    initial = fusion_evidence(archive, observations[0], rpcs[0], worker_rpcs[0])
    actual_records = [record for record in records if record['actual_observation']]
    assert len({rec['actual_observation']['frame']['frame_id'] for rec in actual_records}) == len(actual_records)
    postfall = json.loads(STATIC_REPORT.read_text())
    fusion_verified = lambda rec: (rec['actual_observation']['worker_accepted']
                                  and not rec['actual_observation']['worker_duplicate'])
    backend_paths = [WORKSPACE / name for name in (
        'config/csm_primary.launch', 'runtime/csm_odometry_bridge.py', 'runtime/rtabmap_mapping_backend.py',
        'nso_bridge/nso_bridge/optimized_primitive_feedback_v1.py',
        'nso_bridge/nso_bridge/primitive_geometry_rebase_v1.py')]
    backend_paths.extend([Path('/opt/ros/noetic/lib/laser_scan_matcher/laser_scan_matcher_node'),
                          Path('/opt/ros/noetic/lib/rtabmap_slam/rtabmap')])
    generations = [rec['actual_observation']['rtab_boundary_generation'] for rec in actual_records]
    assert [g['generation'] for g in generations] == list(range(1, len(actual_records) + 1))
    source_manifest = json.loads(files['source_manifest.json'].read_text())
    recorded_code = {item['path']: item['sha256'] for item in source_manifest['files']}
    return {
        'schema': 'nso.archived_hardware_pose_evidence/1', 'read_only': True,
        'physical_execution_requested_or_started': False, 'archive': str(archive),
        'source_files': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in files.items()},
        'source_backend_files': {str(path): sha256(path) for path in backend_paths if path.is_file()},
        'code_equality_with_archived_source_manifest': {
            str(path.relative_to(WORKSPACE)): sha256(path) == recorded_code[str(path.relative_to(WORKSPACE))]
            for path in backend_paths if path.is_file() and WORKSPACE in path.parents
            and str(path.relative_to(WORKSPACE)) in recorded_code},
        'session': {key: result[key] for key in (
            'session_id', 'mode', 'status', 'reason', 'navigation_source', 'primitive_verified_arrivals',
            'macro_verified_arrivals', 'actual_CAN_distance_m', 'elapsed_run_s', 'full_optimized_geometry_replay')},
        'count': {
            'nso_dispatched_goals': len(records), 'ros_terminal_results': len(terminals),
            'ros_terminal_statuses': Counter(rec['ros_terminal_status'] for rec in records),
            'nso_primitive_outcomes': len(outcomes),
            'fresh_accepted_worker_fusions_after_dispatch': sum(map(fusion_verified, actual_records)),
            'verified_arrival_and_accepted_observation': sum(rec['arrival_verified'] and fusion_verified(rec) for rec in actual_records),
            'verified_arrival_with_logged_can_motion_and_accepted_observation': sum(
                rec['arrival_verified'] and bool(rec['can_moving_samples']) and fusion_verified(rec) for rec in actual_records),
            'paired_sensor_receipts_after_ros_terminal': sum(rec['sensor_receipt_after_terminal_by_monotonic_clock'] for rec in actual_records),
            'paired_sensor_capture_after_goal': sum(rec['sensor_capture_after_goal_by_sensor_unix_stamps'] for rec in actual_records),
            'distinct_rtab_graph_revisions_on_action_observations': len({rec['actual_observation']['frame']['rtabmap_revision_sha256']
                                                                        for rec in actual_records if rec['actual_observation']['frame']['rtabmap_revision_sha256']}),
            'geometry_rebase_statuses': Counter(rec['actual_observation']['rtab_rebase_status'] for rec in actual_records),
            'matched_rtabmap_rebuild_boundary_receipts': len(generations),
            'rtabmap_metric_planning_state_changed_after_rebuild': sum(g['metric_state_changed'] for g in generations),
            'rtabmap_rebuilt_generations_containing_the_next_measured_action': sum(
                g['next_actual_action_written_into_rebased_generation'] for g in generations),
        },
        'rtabmap_rebuild_status_scope': ('geometry_rebase_candidate_not_activated is emitted by the rebuild helper. '
                                        'Inspect outer planning_state_after_rebase, successive generation IDs and '
                                        'presence of the next measured action in a rebuilt generation for active use.'),
        'separate_final_generic_observe_summary': result['final_worker_observation_receipt'],
        'scope': {
            'independent_hardware_position_reference_available': False,
            'calibrated_reconstruction_accuracy_verified': False,
            'current_regional_method_hardware_rerun': False,
        },
        'latest_available_postfall_static_record': {
            'path': str(STATIC_REPORT), 'sha256': sha256(STATIC_REPORT), 'time_utc': utc(postfall['time']),
            'physical_motion_requested': postfall['physical_motion_requested'],
            'camera_rgbd_received': postfall['summary']['depth']['received'],
            'camera_integrated_imu_received': postfall['summary']['imu']['received'],
            'can_rx_no_tx': postfall['CAN']['no_transmit'], 'network': postfall['network'],
            'limitations': postfall['limitations'],
        },
        'bootstrap_accepted_observation': initial, 'actions': records,
    }


def write_csv(path, records):
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['action_id', 'action', 'goal_id', 'goal_unix_s', 'terminal_monotonic_s',
                         'ros_status', 'arrival_verified', 'can_moving_samples', 'can_distance_m',
                         'rgbd_unix_s', 'scan_unix_s', 'new_worker_fusion', 'new_lidar_area_m2',
                         'new_camera_area_m2', 'measured_support_proxy_m2', 'rtab_revision_sha256'])
        for rec in records:
            actual = rec['actual_observation'] or {}
            frame = actual.get('frame', {})
            gain = actual.get('worker_gain', {})
            writer.writerow([rec[key] for key in (
                'action_id', 'action', 'goal_id', 'goal_stamp_unix_s', 'terminal_receipt_monotonic',
                'ros_terminal_status', 'arrival_verified', 'can_moving_samples', 'can_distance_delta_m')] + [
                frame.get('rgbd_timestamp_s'), frame.get('scan_timestamp_s'),
                actual.get('worker_accepted') and not actual['worker_duplicate'],
                gain.get('new_observed_2d_area_m2'), gain.get('new_camera_observed_2d_area_m2'),
                gain.get('measured_support_proxy_m2'), frame.get('rtabmap_revision_sha256')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    summary = audit(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'hardware_chain_receipt.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    write_csv(args.output / 'hardware_chain.csv', summary['actions'])
    print(json.dumps(summary['count'], indent=2))
