"""Summarize the successful data-collection probe without claiming a repair."""
import json
import math
from pathlib import Path
import numpy as np
from geometry_msgs.msg import PoseStamped

P = Path(__file__).resolve().parent
REPORT = P / 'report'
result = json.loads((REPORT / 'result.json').read_text())
telemetry = json.loads((P / 'telemetry.json').read_text())
cfg = json.loads((P / 'configuration.json').read_text())
events = [json.loads(s) for s in (REPORT / 'adapter/events.jsonl').read_text().splitlines()]
goal = result['goals'][0]
terminal = result['goal_results'][0]
start, end = goal['receipt_monotonic'], terminal['receipt_monotonic']
provenance = {t['control_provenance']['csm_source_stamp_ns']: t['control_provenance']
              for t in telemetry if t.get('control_provenance')}
active = [p for p in provenance.values()
          if start <= p['can_pair']['selected_monotonic_ns'] / 1e9 <= end]
dwa = cfg['applied_DWA']

def inverted(v):
    return (max(dwa['min_vel_x'], v['vx'] - dwa['acc_lim_x'] * .1) >
            min(dwa['max_vel_x'], v['vx'] + dwa['acc_lim_x'] * .1) or
            max(-dwa['max_vel_theta'], v['w'] - dwa['acc_lim_theta'] * .1) >
            min(dwa['max_vel_theta'], v['w'] + dwa['acc_lim_theta'] * .1))

def valid(p):
    s = p['can_pair']
    return all(0 <= d <= 150_000_000 for d in (
        s['scan_stamp_ns'] - s['kernel_stamp_ns'],
        s['selected_wall_ns'] - s['kernel_stamp_ns'],
        s['selected_monotonic_ns'] - s['receipt_monotonic_ns']))

z = np.load(P / 'failure_scan_ring.npz')
meta = json.loads((P / 'failure_scan_ring.json').read_text())
poses = []
for row in meta['messages']:
    if row['topic'] != '/nso_csm/pose_stamped':
        continue
    m = PoseStamped().deserialize(z[row['key']].tobytes())
    poses.append((m.header.stamp.to_nsec(), m.pose.position.x, m.pose.position.y))
steps = []
for a, b in zip(poses, poses[1:]):
    dt = (b[0] - a[0]) / 1e9
    steps.append(dict(stamp_ns=b[0], dt_s=dt,
                      distance_m=math.hypot(b[1] - a[1], b[2] - a[2]),
                      original_guard_bound_m=.02 + .2 * dt))
arrival = result['standard_navigation_setup_evidence']
finish = next(e for e in events if e['event'] == 'finished')
commands = {t['actuator']['last_command']['at_monotonic']: t['actuator']['last_command']
            for t in telemetry if t.get('actuator', {}).get('last_command') and
            start <= t['actuator']['last_command']['at_monotonic'] <= end}
analysis = dict(
    probe='control_can_offset_07', scope='Standard ROS diagnostic, unchanged CAN '
    'control profile; different actual start from05/06, not a matched repair test '
    'or NSO credited action.',
    summary=json.loads((REPORT / 'summary.json').read_text()),
    actual_CAN_distance_m=result['actual_CAN_distance_m'],
    actual_CAN_absolute_yaw_rad=result['actual_CAN_absolute_yaw_rad'],
    DWA_xy_tolerance_m=dwa['xy_goal_tolerance'],
    supervisor_verification_xy_tolerance_m=arrival['xy_tolerance_m'],
    terminal_error_m=arrival['target_error_m'],
    terminal_yaw_error_rad=arrival['target_yaw_error_rad'],
    terminal_within_DWA_xy_tolerance=arrival['target_error_m'] <= dwa['xy_goal_tolerance'],
    goal_to_terminal_s=end-start,
    unique_provenance_pairs=len(provenance), active_provenance_pairs=len(active),
    pair_validity_failures=sum(not valid(p) for p in provenance.values()),
    input_window_audit={k:sum(inverted(p[k]) for p in active)
                        for k in ('raw_csm_twist', 'control_twist')},
    window_audit_scope='Sampled input feasibility at0.1s; raw CSM is a '
                       'counterfactual input, not the actual DWA sampled state.',
    command_records=len(commands), command_bound_violations=sum(
        c['v'] < -1e-8 or c['v'] > dwa['max_vel_x'] + 1e-8 or
        abs(c['w']) > dwa['max_vel_theta'] + 1e-8 for c in commands.values()),
    scan_ring_pose_count=len(poses),
    maximum_recorded_pose_step=max(steps, key=lambda s:s['distance_m']),
    original_jump_bound_violations=sum(s['distance_m'] > s['original_guard_bound_m'] for s in steps),
    adapter_final_reason=finish['reason'],
    adapter_finish_after_arrival_s=finish['closed_monotonic'] - end,
    native_fault8_meaning='external_guard; recorded after goal completion, '
                         'during stop cleanup. Not a chassis error code.',
    stop_verified_before_close=result['feedback_zero_before_close'],
    stop_verified_after_close=result['feedback_zero_after_close'],
    physical_stop_independently_observed=result['physical_stop_independently_observed'],
    CAN_capture_audit='../../can_source_audit07/summary.json',
    limits=['One arrival under unchanged configuration does not fix the '
            'reproduced intermittent CSM registration jump.',
            'Controller/supervisor use CSM for arrival evidence; no independent '
            'physical position accuracy was measured.',
            'Full-rate bounded CAN capture is available, but telemetry remains '
            'sampled asynchronous latest-state observations.'])
print(json.dumps(analysis, indent=2))
