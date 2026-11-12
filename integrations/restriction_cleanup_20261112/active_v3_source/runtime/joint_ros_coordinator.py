"""ROS execution boundary for the restored observed-map joint v10 method.

The science worker owns candidates, scores, action accounting and return.
Legacy execution uses each fixed metric endpoint. An explicitly selected
measured_in_place_turn_v1 adapter may execute a first non-anchor turn at the
current measured XY within that same pending cell, keeping its absolute yaw.
Logical endpoints and scientific cell/heading validation are unchanged. A single
nominal retry can recover a ROS-successful measured turn with target pose error;
it preserves both attempts and the original final acceptance criteria.
The different metric execution target is recorded; method equivalence is not
asserted. Importing this module creates no subscriptions or motion commands.
"""
import math
import time
from contextlib import nullcontext

from nso_ros_coordinator import matrix, reference_pose_error, wrap


def measured_grid_pose(odom_pose, grid):
    """Map an actual base pose, never an assigned destination, to v10 cells."""
    import numpy as np
    transform = np.asarray(grid['planning_from_odom'], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('explicit finite planning_from_odom required')
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError('planning_from_odom must be rigid')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.):
        raise ValueError('planning_from_odom must be rigid')
    if not np.isfinite(odom_pose).all() or len(odom_pose) != 3:
        raise ValueError('actual finite odom base pose required')
    shape, origin = grid['shape'], grid['origin_xy_m']
    resolution = grid['resolution_m']
    if resolution != .2 or list(origin) != [0., 0.]:
        raise ValueError('original v10 requires .2m grid and planning origin zero')
    point = transform @ [odom_pose[0], odom_pose[1], 0., 1.]
    col = math.floor((point[0] - origin[0]) / resolution)
    row = shape[0] - 1 - math.floor((point[1] - origin[1]) / resolution)
    yaw = wrap(odom_pose[2] + math.atan2(transform[1, 0], transform[0, 0]))
    # Original headings are north, east, south, west, not ROS yaw indices.
    heading = min(range(4), key=lambda h: abs(wrap(yaw - (math.pi / 2 - h * math.pi / 2))))
    return [row, col, heading]


def endpoint_evidence(target, actual_odom, grid, expected_grid_pose, xy_tolerance, yaw_tolerance):
    """Validate the original pending grid state; retain metric error as evidence.

    ROS already completes the navigation goal with its configured tolerance.
    A second metric tolerance in odom can repeatedly reissue an already reached
    grid state when gmapping corrects map->odom. The science worker independently
    checks the actual exposure-time cell and heading before accepting a packet.
    """
    evidence = reference_pose_error(target, actual_odom, xy_tolerance, yaw_tolerance)
    actual_grid_pose = measured_grid_pose(actual_odom, grid)
    evidence.update(metric_tolerance_met=evidence['verified'],
                    actual_grid_pose=actual_grid_pose, expected_grid_pose=list(expected_grid_pose),
                    grid_pose_matches=actual_grid_pose == list(expected_grid_pose))
    evidence['verified'] = evidence['grid_pose_matches']
    evidence['acceptance_contract'] = 'original_pending_cell_and_heading; metric_errors_diagnostic'
    return evidence


def metric_goal_to_map(odom_pose, map_from_odom):
    """Reproject the same scientific odom endpoint using current SLAM TF."""
    import numpy as np
    transform = np.asarray(map_from_odom, dtype=float)
    target = transform @ [odom_pose[0], odom_pose[1], 0., 1.]
    return [float(target[0]), float(target[1]),
            wrap(odom_pose[2] + math.atan2(transform[1, 0], transform[0, 0]))]


def validate_joint_identity(identity):
    if (not isinstance(identity, dict) or identity.get('schema') != 'nso_joint_runtime_v10/1'
            or identity.get('execution_allowed') is not True):
        raise RuntimeError('physical joint v10 worker identity required')
    for key in ('calibration', 'grid', 'summary'):
        if not isinstance(identity.get(key), dict):
            raise ValueError('joint worker identity lacks ' + key)
    # Validate the transform without borrowing a V35 task anchor or graph.
    measured_grid_pose([0., 0., 0.], identity['grid'])
    profile = identity['calibration'].get('joint_turn_execution_profile', 'grid_center')
    if profile not in ('grid_center', 'measured_in_place_turn_v1'):
        raise ValueError('unknown joint_turn_execution_profile')
    if profile == 'measured_in_place_turn_v1':
        anchor = identity.get('configuration', {}).get('return_anchor')
        if not isinstance(anchor, list) or len(anchor) != 3:
            raise ValueError('measured turn adapter requires declared return_anchor')
    return identity


def execution_metric_endpoint(endpoint, source, identity, *, retry=False):
    """Choose only a versioned execution target; never mutate logical pending."""
    nominal = list(endpoint['target_odom_pose'])
    profile = identity['calibration'].get('joint_turn_execution_profile', 'grid_center')
    reason = 'legacy_grid_center'
    target = list(nominal)
    if profile == 'measured_in_place_turn_v1':
        if endpoint['action'] not in ('left', 'right'):
            reason = 'forward_uses_nominal_endpoint'
        elif endpoint['pose'] == identity['configuration']['return_anchor']:
            reason = 'return_anchor_uses_nominal_endpoint'
        elif retry:
            reason = 'retry_uses_nominal_endpoint'
        elif measured_grid_pose(source['pose'], identity['grid'])[:2] != endpoint['pose'][:2]:
            reason = 'actual_cell_differs_uses_nominal_endpoint'
        else:
            target[:2] = list(source['pose'][:2])
            reason = 'first_turn_measured_xy_in_pending_cell'
    return dict(profile=profile, reason=reason,
                applied=reason == 'first_turn_measured_xy_in_pending_cell',
                logical_target_odom_pose=nominal, execution_target_odom_pose=target,
                scientific_endpoint_contract='original_pending_cell_and_heading',
                original_metric_execution_equivalence=target == nominal)



def nominal_endpoint_retry_reason(terminal, arrival, scientific, metric_execution,
                                  source, actual_odom, *, retry_used):
    """One same-pending recovery; never changes tolerances or accepts a frame."""
    if retry_used or terminal.get('status') != 3:
        return None
    if arrival.get('verified') is True and scientific.get('verified') is False:
        return 'original_ROS_arrival_verified_logical_endpoint_mismatch'
    if (metric_execution.get('profile') != 'measured_in_place_turn_v1'
            or metric_execution.get('applied') is not True
            or metric_execution.get('action') not in ('left', 'right')
            or arrival.get('reason') != 'target_pose_error'):
        return None
    # Source freshness/health remains enforced by capture/check. Here reject
    # invalid evidence rather than creating a new freshness or distance limit.
    for pose in (source, actual_odom):
        try:
            values=pose['pose']
            stamp=pose.get('source_stamp_s')
            stamp_ns=pose.get('source_stamp_ns')
            age=pose.get('source_age_s')
            valid_stamp=(type(stamp_ns) is int and stamp_ns>0) or (
                isinstance(stamp,(int,float)) and math.isfinite(stamp) and stamp>0)
            if (len(values)!=3 or not all(math.isfinite(v) for v in values)
                    or not valid_stamp or not isinstance(age,(int,float))
                    or not math.isfinite(age) or age<0 or pose.get('target_frame')!='odom'
                    or pose.get('source_frame')!='base_link'
                    or pose.get('pose_error') is not None):
                return None
        except (KeyError,TypeError,ValueError,OverflowError):
            return None
    return 'measured_first_turn_ROS_succeeded_target_pose_error_retry_nominal'


def pending_endpoint(decision, state):
    """Check transport consistency; do not rank or alter the worker action."""
    action, pending = decision.get('action'), decision.get('pending')
    if action is None:
        if pending is not None:
            raise RuntimeError('stopped joint decision still has a pending action')
        return None
    if action not in ('forward', 'left', 'right') or not isinstance(pending, dict):
        raise RuntimeError('joint worker returned an unsupported primitive')
    previous_id = state.get('action_id', state.get('step'))
    if type(previous_id) is not int or pending.get('action_id') != previous_id + 1:
        raise RuntimeError('joint pending action sequence differs from accepted state')
    pose, target = pending.get('pose'), pending.get('target_odom_pose')
    if (not isinstance(pose, list) or len(pose) != 3 or any(type(v) is not int for v in pose)
            or pose[2] not in range(4) or not isinstance(target, list) or len(target) != 3
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in target)):
        raise RuntimeError('joint worker must declare its grid destination and metric odom endpoint')
    if 'action' in pending and pending['action'] != action:
        raise RuntimeError('joint pending primitive differs from selected action')
    return dict(action=action, action_id=pending['action_id'], pose=pose,
                target_odom_pose=list(target), selected=decision.get('selected'))


def run_joint(ctx):
    """Execute worker primitives with declared metric adaptation and shared protection."""
    identity = validate_joint_identity(ctx['task_receipt'])
    import rospy
    from move_base_msgs.msg import MoveBaseGoal
    from mission import arrival_evidence

    worker, client = ctx['worker'], ctx['client']
    grid = identity['grid']
    recover_failed_attempt = identity.get('configuration', {}).get('policy_variant') == 'ros_checked_exploration_v2'
    execution_frame = ctx.get('execution_frame', 'map')
    if execution_frame not in ('map', 'odom'):
        raise ValueError('joint execution_frame must be map or odom')
    capture_execution_pose = ctx['capture_odom_pose'] if execution_frame == 'odom' else ctx['capture_pose']
    actions = []
    ctx['result'].update(joint_actions=actions, joint_identity=identity,
        joint_turn_execution_profile=identity['calibration'].get('joint_turn_execution_profile', 'grid_center'))

    def record(event, **fields):
        ctx['journal'].add(dict(event=event, **fields))

    def heartbeat():
        ctx['check'](True)
        ctx['progress']()
        if time.monotonic() - ctx['run_start'] >= ctx['duration']:
            raise RuntimeError('external_execution_duration_complete')

    def call(method, params):
        rpc_started = time.monotonic()
        response = worker.call(method, params, heartbeat)
        rpc_elapsed_s = time.monotonic() - rpc_started
        record('NSO_joint_rpc', method=method, params=params, response=response, rpc_elapsed_s=rpc_elapsed_s)
        if 'summary' in response:
            ctx['result']['joint_summary'] = response['summary']
        return response

    def fail(reason, evidence):
        record('NSO_joint_execution_failure', reason=reason, evidence=evidence,
               core_observation_accepted=False)
        ctx['result'].update(status='navigation_execution_failed', execution_interrupted=True,
                             reason=reason, method_stop=False)

    def capture(action_id, action, endpoint=None, *, failed_reason=None):
        heartbeat()
        before = ctx['capture_odom_pose']()
        # No ideal cell or pose is supplied to the measured sensor capture.
        # These polls bracket processing, not the historical exposure time.
        # The worker validates the NPZ's actual historical base cell against
        # its pending primitive before fusion; do not replace that contract
        # with newer odometry after compression/map resampling.
        path = ctx['capture_observation'](identity['calibration'], action_id, None, action)
        after = ctx['capture_odom_pose']()
        params=dict(path=str(path), action_id=action_id, action=action)
        method='observe'
        if failed_reason is not None:
            if not recover_failed_attempt:
                raise RuntimeError('failed attempt observation requires explicit v2 profile')
            method='observe_failed_action'
            params['reason']=failed_reason
        response = call(method, params)
        if failed_reason is not None:
            outcome=response.get('failed_execution_attempt', {})
            if (outcome.get('arrival_credited') is not False
                    or outcome.get('paid_attempt_charged') is not True):
                raise RuntimeError('failed attempt RPC lacks no-arrival/charged-attempt receipt')
        record('NSO_joint_failed_attempt_observed' if failed_reason is not None else 'NSO_joint_observation_accepted',
               arrival_credited=False if failed_reason is not None else None, failed_reason=failed_reason,
               action_id=action_id, action=action,
               capture_timings=dict(ctx.get('capture_timings', {})), path=str(path), actual_odom_before=before, actual_odom_after=after,
               processing_pose_polls_are_diagnostic=True,
               exposure_pose_source='joint_NPZ_actual_historical_base_TF_validated_by_worker',
               state=response['state'])
        return response['state']

    if identity.get('current_state') is not None or identity.get('pending') is not None:
        raise RuntimeError('joint resume requires an explicit executed-history adapter')
    state = capture(0, None)
    if state is None:
        return
    while True:
        heartbeat()
        decision = call('next_action', {})
        endpoint = pending_endpoint(decision, state)
        if endpoint is None:
            termination = decision.get('summary', {}).get('runtime', {}).get('termination') or {}
            ctx['result'].update(status='nso_joint_core_failed' if termination.get('failed')
                                else 'nso_joint_core_stopped', method_stop=True,
                                reason=termination.get('reason'),
                                method_failed=bool(termination.get('failed')))
            return
        previous_projection_goal = None
        recovered_state = False
        while True:
            # Opt-in execution capability exists only during the existing
            # same-pending nominal retry; first attempts never enter this scope.
            scope = (ctx['correction_reverse_scope'](endpoint,record)
                     if previous_projection_goal is not None and ctx.get('correction_reverse_scope')
                     else nullcontext())
            with scope:
                heartbeat()
                source = ctx['capture_odom_pose']()
                metric_execution = execution_metric_endpoint(endpoint, source, identity,
                    retry=previous_projection_goal is not None)
                odom_execution_target = metric_execution['execution_target_odom_pose']
                map_projection = None
                if execution_frame == 'map':
                    if type(source.get('source_stamp_ns')) is int:
                        seconds, nanoseconds = divmod(source['source_stamp_ns'], 10**9)
                        query = rospy.Time(seconds, nanoseconds)
                    else:
                        query = rospy.Time.from_sec(source['source_stamp_s'])
                    tf = ctx['tfbuf'].lookup_transform('map', 'odom', query, rospy.Duration(.3))
                    transform = matrix(tf.transform)
                    target = metric_goal_to_map(odom_execution_target, transform)
                    map_projection = dict(matrix=transform.tolist(), query_stamp_ns=query.to_nsec(),
                        response_stamp_ns=tf.header.stamp.to_nsec(), source_frame=tf.child_frame_id,
                        target_frame=tf.header.frame_id,
                        timestamp_semantics='actual TF lookup at odometry source time; may be interpolated')
                else:
                    target = list(odom_execution_target)
                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = execution_frame
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x, goal.target_pose.pose.position.y = target[:2]
                goal.target_pose.pose.orientation.z = math.sin(target[2]/2)
                goal.target_pose.pose.orientation.w = math.cos(target[2]/2)
                before, can_start = capture_execution_pose(), ctx['can_snapshot']()
                client.send_goal(goal)
                goal_id = client.gh.comm_state_machine.action_goal.goal_id.id
                sent = dict(id=goal_id, frame=execution_frame, x=target[0], y=target[1], yaw=target[2])
                execution = dict(action=endpoint['action'], action_id=endpoint['action_id'],
                    grid_pose=endpoint['pose'], target_odom_pose=endpoint['target_odom_pose'],
                    selected=endpoint['selected'], goal=sent, source_odom_pose=source,
                    planning_from_odom=grid['planning_from_odom'], execution_frame=execution_frame,
                    metric_execution=metric_execution)
                if map_projection is not None:
                    execution['map_from_odom'] = map_projection
                if previous_projection_goal is not None:
                    execution.update(fixed_endpoint_reprojected=execution_frame == 'map',
                                     fixed_endpoint_retried=True, previous_goal=previous_projection_goal)
                record('NSO_joint_goal_sent', **execution)
                while not client.wait_for_result(rospy.Duration(.05)):
                    heartbeat()
                terminal = dict(id=goal_id, status=client.get_state())
                after, actual_odom, can_end = capture_execution_pose(), ctx['capture_odom_pose'](), ctx['can_snapshot']()
                arrival_args = (sent, terminal, before, after, can_start, can_end)
                arrival = (arrival_evidence(*arrival_args, expected_frame='odom')
                           if execution_frame == 'odom' else arrival_evidence(*arrival_args))
                scientific = endpoint_evidence(endpoint['target_odom_pose'], actual_odom['pose'], grid,
                    endpoint['pose'], ctx['xy_tolerance_m'], ctx['yaw_tolerance_rad'])
                execution.update(arrival=arrival, actual_odom_pose=actual_odom, endpoint_error=scientific)
                actions.append(execution)
                record('NSO_joint_action_result', **execution)
                if arrival['verified'] and scientific['verified']:
                    break
                if recover_failed_attempt and terminal['status'] in (3, 4):
                    reason=('ROS_ABORTED' if terminal['status'] == 4 else 'ROS_success_endpoint_not_verified')
                    execution.update(failed_execution_attempt=True,arrival_credited=False,
                                     recovery='fresh_actual_observation_then_replan')
                    state=capture(endpoint['action_id'], endpoint['action'], failed_reason=reason)
                    record('NSO_joint_failed_attempt_replan', action_id=endpoint['action_id'],
                           reason=reason, terminal=terminal, actual_state=state,
                           arrival_credited=False, nominal_retry_skipped=True)
                    recovered_state = True
                    break
                retry_reason = nominal_endpoint_retry_reason(terminal, arrival, scientific,
                    dict(metric_execution, action=endpoint['action']), source, actual_odom,
                    retry_used=previous_projection_goal is not None)
                if retry_reason is not None:
                    # Retry the same pending nominal endpoint once; no observation
                    # is committed until the unchanged final acceptance succeeds.
                    record('NSO_joint_fixed_endpoint_reprojection' if execution_frame == 'map'
                           else 'NSO_joint_fixed_endpoint_retry', action_id=endpoint['action_id'],
                        target_odom_pose=endpoint['target_odom_pose'], previous_goal=goal_id,
                        endpoint_error=scientific, retry_reason=retry_reason,
                        first_attempt_evidence=execution, core_observation_accepted=False)
                    previous_projection_goal = goal_id
                    continue
                fail('ROS_action_or_joint_endpoint_not_verified', execution)
                return
        if recovered_state:
            if state is None:
                return
            continue
        state = capture(endpoint['action_id'], endpoint['action'], endpoint)
        if state is None:
            return
