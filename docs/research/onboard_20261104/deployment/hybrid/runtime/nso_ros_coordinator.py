"""Thin ROS execution boundary for the unchanged V35 reference controller.

The former four-structure greedy worker is not the reference method and is
not available through this runtime.  Geometry conversion remains reusable by
manual sensor inspection tools; importing this module never imports ROS.
"""
import hashlib
import json
from pathlib import Path
import math
import time


LEGACY_METHOD_VARIANT = 'onboard_adaptation_four_structure_greedy'
MISSING_REFERENCE_TASK_INPUTS = (
    'two public templates for the real task',
    'a shared safe pose graph with its initial position and orientation',
    'an aligned scan/RGB-D observation and its physical-to-reference frame binding',
)


def validate_experiment_config(value):
    """Validate identity metadata, not readiness or a reference task contract."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get('schema') != 'nso_physical_experiment/1':
        raise ValueError('explicit nso_physical_experiment/1 config required')
    if not isinstance(value.get('experiment_id'), str) or not value['experiment_id'].strip():
        raise ValueError('nonempty experiment_id required')
    if value.get('navigation_source') != 'lidar_rgbd':
        raise ValueError('experiment navigation_source must be lidar_rgbd')
    if value.get('method_variant') == LEGACY_METHOD_VARIANT:
        raise ValueError('historical four-structure greedy configuration is not the NSO reference method; '
                         'legacy execution has been removed')
    return json.loads(json.dumps(value, allow_nan=False))


def ensure_nso_runtime_ready(task_receipt=None):
    """Fail before starting any ROS data stack or chassis process.

    Vendoring the reference core does not provide its real-world task inputs.
    The receipt comes from a successful reference worker initialization, not
    from experiment identity metadata. Task validation belongs to its loader.
    """
    if not isinstance(task_receipt, dict) or not all(key in task_receipt for key in
            ('task', 'poses', 'anchor', 'binding', 'calibration', 'summary')):
        raise RuntimeError('NSO runtime unavailable: missing reference task inputs: '
                           + '; '.join(MISSING_REFERENCE_TASK_INPUTS)
                           + '. No legacy fallback is available. Standard ROS navigation remains available.')
    provenance = task_receipt['task']
    if (not isinstance(provenance, dict) or provenance.get('scope') != 'physical_experiment'
            or provenance.get('execution_allowed') is not True):
        raise RuntimeError('NSO runtime unavailable: reference task is not approved for physical execution')
    poses, anchor = task_receipt['poses'], task_receipt['anchor']
    if not isinstance(poses, list) or type(anchor) is not int or not 0 <= anchor < len(poses):
        raise ValueError('reference worker initialization lacks a valid anchor pose')
    graph_pose_to_odom(poses[anchor], task_receipt['binding'])
    return task_receipt


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def graph_pose_to_odom(pose, binding):
    """Original metre-grid convention: +Y, +X, -Y, -X; right increments h."""
    if (len(pose) != 3 or any(type(value) is not int for value in pose)
            or not 0 <= pose[2] < 4):
        raise ValueError('integer V35 graph pose required')
    if (binding.get('physical_frame_id') != 'odom'
            or binding.get('meters_per_grid_unit') != 1.0):
        raise ValueError('fixed metre-grid to odom task binding required')
    tx, ty = binding['translation_xy_m']
    phi = binding['yaw_rad']
    if not all(math.isfinite(value) for value in (tx, ty, phi)):
        raise ValueError('finite task rigid binding required')
    c, s = math.cos(phi), math.sin(phi)
    x, y, h = pose
    return [c*x-s*y+tx, s*x+c*y+ty, wrap(phi+math.pi/2-h*math.pi/2)]


def reference_pose_error(expected_odom, actual_odom, xy_tolerance_m, yaw_tolerance_rad):
    """Assign the issued graph node only after a measured endpoint check.

    Tolerances are supplied by the existing ROS arrival contract. This does
    not search for another node or re-anchor the fixed task transformation.
    """
    if not all(math.isfinite(value) for value in
               list(expected_odom)+list(actual_odom)+[xy_tolerance_m, yaw_tolerance_rad]):
        raise ValueError('finite reference arrival evidence required')
    if xy_tolerance_m < 0 or yaw_tolerance_rad < 0:
        raise ValueError('nonnegative ROS arrival tolerances required')
    xy = math.hypot(actual_odom[0]-expected_odom[0], actual_odom[1]-expected_odom[1])
    yaw = abs(wrap(actual_odom[2]-expected_odom[2]))
    return dict(expected_odom_pose=list(expected_odom), actual_odom_pose=list(actual_odom),
                position_error_m=xy, yaw_error_rad=yaw, xy_tolerance_m=xy_tolerance_m,
                yaw_tolerance_rad=yaw_tolerance_rad,
                verified=xy <= xy_tolerance_m and yaw <= yaw_tolerance_rad,
                scope='measured endpoint within declared ROS tolerance; not exact mathematical pose')



def verified_pending_resume(task, attempt, xy_tolerance, yaw_tolerance, arrival_verifier):
    """Bind a replayed pending RPC to its already measured physical arrival.

    This validates existing execution evidence, never selects an action.
    An unfinished, originally issued goal can only resume that same selection;
    observation acceptance still requires a measured arrival and current pose.
    """
    attempt = Path(attempt).resolve()
    source = task.get('resume_source')
    source_path = source if isinstance(source, str) else (
        source.get('path', source.get('log_path')) if isinstance(source, dict) else None)
    log_path = attempt / 'nso' / 'decisions.jsonl'
    if not source_path or Path(source_path).resolve() != log_path:
        raise RuntimeError('resume worker replay source differs from prior attempt')
    if isinstance(source, dict) and source.get('sha256') is not None:
        if hashlib.sha256(log_path.read_bytes()).hexdigest() != source['sha256']:
            raise RuntimeError('resume replay source hash changed')
    prior = json.loads((attempt / 'result.json').read_text())
    manifest = task['task'].get('manifest', {})
    old_manifest = prior.get('nso_task', {}).get('manifest', {})
    if (not manifest.get('sha256') or manifest.get('sha256') != old_manifest.get('sha256')
            or not manifest.get('path')
            or hashlib.sha256(Path(manifest['path']).read_bytes()).hexdigest() != manifest['sha256']):
        raise RuntimeError('resume task manifest differs from executed task')
    state, pending, selected = (task.get(k) for k in ('current_state', 'pending', 'selected'))
    if not all(isinstance(v, dict) for v in (state, pending, selected)):
        raise RuntimeError('resume requires replayed current state and pending selected action')
    node = pending.get('node')
    poses = task['poses']
    if type(node) is not int or not 0 <= node < len(poses):
        raise RuntimeError('resume pending graph node is invalid')
    action, destination = pending.get('action'), poses[node]
    if (action not in ('forward', 'left', 'right') or type(state.get('step')) is not int
            or selected.get('action') != action or selected.get('node') != node
            or selected.get('step') != state['step']
            or selected.get('from_node') != state.get('node')
            or selected.get('pose') != destination):
        raise RuntimeError('resume pending action differs from replayed selection')
    previous = prior.get('nso_actions', [])
    completed = previous[-1] if previous else {}
    execution_required = (completed.get('step') != state['step']+1
        or completed.get('arrival', {}).get('verified') is not True
        or completed.get('reference_error', {}).get('verified') is not True)
    execution = completed
    if execution_required:
        events_path = attempt / 'events.jsonl'
        if not events_path.is_file() or prior.get('execution_interrupted') is not True:
            raise RuntimeError('resume executed action differs: no interrupted issued-goal evidence')
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        issued = [row for row in events if row.get('event') == 'NSO_reference_goal_sent']
        if not issued:
            raise RuntimeError('resume has no original issued pending goal')
        execution = issued[-1]
        if any(row.get('event') == 'NSO_reference_observation_accepted'
               and row.get('step', -1) >= state['step']+1 for row in events):
            raise RuntimeError('resume pending action already has a later observation')
        # Replaying the log must end at this very selection, without a newer
        # observation or a second, different controller decision after it.
        rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        successful = [row for row in rows if row.get('response', {}).get('ok') is True
                      and row.get('request', {}).get('method') in ('observe', 'next_action')]
        if not successful or successful[-1]['request']['method'] != 'next_action':
            raise RuntimeError('resume original pending decision is not the last successful RPC')
        decision = successful[-1]['response']['result']
        if decision.get('pending') != pending or decision.get('selected') != selected:
            raise RuntimeError('resume issued action differs from last logged decision')
        goal = execution.get('goal', {})
        if not isinstance(goal.get('id'), str) or not goal['id']:
            raise RuntimeError('resume original issued goal ID is missing')
    if (execution.get('action') != action or execution.get('step') != state['step']+1
            or execution.get('graph_pose') != destination or execution.get('binding') != task['binding']):
        raise RuntimeError('resume executed action differs from pending action or fixed binding')
    expected = graph_pose_to_odom(destination, task['binding'])
    if execution.get('expected_odom_pose') != expected:
        raise RuntimeError('resume executed endpoint differs from fixed task graph')
    receipt = dict(source_attempt=str(attempt), source_result_sha256=hashlib.sha256(
        (attempt / 'result.json').read_bytes()).hexdigest(),
        source_log=str(log_path), action=action, step=state['step']+1,
        graph_pose=destination, execution=execution, replayed_state=state,
        execution_required=execution_required, previous_goal=execution.get('goal', {}).get('id'),
        previous_failed_execution=completed if completed.get('step') == state['step']+1
            and execution_required else None)
    if execution_required:
        receipt['source_events_sha256'] = hashlib.sha256(events_path.read_bytes()).hexdigest()
        return receipt
    arrival, reference = execution.get('arrival', {}), execution.get('reference_error', {})
    if arrival.get('verified') is not True or reference.get('verified') is not True:
        raise RuntimeError('resume prior physical arrival was not verified')
    try:
        measured = arrival_verifier(*(arrival[k] for k in
            ('goal', 'terminal', 'before', 'after', 'can_start', 'can_end')))
        endpoint = reference_pose_error(expected, execution['actual_odom_pose']['pose'],
                                        xy_tolerance, yaw_tolerance)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError('resume physical arrival evidence is incomplete') from error
    if not measured['verified'] or not endpoint['verified']:
        raise RuntimeError('resume prior CAN or measured endpoint evidence does not verify')
    return receipt


def run_nso(ctx=None):
    """Execute only the next primitive issued by an initialized reference core.

    Mission owns process lifetime, source health, the actuator and termination.
    The task loader owns models, calibration and graph validity. No candidate
    sampling, route-cost estimate, alternative action or method stop lives here.
    """
    task = ensure_nso_runtime_ready(ctx.get('task_receipt') if isinstance(ctx, dict) else None)
    import rospy
    from move_base_msgs.msg import MoveBaseGoal
    from mission import arrival_evidence

    worker, client = ctx['worker'], ctx['client']
    poses, binding = task['poses'], task['binding']
    xy_tolerance, yaw_tolerance = ctx['xy_tolerance_m'], ctx['yaw_tolerance_rad']
    actions = []
    ctx['result']['nso_actions'] = actions
    ctx['result']['nso_task'] = task['task']

    def record(event, **fields):
        ctx['journal'].add(dict(event=event, **fields))

    def heartbeat():
        ctx['check'](True)
        ctx['progress']()
        # A supervisor time limit interrupts execution; it is not core budget.
        if time.monotonic()-ctx['run_start'] >= ctx['duration']:
            raise RuntimeError('external_execution_duration_complete')

    def call(method, params):
        response = worker.call(method, params, heartbeat)
        record('NSO_reference_rpc', method=method, params=params, response=response)
        if 'summary' in response:
            ctx['result']['nso_summary'] = response['summary']
        return response

    def pose_check(graph_pose, measured):
        return reference_pose_error(graph_pose_to_odom(graph_pose, binding), measured['pose'],
                                    xy_tolerance, yaw_tolerance)

    def fail(reason, evidence):
        record('NSO_reference_execution_failure', reason=reason, evidence=evidence,
               core_observation_accepted=False)
        ctx['result'].update(status='navigation_execution_failed', execution_interrupted=True,
                             reason=reason, method_stop=False)

    def observe(step, graph_pose, action):
        heartbeat()
        before = ctx['capture_odom_pose']()
        before_error = pose_check(graph_pose, before)
        if not before_error['verified']:
            fail('reference_pose_not_reached_before_observation', dict(pose=before, error=before_error))
            return None
        path = ctx['capture_observation'](task['calibration'], step, list(graph_pose), action)
        after = ctx['capture_odom_pose']()
        after_error = pose_check(graph_pose, after)
        receipt = dict(step=step, graph_pose=list(graph_pose), action=action, path=str(path),
                       before=before, after=after, before_error=before_error, after_error=after_error)
        if not after_error['verified']:
            fail('reference_pose_changed_during_observation', receipt)
            return None
        outcome = call('observe', dict(path=str(path), step=step, pose=list(graph_pose), action=action))
        record('NSO_reference_observation_accepted', **receipt, state=outcome['state'],
               sensor_sha256=outcome['sensor_sha256'])
        return outcome['state']

    # A task binding is fixed before execution. Do not assign arbitrary current
    # odometry to the anchor or reset alignment after a navigation error.
    resume_pending_decision = None
    resumed = None
    if ctx.get('resume_attempt') is not None:
        resumed = verified_pending_resume(task, ctx['resume_attempt'], xy_tolerance,
                                          yaw_tolerance, arrival_evidence)
        ctx['result']['nso_resume'] = resumed
        if resumed['execution_required']:
            record('NSO_reference_pending_execution_resume', **resumed,
                   execution_resumed=True, next_action_repeated=False)
            state = task['current_state']
            resume_pending_decision = dict(action=task['pending']['action'],
                pending=task['pending'], selected=task['selected'])
        else:
            record('NSO_reference_pending_arrival_resume', **resumed,
                   motion_repeated=False, next_action_repeated=False)
            state = observe(resumed['step'], resumed['graph_pose'], resumed['action'])
    else:
        state = observe(0, poses[task['anchor']], None)
    if state is None:
        return
    while True:
        heartbeat()
        execution_resumed = resume_pending_decision is not None
        decision = resume_pending_decision if execution_resumed else call('next_action', {})
        resume_pending_decision = None
        action = decision['action']
        if action is None:
            ctx['result'].update(status='nso_core_stopped', method_stop=True)
            return
        selected, pending = decision['selected'], decision['pending']
        if (action not in ('forward', 'left', 'right') or not isinstance(pending, dict)
                or pending.get('action') != action or selected.get('action') != action
                or selected.get('step') != state['step'] or selected.get('from_node') != state['node']
                or selected.get('node') != pending.get('node')):
            raise RuntimeError('reference worker action receipt is inconsistent')
        destination = poses[pending['node']]
        if list(destination) != selected['pose']:
            raise RuntimeError('reference worker destination differs from its public graph')
        expected_odom = graph_pose_to_odom(destination, binding)
        previous_projection_goal = None
        while True:
            heartbeat()
            source_pose = ctx['capture_odom_pose']()
            if type(source_pose.get('source_stamp_ns')) is int:
                seconds, nanoseconds = divmod(source_pose['source_stamp_ns'], 10**9)
                query = rospy.Time(seconds, nanoseconds)
            else:
                query = rospy.Time.from_sec(source_pose['source_stamp_s'])
            tf = ctx['tfbuf'].lookup_transform('map', 'odom', query, rospy.Duration(.3))
            transform = matrix(tf.transform)
            target = transform @ [expected_odom[0], expected_odom[1], 0., 1.]
            map_yaw = wrap(expected_odom[2]+math.atan2(transform[1, 0], transform[0, 0]))
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = 'map'
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = float(target[0])
            goal.target_pose.pose.position.y = float(target[1])
            goal.target_pose.pose.orientation.z = math.sin(map_yaw/2)
            goal.target_pose.pose.orientation.w = math.cos(map_yaw/2)
            before, can_start = ctx['capture_pose'](), ctx['can_snapshot']()
            client.send_goal(goal)
            goal_id = client.gh.comm_state_machine.action_goal.goal_id.id
            sent = dict(id=goal_id, frame='map', x=float(target[0]), y=float(target[1]), yaw=map_yaw)
            tf_record = dict(query_stamp_ns=query.to_nsec(), response_stamp_ns=tf.header.stamp.to_nsec(),
                             target_frame=tf.header.frame_id, source_frame=tf.child_frame_id,
                             translation_m=[tf.transform.translation.x, tf.transform.translation.y,
                                            tf.transform.translation.z],
                             rotation_xyzw=[tf.transform.rotation.x, tf.transform.rotation.y,
                                            tf.transform.rotation.z, tf.transform.rotation.w],
                             matrix=transform.tolist(),
                             timestamp_semantics='TF lookup response at requested odometry time; may be interpolated')
            execution = dict(action=action, step=state['step']+1, graph_pose=list(destination),
                             expected_odom_pose=expected_odom, binding=binding,
                             map_from_odom=tf_record, source_odom_pose=source_pose, goal=sent)
            if execution_resumed:
                execution.update(execution_resumed=True, previous_goal=resumed['previous_goal'])
            if previous_projection_goal is not None:
                execution.update(fixed_endpoint_reprojected=True, previous_goal=previous_projection_goal)
            record('NSO_reference_goal_sent', **execution)
            while not client.wait_for_result(rospy.Duration(.05)):
                heartbeat()
            terminal = dict(id=goal_id, status=client.get_state())
            after, actual_odom, can_end = ctx['capture_pose'](), ctx['capture_odom_pose'](), ctx['can_snapshot']()
            evidence = arrival_evidence(sent, terminal, before, after, can_start, can_end)
            reference_error = pose_check(destination, actual_odom)
            execution.update(arrival=evidence, actual_odom_pose=actual_odom, reference_error=reference_error)
            actions.append(execution)
            record('NSO_reference_action_result', **execution)
            if evidence['verified'] and reference_error['verified']:
                break
            if terminal['status'] == 3 and evidence['verified'] and not reference_error['verified']:
                # Finish the same fixed odom endpoint using the current map transform.
                # This is execution-frame correction; no new core action is selected.
                record('NSO_reference_fixed_endpoint_reprojection', step=state['step']+1,
                       action=action, graph_pose=list(destination), expected_odom_pose=expected_odom,
                       previous_goal=goal_id, reference_error=reference_error,
                       core_observation_accepted=False)
                previous_projection_goal = goal_id
                continue
            fail('ROS_action_or_reference_endpoint_not_verified', execution)
            return
        # No observation is credited for an aborted/unreached action. A failed
        # navigation is not a sensed collision and cannot select another edge.
        state = observe(state['step']+1, destination, action)
        if state is None:
            return


def matrix(transform):
    import numpy as np
    t, q = transform.translation, transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = [t.x, t.y, t.z]
    return result
