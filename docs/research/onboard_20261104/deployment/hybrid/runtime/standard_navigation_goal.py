"""One explicit standard ROS setup goal; never credited as an NSO action."""
import math


def validate_goal_odom(value):
    try:
        if len(value) != 3:
            raise ValueError('navigation goal requires three finite odom coordinates')
        goal = [float(v) for v in value]
        if not all(math.isfinite(v) for v in goal):
            raise ValueError('navigation goal requires three finite odom coordinates')
    except (TypeError, OverflowError) as exc:
        raise ValueError('navigation goal requires three finite odom coordinates') from exc
    return goal


def run_standard_navigation_goal(ctx):
    import rospy
    from move_base_msgs.msg import MoveBaseGoal
    from mission import arrival_evidence

    target = validate_goal_odom(ctx['goal_odom'])
    result, client = ctx['result'], ctx['client']
    result['status'] = 'standard_navigation_setup_running'
    scope = dict(purpose='experimental_setup_reset', NSO_action=False,
                 credited_to_NSO=False, execution_frame='odom')
    ctx['heartbeat']()
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'odom'
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x, goal.target_pose.pose.position.y = target[:2]
    goal.target_pose.pose.orientation.z = math.sin(target[2] / 2)
    goal.target_pose.pose.orientation.w = math.cos(target[2] / 2)
    before, can_start = ctx['capture_odom_pose'](), ctx['can_snapshot']()
    client.send_goal(goal)
    goal_id = client.gh.comm_state_machine.action_goal.goal_id.id
    sent = dict(id=goal_id, frame='odom', x=target[0], y=target[1], yaw=target[2])
    ctx['journal'].add(dict(event='standard_navigation_setup_goal_sent', goal=sent, **scope))
    while not client.wait_for_result(rospy.Duration(.05)):
        ctx['heartbeat']()
    ctx['heartbeat']()
    terminal = dict(id=goal_id, status=client.get_state())
    after, can_end = ctx['capture_odom_pose'](), ctx['can_snapshot']()
    evidence = arrival_evidence(sent, terminal, before, after, can_start, can_end,
                                expected_frame='odom')
    evidence.update(scope)
    result['standard_navigation_setup_evidence'] = evidence
    result['status'] = ('standard_navigation_setup_goal_reached' if evidence['verified']
                        else 'standard_navigation_setup_goal_failed')
    ctx['journal'].add(dict(event='standard_navigation_setup_goal_completed',
                            status=result['status'], evidence=evidence, **scope))
    return evidence
