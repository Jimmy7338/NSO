"""Declared original-primitive feedback pilot through existing ROS navigation.

The optional pilot count ends an integration experiment, not the NSO method.
No finite policy budget, return rule, candidate filter or new obstacle gate.
"""
import math,time
from joint_ros_coordinator import validate_joint_identity,metric_goal_to_map
from nso_ros_coordinator import matrix,reference_pose_error
PROFILE='primitive_feedback_unbounded_v1'

def run_primitive_feedback(ctx):
    identity=validate_joint_identity(ctx['task_receipt'])
    if identity['configuration'].get('policy_variant')!=PROFILE:raise ValueError('explicit primitive feedback identity required')
    if ctx.get('duration')is not None:raise ValueError('primitive feedback profile has no episode time budget')
    pilot=ctx.get('primitive_feedback_pilot_attempts')
    if pilot is not None and (type(pilot)is not int or pilot<1):raise ValueError('positive explicit experiment-harness count required')
    import rospy
    from move_base_msgs.msg import MoveBaseGoal
    worker,client=ctx['worker'],ctx['client'];result=ctx['result'];attempts=[]
    result.update(primitive_feedback_attempts=attempts,primitive_feedback_profile=PROFILE,primitive_feedback_pilot_attempt_limit=pilot,pilot_limit_is_method_budget=False,primitive_feedback_original_gain_enabled=True,full_optimized_geometry_replay=False)
    def record(event,**fields):ctx['journal'].add(dict(event=event,**fields))
    def heartbeat():ctx['check'](True);ctx['progress']()
    def call(method,params):
        response=worker.call(method,params,heartbeat);record('NSO_primitive_feedback_rpc',method=method,params=params,response=response)
        if 'summary'in response:result['joint_summary']=response['summary']
        return response
    def observation(action_id,status,reason):
        heartbeat();path=ctx['capture_observation'](identity['calibration'],action_id,None,None)
        response=call('observe_primitive_result',dict(path=str(path),action_id=action_id,ros_status=status,reason=reason))
        record('NSO_primitive_feedback_actual_observation',action_id=action_id,path=str(path),ros_status=status,reason=reason,state=response.get('state'),primitive_outcome=response.get('primitive_outcome'),synthetic_observations=0)
        return response
    observation(0,None,'bootstrap_actual_observation')
    last=0
    while True:
        heartbeat();decision=call('next_action',{});pending=decision.get('pending')
        if pending is None:
            if decision.get('action')is not None:raise RuntimeError('primitive action lacks pending endpoint')
            result.update(status='nso_primitive_feedback_core_stopped',method_stop=True,reason='no_positive_current_option');return
        if pending.get('target_world_frame')!='odom':raise ValueError('pilot must declare raw odom endpoint')
        action_id=pending.get('action_id');raw_target=pending.get('target_odom_pose')
        if type(action_id)is not int or action_id!=last+1 or pending.get('action')not in ('forward','left','right'):raise ValueError('one exact original pending primitive required')
        if not isinstance(raw_target,list)or len(raw_target)!=3 or not all(type(x)in(int,float)and math.isfinite(x)for x in raw_target):raise ValueError('finite actual raw-odom goal required')
        source=ctx['capture_odom_pose']();frame=ctx.get('execution_frame','odom');projection=None
        target=list(raw_target)
        if frame!='odom':
            stamp=source.get('source_stamp_ns')
            query=rospy.Time(*divmod(stamp,10**9))if type(stamp)is int else rospy.Time.from_sec(source['source_stamp_s'])
            transform=matrix(ctx['tfbuf'].lookup_transform(frame,'odom',query,rospy.Duration(.3)).transform)
            target=metric_goal_to_map(raw_target,transform);projection=dict(matrix=transform.tolist(),query_stamp_ns=query.to_nsec(),source_frame='odom',target_frame=frame)
        goal=MoveBaseGoal();goal.target_pose.header.frame_id=frame;goal.target_pose.header.stamp=rospy.Time.now()
        goal.target_pose.pose.position.x,goal.target_pose.pose.position.y=target[:2]
        goal.target_pose.pose.orientation.z=math.sin(target[2]/2);goal.target_pose.pose.orientation.w=math.cos(target[2]/2)
        started=time.monotonic();can_before=ctx['can_snapshot']();client.send_goal(goal)
        goal_id=client.gh.comm_state_machine.action_goal.goal_id.id
        execution=dict(action_id=action_id,action=pending['action'],goal=dict(id=goal_id,frame=frame,x=target[0],y=target[1],yaw=target[2]),raw_odom_target=raw_target,original_pending=pending,actual_raw_odom_before=source,CAN_before=can_before,started_monotonic=started,map_projection=projection)
        record('NSO_primitive_feedback_goal_sent',**execution)
        try:
            while not client.wait_for_result(rospy.Duration(.05)):heartbeat()
            heartbeat()
        except BaseException:
            client.cancel_goal();raise
        status=client.get_state()
        if status not in (2,3,4,5,8,9):raise RuntimeError('actual terminal ROS status required')
        after=ctx['capture_odom_pose']();reason='ROS_terminal_status_'+str(status)
        execution.update(terminal=dict(id=goal_id,status=status),actual_raw_odom_after=after,CAN_after=ctx['can_snapshot'](),elapsed_s=time.monotonic()-started,metric_error_diagnostic=reference_pose_error(raw_target,after['pose'],ctx['xy_tolerance_m'],ctx['yaw_tolerance_rad']))
        outcome=observation(action_id,status,reason);execution['primitive_outcome']=outcome.get('primitive_outcome')
        attempts.append(execution);record('NSO_primitive_feedback_goal_result',**execution);last=action_id
        result['primitive_verified_arrivals']=sum(x.get('primitive_outcome',{}).get('primitive_arrival_verified',False)for x in attempts)
        result['macro_verified_arrivals']=(outcome.get('summary')or{}).get('macro_arrivals_verified',0)
        if pilot is not None and last>=pilot:
            result.update(status='primitive_feedback_pilot_complete',reason='explicit_integration_experiment_harness_count_complete',method_stop=False,pilot_complete=True,full_exploration_complete=False,full_macro_success_claimed=False)
            return
