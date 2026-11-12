"""Viewpoint-selection research policy with ROS-owned continuous navigation.

Original candidate scores/geometry are reused. Nominal grid route cost is a
forecast only; each actual terminal waypoint attempt acquires one real frame.
"""
from copy import deepcopy
import math
import numpy as np
from nso_bridge.joint_engine import JointSession
from nso_bridge.ros_checked_exploration_v2 import ROSCheckedCPUFourModules, identity as admission_identity

PROFILE='ros_viewpoint_exploration_v3'

class HorizonLedgerView:
    def __init__(self,ledger,horizon):self._ledger=ledger;self.remaining_budget=horizon
    def __getattr__(self,name):return getattr(self._ledger,name)

class ViewpointCPU(ROSCheckedCPUFourModules):
    policy_variant=PROFILE
    def select_target(self,scene_idx):
        s=self.scenes[scene_idx];actual=s['ledger']
        s['ledger']=HorizonLedgerView(actual,self.nominal_horizon)
        try:return super().select_target(scene_idx)
        finally:s['ledger']=actual
    def start_scene(self,scene_idx,**kwargs):
        super().start_scene(scene_idx,**kwargs)
        self.scenes[scene_idx]['feedback']['accounting_unit']='actual_waypoint_attempt'
    def compute_reward(self,scene_idx):
        s=self.scenes[scene_idx];p=s['packet']
        event=s['ledger'].observe(s['mapper'],frame=p.frame,frame_id=p.frame_id,
            action_id=p.action_id,map_version=s['mapper'].frames,action_cost=1,
            collision=p.collision,arrived=False,done=False,failed=False,
            scene_id=p.scene_id,episode_id=p.episode_id)
        event.update(accounting_unit='actual_waypoint_attempt',primitive_action_credit=False,
            primitive_gain_feedback='disabled; macro is not a forward/turn primitive')
        s['feedback']=event;s['planning_direction_support']=deepcopy(s['current_direction_support'])
        self._record(scene_idx,'IGCR','observe_waypoint_result',dict(attempt_id=p.action_id),event)
        return event['reward'],event['parts']

class ViewpointSession(JointSession):
    def __init__(self,configuration,**kwargs):
        c=deepcopy(configuration)
        if c.get('policy_variant')!=PROFILE:raise ValueError('explicit viewpoint profile required')
        horizon=c.pop('nominal_planning_horizon',40)
        if type(horizon) is not int or horizon<1:raise ValueError('positive integer nominal grid horizon required')
        max_attempts=c.pop('max_viewpoint_attempts',c['total_budget'])
        if type(max_attempts) is not int or not 1<=max_attempts<=40:raise ValueError('waypoint attempts must be1..40')
        c['total_budget']=max_attempts;c['policy_variant']='ros_checked_exploration_v2'
        super().__init__(c,**kwargs)
        self.policy_variant=PROFILE;self.configuration['policy_variant']=PROFILE
        self.configuration.update(nominal_planning_horizon=horizon,max_viewpoint_attempts=max_attempts)
        self.backend=ViewpointCPU(self.args,1,self.shape);self.backend.nominal_horizon=horizon
        self.components._cpu_backend=self.backend
        self.nominal_horizon=horizon;self.max_attempts=max_attempts
        self.viewpoint_pending=None;self.waypoint_outcomes=[];self._waypoint_result=None
    def identity(self):
        result=super().identity()
        result['receipt']['viewpoint_execution']=dict(profile=PROFILE,original_method_equivalence=False,
            research_policy_changed=True,admission={**{k:v for k,v in admission_identity().items() if k!='no_option_fallback'},
                'no_option_fallback':'v3 best-effort anchor viewpoint only; no forward/left/right primitive fallback executed'},
            actual_resource='external600s episode plus at most40 actual waypoint attempts',
            max_viewpoint_attempts=self.max_attempts,nominal_planning_horizon=self.nominal_horizon,
            nominal_grid_cost='internal prediction units only, not physical paid primitives',
            actual_observations='one fresh acquired packet after each goal completion/failure; no synthetic intermediate frames',
            primitive_gain_calibration='disabled, no macro-as-forward or turn feedback',
            scorer='unchanged inherited NSO candidate score formulas',
            candidate_pool_change='none beyond declared inherited v2 admission; exact original chooser output reused, failed goals only recorded',
            return_request='explicit caller return_requested or original no-positive-option fallback; no automatic time-margin rule or guaranteed return',
            recovery_cost='separate interventions; actual CAN distance/time include their motion')
        result['pending']=deepcopy(self.viewpoint_pending)
        return result
    def summary(self):
        result=super().summary();result.pop('actual_paid_actions',None);result.pop('total_budget',None)
        result.update(accounting_unit='actual_waypoint_attempt',waypoint_attempts=len(self.waypoint_outcomes),
            waypoint_ROS_successes=sum(x['ros_status']==3 for x in self.waypoint_outcomes),
            waypoint_successes=sum(x['ros_status']==3 for x in self.waypoint_outcomes),
            waypoint_success_definition='actual ROS terminal status3; measured pose retained, no cardinal-grid acceptance gate',
            max_viewpoint_attempts=self.max_attempts,nominal_planning_horizon=self.nominal_horizon,
            primitive_arrivals_credited=0,primitive_gain_feedback_disabled=True,
            legacy_ledger_field_semantics='internal action/paid counter represents waypoint attempts, not grid primitives')
        return result
    def next_action(self):raise ValueError('viewpoint profile requires next_viewpoint RPC; no primitive execution')
    def _goal(self,pose,selected,role):
        x,y=self.transform.cell_to_world(tuple(pose[:2]));yaw=math.pi/2-pose[2]*math.pi/2
        inv=np.linalg.inv(self.planning_from_odom);point=inv@[x,y,0.,1.]
        yaw+=math.atan2(inv[1,0],inv[0,0])
        return dict(attempt_id=len(self.waypoint_outcomes)+1,pose=list(pose),
            target_odom_pose=[float(point[0]),float(point[1]),math.atan2(math.sin(yaw),math.cos(yaw))],
            selected=deepcopy(selected),goal_role=role,
            predicted_nominal_grid_cost=None if selected is None else selected['cost'],
            predicted_nominal_outbound_cost=None if selected is None else selected['outbound_cost'],
            cost_unit='predicted_nominal_grid_motion_only',ROS_owns_path=True)
    def next_viewpoint(self,remaining_time_s=None,return_requested=False):
        if self.last_packet is None:raise ValueError('bootstrap actual observation first')
        if self.runtime.states[0]['closed']:return dict(target=None,summary=self.summary())
        if self.viewpoint_pending is not None:raise ValueError('observe terminal waypoint result before next selection')
        if remaining_time_s is not None and (not isinstance(remaining_time_s,(int,float)) or not math.isfinite(remaining_time_s)):
            raise ValueError('finite remaining time required')
        if len(self.waypoint_outcomes)>=self.max_attempts or (remaining_time_s is not None and remaining_time_s<=0):
            reason='waypoint_attempt_budget_exhausted' if len(self.waypoint_outcomes)>=self.max_attempts else 'external_time_budget_exhausted'
            self.runtime.close_sensor_episode(0,reason=reason)
            return dict(target=None,summary=self.summary())
        if self.runtime.states[0]['closed']:return dict(target=None,summary=self.summary())
        selected=None
        if not return_requested:
            self.backend.select_target(0)
            selected=deepcopy(self.backend.scenes[0].get('selected'))
        if selected is None:
            if tuple((*self.last_packet.position,self.last_packet.heading))==self.anchor:
                self.runtime.close_sensor_episode(0,reason='no_positive_viewpoint_at_anchor')
                return dict(target=None,summary=self.summary())
            target=self._goal(self.anchor,None,'best_effort_return')
        else:target=self._goal(selected['pose'],selected,'NSO_selected_viewpoint')
        self.viewpoint_pending=target
        self.runtime.audit.append(dict(event='viewpoint_goal_selected',target=deepcopy(target),
            nominal_horizon=self.nominal_horizon,remaining_time_s=remaining_time_s))
        return dict(target=deepcopy(target),selection=self.backend.scenes[0].get('last_selection'),summary=self.summary())
    def observe_viewpoint_result(self,path,attempt_id,ros_status=None,reason=None):
        if self._prefix_failed or (self.runtime.states[0] is not None and self.runtime.states[0]['closed']):
            raise ValueError('waypoint worker closed; create a fresh session, no partial-fusion retry')
        if attempt_id==0:
            if self.last_packet is not None:raise ValueError('bootstrap already acquired')
            if ros_status is not None:raise ValueError('bootstrap has no ROS terminal status')
            result=self.observe(path,0)
            return {**result,'outcome':dict(attempt_id=0,bootstrap=True,waypoint_attempt_charged=False)}
        pending=self.viewpoint_pending
        if pending is None or type(attempt_id) is not int or attempt_id!=pending['attempt_id']:
            raise ValueError('one exact pending waypoint attempt required')
        if type(ros_status) is not int or ros_status not in (2,3,4,5,8,9):
            raise ValueError('actual terminal ROS status required')
        self._waypoint_result=dict(attempt_id=attempt_id,ros_status=ros_status,reason=reason,
            target=deepcopy(pending),waypoint_attempt_charged=True,primitive_arrival_credit=False)
        try:result=self.observe(path,attempt_id,action=None)
        finally:self._waypoint_result=None
        return {**result,'outcome':deepcopy(self.waypoint_outcomes[-1])}
    def _fuse_packet(self,packet,frame,scan,action_id,action,collision,done):
        if self.last_packet is None:
            try:return super()._fuse_packet(packet,frame,scan,action_id,action,collision,done)
            except Exception:
                self._prefix_failed=True  # Bootstrap fusion is also nontransactional.
                if self.runtime.states[0] is not None:self.runtime.states[0]['closed']=True
                raise
        if self._waypoint_result is None:raise ValueError('macro observations require terminal waypoint-result RPC')
        state=self.runtime.states[0];previous=self.last_packet
        if packet.action_id!=previous.action_id+1 or packet.frame_id in state['packet_hashes']:
            raise ValueError('waypoint observation must be fresh and sequential')
        if packet.frame.timestamp_s<=previous.frame.timestamp_s:raise ValueError('waypoint frame timestamp must advance')
        if state['closed']:raise ValueError('waypoint episode already closed')
        try:
            self.mapper.update(frame,scan)
            state['packet']=packet;state['packet_hashes'][packet.frame_id]=packet.sha256()
            state['action_frames'][packet.action_id]=packet.frame_id
            self.backend.bind_packet(0,packet,arrived=False)
            self.runtime._update_sensor_modules(0)
        except Exception:
            # Fusion/feedback is not transactional. Retain the partial state,
            # poison this worker, and never let an RPC retry credit it twice.
            state['closed']=True
            state['termination']=dict(reason='waypoint_fusion_or_feedback_exception',failed=True,
                returned_to_anchor=False,final_frame_id=previous.frame_id,
                attempted_frame_id=packet.frame_id,final_action_id=previous.action_id,
                attempted_waypoint_id=packet.action_id,rollback_performed=False)
            self.runtime.audit.append(dict(event='failed_during_waypoint_fusion',**state['termination']))
            raise
        actual=[*packet.position,packet.heading];outcome=deepcopy(self._waypoint_result)
        outcome.update(actual_discrete_pose=actual,actual_base_pose_planning=list(packet.base_pose_planning),
            ROS_success=outcome['ros_status']==3,actual_goal_pose_verified=actual==outcome['target']['pose'])
        outcome['candidate_exclusion_applied']=False
        self.waypoint_outcomes.append(outcome);self.viewpoint_pending=None
        self.runtime.audit.append(dict(event='actual_waypoint_result_observed',**outcome))
        self.last_packet=packet
        if self._attempt_semantic_receipt is not None:self.semantic_receipt=self._attempt_semantic_receipt
        return dict(accepted=True,duplicate=False,state=self._state(),summary=self.summary())
