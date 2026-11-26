"""Explicit user-authorized research admission variant; original vendor unchanged.

Unknown cells admit planning, never measured free-space credit. ROS checks the
actual configured footprint. Research return paths are best effort, not certificates.
"""
import numpy as np
from scipy.sparse.csgraph import dijkstra
from nso.execution_guard_v8 import ObservedExecutionGuard, ActionAssessment, ReturnPlan
from nso.route_coverage_v2 import orientation_graph, recover_actions
from utils.grid_geometry import DIRECTIONS, inflated_obstacles
from nso_bridge.exploration_joint_policy import ExplorationJointCPUFourModules, ExplorationJointRuntimeIntegration

PROFILE='ros_checked_exploration_v2'

def identity():
    return dict(profile=PROFILE, research_policy_changed=True, original_method_equivalence=False,
        unknown_navigation='traversable in research graph; never relabelled free',
        research_collision_admission='center-cell not-known-occupied; research circular-footprint admission removed; actual existing ROS rectangle remains authoritative',
        current_start_exception='current cell virtually admitted for rotation/escape; successor translation cannot enter known occupied center',
        return_contract='best effort through not-known-occupied graph; optimistic return reserved when reachable; best effort without known-free certificate when unreachable',
        radius_changed=False, scorer_changed=False, object_semantics_changed=False,
        no_option_fallback='budgeted best-effort return or forward/left/right admissible primitive; explicitly unscored',
        measurement_or_occupancy_clearing=False, ROS_and_hardware_protection='unchanged outside research layer',
        evidence_scope='modified-policy integration; cannot attribute outcomes to original NSO')

class ROSCheckedGuard(ObservedExecutionGuard):
    current=None
    def safe_grid(self,belief):
        belief=np.asarray(belief)
        if belief.ndim!=2 or not np.isin(belief,[-1,0,1]).all():raise ValueError('invalid measured belief')
        safe=(belief!=1).copy()
        if self.current is not None and self._inside(safe,self.current):safe[self.current]=True
        return safe
    def assess(self,belief,position,heading,action,remaining_actions):
        self._heading(heading);position=tuple(position);target=position
        if action not in ('forward','left','right'):raise ValueError('unknown paid primitive')
        if action=='forward':
            dr,dc=DIRECTIONS[heading];target=(position[0]+int(dr),position[1]+int(dc))
        if remaining_actions<1:return ActionAssessment(False,'no_action_budget',target)
        raw=np.asarray(belief)
        if not self._inside(raw,position) or not self._inside(raw,target):return ActionAssessment(False,'outside_resource_grid',target)
        if target!=position and raw[target]==1:return ActionAssessment(False,'next_center_known_occupied',target)
        return ActionAssessment(True,'ROS_checked_unknown_admission',target)
    def return_plan(self,belief,position,heading,anchor,remaining_actions):
        self._heading(heading);position=tuple(position);goal=tuple(anchor[:2]);safe=self.safe_grid(belief)
        if not self._inside(safe,position) or not self._inside(safe,goal) or not safe[goal]:
            return ReturnPlan(False,'best_effort_anchor_not_admitted',(),0)
        safe[position]=True
        graph,cells,ids=orientation_graph(safe);start=int(ids[position])*4+heading;end=int(ids[goal])*4+anchor[2]
        costs,pred=dijkstra(graph,directed=True,indices=start,return_predecessors=True)
        if not np.isfinite(costs[end]):return ReturnPlan(False,'best_effort_anchor_disconnected',(),0)
        actions=tuple(recover_actions(pred,start,end,cells));cost=len(actions)
        # Never certify a return or exceed the remaining paid budget.
        bounded=actions[:max(0,int(remaining_actions))]
        return ReturnPlan(bool(bounded) or start==end,'best_effort_return_not_certified',bounded,cost)

class ROSCheckedCPUFourModules(ExplorationJointCPUFourModules):
    policy_variant=PROFILE
    admission_profile=PROFILE
    def start_scene(self,scene_idx,**kwargs):
        super().start_scene(scene_idx,**kwargs);s=self.scenes[scene_idx]
        s['guard']=ROSCheckedGuard(s['mapper'].config.resolution_m,s['mapper'].config.robot_radius_m)
        s['guard'].current=tuple(s['packet'].position)
    def update_topo(self,scene_idx):
        s=self.scenes[scene_idx];s['guard'].current=tuple(s['packet'].position)
        return super().update_topo(scene_idx)
    def assess_action(self,scene_idx,action):
        s=self.scenes[scene_idx];p=s['packet'];budget=s['ledger'].remaining_budget
        check=s['guard'].assess(s['mapper'].belief,p.position,p.heading,action,budget)
        heading=(p.heading+(1 if action=='right' else -1 if action=='left' else 0))%4
        predicted=None
        if check.allowed:
            predicted=s['gain'].prepare_action(s['mapper'],position=p.position,heading=p.heading,
                action=action,action_id=p.action_id+1,planning_camera_poses=s['ledger'].planning_camera_poses())
        result=dict(allowed=bool(check.allowed),reason=check.reason,action=action,next_pose=[*check.target,heading],
            remaining_budget=budget,reserved_return_cost=None,return_certificate=False,
            admission_profile=PROFILE,learned_probability=None,calibrated_uncertainty=None,
            observed_gain_prediction=predicted)
        self._record(scene_idx,'RPN-UQ','assess_local_action',dict(anchor=s['return_anchor'],remaining_budget=budget),result)
        return result

class ROSCheckedRuntimeIntegration(ExplorationJointRuntimeIntegration):
    policy_variant=PROFILE
    def next_local_action(self,scene_idx):
        s=self.states[scene_idx]
        if s is None or s['closed']:return None
        if s['pending'] is not None:raise RuntimeError('observe pending attempt before replanning')
        b=self.components._cpu_backend.scenes[scene_idx];p=s['packet']
        if b['ledger'].remaining_budget==0:
            self.close_sensor_episode(scene_idx,reason='budget_exhausted');return None
        if not s['active_actions']:
            self.choose_goal(scene_idx,list(p.position),(0,self.full_shape[0],0,self.full_shape[1]))
        action=s['active_actions'][0] if s['active_actions'] else None
        check=None if action is None else b['guard'].assess(s['mapper'].belief,p.position,p.heading,action,b['ledger'].remaining_budget)
        if check is not None and not check.allowed:
            s['active_actions']=[];s['option']=None
            self.choose_goal(scene_idx,list(p.position),(0,self.full_shape[0],0,self.full_shape[1]))
            self.audit.append(dict(event='ros_checked_replan_after_rejection',reason=check.reason,frame_id=p.frame_id))
            action=s['active_actions'][0] if s['active_actions'] else None
        if action is None:
            returning=self.components.plan_observed_return(scene_idx)
            if returning.available and returning.actions:
                s['active_actions']=list(returning.actions);s['option']=None;s['phase']='return_best_effort'
                action=s['active_actions'][0]
            else:
                # Explicit finite-budget escape/search attempt, never a sensor
                # free-space assertion or a claimed scored original option.
                action=next((a for a in ('forward','left','right') if b['guard'].assess(
                    s['mapper'].belief,p.position,p.heading,a,b['ledger'].remaining_budget).allowed),None)
                if action is None:
                    self.close_sensor_episode(scene_idx,reason='outside_resource_grid');return None
                s['option']=None;s['active_actions']=[action];s['phase']='ROS_best_effort'
                self.audit.append(dict(event='ROS_best_effort_without_scored_option',frame_id=p.frame_id,
                    return_reason=returning.reason,action=action,profile=PROFILE))
        self.components.bind_execution_option(scene_idx,s['option'])
        assessment=self.components.assess_local_action(scene_idx,action)
        if not assessment['allowed']:
            # A fresh goal may encounter a known occupied successor; rotations
            # remain permissible from the actual current start under ROS.
            s['option']=None;s['active_actions']=['left'];action='left'
            assessment=self.components.assess_local_action(scene_idx,action)
        s['pending']=assessment
        self.audit.append(dict(event='paid_action_authorized',frame_id=p.frame_id,next_action_id=p.action_id+1,
            phase=s['phase'],assessment=assessment,option_id=None if s['option'] is None else s['option']['option_id']))
        return action

    def close_sensor_episode(self,scene_idx,*,reason):
        s=self.states[scene_idx]
        if s['pending'] is not None:raise RuntimeError('consume terminal observation before closing')
        if s['closed']:return self.sensor_episode_summary(scene_idx)
        b=self.components._cpu_backend.scenes[scene_idx];p=s['packet']
        returned=tuple((*p.position,p.heading))==b['return_anchor']
        actual_safe=~inflated_obstacles(np.asarray(s['mapper'].belief)!=0,
            s['mapper'].config.robot_radius_m/s['mapper'].config.resolution_m)
        failed=not returned or p.collision
        s['closed']=True
        s['termination']=dict(reason=reason,returned_to_anchor=returned,failed=failed,
            current_footprint_known_safe=bool(actual_safe[p.position]),admission_profile=PROFILE,
            return_certificate=False,final_frame_id=p.frame_id,final_action_id=p.action_id)
        terminal=b['ledger'].finish(reason=reason,failed=failed);b['feedback']=terminal
        self.components._cpu_backend._record(scene_idx,'IGCR','finish',dict(reason=reason,failed=failed),terminal)
        self.audit.append(dict(event='close_after_final_observation',**s['termination']))
        return self.sensor_episode_summary(scene_idx)
