"""Explicit unlimited waypoint exploration; no time/count/roundtrip budget gate."""
from copy import deepcopy
import math
import numpy as np
from nso_bridge.ros_viewpoint_exploration_v3 import ViewpointSession,ViewpointCPU,HorizonLedgerView
from nso.observed_feedback_v10 import FeedbackLedger
from nso_bridge.ros_checked_exploration_v2 import ROSCheckedCPUFourModules
PROFILE='ros_viewpoint_exploration_v4_unbounded'

class UnboundedActualLedger(FeedbackLedger):
    @property
    def remaining_budget(self):return math.inf  # Arithmetic only; RPC fields normalized to None below.
    def observe(self,*args,**kwargs):
        value=super().observe(*args,**kwargs);value['remaining_budget']=None
        if self._events:self._events[-1]['remaining_budget']=None
        return value
    def snapshot(self):
        value=super().snapshot();value.update(total_budget=None,remaining_budget=None,accounting_unit='actual_waypoint_attempt_unbounded');return value
    def finish(self,*args,**kwargs):
        value=super().finish(*args,**kwargs);value['remaining_budget']=None
        if self._events:self._events[-1]['remaining_budget']=None
        return value

class UnboundedViewpointCPU(ViewpointCPU):
    policy_variant=PROFILE
    admission_profile=PROFILE
    def start_scene(self,scene_idx,**kwargs):
        kwargs=dict(kwargs,total_budget=1)  # Bootstrap constructor seed only, never an action limit.
        super().start_scene(scene_idx,**kwargs);s=self.scenes[scene_idx];old=s['ledger']
        unlimited=UnboundedActualLedger.__new__(UnboundedActualLedger)
        unlimited.__dict__.update(old.__dict__);unlimited.total_budget=None
        s['ledger']=unlimited;s['feedback']['remaining_budget']=None
    def select_target(self,scene_idx):
        s=self.scenes[scene_idx];actual=s['ledger'];s['ledger']=HorizonLedgerView(actual,None)
        try:return ROSCheckedCPUFourModules.select_target(self,scene_idx)
        finally:s['ledger']=actual

class UnboundedViewpointSession(ViewpointSession):
    def __init__(self,configuration,**kwargs):
        original=deepcopy(configuration)
        if original.get('policy_variant')!=PROFILE:raise ValueError('explicit unbounded profile required')
        for k in ('max_viewpoint_attempts','nominal_planning_horizon'):
            if original.get(k) is not None:raise ValueError(k+' must be None in unbounded profile')
        c={k:v for k,v in original.items() if k not in ('execution_limits_profile','return_policy')}
        # One is solely a constructor/bootstrap compatibility seed. Before any
        # selected action, the actual ledger is replaced by a truly unbounded one.
        c.update(policy_variant='ros_viewpoint_exploration_v3',total_budget=1,max_viewpoint_attempts=1,nominal_planning_horizon=1)
        super().__init__(c,**kwargs)
        self.policy_variant=PROFILE;self.configuration=original
        self.configuration.update(total_budget=None,max_viewpoint_attempts=None,nominal_planning_horizon=None,
            execution_limits_profile='exploration_until_stopped_v1',return_policy='no_automatic_return')
        self.max_attempts=None;self.nominal_horizon=None
        self.backend=UnboundedViewpointCPU(self.args,1,self.shape);self.components._cpu_backend=self.backend
    def identity(self):
        value=super().identity();v=value['receipt']['viewpoint_execution']
        v.update(profile=PROFILE,actual_resource='no episode time, waypoint-count or nominal-route budget; explicit stop/hardware protection remain',
            max_viewpoint_attempts=None,nominal_planning_horizon=None,
            return_request='no automatic return, no anchor fallback or roundtrip filtering',
            nominal_grid_cost='outbound route prediction/tie-break only; return suffix removed',
            scorer='inherited formulas use outbound states and outbound cost unchanged; candidate cost/tie-break now outbound-only')
        v['admission'].update(return_contract='no return planning or reserved budget',no_option_fallback='none; close at actual pose if no positive option')
        return value
    def next_viewpoint(self,remaining_time_s=None,return_requested=False):
        if self.last_packet is None:raise ValueError('bootstrap actual observation first')
        state=self.runtime.states[0]
        if state['closed']:return dict(target=None,summary=self.summary())
        if self.viewpoint_pending is not None:raise ValueError('observe terminal waypoint before selection')
        self.backend.select_target(0);scene=self.backend.scenes[0];selected=deepcopy(scene.get('selected'))
        if selected is None:
            p=self.last_packet;state['closed']=True
            state['termination']=dict(reason='no_positive_current_viewpoint',failed=False,
                returned_to_anchor=tuple((*p.position,p.heading))==self.anchor,return_required=False,
                final_frame_id=p.frame_id,final_action_id=p.action_id,
                coverage_complete_verified=False,completion_scope='current original candidate set only')
            event=scene['ledger'].finish(reason='no_positive_current_viewpoint',failed=False);scene['feedback']=event
            return dict(target=None,summary=self.summary())
        target=self._goal(selected['pose'],selected,'NSO_selected_viewpoint_no_return')
        self.viewpoint_pending=target
        self.runtime.audit.append(dict(event='viewpoint_goal_selected',target=deepcopy(target),
            time_limit=None,waypoint_limit=None,roundtrip_limit=None,return_requested_ignored=bool(return_requested)))
        return dict(target=deepcopy(target),selection=scene.get('last_selection'),summary=self.summary())
