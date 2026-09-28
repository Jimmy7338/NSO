"""Paid, persistent observed-map options; no world, Q or learned reward.

An option ends after observation waypoints, not after its reserved return.
The latter is revalidated every step and reported separately from paid cost.
"""
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np

from nso.observed_state_v26 import GeometryStateV26, SemanticCueV26
from nso.observed_planner_v26 import ObservedPlannerV26
from nso.execution_guard_v8 import ObservedExecutionGuard
from nso.observed_feedback_v27 import make_feedback_support_v27
from utils.grid_geometry import DIRECTIONS

VERSION = 'persistent-observation-option-v29-1'


def _full_support(state, full):
    if not isinstance(state, GeometryStateV26) or not isinstance(full, GeometryStateV26):
        raise TypeError('typed observed states required')
    for field in state.__dataclass_fields__:
        if field in ('patches', 'geometry_sha256'):
            continue
        equal = np.array_equal(state.belief, full.belief) if field == 'belief' else (
            getattr(state, field) == getattr(full, field))
        if not equal:
            raise ValueError('full observation mismatch: '+field)
    sample = tuple(full.patches[i] for i in np.linspace(0, len(full.patches)-1, 256, dtype=int)) \
        if len(full.patches) > 256 else full.patches
    if sample != state.patches:
        raise ValueError('planning sample does not match full support')
    return make_feedback_support_v27(state.action_id, state.geometry_sha256, full.patches)


def _pose(value):
    if (len(value) != 3 or any(type(x) is not int or x < 0 for x in value)
            or value[2] > 3):
        raise ValueError('integer row,column,heading required')
    return tuple(value)


def remaining_route(state, waypoints):
    """Forecast on one observed map, without inventing future observations."""
    current = state
    outbound = []
    last_return = None
    for target in waypoints:
        if (*current.position, current.heading) == target:
            continue
        route = ObservedPlannerV26._space(current).route(target, group='persistent_observation_v29')
        if route is None:
            return None
        outbound.extend(route['outbound_actions'])
        last_return = route['return_actions']
        # Routing forecast only; never used as an accepted measured state.
        current = replace(current, position=target[:2], heading=target[2],
                          remaining_budget=current.remaining_budget-route['outbound_cost'])
    if not outbound:
        return None
    return dict(outbound_actions=outbound, outbound_cost=len(outbound),
                return_cost=len(last_return), total_reserved_cost=len(outbound)+len(last_return))


class PersistentObservationOptionV29:
    def __init__(self):
        self.state = self.support = None
        self.pending = None
        self._active = None
        self._serial = 0
        self._return_only = False
        self._terminal = False
        self.events = []
        self.closed = []
        self.paid_ledger = []

    @property
    def active(self):
        return None if self._active is None else deepcopy(self._active['record'])

    @property
    def return_only(self):
        return self._return_only

    def _event(self, module, operation, **payload):
        self.events.append(dict(module=module, operation=operation,
            action_id=self.state.action_id, **payload))

    def initialize(self, state, full_state):
        if self.state is not None or state.action_id != 0:
            raise ValueError('one action-zero initialization required')
        support = _full_support(state, full_state)
        self.state, self.support = state, support

    def open(self, selected, cues=(), *, waypoints=None):
        if (self.state is None or self.pending is not None or self._active is not None
                or self._return_only or self._terminal):
            raise ValueError('cannot open an option in current lifecycle state')
        targets = tuple(_pose(p) for p in (waypoints if waypoints is not None else (selected['pose'],)))
        if not targets or targets[-1] != _pose(selected['pose']):
            raise ValueError('nonempty ordered waypoints must end at selected pose')
        # Skip a current opening waypoint without granting an observation reward.
        while targets and targets[0] == (*self.state.position, self.state.heading):
            targets = targets[1:]
        route = remaining_route(self.state, targets)
        if route is None:
            raise ValueError('no paid complete observation and return route')
        cues = tuple(cues)
        if any(not isinstance(c, SemanticCueV26) or c.action_id > self.state.action_id for c in cues):
            raise ValueError('future/untyped semantic evidence forbidden')
        by_id = {c.cue_id:c for c in cues}
        if len(by_id) != len(cues):
            raise ValueError('duplicate cue identity')
        frozen = []
        for evidence in selected.get('semantic_evidence', ()):
            gain = evidence['hypothesis_gain']
            if not math.isfinite(gain) or gain < 0:
                raise ValueError('finite nonnegative prediction proxy required')
            if gain == 0:
                continue
            if evidence['cue_id'] not in by_id:
                raise ValueError('prediction must reference an observed cue')
            cue = by_id[evidence['cue_id']]
            frozen.append(dict(cue_id=cue.cue_id, center=list(cue.center), outward=list(cue.outward),
                observed_class_id=cue.class_id, planning_class_id=evidence.get('class_id',cue.class_id),
                confidence=cue.confidence, cue_action_id=cue.action_id,
                hypothesis_gain=float(gain), sector=evidence.get('sector'), calibrated=False))
        self._serial += 1
        record = dict(option_id=self._serial, opened_at_action=self.state.action_id,
            target_pose=list(targets[-1]), waypoints=[list(p) for p in targets],
            group=str(selected['group']), frozen_prediction=frozen,
            start_geometry_sha256=self.state.geometry_sha256,
            start_support_sha256=self.support.support_sha256,
            initial_outbound_cost=route['outbound_cost'],
            initial_return_reserve=route['return_cost'], initial_total_reserved=route['total_reserved_cost'])
        self._active = dict(record=record, targets=targets, index=0, start=self.support,
            start_pose=(*self.state.position, self.state.heading), poses={(*self.state.position,self.state.heading)},
            paid_actions=[], route_refreshes=0, max_translation_baseline_m=0.)
        self._event('IGCR', 'option_opened', **deepcopy(record))
        return self._serial

    def next_action(self):
        if self.state is None or self.pending is not None:
            raise ValueError('initialize and consume pending action before another request')
        if self._terminal:
            return None
        if self._active is not None:
            option = self._active
            route = remaining_route(self.state, option['targets'][option['index']:])
            if route is None:
                self._close('cancelled', 'remaining_observation_or_paid_return_unavailable')
                self._return_only = True
            else:
                guard = ObservedExecutionGuard(self.state.resolution_m,self.state.robot_radius_m)
                action = route['outbound_actions'][0]
                check = guard.assess(self.state.belief,self.state.position,self.state.heading,
                                     action,self.state.remaining_budget)
                if check.allowed:
                    option['route_refreshes'] += 1
                    self._event('RPN-UQ', 'persistent_target_route_revalidated',
                        option_id=option['record']['option_id'], proposed_action=action,
                        remaining_outbound_cost=route['outbound_cost'], return_reserve=route['return_cost'],
                        total_reserved_cost=route['total_reserved_cost'], uncertainty_calibrated=False)
                    self.pending = action
                    return action
                self._close('cancelled', 'observed_action_guard_rejected')
                self._return_only = True
        guard = ObservedExecutionGuard(self.state.resolution_m,self.state.robot_radius_m)
        home = guard.return_plan(self.state.belief,self.state.position,self.state.heading,
                                 self.state.anchor,self.state.remaining_budget)
        action = home.actions[0] if home.available and home.actions else None
        assessment = guard.assess(self.state.belief,self.state.position,self.state.heading,
                                 action,self.state.remaining_budget) if action is not None else None
        if assessment is not None and not assessment.allowed:
            action = None
        self._event('RPN-UQ', 'guarded_return_or_stop', proposed_action=action, return_only=self._return_only,
            return_available=home.available, remaining_return_cost=home.paid_cost,
            reason=assessment.reason if assessment is not None and not assessment.allowed else home.reason)
        self.pending = action
        return action

    def begin_return(self, reason='no_positive_observed_option'):
        if self.state is None or self.pending is not None or self._active is not None:
            raise ValueError('cannot enter return with a pending or active option')
        self._return_only = True
        self._event('STGHP', 'return_mode_entered', reason=reason)

    def validate_transition(self, action_id, position, heading, action, collision=False):
        if (self.state is None or self.pending is None or self._terminal
                or action_id != self.state.action_id+1 or action != self.pending):
            raise ValueError('observation does not consume the issued paid action')
        r,c = self.state.position
        h = self.state.heading
        if action == 'forward':
            dr,dc = DIRECTIONS[h]; r,c = r+dr,c+dc
        else:
            h = (h+(1 if action == 'right' else -1)) % 4
        if not collision and (tuple(position), heading) != ((r,c),h):
            raise ValueError('observation pose disagrees with issued action')

    def consume(self, state, full_state, *, action, collision=False, stop_reason=None):
        self.validate_transition(state.action_id,state.position,state.heading,action,collision)
        if state.anchor != self.state.anchor or state.remaining_budget != self.state.remaining_budget-1:
            raise ValueError('anchor or paid budget changed')
        for field in ('resolution_m','robot_radius_m','max_depth_m','fov_deg','width_px','height_px','camera_height_m'):
            if getattr(state,field) != getattr(self.state,field):
                raise ValueError('public sensor/map contract changed: '+field)
        if state.belief.shape != self.state.belief.shape:
            raise ValueError('public map shape changed')
        support = _full_support(state, full_state)
        self.state,self.support,self.pending = state,support,None
        option = self._active
        ledger = dict(action_id=state.action_id, action=action,
            option_id=None if option is None else option['record']['option_id'],
            phase='return_without_observation_option' if option is None else 'observation_option',
            collision=collision)
        self.paid_ledger.append(ledger)
        self._event('IGCR', 'paid_action_accounted', **{k:v for k,v in ledger.items() if k != 'action_id'})
        if option is not None:
            pose = (*state.position,state.heading)
            option['paid_actions'].append(action)
            option['poses'].add(pose)
            baseline = math.hypot(pose[0]-option['start_pose'][0],pose[1]-option['start_pose'][1])*state.resolution_m
            option['max_translation_baseline_m'] = max(option['max_translation_baseline_m'],baseline)
            self._event('IGCR', 'paid_transition_attributed',option_id=option['record']['option_id'],action=action)
        if collision:
            self._close('cancelled', 'collision')
            self._terminal = True
            return
        if option is not None:
            while option['index'] < len(option['targets']) and option['targets'][option['index']] == pose:
                option['index'] += 1
                self._event('IGCR','observation_waypoint_reached',option_id=option['record']['option_id'],
                            completed_waypoints=option['index'])
            if option['index'] == len(option['targets']):
                self._close('completed', 'all_observation_waypoints_reached')
        if stop_reason or state.remaining_budget == 0:
            # Actual arrival counts even when the same frame ends sensing.
            # Mission return/success remains a separate runtime assessment.
            self._close('cancelled', stop_reason or 'budget_end')
            self._terminal = True

    def _close(self, status, reason):
        if self._active is None:
            return
        option = self._active
        features = []
        for cue in option['record']['frozen_prediction']:
            center = cue['center']
            nearby = lambda p: math.hypot(p.point[0]-center[0],p.point[1]-center[1]) <= 1.5
            old = {p.key:p for p in option['start'].patches if nearby(p)}
            current = {p.key:p for p in self.support.patches if nearby(p)}
            common = old.keys() & current.keys()
            direction = {k for k in common if current[k].bits & ~old[k].bits}
            closer = {k for k in common if current[k].best_range < old[k].best_range-.01}
            features.append(dict(cue_id=cue['cue_id'], frozen_center=center,
                start_local_patches=len(old),end_local_patches=len(current),comparable_patches=len(common),
                new_keys=len(current.keys()-old.keys()),missing_start_keys=len(old.keys()-current.keys()),
                direction_improved_patches=len(direction),range_improved_patches=len(closer),
                improved_common_patches=len(direction|closer)))
        closed = dict(**deepcopy(option['record']), status=status,reason=reason,
            closed_at_action=self.state.action_id, actual_paid_actions=len(option['paid_actions']),
            actions=list(option['paid_actions']), completed_waypoints=option['index'],
            distinct_observed_base_poses_including_start=len(option['poses']),
            max_translation_baseline_from_start_m=option['max_translation_baseline_m'],
            route_refreshes=option['route_refreshes'], outcome_features=features,
            end_support_sha256=self.support.support_sha256, outcome_scope='whole_option_endpoint_net_change',
            evaluation_Q_used=False, calibrated_quality_residual=None, learning_reward=None,
            quality_feedback_applied=False, return_reserve_is_paid_cost=False)
        self.closed.append(closed)
        self._event('IGCR','option_closed',**deepcopy(closed))
        self._active = None
