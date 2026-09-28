"""Sensor-packet adapter for persistent V29 options and fixed V28r1 scores.

This has not been validated in a new autonomous physical/simulated episode.
IGCR exports whole-option measured features, not a trained quality correction.
"""
from copy import deepcopy

from nso.cpu_sensor_contract_v10 import GridTransform
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.observed_state_v26 import geometry_state_v26, VisibleSemanticMemoryV26
from nso.observed_planner_v26 import ObservedPlannerV26
from nso.observed_debt_v28_r1 import build_context, rank_context
from nso.persistent_observation_option_v29 import PersistentObservationOptionV29, VERSION


class ObservedANSRuntimeV29:
    def __init__(self, shape, sensor_config, total_budget, mode='G', replan_interval=5):
        if type(total_budget) is not int or total_budget < 1:
            raise ValueError('positive paid action budget required')
        if type(replan_interval) is not int or replan_interval < 1 or mode not in ('G','O','S','X'):
            raise ValueError('valid periodic check interval and mode required')
        self.config,self.budget,self.mode = sensor_config,total_budget,mode
        self.transform = GridTransform(tuple(shape), sensor_config.resolution_m)
        self.transform.validate_cpu_mapper(sensor_config)
        self.mapper = ObservedRuntimeMapperV10(shape,sensor_config)
        self.semantic = None if mode == 'G' else VisibleSemanticMemoryV26()
        self.geometry_planner = ObservedPlannerV26('G')
        self.options = PersistentObservationOptionV29()
        self.packet = self.state = self.full_state = self.anchor = None
        self.cues = ()
        self.terminal_reason = None
        self.replan_interval = replan_interval
        self.seen_frame_ids = set()
        self.plans,self.calls = [],[]
        self._event_cursor = 0

    @property
    def pending(self):
        return self.options.pending

    def _flush_events(self):
        self.calls.extend(deepcopy(self.options.events[self._event_cursor:]))
        self._event_cursor = len(self.options.events)

    def accept(self, packet):
        packet.validate(self.transform,self.config)
        if self.terminal_reason is not None or packet.frame_id in self.seen_frame_ids:
            raise ValueError('stopped episode or already consumed frame')
        if self.packet is None:
            if packet.action_id != 0 or packet.action is not None:
                raise ValueError('action-zero cold start required')
            anchor = (*packet.position,packet.heading)
        else:
            old = self.packet
            if (packet.scene_id != old.scene_id or packet.episode_id != old.episode_id
                    or packet.frame.timestamp_s <= old.frame.timestamp_s):
                raise ValueError('foreign episode or nonmonotonic time')
            self.options.validate_transition(packet.action_id,packet.position,packet.heading,
                                             packet.action,packet.collision)
            anchor = self.anchor
        if packet.action_id > self.budget:
            raise ValueError('paid action budget exceeded')
        self.mapper.update(packet.frame,packet.scan)
        state = geometry_state_v26(self.mapper,packet,anchor,self.budget-packet.action_id)
        full = geometry_state_v26(self.mapper,packet,anchor,self.budget-packet.action_id,max_patches=2**31-1)
        if self.packet is None:
            self.options.initialize(state,full)
        else:
            self.options.consume(state,full,action=packet.action,collision=packet.collision,
                stop_reason='sensor_end' if packet.done else None)
        if self.semantic is not None:
            self.cues = self.semantic.update(packet)
        self.packet,self.state,self.full_state,self.anchor = packet,state,full,anchor
        self.seen_frame_ids.add(packet.frame_id)
        self.calls.append(dict(module='OV-SDF',operation='observed_geometry_and_full_option_support',
            action_id=state.action_id,geometry_sha256=state.geometry_sha256,
            full_support_patches=len(full.patches),evaluation_Q_used=False))
        self._flush_events()
        if packet.collision:
            self.terminal_reason = 'collision_observed'
        elif packet.done or packet.action_id == self.budget:
            self.terminal_reason = 'sensor_or_budget_end_returned' if (
                *packet.position,packet.heading) == self.anchor else 'sensor_or_budget_end_without_return'
        return state

    def _plan(self):
        base = self.geometry_planner.plan(self.state)
        if self.mode == 'G':
            answer = base
        else:
            context = build_context(self.state,self.full_state,self.cues,base)
            answer = rank_context(context,self.mode)
        self.calls.append(dict(module='STGHP',operation='select_new_persistent_option',
            action_id=self.state.action_id,mode=self.mode,candidate_count=len(answer['candidates']),
            evaluation_Q_used=False,score_version='unchanged_v28_r1'))
        return answer

    def next_action(self):
        if self.state is None or self.pending is not None:
            raise ValueError('initialize and consume previous paid observation first')
        if self.terminal_reason is not None:
            return None
        active = self.options.active
        if active is None and not self.options.return_only:
            result = self._plan()
            self.plans.append(deepcopy(result))
            if result['selected'] is None:
                self.options.begin_return()
            else:
                self.options.open(result['selected'],self.cues)
        elif active is not None:
            elapsed = self.state.action_id-active['opened_at_action']
            if elapsed and elapsed % self.replan_interval == 0:
                self.calls.append(dict(module='STGHP',operation='preserve_option_at_periodic_check',
                    action_id=self.state.action_id,option_id=active['option_id'],
                    original_open_action=active['opened_at_action'],target_reselected=False))
        action = self.options.next_action()
        self._flush_events()
        if action is None:
            self.terminal_reason = 'returned_or_no_safe_action' if (
                *self.state.position,self.state.heading) == self.anchor else 'observed_return_unavailable'
        return action

    def summary(self):
        return dict(version=VERSION,mode=self.mode,budget=self.budget,
            actual_paid_actions=len(self.options.paid_ledger),
            actions_without_observation_option=sum(x['option_id'] is None for x in self.options.paid_ledger),
            global_plans=len(self.plans),closed_options=len(self.options.closed),
            active_option=self.options.active,terminal_reason=self.terminal_reason,
            whole_option_outcome_features=True,quality_feedback_applied=False,
            trained_RPN_UQ=False,trained_IGCR=False,full_method_efficacy_proven=False,
            semantic_source='none' if self.semantic is None else 'artificial_RGB_marker')
