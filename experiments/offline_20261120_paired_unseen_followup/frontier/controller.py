"""Frontier action supplier with the frozen B11 guard and observation owner.

No world, reference mesh or evaluator object enters this interface.
"""
from dataclasses import asdict


class FrontierDriver:
    def __init__(self, components, runtime, policy, observation):
        self.components = components
        self.runtime = runtime
        self.policy = policy
        self.observation = observation
        self.events = []
        self.last_decision = None

    def _choose(self):
        d = self.policy.act(self.observation())
        self.last_decision = asdict(d)
        return d.action

    def _return(self):
        s = self.runtime.states[0]
        s['phase'] = 'return'
        s['option'] = None
        self.components.bind_execution_option(0, None)
        returning = self.components.plan_observed_return(0)
        if not returning.available or not returning.actions:
            self.runtime.close_sensor_episode(0, reason=returning.reason if not returning.available
                                              else 'frontier_policy_stopped_at_anchor')
            return None
        s['active_actions'] = list(returning.actions)
        return s['active_actions'][0]

    def next_action(self):
        s = self.runtime.states[0]
        if s['closed']:
            return None
        if s['pending'] is not None:
            raise RuntimeError('Previous paid action has no accepted observation')
        ledger = self.components._cpu_backend.scenes[0]['ledger']
        if ledger.remaining_budget == 0:
            self.runtime.close_sensor_episode(0, reason='budget_exhausted')
            return None
        self.last_decision = None
        self.components.bind_execution_option(0, None)
        if s['phase'] == 'return':
            action = s['active_actions'][0] if s['active_actions'] else self._return()
        else:
            s['phase'] = 'outbound'
            action = self._choose()
            if action == 'stop':
                self.events.append(dict(event='frontier_policy_stop', frame_id=s['packet'].frame_id))
                action = self._return()
        if action is None:
            return None
        check = self.components.assess_local_action(0, action)
        if not check['allowed']:
            self.events.append(dict(event='guard_rejection', frame_id=s['packet'].frame_id,
                                    phase=s['phase'], assessment=check))
            if s['phase'] != 'return':
                # Same B11 scheduling contract: one fresh target selection after
                # rejection, then observed-safe return if it still fails. This
                # does not delete candidates or change the baseline's ranking.
                self.policy.goal = None
                self.policy._path = []
                action = self._choose()
                self.events.append(dict(event='frontier_replan_after_rejection',
                                        frame_id=s['packet'].frame_id, decision=self.last_decision))
                check = self.components.assess_local_action(0, action) if action != 'stop' else None
            if check is None or not check['allowed']:
                s['active_actions'] = []
                action = self._return()
                if action is None:
                    return None
                check = self.components.assess_local_action(0, action)
                if not check['allowed']:
                    self.events.append(dict(event='return_guard_rejection',
                                            frame_id=s['packet'].frame_id, assessment=check))
                    self.runtime.close_sensor_episode(0, reason=check['reason'])
                    return None
        if s['phase'] != 'return':
            s['active_actions'] = [action]
        s['option'] = None
        s['pending'] = check
        self.runtime.audit.append(dict(event='paid_action_authorized', frame_id=s['packet'].frame_id,
            next_action_id=s['packet'].action_id + 1, phase=s['phase'], assessment=check,
            baseline='F_nearest', option_id=None, selection_call_id=None))
        return action
