"""Opt-in CPU runtime: one fresh global plan after an outbound safety denial.

All motion still passes the original footprint and successor-return guards.
Frozen runtimes remain unchanged. Recovery is a common execution mechanism,
not a semantic scorer or a reason to weaken the observed-map safety contract.
"""
from nso.runtime_integration import NSORuntimeIntegration
from nso.semantic_opportunities_v14 import candidate_capacity


class RecoveryRuntimeV14(NSORuntimeIntegration):
    def choose_goal(self, scene_idx, proposed_local, bounds):
        backend = self.components._cpu_backend
        if backend is not None and backend.scenes[scene_idx] is not None:
            backend.args.cpu_max_candidates = candidate_capacity(
                len(backend.scenes[scene_idx]['assets']),
                int(getattr(backend.args, 'cpu_coverage_slots', 4)))
        return super().choose_goal(scene_idx, proposed_local, bounds)

    def next_local_action(self, scene_idx):
        s = self.states[scene_idx]
        if s is None or s['closed']:
            return None
        if s['pending'] is not None:
            raise RuntimeError('pending action has no sensor observation yet')
        comp, p = self.components, s['packet']
        ledger = comp._cpu_backend.scenes[scene_idx]['ledger']
        if ledger.remaining_budget == 0:
            self.close_sensor_episode(scene_idx, reason='budget_exhausted')
            return None
        bounds = (0, self.full_shape[0], 0, self.full_shape[1])
        if not s['active_actions'] and s['phase'] != 'return':
            self.choose_goal(scene_idx, list(p.position), bounds)
        if not s['active_actions']:
            returning = comp.plan_observed_return(scene_idx)
            if not returning.available or not returning.actions:
                reason = returning.reason if not returning.available else s.get(
                    'v14_return_reason', 'no_positive_option_at_anchor')
                self.close_sensor_episode(scene_idx, reason=reason)
                return None
            s['phase'] = 'return'
            s['option'] = None
            s['active_actions'] = list(returning.actions)

        action = s['active_actions'][0]
        comp.bind_execution_option(scene_idx, s['option'])
        check = comp.assess_local_action(scene_idx, action)
        if not check['allowed']:
            was_outbound = s['phase'] == 'outbound'
            event = dict(event='v14_route_denied', action_id=p.action_id,
                         phase=s['phase'], denial=check, remaining_budget=ledger.remaining_budget,
                         recovery_attempted=False, recovery_accepted=False)
            s['active_actions'] = []
            s['option'] = None
            comp.bind_execution_option(scene_idx, None)
            returning = comp.plan_observed_return(scene_idx)
            # No free retries on the same observation. Return-phase denials
            # only refresh the return path; they never restart exploration.
            if was_outbound and returning.available and len(returning.actions) < ledger.remaining_budget:
                event['recovery_attempted'] = True
                s['phase'] = 'planning'
                self.choose_goal(scene_idx, list(p.position), bounds)
                if s['active_actions']:
                    action = s['active_actions'][0]
                    comp.bind_execution_option(scene_idx, s['option'])
                    check = comp.assess_local_action(scene_idx, action)
                    event['recovery_assessment'] = check
                    event['recovery_accepted'] = bool(check['allowed'])
            if not event['recovery_accepted']:
                s['active_actions'] = []
                s['option'] = None
                s['phase'] = 'return'
                s['v14_return_reason'] = 'returned_after_route_denial_without_safe_recovery'
                comp.bind_execution_option(scene_idx, None)
                returning = comp.plan_observed_return(scene_idx)
                if not returning.available or not returning.actions:
                    self.audit.append(event)
                    self.close_sensor_episode(scene_idx, reason=(returning.reason if not returning.available
                                              else s['v14_return_reason']))
                    return None
                s['active_actions'] = list(returning.actions)
                action = s['active_actions'][0]
                check = comp.assess_local_action(scene_idx, action)
                if not check['allowed']:
                    self.audit.append(event)
                    self.close_sensor_episode(scene_idx, reason=check['reason'])
                    return None
            self.audit.append(event)
        s['pending'] = check
        self.audit.append(dict(event='paid_action_authorized', frame_id=p.frame_id,
            next_action_id=p.action_id + 1, phase=s['phase'], assessment=check,
            option_id=None if s['option'] is None else s['option']['option_id'],
            selection_call_id=None if s['option'] is None else s['option']['selection_call_id']))
        return action
