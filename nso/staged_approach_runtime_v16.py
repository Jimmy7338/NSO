"""A declared one-option physical probe through the existing sensor owner.

The external prefix is charged and guarded. An outbound denial ends the
attempt; the caller then observes the current map and requests guarded return.
No second inspection is executed by this diagnostic runtime.
"""
from copy import deepcopy
from nso.runtime_integration import NSORuntimeIntegration
from nso.cpu_sensor_contract_v10 import digest


class StagedApproachRuntimeV16(NSORuntimeIntegration):
    def authorize_probe_action(self, action, *, phase):
        s = self.states[0]
        if s['closed'] or s['pending'] is not None:
            raise RuntimeError('probe requires an open state without pending observation')
        if phase not in ('external_prefix', 'outbound'):
            raise ValueError('use the original runtime for return execution')
        if phase == 'external_prefix':
            s.update(option=None, active_actions=[action], phase=phase)
        elif s['phase'] != phase or not s['active_actions'] or s['active_actions'][0] != action:
            raise ValueError('action is not the next declared outbound primitive')
        self.components.bind_execution_option(0, s['option'])
        check = self.components.assess_local_action(0, action)
        self.audit.append(dict(event='probe_action_assessment', action_id=s['packet'].action_id,
            phase=phase, assessment=deepcopy(check)))
        if not check['allowed']:
            return None
        s['pending'] = check
        return action

    def install_candidate(self, declaration, pool_sha256):
        s = self.states[0]
        self.choose_goal(0, list(s['packet'].position), (0,self.full_shape[0],0,self.full_shape[1]))
        backend = self.components._cpu_backend; b = backend.scenes[0]
        selection = b['last_selection']
        if digest(selection['candidates']) != pool_sha256:
            raise ValueError('observed prefix no longer yields declared shared candidate pool')
        route = next((x for x in selection['candidates'] if x['candidate_id'] == declaration['candidate_id']), None)
        if route != declaration:
            raise ValueError('declared candidate differs')
        nominal = deepcopy(s['option'])
        if nominal is None:
            raise ValueError('probe requires nominal option metadata')
        option = deepcopy(route)
        for key in ('option_id','selection_call_id','selected_map_version','selected_feedback_version','parent_region'):
            option[key] = nominal[key]
        s.update(option=option, active_actions=list(option['outbound_actions']), phase='outbound')
        b['selected'] = deepcopy(option)
        receipt = dict(event='declared_staged_intervention', action_id=s['packet'].action_id,
            candidate_pool_sha256=pool_sha256, candidate_id=route['candidate_id'],
            nominal_candidate_id=nominal['candidate_id'], original_guards=True,
            option=deepcopy(option), external_prefix=True)
        self.audit.append(receipt)
        return receipt

    def begin_return(self):
        s = self.states[0]
        if s['pending'] is not None:
            raise RuntimeError('consume the last observation before return')
        s.update(option=None, active_actions=[], phase='return')
        self.components.bind_execution_option(0, None)
