"""One declared candidate intervention inside the original global call.

No world or evaluator access. Nominal choices remain in decision snapshots;
an explicit intervention receipt distinguishes proposal from execution.
"""
from copy import deepcopy
from nso.cpu_sensor_contract_v10 import digest
from nso.decision_capture_v14 import DecisionCaptureRuntimeV14


class CandidateInterventionRuntimeV15(DecisionCaptureRuntimeV14):
    def configure_intervention(self, checkpoint, candidate_id):
        if hasattr(self, 'intervention_checkpoint'):
            raise ValueError('one intervention per episode')
        self.intervention_checkpoint = deepcopy(checkpoint)
        self.intervention_candidate_id = candidate_id
        self.intervention_receipt = None

    def choose_goal(self, scene_idx, proposed_local, bounds):
        result = super().choose_goal(scene_idx, proposed_local, bounds)
        checkpoint = self.intervention_checkpoint
        observed = self.decision_snapshots[-1]
        if self.intervention_receipt is not None:
            return result
        if observed['action_id'] > checkpoint['action_id']:
            raise ValueError('missed intervention checkpoint')
        if observed['action_id'] != checkpoint['action_id']:
            return result
        if digest(observed) != digest(checkpoint):
            raise ValueError('intervention before/after state or full pool differs')
        state = self.states[scene_idx]
        backend = self.components._cpu_backend
        bs = backend.scenes[scene_idx]
        if state['pending'] is not None or state['closed']:
            raise ValueError('intervention crossed action authorization boundary')
        matches = [c for c in observed['candidates'] if c['candidate_id'] == self.intervention_candidate_id]
        if len(matches) != 1:
            raise ValueError('candidate not uniquely present')
        candidate = matches[0]
        if (not candidate['outbound_actions'] or len(candidate['outbound_actions']) != candidate['outbound_cost']
                or candidate['cost'] > bs['ledger'].remaining_budget):
            raise ValueError('invalid route or return reservation')
        nominal = state['option']
        if nominal is None:
            raise ValueError('pilot requires existing nominal option metadata')
        # Preserve the original plan and call identity; no extra selection or
        # assessment is issued. Existing next_local_action guards the new route.
        option = deepcopy(candidate)
        for key in ('option_id', 'selection_call_id', 'selected_map_version',
                    'selected_feedback_version', 'parent_region'):
            option[key] = nominal[key]
        state['option'] = deepcopy(option)
        state['active_actions'] = list(option['outbound_actions'])
        state['phase'] = 'outbound'
        bs['selected'] = deepcopy(option)
        self.intervention_receipt = dict(action_id=observed['action_id'],
            decision_ordinal=observed['ordinal'], candidate_id=candidate['candidate_id'],
            nominal_candidate_id=nominal['candidate_id'],
            candidate_pool_sha256=observed['candidate_sha256'],
            checkpoint_sha256=digest(observed), option=deepcopy(option),
            extra_global_selection_calls=0, guards_unchanged=True)
        self.audit.append(dict(event='v15_candidate_intervention', **deepcopy(self.intervention_receipt)))
        return list(option['pose'][:2])
