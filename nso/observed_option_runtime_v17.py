"""Explicit first-stage intervention, then normal guarded runtime continuation."""
from nso.staged_approach_runtime_v16 import StagedApproachRuntimeV16


class ObservedOptionRuntimeV17(StagedApproachRuntimeV16):
    def install_two_stage_candidate(self, declaration, pool_sha256):
        state = self.states[0]
        if state['closed'] or state['pending'] is not None or state['active_actions']:
            raise RuntimeError('two-stage installation requires an idle observation boundary')
        backend = self.components._cpu_backend
        if backend.scenes[0].get('region_continuation') is not None:
            raise RuntimeError('two-stage commitment already pending')
        receipt = self.install_candidate(declaration, pool_sha256)
        backend.arm_region_continuation(0, declaration)
        self.audit.append(dict(event='observed_two_stage_armed',
            action_id=state['packet'].action_id, candidate_id=declaration['candidate_id'],
            conditional_future_view=True, original_guards=True))
        return receipt

    def begin_return(self):
        super().begin_return()
        self.components._cpu_backend.scenes[0]['region_continuation'] = None
