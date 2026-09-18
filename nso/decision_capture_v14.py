"""Capture every global decision, including internal recovery, from live state.

No simulator, evaluator, future trajectory or reward enters this observer.
Snapshots are equality certificates; TSDF restoration still requires replay
of the complete prefix. Capturing does not choose or execute a different action.
"""
from nso.cpu_sensor_contract_v10 import digest, json_value
from nso.decision_replay_v13 import decision_state
from nso.recovery_runtime_v14 import RecoveryRuntimeV14
from nso.semantic_opportunities_v14 import observed_descriptors, route_instance_features


class DecisionCaptureRuntimeV14(RecoveryRuntimeV14):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_snapshots = []

    def choose_goal(self, scene_idx, proposed_local, bounds):
        if scene_idx != 0 or len(self.states) != 1:
            raise ValueError('V14 capture contract requires one CPU scene')
        state = self.states[scene_idx]
        if state is None or state['closed']:
            return super().choose_goal(scene_idx, proposed_local, bounds)
        if state['pending'] is not None:
            raise RuntimeError('cannot capture a decision before pending observation')
        before = decision_state(self)
        result = super().choose_goal(scene_idx, proposed_local, bounds)
        backend = self.components._cpu_backend
        selected = backend.scenes[scene_idx]['last_selection']
        candidates = selected['candidates']
        after = decision_state(self)
        assets, descriptors = observed_descriptors(self, candidates)
        features = route_instance_features(state['mapper'], candidates, assets, descriptors)
        if decision_state(self)['sha256'] != after['sha256']:
            raise RuntimeError('feature observer changed decision state')
        self.decision_snapshots.append(json_value(dict(
            ordinal=len(self.decision_snapshots) + 1, action_id=state['packet'].action_id,
            state_before=before, state_after_choice=after,
            remaining_budget=backend.scenes[scene_idx]['ledger'].remaining_budget,
            proposed_local=list(proposed_local), bounds=list(bounds),
            candidate_sha256=digest(candidates), candidates=candidates,
            selected_option=state['option'], descriptors=descriptors, features=features,
            observed_instance_count=len(assets),
            candidate_cap=backend.args.cpu_max_candidates,
            candidate_audit=selected['candidate_audit'],
            feedback_state=backend.scenes[scene_idx]['ledger'].snapshot(),
            observation_only=True, future_rewards_read=False)))
        return result
