"""Predetermined R2 motion check with the same measured safety guard.

This is deliberately not an autonomous semantic planning policy.
"""
from copy import deepcopy

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_safety_v43 import ObservedSafetyV43


class DiagnosticPolicyV43:
    def __init__(self, graph, *, home, budget, actions):
        self.graph = deepcopy(graph)
        self.graph.validate_state(home)
        if type(budget) is not int or budget < len(actions) or not actions:
            raise ValueError('diagnostic sequence must fit declared paid budget')
        check = home
        for action in actions:
            check = self.graph.successor(check, action)
        if check != home:
            raise ValueError('diagnostic must return to exact home position and heading')
        self.home, self.actions = home, tuple(actions)
        self.state, self.paid_step, self.pending = None, -1, None
        self.frames = set(); self.failed = False
        self.safety = ObservedSafetyV43(self.graph, unknown_space_policy='provided_navigation_prior')

    def accept(self, observation, mapper, *, execution_outcome='success'):
        if type(observation) is not PaidRGBDObservationV40:
            raise TypeError('strict paid observation required')
        if observation.frame_id in self.frames or observation.paid_step != self.paid_step+1:
            raise ValueError('diagnostic requires consecutive unique packets')
        state = self.graph.state_from_observation(observation)
        snapshot = mapper.snapshot()
        if (snapshot['frames'] != observation.paid_step+1 or not snapshot['receipts']
                or snapshot['receipts'][-1]['observation_sha256'] != observation.sha256()):
            raise ValueError('mapper must have integrated this exact packet once')
        if self.paid_step == -1:
            if state != self.home or execution_outcome != 'success':
                raise ValueError('diagnostic initial state mismatch')
        else:
            if self.pending is None:
                raise ValueError('diagnostic packet needs prior submitted action')
            expected = self.graph.successor(self.state, self.pending)
            if execution_outcome == 'collision' and self.pending == 'forward' and state == self.state:
                self.graph.block_observed_edge(self.state.node, expected.node)
                self.failed = True
            elif execution_outcome != 'success' or state != expected:
                raise ValueError('diagnostic execution disagrees with submitted action')
        safety = self.safety.update(snapshot, mapper.occupancy_arrays(), state)
        self.state, self.paid_step, self.pending = state, observation.paid_step, None
        self.frames.add(observation.frame_id)
        return dict(phase='R2_fixed_motion_diagnostic', autonomous_policy=False,
                    observation_sha256=observation.sha256(), safety=safety)

    def choose(self):
        if self.state is None or self.pending is not None:
            raise ValueError('accept current diagnostic packet before selecting action')
        if self.failed:
            return dict(action='blocked', reason='diagnostic_collision', paid_step=self.paid_step)
        if self.paid_step == len(self.actions):
            return dict(action='stop' if self.state == self.home else 'blocked',
                        reason='fixed_diagnostic_complete', paid_step=self.paid_step)
        action = self.actions[self.paid_step]
        try:
            successor = self.graph.successor(self.state, action)
        except ValueError:
            return dict(action='blocked', reason='diagnostic_observed_edge_block', paid_step=self.paid_step)
        route = self.graph.route(successor, self.home)
        remaining = len(self.actions)-self.paid_step
        if route is None or route.cost+1 > remaining:
            return dict(action='blocked', reason='diagnostic_return_reserve', paid_step=self.paid_step)
        guard = self.safety.guard_action(action)
        if not guard['allowed']:
            return dict(action='blocked', reason=guard['reason'], safety=guard, paid_step=self.paid_step)
        self.pending = action
        return dict(action=action, reason='predeclared_R2_diagnostic', paid_step=self.paid_step,
                    autonomous_policy=False, safety=guard, return_cost_after_action=route.cost)
