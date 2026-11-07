"""Observation-only CPU ANS interfaces for the controlled V35 experiment.

The driver owns mapping, rendering and evaluation. This runtime receives a
sanitized paid observation and a digest of the measured map, never a World,
episode configuration, actual template index or evaluation reference. Both
public templates and the common prefix are explicit task priors. Potential
surface masks are forecasts; they never stand in for measured map quality.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import re

import numpy as np

from nso.observation_belief_v35 import MODES_V35, ObservationBeliefV35
from nso.pixel_information_v34 import arrays_sha256


ACTION_NAMES = ('forward', 'left', 'right')


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'integer {name} required')
    return int(value)


@dataclass(frozen=True, slots=True)
class ObservationV35:
    """Only observed sensor arrays, paid odometry and an opaque identity.

    A driver must construct this value explicitly. A complete SensorPacket
    is deliberately not accepted because its scene/episode metadata can
    disclose the hidden scenario. Arrays are copied and made read-only.
    """
    frame_id: str
    step: int
    pose: tuple
    action: str | None
    depth: np.ndarray
    rgb: np.ndarray
    ranges: np.ndarray
    collision: bool = False
    measured_map_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError('nonempty opaque frame identity required')
        step = _integer(self.step, 'paid step')
        if step < 0:
            raise ValueError('nonnegative paid step required')
        pose = tuple(_integer(v, 'pose component') for v in self.pose)
        if len(pose) != 3 or not 0 <= pose[2] < 4:
            raise ValueError('public (x,y,heading) odometry required')
        if self.action is not None and self.action not in ACTION_NAMES:
            raise ValueError('unknown paid action')
        if type(self.collision) is not bool:
            raise ValueError('boolean collision required')
        if self.measured_map_sha256 is not None and (
                not isinstance(self.measured_map_sha256, str) or
                re.fullmatch('[0-9a-f]{64}', self.measured_map_sha256) is None):
            raise ValueError('measured map digest must be SHA256 hex')
        depth = np.asarray(self.depth)
        ranges = np.asarray(self.ranges)
        rgb = np.asarray(self.rgb)
        if depth.ndim != 2 or not depth.size or ranges.ndim != 1 or not ranges.size:
            raise ValueError('nonempty depth image and planar scan required')
        if rgb.dtype != np.uint8 or rgb.shape != depth.shape + (3,):
            raise ValueError('aligned uint8 RGB required')
        for name, value in (('depth', depth), ('ranges', ranges)):
            if value.dtype.kind not in 'fiu' or not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f'finite nonnegative {name} required')
        object.__setattr__(self, 'step', step)
        object.__setattr__(self, 'pose', pose)
        for name, value in (('depth', depth), ('rgb', rgb), ('ranges', ranges)):
            frozen = np.array(value, copy=True)
            frozen.flags.writeable = False
            object.__setattr__(self, name, frozen)


class CPUFourModuleControllerV35:
    """Global belief planning and local public-graph return protection.

    Mode changes only the posterior's RGB/geometry intervention. Every mode
    has the same graph, two public sensor/visibility templates, action costs,
    common prefix and planning algorithm. It is a controlled CPU mechanism,
    not the trained four-module network or unknown-map navigation.
    """
    def __init__(self, models, templates, prefix_actions, *, mode='S',
                 total_budget=42, informative_nodes=None, planner=None):
        self.models = tuple(models)
        if len(self.models) != 2 or mode not in MODES_V35:
            raise ValueError('two public models and a declared mode required')
        self.budget = _integer(total_budget, 'total budget')
        if self.budget < 1:
            raise ValueError('positive total budget required')
        a, b = self.models
        self.poses = tuple(tuple(pose) for pose in a.poses)
        self.edges = tuple(tuple((name, int(node)) for name, node in row) for row in a.edges)
        self.anchor = int(a.anchor)
        if (a.poses, a.edges, a.anchor) != (b.poses, b.edges, b.anchor):
            raise ValueError('both public hypotheses must share the legal graph')
        if tuple(templates.poses) != self.poses:
            raise ValueError('public sensor and reward template node order differs')
        if not self.poses or len(set(self.poses)) != len(self.poses) or not 0 <= self.anchor < len(self.poses):
            raise ValueError('unique public graph and anchor required')
        self.node_for_pose = {pose: n for n, pose in enumerate(self.poses)}
        self.links = []
        if len(self.edges) != len(self.poses):
            raise ValueError('one edge row per public pose required')
        for row in self.edges:
            links = dict(row)
            if len(links) != len(row) or any(action not in ACTION_NAMES or
                    not 0 <= node < len(self.poses) for action, node in row):
                raise ValueError('unambiguous legal action graph required')
            self.links.append(links)
        self.return_distances = self._return_distances()
        self.prefix_actions = tuple(prefix_actions)
        if len(self.prefix_actions) > self.budget:
            raise ValueError('common prefix exceeds total budget')
        cursor = self.anchor
        for action in self.prefix_actions:
            if action not in self.links[cursor]:
                raise ValueError('common prefix is not legal on the public graph')
            cursor = self.links[cursor][action]
        if cursor != self.anchor:
            raise ValueError('common prefix must restore the anchor pose')
        self.mode = mode
        self.belief = ObservationBeliefV35(templates, mode)
        if informative_nodes is None:
            informative_nodes = tuple(n for n in range(len(self.poses))
                if templates.template_information(n)['informative'])
        self.informative_nodes = frozenset(_integer(n, 'information node') for n in informative_nodes)
        if any(n < 0 or n >= len(self.poses) for n in self.informative_nodes):
            raise ValueError('information node outside public graph')
        if planner is None:
            from nso.online_planner_v35 import ForecastBeliefPlannerV35
            planner = ForecastBeliefPlannerV35(self.models, self.informative_nodes)
        self.planner = planner
        self.observation = self.state = None
        self.pending = self.selected = None
        self.terminal_reason = None
        self.masks = (0, 0)
        self.visited_nodes = set()
        self.seen_frame_ids = set()
        self.calls = []
        self.plans = []
        self.posterior_receipts = []
        self._accepting_observation = None
        self._topology_receipt = self._action_assessment = None

    def _return_distances(self):
        reverse = [[] for _ in self.poses]
        for node, row in enumerate(self.edges):
            for _, destination in row:
                reverse[destination].append(node)
        distance = {self.anchor: 0}; queue = deque([self.anchor])
        while queue:
            node = queue.popleft()
            for previous in reverse[node]:
                if previous not in distance:
                    distance[previous] = distance[node] + 1
                    queue.append(previous)
        return distance

    def _record(self, module, operation, **fields):
        row = {'module': module, 'operation': operation, **deepcopy(fields)}
        self.calls.append(row)
        return row

    def update_semantic(self, observation):
        """OV-SDF semantic prior and IGCR residuals share one sensor update."""
        if observation is not self._accepting_observation:
            raise ValueError('semantic update requires a validated paid observation')
        receipt = self.belief.update(pose=observation.pose,
            depth=observation.depth, rgb=observation.rgb,
            ranges=observation.ranges, step=observation.step)
        self.posterior_receipts.append(deepcopy(receipt))
        self._record('OV-SDF', 'observed_semantic_and_map_state',
            step=observation.step, measured_map_sha256=observation.measured_map_sha256,
            sensor_sha256=arrays_sha256(depth=observation.depth,
                rgb=observation.rgb, ranges=observation.ranges),
            semantic_log_odds=receipt['semantic_log_odds'],
            class_distinct_xy=receipt['class_distinct_xy'],
            class_conflict=receipt['class_conflict'],
            semantic_enabled=self.mode != 'G',
            measured_quality_available=False,
            measured_map_source='driver observed mapper; digest only',
            prior_source='same two public templates for every mode')
        self._record('IGCR', 'paid_geometry_residual_feedback',
            step=observation.step, geometry_log_odds=receipt['geometry_log_odds'],
            geometry_applied_log_odds=receipt['geometry_applied_log_odds'],
            new_geometry_pose=receipt['new_geometry_pose'],
            residuals=receipt['residuals'], probabilities=receipt['probabilities'],
            feedback_enabled=self.mode != 'swapped_no_feedback',
            actual_reconstruction_quality_credit=False)
        return receipt

    def compute_reward(self, previous_masks=None):
        """Expected public potential only, never a measured mesh score."""
        rows = [model.terminal(mask) for model, mask in zip(self.models, self.masks)]
        probabilities = self.belief.probabilities
        expected = sum(p * float(row['joint']) for p, row in zip(probabilities, rows))
        gains = None if previous_masks is None else [
            float(row['joint']) - float(model.terminal(old)['joint'])
            for row, model, old in zip(rows, self.models, previous_masks)]
        return {'hypothesis_potential': deepcopy(rows),
                'probability_weighted_potential': expected,
                'new_pose_potential_gain': gains,
                'scope': 'public visibility forecast; not measured C_map/F1/quality'}

    def update_topo(self):
        if self.state is None:
            raise ValueError('paid observation required before topology update')
        if self._topology_receipt is not None and self._topology_receipt['step'] == self.state['step']:
            return deepcopy(self._topology_receipt)
        self._topology_receipt = self._record('STGHP', 'update_public_topology_state',
            step=self.state['step'], node=self.state['node'],
            visited_nodes=sorted(self.visited_nodes),
            masks_hex=[hex(mask) for mask in self.masks],
            return_distance=self.return_distances.get(self.state['node']),
            remaining_budget=self.budget - self.state['step'],
            source='same known safe graph and prior potential for all modes')
        return deepcopy(self._topology_receipt)

    def accept(self, observation):
        if type(observation) is not ObservationV35:
            raise TypeError('explicit sanitized ObservationV35 required')
        if self.terminal_reason is not None:
            raise ValueError('episode already stopped')
        if observation.frame_id in self.seen_frame_ids:
            raise ValueError('frame identity already consumed')
        if observation.step > self.budget:
            raise ValueError('paid action budget exceeded')
        if observation.pose not in self.node_for_pose:
            raise ValueError('observed odometry outside public graph')
        node = self.node_for_pose[observation.pose]
        if self.observation is None:
            if observation.step != 0 or observation.action is not None or node != self.anchor or observation.collision:
                raise ValueError('real action-zero observation at anchor required')
        else:
            old = self.observation
            if self.pending is None or observation.step != old.step + 1 or observation.action != self.pending['action']:
                raise ValueError('new observation does not match issued paid action')
            expected = self.state['node'] if observation.collision else self.pending['node']
            if node != expected:
                raise ValueError('odometry disagrees with issued paid action')
        # All boundary checks run before changing masks, belief or receipts.
        self._accepting_observation = observation
        try:
            receipt = self.update_semantic(observation)
        finally:
            self._accepting_observation = None
        previous = self.masks
        self.masks = tuple(old | int(model.observed_masks[node])
            for old, model in zip(previous, self.models))
        repeated = node in self.visited_nodes
        self.visited_nodes.add(node)
        self.observation = observation
        self.pending = None
        self.seen_frame_ids.add(observation.frame_id)
        self.state = {'step': observation.step, 'pose': list(observation.pose),
            'node': node, 'remaining_budget': self.budget - observation.step,
            'probabilities': receipt['probabilities'],
            'masks_hex': [hex(mask) for mask in self.masks],
            'repeated_pose': repeated, 'potential': self.compute_reward(previous),
            'measured_map_sha256': observation.measured_map_sha256}
        self.update_topo()
        if observation.collision:
            self.terminal_reason = 'collision_observed'
        elif observation.step == self.budget:
            self.terminal_reason = ('sensor_or_budget_end_returned' if node == self.anchor
                                    else 'budget_or_sensor_end_without_return')
        return deepcopy(self.state)

    def select_target(self):
        if self.state is None:
            raise ValueError('consume action-zero observation first')
        if self.pending is not None:
            raise ValueError('consume previous paid observation before another target')
        if self.terminal_reason is not None:
            raise ValueError('episode already stopped')
        step, node = self.state['step'], self.state['node']
        if self.selected is not None and self.selected['step'] == step:
            return deepcopy(self.selected)
        if step < len(self.prefix_actions):
            action = self.prefix_actions[step]
            result = {'action': action, 'phase': 'common_forced_prefix',
                      'posterior_used_to_choose_action': False}
        else:
            result = deepcopy(self.planner.select(node,
                self.budget - step, self.masks, self.belief.probabilities[0],
                geometry_feedback=self.mode != 'swapped_no_feedback',
                excluded_information_nodes=tuple(sorted(self.visited_nodes))))
            if not isinstance(result, dict) or 'action' not in result:
                raise ValueError('online planner must return an action receipt')
            result['phase'] = 'online_belief_global_planning'
            result['posterior_used_to_choose_action'] = True
        action = result['action']
        destination = self.links[node].get(action) if action is not None else None
        selected = {'step': step, 'from_node': node, 'action': action,
            'node': destination,
            'pose': None if destination is None else list(self.poses[destination]),
            'probabilities': list(self.belief.probabilities),
            'planning': result}
        self.selected = deepcopy(selected)
        self.plans.append(deepcopy(selected))
        self._record('STGHP', 'select_online_global_next_pose', **selected)
        return deepcopy(selected)

    def assess_action(self, selected):
        if self.state is None:
            raise ValueError('paid observation required before local validation')
        if (self.selected is None or selected != self.selected or
                selected['step'] != self.state['step'] or
                selected['from_node'] != self.state['node']):
            raise ValueError('current global selection required before local validation')
        if self._action_assessment is not None and self._action_assessment['step'] == self.state['step']:
            return deepcopy(self._action_assessment)
        action = selected['action']
        remaining = self.budget - self.state['step']
        destination = self.links[self.state['node']].get(action) if action is not None else None
        return_cost = self.return_distances.get(destination)
        allowed = (action is not None and destination is not None and
                   return_cost is not None and 1 + return_cost <= remaining)
        receipt = self._record('RPN-UQ', 'validate_public_edge_and_return_budget',
            step=self.state['step'], action=action, node=destination,
            remaining_budget=remaining, return_actions=return_cost, allowed=allowed,
            learned_uncertainty=False, safety_source='public safe graph; not noisy-map autonomous safety')
        self._action_assessment = receipt
        return deepcopy(receipt)

    def next_action(self):
        if self.state is None:
            raise ValueError('consume action-zero observation first')
        if self.pending is not None:
            raise ValueError('consume previous paid observation before another action')
        if self.terminal_reason is not None:
            return None
        selected = self.select_target()
        assessment = self.assess_action(selected)
        if not assessment['allowed']:
            self.terminal_reason = ('returned_no_affordable_action' if self.state['node'] == self.anchor
                                    else 'no_affordable_public_return_action')
            return None
        self.pending = {'action': selected['action'], 'node': assessment['node']}
        return selected['action']

    def summary(self):
        return {'mode': self.mode, 'total_budget': self.budget,
            'actual_paid_actions': 0 if self.state is None else self.state['step'],
            'terminal_reason': self.terminal_reason,
            'current_pose': None if self.state is None else self.state['pose'],
            'anchor_pose': list(self.poses[self.anchor]),
            'common_prefix_actions': len(self.prefix_actions),
            'global_plans': len(self.plans),
            'online_global_plans': sum(row['planning']['phase'] == 'online_belief_global_planning' for row in self.plans),
            'module_calls': len(self.calls),
            'modules_called': sorted({row['module'] for row in self.calls}),
            'probabilities': list(self.belief.probabilities),
            'public_prior_template_count': 2,
            'actual_hypothesis_input': False, 'evaluation_quality_input': False,
            'trained_quality_head': False, 'trained_RPN_UQ': False,
            'full_method_efficacy_proven': False,
            'scope': 'online controlled two-template CPU mechanism with public safe graph and artificial RGB categories'}


ObservedANSRuntimeV35 = CPUFourModuleControllerV35
