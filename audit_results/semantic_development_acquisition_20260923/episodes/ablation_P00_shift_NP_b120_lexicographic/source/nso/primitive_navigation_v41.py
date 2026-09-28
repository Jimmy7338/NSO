"""Primitive-cost navigation and return guard for a supplied public graph.

This module does not generate semantic/view-quality utilities or query a World.
Utilities must come from a separate observed-data predictor. The input graph is
a shared navigation prior, not a claim of navigation without prior knowledge.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import heapq
import json
import math

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40


@dataclass(frozen=True)
class PrimitiveStateV41:
    node: str
    heading: int

    def __post_init__(self):
        if not isinstance(self.node, str) or not self.node:
            raise ValueError('nonempty public node identifier required')
        if type(self.heading) is not int or not 0 <= self.heading < 12:
            raise ValueError('integer heading in 0..11 required')


@dataclass(frozen=True)
class PrimitiveRouteV41:
    actions: tuple
    states: tuple

    @property
    def cost(self):
        return len(self.actions)


class PublicPrimitiveGraphV41:
    """Expand declared coarse edges into 0.25 m moves and 30 degree turns."""
    def __init__(self, spec, *, camera_height_m=.9):
        if isinstance(camera_height_m, bool) or not isinstance(camera_height_m, (int, float)) or not math.isfinite(camera_height_m) or camera_height_m <= 0:
            raise ValueError('positive finite public camera height required')
        self.camera_height_m = float(camera_height_m)
        keys = {'schema_version', 'source_kind', 'nodes', 'edges'}
        if not isinstance(spec, Mapping) or set(spec) != keys:
            raise ValueError('public graph field whitelist violation')
        if spec['schema_version'] != 'v41.public_navigation.v1' or spec['source_kind'] != 'provided_navigation_prior':
            raise ValueError('explicit shared navigation prior required')
        nodes = spec['nodes']
        if not isinstance(nodes, Mapping) or not 1 <= len(nodes) <= 256:
            raise ValueError('bounded nonempty public node mapping required')
        self.positions, self.original_nodes = {}, set(nodes)
        self._position_names = {}
        for name, xy in nodes.items():
            if not isinstance(name, str) or not name or name.startswith('@'):
                raise ValueError('nonempty public node name without reserved @ prefix required')
            if type(xy) not in (list, tuple) or len(xy) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in xy):
                raise ValueError('finite XY pair required')
            key = tuple(round(float(x), 9) for x in xy)
            if key in self._position_names:
                raise ValueError('duplicate public node position')
            self.positions[name] = key
            self._position_names[key] = name
        edges = spec['edges']
        if type(edges) is not list or len(edges) > 1024:
            raise ValueError('bounded edge list required')
        self._forward, self._blocked = {}, set()
        seen = set()
        for edge in edges:
            if type(edge) not in (list, tuple) or len(edge) != 2 or any(x not in self.original_nodes for x in edge) or edge[0] == edge[1]:
                raise ValueError('edge must connect two declared distinct public nodes')
            edge_key = tuple(sorted(edge))
            if edge_key in seen:
                raise ValueError('duplicate undirected public edge')
            seen.add(edge_key)
            a, b = map(lambda name: np.asarray(self.positions[name]), edge)
            delta = b - a
            length = float(np.linalg.norm(delta))
            steps = round(length / .25)
            if not 1 <= steps <= 200 or not math.isclose(length, .25 * steps, abs_tol=1e-8):
                raise ValueError('public edge must be an integer number of 0.25 m moves')
            angle = math.atan2(delta[1], delta[0])
            heading = round(angle / (math.pi / 6)) % 12
            expected = .25 * steps * np.array([math.cos(heading * math.pi / 6), math.sin(heading * math.pi / 6)])
            if not np.allclose(delta, expected, rtol=0, atol=1e-8):
                raise ValueError('public edge bearing must be a multiple of 30 degrees')
            chain = [edge[0]]
            for step in range(1, steps):
                key = tuple(round(float(x), 9) for x in a + delta * step / steps)
                if key not in self._position_names:
                    name = '@' + hashlib.sha256(repr(key).encode()).hexdigest()[:16]
                    self.positions[name] = key
                    self._position_names[key] = name
                chain.append(self._position_names[key])
            chain.append(edge[1])
            for left, right in zip(chain, chain[1:]):
                for start, end, direction in ((left, right, heading), (right, left, (heading + 6) % 12)):
                    state = PrimitiveStateV41(start, direction)
                    if state in self._forward and self._forward[state] != end:
                        raise ValueError('ambiguous public forward transition')
                    self._forward[state] = end
        if len(self.positions) > 4096:
            raise ValueError('expanded graph exceeds node cap')
        self.input_sha256 = hashlib.sha256(json.dumps({'graph': spec, 'camera_height_m': self.camera_height_m},
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def validate_state(self, state):
        if type(state) is not PrimitiveStateV41 or state.node not in self.positions:
            raise ValueError('state must belong to expanded public graph')

    def block_observed_edge(self, left, right):
        """Caller may remove a known edge; this cannot add oracle traversability."""
        if left not in self.positions or right not in self.positions:
            raise ValueError('unknown public edge endpoint')
        if not any(s.node == left and target == right for s, target in self._forward.items()):
            raise ValueError('edge was never publicly declared')
        self._blocked.add(frozenset((left, right)))

    def successor(self, state, action):
        self.validate_state(state)
        if action == 'left':
            return PrimitiveStateV41(state.node, (state.heading + 1) % 12)
        if action == 'right':
            return PrimitiveStateV41(state.node, (state.heading - 1) % 12)
        if action == 'observe':
            return state
        if action == 'forward':
            target = self._forward.get(state)
            if target is not None and frozenset((state.node, target)) not in self._blocked:
                return PrimitiveStateV41(target, state.heading)
            raise ValueError('forward motion is absent or blocked in the public graph')
        raise ValueError('unknown primitive action')

    def route(self, start, goal):
        self.validate_state(start)
        self.validate_state(goal)
        frontier = [(0, start.node, start.heading)]
        distance, previous = {start: 0}, {}
        while frontier:
            cost, node, heading = heapq.heappop(frontier)
            state = PrimitiveStateV41(node, heading)
            if cost != distance[state]:
                continue
            if state == goal:
                actions, states = [], [state]
                while state != start:
                    state, action = previous[state]
                    actions.append(action)
                    states.append(state)
                return PrimitiveRouteV41(tuple(reversed(actions)), tuple(reversed(states)))
            for action in ('forward', 'left', 'right'):
                try:
                    successor = self.successor(state, action)
                except ValueError:
                    continue
                new_cost = cost + 1
                if new_cost < distance.get(successor, math.inf):
                    distance[successor] = new_cost
                    previous[successor] = (state, action)
                    heapq.heappush(frontier, (new_cost, successor.node, successor.heading))
        return None

    def state_from_observation(self, observation):
        if type(observation) is not PaidRGBDObservationV40:
            raise TypeError('strict paid RGB-D observation required')
        if abs(observation.world_from_camera[2, 3] - self.camera_height_m) > 1e-6:
            raise ValueError('observed camera height changed outside the motion contract')
        position = observation.world_from_camera[:2, 3]
        nearby = [node for node, xy in self.positions.items() if np.linalg.norm(position - xy) <= 1e-6]
        if len(nearby) != 1:
            raise ValueError('observed pose does not identify one public primitive node')
        forward = observation.world_from_camera[:3, 2]
        if abs(forward[2]) > 1e-6:
            raise ValueError('this navigation adapter requires a level optical camera')
        angle = math.atan2(forward[1], forward[0])
        heading = round(angle / (math.pi / 6)) % 12
        wrapped = math.atan2(math.sin(angle-heading*math.pi/6), math.cos(angle-heading*math.pi/6))
        if abs(wrapped) > 1e-6:
            raise ValueError('observed yaw is not a declared 30 degree state')
        yaw = heading * math.pi/6
        expected_rotation = np.array([[math.sin(yaw), 0, math.cos(yaw)],
                                      [-math.cos(yaw), 0, math.sin(yaw)], [0, -1, 0]])
        if not np.allclose(observation.world_from_camera[:3, :3], expected_rotation, rtol=0, atol=1e-6):
            raise ValueError('observed camera roll or pitch changed outside the motion contract')
        return PrimitiveStateV41(nearby[0], heading)


class ReturnAwarePrimitiveRouterV41:
    """Action-0 routing after observation, with full terminal XY/yaw reserve.

The upstream caller supplies predicted utilities; this guard does not establish
their provenance or implement semantic quality forecasting.
"""
    def __init__(self, graph, *, home, budget):
        if type(graph) is not PublicPrimitiveGraphV41:
            raise TypeError('validated public primitive graph required')
        graph.validate_state(home)
        if type(budget) is not int or budget < 1:
            raise ValueError('positive integer primitive budget required')
        self.graph, self.home, self.budget = graph, home, budget
        self.state, self.step, self.pending = None, -1, None
        self.frames = set()

    def accept(self, observation, *, execution_outcome='success'):
        if type(observation) is not PaidRGBDObservationV40:
            raise TypeError('strict paid RGB-D observation required')
        if observation.paid_step != self.step + 1 or observation.frame_id in self.frames:
            raise ValueError('one unique observation per consecutive paid step required')
        if observation.paid_step > self.budget:
            raise ValueError('paid action budget exceeded')
        actual = self.graph.state_from_observation(observation)
        if self.step == -1:
            if actual != self.home or execution_outcome != 'success':
                raise ValueError('initial grant must match declared home pose and orientation')
        else:
            if self.pending is None:
                raise ValueError('observation requires a previously submitted action')
            if execution_outcome == 'collision' and self.pending['action'] == 'forward':
                if actual != self.state:
                    raise ValueError('collision model must leave pose unchanged')
                target = self.pending['expected_state'].node
                self.graph.block_observed_edge(self.state.node, target)
            elif execution_outcome != 'success' or actual != self.pending['expected_state']:
                raise ValueError('observation disagrees with the submitted primitive')
        self.state, self.step, self.pending = actual, observation.paid_step, None
        self.frames.add(observation.frame_id)
        return dict(paid_step=self.step, remaining=self.budget-self.step,
                    observation_sha256=observation.sha256(), public_graph_sha256=self.graph.input_sha256)

    def choose(self, candidate_utilities):
        if self.state is None or self.pending is not None:
            raise ValueError('accept the current observation before choosing another action')
        if not isinstance(candidate_utilities, Mapping):
            raise TypeError('mapping of public view states to upstream predicted utility required')
        remaining = self.budget - self.step
        scored, feasible = [], []
        for target, utility in candidate_utilities.items():
            self.graph.validate_state(target)
            if isinstance(utility, bool) or not isinstance(utility, (int, float)) or not math.isfinite(utility) or utility < 0:
                raise ValueError('finite nonnegative upstream utility required')
            outbound, inbound = self.graph.route(self.state, target), self.graph.route(target, self.home)
            cost = outbound.cost + 1 + inbound.cost if outbound is not None and inbound is not None else None
            row = dict(node=target.node, heading=target.heading, utility=float(utility),
                       total_with_observation_and_return=cost, feasible=cost is not None and cost <= remaining)
            scored.append(row)
            if row['feasible'] and utility > 0:
                feasible.append((-(utility/(outbound.cost+1)), target.node, target.heading, target, outbound))
        if feasible:
            _, _, _, target, route = min(feasible)
            action = route.actions[0] if route.actions else 'observe'
            reason = 'upstream_goal_with_paid_return_reserve'
        else:
            route = self.graph.route(self.state, self.home)
            if route is None or route.cost > remaining:
                return dict(action='blocked', reason='no_budget_feasible_return', paid_step=self.step, candidates=scored)
            if not route.actions:
                return dict(action='stop', reason='home_xy_and_yaw_reached', paid_step=self.step, candidates=scored)
            target, action, reason = self.home, route.actions[0], 'return'
        expected = self.graph.successor(self.state, action)
        return_route = self.graph.route(expected, self.home)
        if return_route is None or 1 + return_route.cost > remaining:
            raise AssertionError('local primitive violates return reserve')
        self.pending = dict(action=action, expected_state=expected)
        return dict(action=action, reason=reason, paid_step=self.step,
                    target=dict(node=target.node, heading=target.heading),
                    return_cost_after_action=return_route.cost, candidates=scored,
                    utility_source='upstream predictor; provenance must be checked by caller',
                    fixed_prefix_actions=0)
