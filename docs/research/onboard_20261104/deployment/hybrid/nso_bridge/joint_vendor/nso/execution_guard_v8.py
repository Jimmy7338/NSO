"""Sensor-map execution gate for the RPN feasibility interface.

This deterministic CPU component is separate from learned uncertainty. It
certifies only the current observed grid; it never reads simulator truth and
does not promise collision-free motion under map error or localization drift.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.csgraph import dijkstra
from nso.route_coverage_v2 import orientation_graph, recover_actions
from utils.grid_geometry import DIRECTIONS, inflated_obstacles


@dataclass(frozen=True)
class ActionAssessment:
    allowed: bool
    reason: str
    target: tuple


@dataclass(frozen=True)
class ReturnPlan:
    available: bool
    reason: str
    actions: tuple
    paid_cost: int


class ObservedExecutionGuard:
    def __init__(self, resolution_m, robot_radius_m):
        if resolution_m <= 0 or robot_radius_m < 0:
            raise ValueError('invalid geometry units')
        self.resolution_m = float(resolution_m)
        self.robot_radius_m = float(robot_radius_m)

    def safe_grid(self, belief):
        belief = np.asarray(belief)
        if belief.ndim != 2 or not np.isin(belief, [-1, 0, 1]).all():
            raise ValueError('expected unknown/free/occupied observation grid')
        return ~inflated_obstacles(belief != 0, self.robot_radius_m / self.resolution_m)

    @staticmethod
    def _inside(safe, position):
        return (len(position) == 2 and 0 <= position[0] < safe.shape[0]
                and 0 <= position[1] < safe.shape[1])

    @staticmethod
    def _heading(heading):
        if heading not in (0, 1, 2, 3):
            raise ValueError('expected a discrete camera/base heading')

    def assess(self, belief, position, heading, action, remaining_actions):
        self._heading(heading)
        position = tuple(position)
        if action not in ('forward', 'left', 'right'):
            raise ValueError('only paid movement/rotation may pass this gate')
        target = position
        if action == 'forward':
            dr, dc = DIRECTIONS[heading]
            target = (position[0] + int(dr), position[1] + int(dc))
        if remaining_actions < 1:
            return ActionAssessment(False, 'no_action_budget', target)
        safe = self.safe_grid(belief)
        if not self._inside(safe, position) or not safe[position]:
            return ActionAssessment(False, 'current_footprint_not_known_safe', target)
        if not self._inside(safe, target) or not safe[target]:
            return ActionAssessment(False, 'next_footprint_not_known_safe', target)
        return ActionAssessment(True, 'latest_observed_grid_allows_action', target)

    def return_plan(self, belief, position, heading, anchor, remaining_actions):
        """Shortest known-safe return including restoration of anchor heading.

        Every returned action still needs assess() on the latest observation.
        An unavailable plan means stop; no unknown cells or unsafe start cells
        are cleared as an escape exception.
        """
        self._heading(heading)
        if len(anchor) != 3:
            raise ValueError('return anchor must include row, column and heading')
        self._heading(anchor[2])
        if remaining_actions < 0:
            raise ValueError('negative remaining action budget')
        safe = self.safe_grid(belief)
        position, goal = tuple(position), tuple(anchor[:2])
        if not self._inside(safe, position) or not safe[position]:
            return ReturnPlan(False, 'current_footprint_not_known_safe', (), 0)
        if not self._inside(safe, goal) or not safe[goal]:
            return ReturnPlan(False, 'anchor_footprint_not_known_safe', (), 0)
        graph, cells, ids = orientation_graph(safe)
        start = int(ids[position]) * 4 + int(heading)
        end = int(ids[goal]) * 4 + int(anchor[2])
        costs, predecessors = dijkstra(graph, directed=True, indices=start, return_predecessors=True)
        if not np.isfinite(costs[end]):
            return ReturnPlan(False, 'anchor_disconnected_in_latest_map', (), 0)
        if costs[end] > remaining_actions:
            return ReturnPlan(False, 'known_return_exceeds_remaining_budget', (), int(costs[end]))
        actions = tuple(recover_actions(predecessors, start, end, cells))
        if len(actions) != round(costs[end]):
            raise ValueError('return graph cost does not match paid actions')
        return ReturnPlan(True, 'known_safe_return_within_budget', actions, len(actions))
