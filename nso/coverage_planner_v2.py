"""Observation-only feasibility prototype; not a reproduction of TARE/FALCON.

Planning uses known, footprint-safe cells. Observations made while turning are
paid for with primitive actions, just as in the frozen baselines. No room labels,
unseen semantics, evaluator masks, or learned reachability enter this module.
"""
from collections import deque
import math
import numpy as np
from scipy.ndimage import binary_dilation, label

from nso.frontier_policy import CROSS, Decision, shortest_paths, trace_path
from nso.grid_mechanisms import MechanismPolicy, action_cost
from utils.grid_geometry import DIRECTIONS, inflated_obstacles, visible_mask


class CostCorrectedPolicy(MechanismPolicy):
    """Single-factor ablation: correct the cost, retain the legacy controller."""
    def _candidate_score(self, obs, cell, heading, gain, distance, parent, base_score):
        path = [obs.position] + trace_path(parent, obs.position, cell)
        corrected = gain / (1 + action_cost(path, obs.heading, heading))
        return super()._candidate_score(obs, cell, heading, gain, distance, parent, corrected)


def path_actions(start, path, heading):
    """Executable unit-action path; headings and turns have the same unit cost."""
    actions = []
    poses = []
    position = start
    for target in path:
        h = DIRECTIONS.index((target[0] - position[0], target[1] - position[1]))
        while heading != h:
            turn = (h - heading) % 4
            action = 'right' if turn in (1, 2) else 'left'
            heading = (heading + (1 if action == 'right' else -1)) % 4
            actions.append(action)
            poses.append((position, heading))
        position = target
        actions.append('forward')
        poses.append((position, heading))
    return actions, poses, heading


class CoveragePlannerV2:
    def __init__(self, config, max_candidates=24, bundle=True, path_gain=True):
        if max_candidates < 1:
            raise ValueError('max_candidates must be positive')
        self.config = config
        self.max_candidates = max_candidates
        self.bundle, self.path_gain = bundle, path_gain
        self.reset()

    def reset(self):
        self.goal = None
        self.goal_heading = None
        self._actions = deque()
        self._gain = self._distance = 0
        self._score = 0.
        self._scans = 0
        self.goal_count = self.repairs = 0
        self.selection_audit = []
        self.attempts = []
        self._active = None

    def _visible(self, obs, cell, heading):
        return visible_mask(obs.belief == 1, cell, heading,
                            int(self.config.sensor_range_m / self.config.resolution_m),
                            self.config.sensor_fov_deg)

    def _safe(self, obs):
        # Unknown footprint cells are not treated as certified free space.
        safe = ~inflated_obstacles(obs.belief != 0,
                                   self.config.robot_radius_m / self.config.resolution_m)
        safe[obs.position] = True
        return safe

    def _candidates(self, obs, safe, distance):
        margin = math.ceil(self.config.robot_radius_m / self.config.resolution_m + math.sqrt(2)/2)
        boundary = safe & binary_dilation(obs.belief == -1, structure=CROSS,
                                          iterations=margin + 1) & (distance >= 0)
        components, count = label(boundary, structure=CROSS)
        pool = set()
        for k in range(1, count + 1):
            cells = np.argwhere(components == k)
            order = np.lexsort((cells[:, 1], cells[:, 0], distance[tuple(cells.T)]))
            for i in np.linspace(0, len(order)-1, min(4, len(order)), dtype=int):
                pool.add(tuple(map(int, cells[order[i]])))
        # Preserve near and far representatives instead of truncating to nearest.
        ordered = sorted(pool - {obs.position}, key=lambda c: (distance[c], c))
        if len(ordered) > self.max_candidates - 1:
            ordered = [ordered[i] for i in np.linspace(0, len(ordered)-1,
                        self.max_candidates-1, dtype=int)]
        return [obs.position] + ordered

    def _select(self, obs, safe):
        distance, parent = shortest_paths(safe, obs.position)
        unknown = self._gain_mask(obs)
        scored = []
        rows = []
        for cell in self._candidates(obs, safe, distance):
            path = trace_path(parent, obs.position, cell)
            actions, poses, arrival = path_actions(obs.position, path, obs.heading)
            # Known-map predictions only: unknown is transparent for predicted
            # sensing gain, never for collision certification.
            travel_seen = np.zeros_like(unknown)
            if self.path_gain:
                stride = max(1, int(self.config.sensor_range_m / self.config.resolution_m) // 2)
                for j, (position, heading) in enumerate(poses):
                    if j % stride == 0 or j == len(poses)-1:
                        travel_seen |= self._visible(obs, position, heading)
            views = [self._visible(obs, cell, h) for h in range(4)]
            choices = []
            if self.bundle:
                # A complete paid rotation at the endpoint. Arrival view has
                # already been observed; the other three views cost 3 actions.
                choices.append((actions + ['right']*3,
                                np.logical_or.reduce(views) | travel_seen,
                                (arrival+3) % 4))
            else:
                for h in range(4):
                    extra, current = [], arrival
                    while current != h:
                        turn = (h-current) % 4
                        action = 'right' if turn in (1, 2) else 'left'
                        current = (current + (1 if action == 'right' else -1)) % 4
                        extra.append(action)
                    choices.append((actions + extra, views[h] | travel_seen, h))
            for sequence, seen, final_heading in choices:
                if not sequence or len(sequence) > self.config.max_steps - obs.step:
                    continue
                gain = int(np.count_nonzero(unknown & seen))
                if not gain:
                    continue
                score = gain / len(sequence)
                rows.append(dict(step=obs.step, goal=list(cell), heading=final_heading,
                                 gain=gain, actions=len(sequence), score=score,
                                 selected=False))
                scored.append(((-score, len(sequence), cell, final_heading),
                               cell, final_heading, sequence, gain, score, len(rows)-1))
        if not scored:
            return False
        _, self.goal, self.goal_heading, actions, self._gain, self._score, i = min(scored)
        self._distance = int(distance[self.goal])
        self._actions = deque(actions)
        self.goal_count += 1
        rows[i]['selected'] = True
        self.selection_audit.extend(rows)
        self._active = dict(start_step=obs.step, goal=list(self.goal),
                            planned_actions=len(actions), predicted_gain=self._gain)
        return True

    def _gain_mask(self, obs):
        return obs.belief == -1

    def _finish(self, obs, status):
        if self._active is not None:
            self.attempts.append(dict(self._active, status=status, end_step=obs.step,
                                     elapsed_actions=obs.step-self._active['start_step']))
            self._active = None

    def observe_outcome(self, obs, episode_done=False):
        if obs.collision:
            self._finish(obs, 'collision')
            self._actions.clear()
            self.goal = None
        elif not self._actions:
            self._finish(obs, 'reached')
            self.goal = None
        elif episode_done:
            self._finish(obs, 'episode_censored')

    def act(self, obs):
        safe = self._safe(obs)
        if self._actions and self._actions[0] == 'forward':
            dr, dc = DIRECTIONS[obs.heading]
            target = (obs.position[0]+dr, obs.position[1]+dc)
            if not (0 <= target[0] < safe.shape[0] and 0 <= target[1] < safe.shape[1] and safe[target]):
                self._finish(obs, 'safety_replan')
                self._actions.clear()
                self.goal = None
                self.repairs += 1
        new_goal = False
        if not self._actions:
            new_goal = self._select(obs, safe)
            if not new_goal:
                self._scans += 1
                return Decision('right' if self._scans <= 4 else 'stop', None, None, 0, 0, 0., False)
            self._scans = 0
        return Decision(self._actions.popleft(), self.goal, self.goal_heading,
                        self._gain, self._distance, self._score, new_goal)

    def diagnostics(self):
        return dict(goal_changes=self.goal_count, safety_replans=self.repairs,
                    controller_attempts=len(self.attempts), semantic_updates=0, rpn_calls=0)
