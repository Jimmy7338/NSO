"""Metric-aligned frontier ablations on top of the frozen controller.

Only unknown cells that can still become reachable robot-center cells contribute
to estimated gain. Known obstacles can already rule out cells before those cells
are themselves observed. This is not an oracle: unseen obstacles stay unknown.
"""
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from nso.frontier_policy import FrontierPolicy, trace_path
from nso.frontier_policy import CROSS
from nso.coverage_planner_v2 import CoveragePlannerV2
from nso.grid_mechanisms import action_cost
from utils.grid_geometry import inflated_obstacles


class NavigableFrontierPolicy(FrontierPolicy):
    def __init__(self, config, max_candidates=24, metric_gain=True, exact_cost=True):
        self.metric_gain = metric_gain
        self.exact_cost = exact_cost
        super().__init__(config, 'gain_per_cost', max_candidates)

    def _select(self, obs, traversable):
        self._eligible = ~inflated_obstacles(
            obs.belief == 1, self.config.robot_radius_m / self.config.resolution_m)
        return super()._select(obs, traversable)

    def _gain_at(self, belief, cell, heading):
        from utils.grid_geometry import visible_mask
        visible = visible_mask(belief == 1, cell, heading,
            int(self.config.sensor_range_m / self.config.resolution_m), self.config.sensor_fov_deg)
        target = (belief == -1) & (self._eligible if self.metric_gain else True)
        return int(np.count_nonzero(visible & target))

    def _candidate_score(self, obs, cell, heading, gain, distance, parent, base_score):
        if not self.exact_cost:
            return base_score
        path = [obs.position] + trace_path(parent, obs.position, cell)
        return gain / (1 + action_cost(path, obs.heading, heading))


class ProjectedCoveragePolicy(CoveragePlannerV2):
    """Retreat true frontiers to safe viewpoints, not a wall-crossing band.

    Spatial bins avoid losing a long frontier when just four component samples
    happen to be uninformative. A cheap square-window upper bound screens the
    candidates before exact occlusion-aware ray casting.
    """
    def _gain_mask(self, obs):
        return (obs.belief == -1) & ~inflated_obstacles(
            obs.belief == 1, self.config.robot_radius_m / self.config.resolution_m)

    def _candidates(self, obs, safe, distance):
        reachable = safe & (distance >= 0)
        frontier = (obs.belief == 0) & binary_dilation(obs.belief == -1, structure=CROSS)
        retreat, indices = distance_transform_edt(~reachable, return_indices=True)
        limit = self.config.robot_radius_m / self.config.resolution_m + 2
        frontier &= retreat <= limit
        cells = np.argwhere(frontier)
        radius = int(self.config.sensor_range_m / self.config.resolution_m)
        binsize = max(3, radius // 2)
        pool = {}
        for r, c in cells:
            target = tuple(map(int, indices[:, r, c]))
            key = (target[0]//binsize, target[1]//binsize)
            center = ((key[0]+.5)*binsize, (key[1]+.5)*binsize)
            error = (target[0]-center[0])**2 + (target[1]-center[1])**2
            if key not in pool or (error, target) < pool[key]:
                pool[key] = (error, target)
        gainmask = self._gain_mask(obs)
        integral = np.pad(gainmask, ((1,0),(1,0))).cumsum(0).cumsum(1)
        ranked = []
        for _, cell in pool.values():
            r, c = cell
            r0, r1 = max(0,r-radius), min(safe.shape[0],r+radius+1)
            c0, c1 = max(0,c-radius), min(safe.shape[1],c+radius+1)
            rough = int(integral[r1,c1]-integral[r0,c1]-integral[r1,c0]+integral[r0,c0])
            ranked.append((-rough/(int(distance[cell])+radius), cell))
        ordered = [cell for _, cell in sorted(ranked) if cell != obs.position]
        return [obs.position] + ordered[:self.max_candidates-1]
