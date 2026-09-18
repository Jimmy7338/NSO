"""Observed-map coverage candidates and reusable directed paths for V20.

The task rectangle is public. Unknown cells predict possible observation gain
but never permit motion. All costs include paid turns and an anchor return.
"""
from copy import deepcopy
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from scipy.sparse.csgraph import dijkstra
from nso.execution_guard_v8 import ObservedExecutionGuard
from nso.response_candidates_v7 import _path, _actions
from nso.route_coverage_v2 import orientation_graph
from utils.grid_geometry import visible_mask


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} requires an integer >= {minimum}')
    return int(value)


def _pose(value, name):
    if len(value) != 3:
        raise ValueError(f'{name} must be row, column and heading')
    result = tuple(_integer(item, name) for item in value)
    if result[2] >= 4:
        raise ValueError(f'{name} requires a cardinal heading')
    return result


class ObservedRouteSpaceV20:
    def __init__(self, mapper, position, heading, return_anchor, remaining_budget):
        self.mapper = mapper
        if len(position) != 2:
            raise ValueError('position requires row and column')
        self.current = _pose((*position, heading), 'current pose')
        self.anchor = _pose(return_anchor, 'return anchor')
        self.budget = _integer(remaining_budget, 'remaining budget')
        self.belief = np.asarray(mapper.belief).copy()
        if self.belief.shape != tuple(mapper.shape):
            raise ValueError('observed map and mapper shape differ')
        self.belief.setflags(write=False)
        c = mapper.config
        self.safe = ObservedExecutionGuard(c.resolution_m, c.robot_radius_m).safe_grid(self.belief)
        self._visibility = {}
        self.available = False
        self.reason = 'current_or_anchor_not_known_safe'
        if (not self._inside(self.current[:2]) or not self._inside(self.anchor[:2])
                or not self.safe[self.current[:2]] or not self.safe[self.anchor[:2]]):
            return
        graph, self.cells, self.ids = orientation_graph(self.safe)
        self.start = int(self.ids[self.current[:2]]) * 4 + self.current[2]
        self.home = int(self.ids[self.anchor[:2]]) * 4 + self.anchor[2]
        self.outward, self.prev = dijkstra(graph, directed=True, indices=self.start, return_predecessors=True)
        self.backward, self.backprev = dijkstra(graph.T.tocsr(), directed=True, indices=self.home, return_predecessors=True)
        self.available = bool(np.isfinite(self.backward[self.start]) and self.backward[self.start] <= self.budget)
        self.reason = 'available' if self.available else 'no_affordable_observed_return'

    def _inside(self, cell):
        return len(cell) == 2 and all(0 <= item < limit for item, limit in zip(cell, self.safe.shape))

    def visibility(self, cell):
        """Unique possible 2D unknown cells; camera and radar share V19 range."""
        if len(cell) != 2:
            raise ValueError('visibility cell requires row and column')
        key = tuple(_integer(item, 'visibility cell') for item in cell)
        if not self._inside(key):
            raise ValueError('visibility cell lies outside observed map')
        if key not in self._visibility:
            c = self.mapper.config
            self._visibility[key] = visible_mask(self.belief == 1, key, 0,
                int(c.max_depth_m / c.resolution_m), 360.) & (self.belief == -1)
            self._visibility[key].setflags(write=False)
        return self._visibility[key]

    def route_mask(self, route):
        result = np.zeros(self.belief.shape, bool)
        for pose in route['outbound_states'][1:]:
            result |= self.visibility(pose[:2])
        return result

    def route(self, pose, *, group='coverage_v20', candidate_id=0, asset_index=None):
        pose = _pose(pose, 'target pose')
        if not self.available or not self._inside(pose[:2]) or not self.safe[pose[:2]]:
            return None
        node = int(self.ids[pose[:2]]) * 4 + pose[2]
        cost = self.outward[node] + self.backward[node]
        if not np.isfinite(cost) or self.outward[node] < 1 or cost > self.budget:
            return None
        out = _path(self.prev, self.start, node, self.cells)
        back = _path(self.backprev, self.home, node, self.cells, reverse=True)
        oa, ba = _actions(out), _actions(back)
        if (out[0] != self.current or out[-1] != pose or back[0] != pose or back[-1] != self.anchor
                or len(oa) + len(ba) != int(round(cost)) or not oa or len(oa) + len(ba) > self.budget):
            raise RuntimeError('directed route and paid return accounting disagree')
        return dict(candidate_id=candidate_id, group=group, asset_index=asset_index,
            pose=list(pose), states=[list(p) for p in out + back[1:]], actions=oa + ba,
            outbound_states=[list(p) for p in out], outbound_actions=oa,
            return_states=[list(p) for p in back], return_actions=ba,
            outbound_cost=len(oa), return_cost=len(ba), cost=len(oa) + len(ba),
            arrival_action=len(oa), return_anchor=list(self.anchor),
            is_coverage_anchor=group.startswith('coverage_'), endpoint_measured_fov_support=0.)

    def refresh(self, original):
        route = self.route(original['pose'], group=original['group'],
            candidate_id=original['candidate_id'], asset_index=original.get('asset_index'))
        if route is None:
            return None
        # Keep intention and attribution, replace only directed path fields.
        updated = deepcopy(original)
        updated.update(route)
        return updated

    def coverage_candidates(self, slots=8, separation_m=1.6):
        slots = _integer(slots, 'coverage slots', 1)
        if not np.isfinite(separation_m) or separation_m < 0:
            raise ValueError('nonnegative finite candidate separation required')
        audit = dict(status=self.reason, class_used=False, truth_used=False,
            coverage_gain='unique_unknown_only', separation_m=separation_m,
            candidate_gain_scope='endpoint_visible_unique_unknown; route_mask unions outbound observations',
            same_cell_rotation_excluded=True,
            diversity='observed_frontier_group_priority_then_spatial_separation',
            frontier_groups_are_room_labels=False, first_translation_used_for_selection=False,
            minimum_current_return_cost=None, evaluated_cells=0)
        if not self.available:
            return [], audit
        audit['minimum_current_return_cost'] = int(self.backward[self.start])
        # Unknown cannot be adjacent to an inflated safe footprint at radius>0.
        # Bridge only the known-free safety margin to obtain observed frontier
        # cells. Restrict to safe cells connected to the current observed pose.
        reachable = np.zeros_like(self.safe)
        reached_cells = self.cells[np.any(np.isfinite(self.outward.reshape(-1, 4)), axis=1)]
        reachable[tuple(reached_cells.T)] = True
        cross = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
        margin_steps = int(np.ceil(self.mapper.config.robot_radius_m / self.mapper.config.resolution_m))
        reachable_margin = (binary_dilation(reachable, structure=cross, iterations=margin_steps,
                            mask=self.belief == 0) if margin_steps else reachable)
        frontier = (self.belief == -1) & binary_dilation(reachable_margin, structure=cross)
        dilation_steps = margin_steps + 1
        frontier_labels, frontier_count = label(frontier, structure=cross)
        frontier_distances = {group: distance_transform_edt(frontier_labels != group)
                              for group in range(1, frontier_count + 1)}
        audit.update(frontier_neighborhood_cells=dilation_steps,
            frontier_groups=[dict(id=group, cells=int(np.count_nonzero(frontier_labels == group)))
                             for group in range(1, frontier_count + 1)],
            frontier_definition='unknown adjacent to reachable known-free safety margin; four-neighbor connected components')
        candidates = []
        costs = self.outward + self.backward
        nodes = np.flatnonzero(np.isfinite(costs) & (self.outward > 0) & (costs <= self.budget))
        # Radar is heading-independent. Keep the cheapest feasible heading at
        # each cell; observation-specific headings remain in the shared pool.
        by_cell = {}
        for node in nodes:
            ci = int(node // 4)
            key = (self.outward[node], costs[node], int(node % 4))
            if ci not in by_cell or key < by_cell[ci][0]:
                by_cell[ci] = key, int(node)
        for ci, (_, node) in by_cell.items():
            cell = tuple(map(int, self.cells[ci]))
            # A 360-degree coverage proposal needs a new sensor position.
            # Same-cell rotations remain supported by route() for inspection.
            if cell == self.current[:2]:
                continue
            visible = self.visibility(cell)
            count = int(visible.sum())
            if count <= 0:
                continue
            out = int(self.outward[node])
            groups, counts = np.unique(frontier_labels[visible & frontier], return_counts=True)
            support = dict(zip(map(int, groups), map(int, counts)))
            primary_group = min(support, key=lambda group: (
                frontier_distances[group][cell], -support[group], group)) if support else None
            candidates.append(dict(pose=(*cell, int(node % 4)), gain=count,
                rate=count / out, outbound=out, cost=int(costs[node]),
                frontier_group=primary_group, visible_frontier_groups=sorted(support)))
        audit['evaluated_cells'] = len(by_cell)
        if not candidates:
            return [], {**audit, 'status': 'no_unknown_gain'}
        # Preserve rate/total anchors, then represent other OBSERVED frontier
        # groups before filling remaining spatially separated alternatives.
        rate_order = sorted(candidates, key=lambda r: (-r['rate'], r['cost'], r['pose']))
        total_order = sorted(candidates, key=lambda r: (-r['gain'], r['outbound'], r['pose']))
        chosen = []
        def add(row, separated=True, reason='spatial_fill'):
            if len(chosen) >= slots:
                return False
            if any(row['pose'][:2] == old['pose'][:2] for old in chosen):
                return False
            if separated and any(np.linalg.norm(np.subtract(row['pose'][:2], old['pose'][:2]))
                    * self.mapper.config.resolution_m < separation_m for old in chosen):
                return False
            row['selection_reason'] = reason
            chosen.append(row)
            return True
        add(rate_order[0], reason='rate_anchor')
        add(total_order[0], separated=False, reason='total_anchor')
        group_order = list(dict.fromkeys(row['frontier_group'] for row in rate_order
                                        if row['frontier_group'] is not None))
        for group in group_order:
            if len(chosen) >= slots:
                break
            if any(row['frontier_group'] == group for row in chosen):
                continue
            for row in rate_order:
                if row['frontier_group'] == group and add(row, reason='new_frontier_group'):
                    break
        for row in rate_order:
            if len(chosen) >= slots:
                break
            add(row)
        routes = [self.route(row['pose'], group=f'coverage_v20_{i}', candidate_id=i)
                  for i, row in enumerate(chosen[:slots])]
        if any(route is None for route in routes):
            raise RuntimeError('selected coverage pose lost its observed route')
        # Initial translation is diagnostic only, not a room/branch selector.
        for row, route in zip(chosen, routes):
            row['first_translation'] = next((p[:2] for p in route['outbound_states'][1:]
                if tuple(p[:2]) != self.current[:2]), None)
        audit.update(status='available', selected=chosen[:slots], candidate_count=len(routes),
            represented_frontier_groups=sorted({row['frontier_group'] for row in chosen
                                                if row['frontier_group'] is not None}))
        return routes, audit
