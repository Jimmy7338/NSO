"""Two-view budgeted coverage lookahead with orientation-state shortest paths.

This bounded CPU probe tests non-greedy viewpoint ordering, not the complete
hierarchical regional route proposed in the redesign document.
"""
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from nso.navigable_frontier_v2 import ProjectedCoveragePolicy
from utils.grid_geometry import DIRECTIONS


def orientation_graph(safe):
    cells = np.argwhere(safe)
    ids = np.full(safe.shape, -1, np.int32)
    ids[tuple(cells.T)] = np.arange(len(cells))
    sources, targets = [], []
    base = np.arange(len(cells))*4
    for h, (dr, dc) in enumerate(DIRECTIONS):
        sources.extend([base+h, base+h])
        targets.extend([base+(h+1)%4, base+(h-1)%4])
        r, c = cells[:,0]+dr, cells[:,1]+dc
        inside = (r>=0)&(r<safe.shape[0])&(c>=0)&(c<safe.shape[1])
        source = base[inside]+h
        dest = ids[r[inside], c[inside]]
        valid = dest>=0
        sources.append(source[valid]); targets.append(dest[valid]*4+h)
    source, target = np.concatenate(sources), np.concatenate(targets)
    graph = csr_matrix((np.ones(len(source)), (source, target)), shape=(4*len(cells),)*2)
    return graph, cells, ids


def recover_actions(predecessors, start, end, cells):
    states = []
    current = end
    while current != start:
        if current < 0:
            raise ValueError('unreachable orientation state')
        states.append(current)
        current = int(predecessors[current])
    states.reverse()
    actions = []
    previous = start
    for current in states:
        if current//4 != previous//4:
            actions.append('forward')
        else:
            actions.append('right' if (current%4-previous%4)%4 == 1 else 'left')
        previous = current
    return actions


class RouteCoveragePolicy(ProjectedCoveragePolicy):
    def __init__(self, config, max_candidates=24, horizon=32):
        if horizon < 1:
            raise ValueError('horizon must be positive')
        self.horizon = horizon
        super().__init__(config, max_candidates, bundle=False, path_gain=False)

    def reset(self):
        super().reset()
        self.transfer_fallbacks = 0

    def _select(self, obs, safe):
        graph, cells, ids = orientation_graph(safe)
        start = int(ids[obs.position])*4 + obs.heading
        costs, predecessors = dijkstra(graph, directed=True, indices=start, return_predecessors=True)
        cell_cost = costs.reshape(-1,4).min(axis=1)
        distances = np.full(safe.shape, -1., dtype=float)
        reachable = np.isfinite(cell_cost)
        distances[tuple(cells[reachable].T)] = cell_cost[reachable]
        eligible = self._gain_mask(obs)
        horizon = min(self.horizon, self.config.max_steps-obs.step)
        candidates = []
        for cell in self._candidates(obs, safe, distances):
            for h in range(4):
                state = int(ids[cell])*4+h
                cost = costs[state]
                if not 0 < cost < horizon:
                    continue
                mask = self._visible(obs, cell, h) & eligible
                gain = int(mask.sum())
                if gain:
                    # Keep meaningful headings; the final score is the expected
                    # covered-cell-time, not this inexpensive screening ratio.
                    candidates.append(dict(cell=cell, heading=h, state=state, cost=int(cost),
                                           mask=mask, gain=gain, rough=gain/(cost+1)))
        candidates.sort(key=lambda c:(-c['rough'], c['cell'], c['heading']))
        candidates = candidates[:self.max_candidates]
        if not candidates:
            # A planning horizon is not a reachability deadline. If every useful
            # frontier lies farther away, travel through known space using the
            # remaining episode budget instead of declaring exploration done.
            self.transfer_fallbacks += 1
            return super()._select(obs, safe)
        between = dijkstra(graph, directed=True, indices=[c['state'] for c in candidates])
        scored = []
        rows = []
        for i, first in enumerate(candidates):
            direct = first['gain']*(horizon-first['cost'])
            best, successor = direct, None
            for j, second in enumerate(candidates):
                if i == j:
                    continue
                arrival = first['cost'] + between[i, second['state']]
                if arrival >= horizon:
                    continue
                marginal = int(np.count_nonzero(second['mask'] & ~first['mask']))
                value = direct + marginal*(horizon-arrival)
                if value > best:
                    best, successor = value, j
            rows.append(dict(step=obs.step, goal=list(first['cell']), heading=first['heading'],
                gain=first['gain'], actions=first['cost'], score=float(best/horizon),
                next_goal=list(candidates[successor]['cell']) if successor is not None else None,
                selected=False))
            scored.append((-best, first['cost'], first['cell'], first['heading'], i))
        i = min(scored)[-1]
        best = candidates[i]
        self.goal, self.goal_heading = best['cell'], best['heading']
        self._gain, self._score = best['gain'], rows[i]['score']
        self._distance = best['cost']
        actions = recover_actions(predecessors, start, best['state'], cells)
        self._actions = deque(actions)
        self.goal_count += 1
        rows[i]['selected'] = True
        self.selection_audit.extend(rows)
        self._active = dict(start_step=obs.step, goal=list(self.goal), planned_actions=len(actions),
                            predicted_gain=self._gain)
        return True

    def diagnostics(self):
        return dict(super().diagnostics(), transfer_fallbacks=self.transfer_fallbacks)
