"""Geometry-only budget witnesses and conservative all-asset visit bounds.

The region-tour dynamic program relaxes within-region entry/exit consistency.
Its value is a lower bound, never a feasible tour or documentation-quality claim.
"""
from collections import deque
import math

import numpy as np

from utils.grid_geometry import DIRECTIONS, supercover_line


def grid_distances(safe, sources):
    safe = np.asarray(safe, dtype=bool)
    distance = np.full(safe.shape, -1, dtype=np.int32)
    queue = deque()
    for value in sources:
        cell = tuple(map(int, value))
        if not (0 <= cell[0] < safe.shape[0] and 0 <= cell[1] < safe.shape[1]) or not safe[cell]:
            raise ValueError('distance source is not safe')
        if distance[cell] < 0:
            distance[cell] = 0
            queue.append(cell)
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < safe.shape[0] and 0 <= nc < safe.shape[1] and safe[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = distance[r, c] + 1
                queue.append((nr, nc))
    return distance


def region_tour_lower_bound(safe, start, regions):
    """Return a necessary translation cost for visiting every region and home.

    For any actual closed walk, order regions by first visit. Every walk segment
    costs at least the minimum safe-grid distance between its endpoint regions.
    Minimizing these segment sums over all orders can only reduce the cost.
    Rotation, sensing and consistent entry/exit inside each region are ignored.
    """
    groups = [np.asarray([start], dtype=int)] + [np.argwhere(np.asarray(region, dtype=bool) & safe) for region in regions]
    if any(len(group) == 0 for group in groups):
        raise ValueError('empty safe service region')
    count = len(regions)
    if count == 0:
        return dict(lower_bound_translation_actions=0, region_distance_matrix=[[0]], relaxed_order=[])
    costs = np.zeros((count + 1, count + 1), dtype=int)
    for i, group in enumerate(groups):
        distance = grid_distances(safe, group)
        for j, other in enumerate(groups):
            values = distance[other[:, 0], other[:, 1]]
            values = values[values >= 0]
            if not len(values):
                raise ValueError('disconnected service regions')
            costs[i, j] = int(values.min())
    states = {(1 << i, i): (int(costs[0, i + 1]), [i]) for i in range(count)}
    for mask in range(1, 1 << count):
        for last in range(count):
            if (mask, last) not in states:
                continue
            value, order = states[mask, last]
            for nxt in range(count):
                if mask & (1 << nxt):
                    continue
                key = mask | (1 << nxt), nxt
                proposal = value + int(costs[last + 1, nxt + 1])
                if key not in states or proposal < states[key][0]:
                    states[key] = proposal, order + [nxt]
    lower, order = min((states[(1 << count) - 1, i][0] + int(costs[i + 1, 0]),
                        states[(1 << count) - 1, i][1]) for i in range(count))
    return dict(lower_bound_translation_actions=int(lower), region_distance_matrix=costs.tolist(), relaxed_order=order)


def conservative_asset_regions(world, range_m=1.2):
    """Supersets of camera cells near any asset exterior; LoS is relaxed.

    Horizontal distance to a containing AABB never exceeds 3D surface distance.
    Ignoring view direction and occlusion further enlarges the service region.
    A tour lower bound on these regions therefore also bounds real close visits.
    """
    if range_m <= 0:
        raise ValueError('positive near-visit range required')
    rows, cols = np.indices(world.shape)
    x = (cols + .5) * world.config.resolution_m
    y = (world.shape[0] - rows - .5) * world.config.resolution_m
    regions, audit = [], []
    for item in world.objects:
        bounds = np.asarray(item.get('near_visit_bounds', item['evaluation_bounds']), dtype=float)
        if bounds.shape != (2, 3) or np.any(bounds[1] < bounds[0]):
            raise ValueError('ordered 3D containing bounds required')
        dx = np.maximum(np.maximum(bounds[0, 0] - x, x - bounds[1, 0]), 0)
        dy = np.maximum(np.maximum(bounds[0, 1] - y, y - bounds[1, 1]), 0)
        region = world.reachable & (np.hypot(dx, dy) <= range_m + 1e-12)
        regions.append(region)
        audit.append(dict(asset_id=int(item['id']), bounds=bounds.tolist(), safe_region_cells=int(region.sum())))
    return regions, audit


def paired_budget(coverage_actions, lower_bounds, fraction=.35):
    if not 0 < fraction < 1:
        raise ValueError('budget interpolation fraction must be inside (0,1)')
    coverage = max(map(int, coverage_actions))
    lower = min(map(int, lower_bounds))
    result = dict(paired_coverage_witness_actions=coverage, paired_all_asset_lower_bound=lower,
                  interpolation_fraction=float(fraction), formula='floor(T_cov + fraction * (L_all - T_cov))')
    if lower <= coverage:
        return dict(result, status='failed', budget=None, reason='all-asset lower bound does not exceed coverage witness')
    budget = math.floor(coverage + fraction * (lower - coverage))
    if not coverage <= budget < lower:
        raise AssertionError('budget must fund coverage while staying below all-asset lower bound')
    return dict(result, status='passed', budget=budget, reason=None)


def exact_region_walk_lower_bound(safe, start, regions):
    """Tighten the region relaxation by BFS over (cell, visited-region mask).

    This is an exact translation-only close-visit walk. Ignoring all heading and
    sensing requirements keeps it a lower bound on the robot's paid action tour.
    It still does not imply a high-quality reconstruction of those assets.
    """
    safe = np.asarray(safe, dtype=bool); cells = np.argwhere(safe)
    index = np.full(safe.shape, -1, dtype=np.int32)
    index[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
    bits = np.zeros(len(cells), dtype=np.int32)
    for i, region in enumerate(regions):
        members = np.asarray(region, dtype=bool)[cells[:, 0], cells[:, 1]]
        if not members.any():
            raise ValueError('empty safe service region')
        bits[members] |= 1 << i
    adjacency = []
    for r, c in cells:
        row = []
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < safe.shape[0] and 0 <= nc < safe.shape[1] and index[nr, nc] >= 0:
                row.append(int(index[nr, nc]))
        adjacency.append(row)
    n = len(cells); home = int(index[tuple(start)]); all_bits = (1 << len(regions)) - 1
    if home < 0:
        raise ValueError('unsafe start')
    distances = np.full(n * (all_bits + 1), -1, dtype=np.int32)
    initial = int(bits[home]) * n + home; distances[initial] = 0; queue = deque([initial])
    expanded = 0
    while queue:
        encoded = queue.popleft(); mask, cell = divmod(encoded, n)
        distance = int(distances[encoded]); expanded += 1
        if cell == home and mask == all_bits:
            return dict(lower_bound_translation_actions=distance, expanded_states=expanded,
                        method='exact safe-grid (cell, visited regions) BFS; rotations and sensing ignored')
        for nxt in adjacency[cell]:
            next_encoded = (mask | int(bits[nxt])) * n + nxt
            if distances[next_encoded] < 0:
                distances[next_encoded] = distance + 1; queue.append(next_encoded)
    raise ValueError('no closed walk visits all service regions')


def scan_known_mask(scan, shape, resolution_m):
    """Known cells from exactly the production mapper's planar-scan ray support."""
    known = np.zeros(shape, dtype=bool)
    def cell(point):
        return shape[0] - 1 - int(np.floor(point[1] / resolution_m)), int(np.floor(point[0] / resolution_m))
    origin = cell(scan.world_from_laser[:3, 3])
    angles = scan.angle_min_rad + np.arange(len(scan.ranges_m)) * scan.angle_increment_rad
    for angle, distance in zip(angles, scan.ranges_m):
        if not np.isfinite(distance) or distance <= 0:
            continue
        local = np.array([np.cos(angle) * distance, np.sin(angle) * distance, 0.])
        target = cell(scan.world_from_laser[:3, :3] @ local + scan.world_from_laser[:3, 3])
        for dr, dc in supercover_line(target[0] - origin[0], target[1] - origin[1]):
            r, c = origin[0] + dr, origin[1] + dc
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                known[r, c] = True
    if 0 <= origin[0] < shape[0] and 0 <= origin[1] < shape[1]:
        known[origin] = True
    return known


def shortest_actions(safe, initial, target, final_heading=None):
    initial = tuple(map(int, initial)); target = tuple(map(int, target))
    queue = deque([initial]); previous = {initial: None}
    while queue:
        state = queue.popleft()
        if state[:2] == target and (final_heading is None or state[2] == final_heading):
            break
        r, c, h = state; dr, dc = DIRECTIONS[h]
        for action, nxt in [('forward', (r + dr, c + dc, h)), ('left', (r, c, (h - 1) % 4)), ('right', (r, c, (h + 1) % 4))]:
            nr, nc, _ = nxt
            if 0 <= nr < safe.shape[0] and 0 <= nc < safe.shape[1] and safe[nr, nc] and nxt not in previous:
                previous[nxt] = state, action
                queue.append(nxt)
    else:
        raise ValueError('witness target is unreachable')
    route = []
    while previous[state] is not None:
        state, action = previous[state]; route.append(action)
    return route[::-1]


def advance_pose(pose, action):
    r, c, h = pose
    if action == 'forward':
        dr, dc = DIRECTIONS[h]; return r + dr, c + dc, h
    if action == 'left':
        return r, c, (h - 1) % 4
    if action == 'right':
        return r, c, (h + 1) % 4
    raise ValueError('unsupported witness action')
