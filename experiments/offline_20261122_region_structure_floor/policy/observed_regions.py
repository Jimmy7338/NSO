"""Observed-only regional priorities and candidate poses for the native planner.

The caller owns oriented-path and return certificates. This module never
changes the belief grid, requests an unobserved goal cell, or executes actions.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from math import ceil, hypot

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label

from utils.grid_geometry import DIRECTIONS, inflated_obstacles, visible_mask


FOUR_CONNECTED = generate_binary_structure(2, 1)


@dataclass(frozen=True)
class RegionConfig:
    min_gap_m: float = 0.6
    max_gap_m: float = 1.8
    max_wall_depth_m: float = 0.65
    min_usable_region_m2: float = 0.48
    matching_distance_m: float = 0.45
    frontier_sample_spacing_m: float = 0.9
    view_search_radius_m: float = 1.2
    max_proposals: int = 24
    max_candidates_per_region: int = 3
    unknown_m2_cap: float = 2.0
    frontier_m_cap: float = 3.0
    frontier_m2_per_m: float = 0.25
    unvisited_bonus_m2: float = 1.25
    undispatched_entrance_bonus_m2: float = 0.35
    roundtrip_action_penalty_m2: float = 0.0125
    repeat_goal_penalty_m2: float = 0.35


def _gaps(belief, resolution, config):
    """Stable, observed free cross-sections with two collinear occupied jambs."""
    openings = []
    for tangent, grid in enumerate((belief, belief.T)):
        layers = []
        for normal, line in enumerate(grid):
            walls = np.flatnonzero(line == 1)
            for left, right in pairwise(walls):
                width = right - left - 1
                if not config.min_gap_m <= round(width * resolution, 8) <= config.max_gap_m:
                    continue
                # The occupied endpoints need an outward occupied extension
                # along the wall. A single piece of clutter is not a jamb.
                if left == 0 or right == len(line) - 1 or line[left - 1] != 1 or line[right + 1] != 1:
                    continue
                if np.any(line[left + 1:right] != 0):
                    continue
                centre = (int(left) + int(right)) / 2
                layers.append((normal, int(left), int(right), centre))
        groups = []
        for row in layers:
            match = next((group for group in groups
                          if row[0] - group[-1][0] <= 1 and abs(row[3] - group[-1][3]) <= 1
                          and abs(row[1] - group[-1][1]) <= 1 and abs(row[2] - group[-1][2]) <= 1), None)
            if match is None:
                groups.append([row])
            else:
                match.append(row)
        for group in groups:
            span = group[-1][0] - group[0][0] + 1
            if span * resolution > config.max_wall_depth_m:
                continue
            n, lo, hi, mid = group[len(group) // 2]
            points = [(row, col) for row, left, right, _ in group for col in range(left + 1, right)]
            if tangent:
                points = [(col, row) for row, col in points]
            centre = (n, round(mid)) if not tangent else (round(mid), n)
            openings.append(dict(axis=tangent, centre=centre, cells=points, width_cells=hi-lo-1,
                                 jambs=((n, lo), (n, hi)) if not tangent else ((lo, n), (hi, n))))
    return openings


class RegionalPlanner:
    def __init__(self, config=RegionConfig()):
        self.config = config
        self.history = {}
        self.entrances = {}
        self.next_region = 1
        self.labels = None
        self.map = None
        self.last_pose = None
        self.goal_attempts = Counter()
        self.rejections = []
        self.latest_proposals = {}
        self.range_m = 3.0
        self.fov_deg = 90
        self.radius_m = 0.2

    def observe(self, belief, pose, paid):
        """Record only realized positions, including return and rejected-plan stops."""
        pose = tuple(map(int, pose))
        paid = int(paid)
        if len(pose) != 3 or not (0 <= pose[0] < belief.shape[0] and 0 <= pose[1] < belief.shape[1]):
            raise ValueError('regional pose outside the observed grid')
        previous = self.history.get(paid)
        if previous is not None and previous != pose:
            raise ValueError('same paid clock contains two different observed poses')
        self.history[paid] = pose
        self.last_pose = pose

    def _stable_regions(self, cores, count, resolution):
        """One-to-one overlap matching, as in the retained ObservedTopology."""
        limit = ceil(self.config.min_usable_region_m2 / resolution**2 - 1e-9)
        areas = np.bincount(cores.ravel(), minlength=count + 1)
        viable = [number for number in range(1, count + 1) if areas[number] >= limit]
        overlap = []
        if self.labels is not None:
            for number in viable:
                ids, sizes = np.unique(self.labels[cores == number], return_counts=True)
                overlap.extend((-int(size), number, int(i)) for i, size in zip(ids, sizes) if i)
        assignment, used = {}, set()
        for _, number, previous_id in sorted(overlap):
            if number not in assignment and previous_id not in used:
                assignment[number] = previous_id
                used.add(previous_id)
        for number in viable:
            if number not in assignment:
                assignment[number] = self.next_region
                self.next_region += 1
        lookup = np.zeros(count + 1, dtype=np.int32)
        for number, region_id in assignment.items():
            lookup[number] = region_id
        return lookup[cores]

    def _side_region(self, gate, direction):
        row, col = gate['centre']
        normal_axis = gate['axis']  # axis 0 scans along row and crosses along row index
        normal_offset = direction * np.arange(1, 5)
        tangent_offset = np.arange(-1, 2)
        if not normal_axis:
            rows, cols = np.meshgrid(row + normal_offset, col + tangent_offset, indexing='ij')
        else:
            rows, cols = np.meshgrid(row + tangent_offset, col + normal_offset, indexing='ij')
        valid = (rows >= 0)&(rows < self.labels.shape[0])&(cols >= 0)&(cols < self.labels.shape[1])
        nearby = self.labels[rows[valid], cols[valid]]
        counts = Counter(int(v) for v in nearby if v)
        return min(counts, key=lambda region: (-counts[region],region)) if counts else None

    def _associate(self, gaps, resolution, paid):
        matched = set()
        for gap in gaps:
            matches = [(hypot(*(a - b for a,b in zip(gap['centre'], door['centre']))), i)
                       for i,door in self.entrances.items() if gap['axis'] == door['axis'] and i not in matched]
            options = [entry for entry in matches if entry[0]*resolution <= self.config.matching_distance_m]
            if options:
                _, identifier = min(options)
                door = self.entrances[identifier]
            else:
                identifier = len(self.entrances) + 1
                door = self.entrances[identifier] = dict(entrance_id=identifier, first_seen_paid=int(paid))
            matched.add(identifier)
            door.update(gap, last_seen_paid=int(paid))
        for identifier, door in self.entrances.items():
            regions = [self._side_region(door, direction) for direction in (-1,1)]
            door['regions'] = regions
            # Keep identity through partial re-observation or a clearly seen
            # closure. A stale opening never generates a new viewpoint.
            occupied = sum(self.map[cell] == 1 for cell in door['cells'])
            width = door['width_cells']
            door['blocked'] = bool(occupied >= max(1, ceil(width/2)))
            door['visible_now'] = identifier in matched
            door['credible'] = bool(identifier in matched and not door['blocked']
                                    and regions[0] != regions[1] and any(regions))

    def _crossings(self, door):
        row, col = door['centre']
        normal = door['axis']
        plane, centre = (row, col) if normal == 0 else (col, row)
        # Actual unit forward transitions include an edge touching the plane.
        # Count completion across it once, without counting tangential motion.
        last_side = 0
        crossings = []
        position_on_plane = None
        for paid, pose in self.history.items():
            offset = pose[normal] - plane
            side = int(np.sign(offset))
            if not side:
                position_on_plane = (pose, paid)
                continue
            if last_side and side != last_side and position_on_plane is not None:
                contact, at = position_on_plane
                if (abs(contact[1-normal]-centre) <= door['width_cells']/2
                        and abs(offset) == 1 and paid - at == 1):
                    crossings.append(paid)
            last_side, position_on_plane = side, None
        return crossings

    def _update(self, belief, resolution, paid):
        if self.map is not None and np.array_equal(belief, self.map):
            return
        self.map = np.asarray(belief).copy()
        gaps = _gaps(self.map, resolution, self.config)
        cuts = np.zeros(self.map.shape, bool)
        for gap in gaps:
            for cell in gap['cells']:
                cuts[cell] = True
        cores, count = label((self.map == 0)&~cuts, structure=FOUR_CONNECTED)
        self.labels = self._stable_regions(cores,count,resolution)
        self.frontiers = (self.map == 0)&binary_dilation(self.map == -1, structure=FOUR_CONNECTED)
        self._associate(gaps,resolution,paid)

    def _room_state(self):
        visited = {}
        arrivals = {}
        previous = 0
        for paid, pose in self.history.items():
            identifier = int(self.labels[pose[:2]])
            if identifier:
                visited.setdefault(identifier, []).append(paid)
                if identifier != previous:
                    arrivals.setdefault(identifier, []).append(paid)
                previous = identifier
        nodes = {}
        for i in map(int, np.unique(self.labels)):
            if i:
                region = self.labels == i
                steps = visited.get(i, [])
                nodes[i] = dict(region_id=i, observed_free_cells=int(np.count_nonzero(region)),
                                remaining_frontier_cells=int(np.count_nonzero(region&self.frontiers)),
                                first_visit_paid=steps[0] if steps else None,
                                last_visit_paid=steps[-1] if steps else None,
                                visit_start_paid_steps=arrivals.get(i, []), visited=bool(steps))
        return nodes

    @staticmethod
    def _priority(nodes, region):
        item = nodes.get(region, {})
        return item.get('visited', True), -item.get('remaining_frontier_cells',0), region or 0

    def _visible(self, pose, resolution):
        cell, heading = pose[:2], pose[2]
        rays = visible_mask(self.map == 1,cell,heading,max(1,int(self.range_m/resolution)),self.fov_deg)
        return int(np.count_nonzero(rays&(self.map == -1)))

    def proposals(self, belief, pose, paid, resolution_m, max_depth_m=3.0, fov_deg=90, robot_radius_m=.2):
        """Propose known footprint-safe views before the caller checks both oriented paths."""
        self.observe(belief,pose,paid)
        self._update(belief,resolution_m,paid)
        self.range_m,self.fov_deg,self.radius_m = max_depth_m,fov_deg,robot_radius_m
        self.latest_proposals = {}
        doors = [door for door in self.entrances.values() if door['credible']]
        if not doors:
            return []
        nodes = self._room_state()
        safe = ~inflated_obstacles(self.map != 0,robot_radius_m/resolution_m)
        current = tuple(map(int,pose))
        radius = ceil(self.config.view_search_radius_m/resolution_m)
        spacing = self.config.frontier_sample_spacing_m/resolution_m
        height,width = safe.shape

        def add_view(source, viewpoint, target_region, entry=None, headings=range(4)):
            r,c = viewpoint
            best = None
            for normal in headings:
                dr,dc = DIRECTIONS[normal]
                choices = [(r-dr*distance,c-dc*distance,normal) for distance in range(1,radius+1)]
                for offer in choices:
                    rr,cc,_ = offer
                    if not (0<=rr<height and 0<=cc<width) or not safe[rr,cc] or offer==current:
                        continue
                    if entry is None and int(self.labels[rr,cc])!=target_region:
                        continue
                    unseen = self._visible(offer,resolution_m)
                    if not unseen:
                        continue
                    # Prefer views looking towards the target from the shortest observed standoff.
                    index=(-unseen, abs(rr-r)+abs(cc-c), offer)
                    if best is None or index < best[0]:
                        best=(index,offer,unseen)
            if best is None:
                return
            _,offer,unseen=best
            if offer not in self.latest_proposals:
                self.latest_proposals[offer] = dict(pose=list(offer), proposal_kind=source,
                                                    region_id=int(self.labels[offer[:2]]),
                                                    target_region_id=target_region,
                                                    entrance_id=entry, visible_unknown_cells=unseen)

        for door in sorted(doors, key=lambda d:(min(self._priority(nodes,i) for i in d['regions']),d['entrance_id'])):
            for direction,region in zip((-1,1),door['regions']):
                centre=list(door['centre'])
                axis=door['axis']
                centre[axis]+=direction*2
                heading = ((0,2) if axis == 0 else (3,1))[direction>0]
                add_view('doorway_view',centre,region,door['entrance_id'],(heading,))
        for region in sorted(nodes, key=lambda i:self._priority(nodes,i)):
            samples=[]
            for r,c in np.argwhere(self.frontiers&(self.labels==region)):
                if all(hypot(r-v,c-w)>=spacing for v,w in samples):
                    samples.append((int(r),int(c)))
                if len(samples)>=self.config.max_candidates_per_region:
                    break
            for target in samples:
                add_view('frontier_view',target,region)
        limited = list(self.latest_proposals.values())[:self.config.max_proposals]
        self.latest_proposals = {tuple(item['pose']):item for item in limited}
        return limited

    def rank(self, belief, pose, paid, resolution_m, native_routes, admitted_proposals):
        """Rank only options already certified by the untouched native feasibility checks."""
        self.observe(belief,pose,paid)
        self._update(belief,resolution_m,paid)
        rooms = self._room_state()
        doorways=[]
        for identifier,door in sorted(self.entrances.items()):
            crossed=self._crossings(door)
            rejected = [item for item in self.rejections if item['entrance_id']==identifier]
            doorways.append({**door,'crossed_paid_steps':crossed,
                             'observed_route_rejections':rejected,
                             'status': 'blocked' if door['blocked'] else 'traversed' if crossed else 'seen'})
        audit = dict(current_pose=list(pose), paid=int(paid), region_count=len(rooms),
                     current_region_id=int(self.labels[tuple(pose[:2])]),
                     regions=list(rooms.values()), entrances=doorways,
                     visible_unvisited_region_ids=[r for r,item in rooms.items() if not item['visited']],
                     proposed=len(self.latest_proposals),
                     admission_rejected=[], ranking=[], local_only_ranking=False, uses_object_semantics=False,
                     native_rerank_options=len(native_routes), prior_native_rejections=list(self.rejections))
        admitted = {tuple(route['pose']):route for route in admitted_proposals}
        known = {tuple(route['pose']):route for route in native_routes}
        audit['admission_rejected'] = [v for p,v in self.latest_proposals.items() if p not in admitted and p not in known]
        if not any(door['credible'] for door in doorways):
            return None, {**audit,'reason':'no_credible_observed_entrance'}
        scores=[]
        native_positions = set(known)
        options = dict(admitted)|known
        params=self.config
        for target,route in options.items():
            r,c,h=target
            explicit=self.latest_proposals.get(target)
            region=(explicit or {}).get('region_id') or int(self.labels[r,c])
            entrance_id=(explicit or {}).get('entrance_id')
            entry=self.entrances.get(entrance_id)
            if entry is not None and entry['blocked']:
                continue
            frontier = rooms.get(region,{}).get('remaining_frontier_cells',0)
            visible=(explicit or {}).get('visible_unknown_cells')
            if visible is None:
                visible=self._visible(target,resolution_m)
            if not visible or not (frontier or entrance_id):
                continue
            unvisited=region in rooms and not rooms[region]['visited']
            outbound,returning=int(route['outbound_cost']),int(route['return_cost'])
            score=(params.unvisited_bonus_m2*unvisited
                   + params.frontier_m2_per_m*min(frontier*resolution_m,params.frontier_m_cap)
                   + min(visible*resolution_m**2,params.unknown_m2_cap)
                   + params.undispatched_entrance_bonus_m2*(entry is not None and not self._crossings(entry))
                   - params.roundtrip_action_penalty_m2*(outbound+returning)
                   - params.repeat_goal_penalty_m2*self.goal_attempts[target])
            row=dict(pose=list(target),region_id=region,entrance_id=entrance_id,visible_unknown_cells=visible,
                     target_region_hint=(explicit or {}).get('target_region_id'),
                     remaining_frontier_cells=frontier, unvisited_region=unvisited, outbound_actions=outbound,
                     return_actions=returning, score_m2=float(score), previous_dispatches=self.goal_attempts[target],
                     origin='native_rerank' if target in native_positions else 'new_regional_pose')
            scores.append((-score,outbound+returning,row['pose'],row,route))
        scores.sort(key=lambda row:row[:3])
        audit['ranking']=[entry[3] for entry in scores]
        if not scores:
            return None, {**audit,'reason':'no_qualified_feasible_regional_view'}
        audit['chosen']=scores[0][3]
        return scores[0][4], {**audit,'reason':'selected_observed_region_view'}

    def note_attempt(self, route, result, paid):
        """Preserve native rejection evidence without asserting that a wall closed."""
        pose=tuple(int(x) for x in route['pose'])
        if result=='dispatched':
            self.goal_attempts[pose]+=1
        else:
            via=route.get('regional_proposal') or self.latest_proposals.get(pose,{})
            self.rejections.append(dict(pose=list(pose), paid=int(paid), reason=str(result),
                                        entrance_id=via.get('entrance_id')))
