"""Discover and inspect navigational ground-evidence gaps from registered packets."""

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.sparse.csgraph import dijkstra

from observed_sweep import DIRECTIONS
from sensor_probe.evidence import (
    MAX_FLOOR_SAMPLE_DISTANCE_M, RANGE_LIMIT_EPS_M, _scan_support, _swept_xy,
)


MISSING_GROUND = "missing_positive_ground_support"
HEIGHT_OFFSETS_M = (-.010, 0., .010)
MAX_VIEWS_PER_EDGE = 3
MAX_VIEWS_WITHOUT_GAIN = 2
MAX_VIEWS_PER_EPISODE = 6
LOOKAHEAD_M = 3.


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def hashes(packet):
    return array_sha(packet.scan.ranges_m), array_sha(packet.frame.depth_m)


def free_line(belief, start, goals):
    """Check known occupied cells along a horizontal proposal ray."""
    if len(goals) == 0:
        return np.empty(0, dtype=bool)
    start, goals = np.asarray(start), np.asarray(goals)
    distances = np.max(np.abs(goals - start), axis=1)
    fractions = np.arange(1, int(distances.max()) * 2 + 1)[None, :] / np.maximum(1, 2 * distances[:, None])
    samples = np.floor(start + np.minimum(fractions, 1)[..., None] * (goals - start)[:, None, :] + .5).astype(int)
    inside = np.any(samples != goals[:, None, :], axis=2)
    inside &= np.any(samples != start, axis=2)
    return ~np.any(inside & (belief[samples[..., 0], samples[..., 1]] == 1), axis=1)


def rotation(heading_delta):
    yaw = -np.pi * heading_delta / 2
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array(((cosine, -sine, 0.), (sine, cosine, 0.), (0., 0., 1.)))


def potential_frontier(belief, target, direction, horizon, already_reachable):
    """Trace a hypothetical branch through measured free cells beyond the edge."""
    queue = deque([(target, 0)])
    visited, frontier = {target}, set()
    while queue:
        (row, column), distance = queue.popleft()
        if distance == horizon:
            continue
        for dr, dc in DIRECTIONS:
            cell = row + dr, column + dc
            if (cell in visited or not 0 <= cell[0] < belief.shape[0] or not 0 <= cell[1] < belief.shape[1]
                    or (cell[0] - target[0]) * direction[0] + (cell[1] - target[1]) * direction[1] < 0):
                continue
            visited.add(cell)
            if belief[cell] == -1:
                frontier.add(cell)
            elif belief[cell] == 0 and not already_reachable[cell]:
                queue.append((cell, distance + 1))
    return len(frontier)


@dataclass
class Offer:
    edge: tuple
    heading: int
    missing: np.ndarray
    proof: dict
    opportunity: int


class MeasuredPaths:
    def __init__(self, graph, current, anchor):
        self.matrix, self.cells, self.ids = graph.orientation_graph()
        self.reverse = self.matrix.T.tocsr()
        self.current, self.home = self.node(current), self.node(anchor)
        if min(self.current, self.home) >= 0:
            self.cost, self.previous = dijkstra(self.matrix, indices=self.current, return_predecessors=True)
            self.home_cost = self.cost_to(self.home)

    def cost_to(self, node):
        return dijkstra(self.reverse, indices=node)

    def node(self, pose):
        return int(self.ids[pose[:2]]) * 4 + pose[2]

    def pose(self, node):
        return (*map(int, self.cells[node // 4]), int(node % 4))

    def route(self, node):
        from nso.route_coverage_v2 import recover_actions

        reverse_path = [node]
        while reverse_path[-1] != self.current:
            reverse_path.append(int(self.previous[reverse_path[-1]]))
        return [self.pose(item) for item in reversed(reverse_path)], recover_actions(
            self.previous, self.current, node, self.cells)


class ActiveGroundObserver:
    """Use measured navigation and geometry only; stop auxiliary movement on any veto."""

    def __init__(self, binding):
        self.binding = binding
        self.evidence = binding.evidence
        self.seen = set()
        self.views = set()
        self.attempted = defaultdict(int)
        self.spent_views = 0
        self.rejected = set()
        self.offer = self.route = self.pending = None
        self.stage = None
        self.seq = 0
        self.consecutive_no_gain = 0
        self.record()

    @property
    def packet(self):
        return self.binding.state["packet"]

    @property
    def pose(self):
        return (*self.packet.position, self.packet.heading)

    @property
    def raw(self):
        return self.evidence.raw

    def record(self):
        signature = hashes(self.packet)
        fresh = (self.pose, signature) not in self.seen
        self.views.add(self.pose)
        self.seen.add((self.pose, signature))
        return fresh, signature

    def event(self, name, **fields):
        self.binding.audit.append({"event": "ground_reacquisition_" + name, "inspection_id": self.seq,
                                   "paid": self.packet.action_id, **fields})

    def reject(self, edge, cause, **fields):
        key = (tuple(sorted(edge)), cause)
        if key not in self.rejected:
            self.rejected.add(key)
            self.event("rejected", edge=edge, reason=cause, **fields)

    def proof(self, edge):
        return self.raw.certify(*(self.evidence.xy(cell) for cell in edge), self.evidence.radius)

    def probes(self, edge):
        xy = _swept_xy(*(self.evidence.xy(cell) for cell in edge), self.evidence.radius)
        floor = self.raw._trees[2]
        if floor is None:
            return xy
        return xy[floor.query(xy)[0] > MAX_FLOOR_SAMPLE_DISTANCE_M]

    def finite_laser(self, xy):
        supported = np.zeros(len(xy), dtype=bool)
        for _, scan in reversed(self.raw._observations):
            remaining = np.flatnonzero(~supported)
            supported[remaining] = _scan_support(scan, xy[remaining])
            if supported.all():
                return True
        return False

    def find_offers(self, belief, graph, paths):
        offers = []
        source = self.pose[:2]
        radius, step = self.evidence.radius, self.evidence.resolution
        r, c = source
        reach = int(np.ceil(LOOKAHEAD_M / step))
        reachable = np.zeros(belief.shape, dtype=bool)
        reachable[tuple(paths.cells.T)] = np.isfinite(paths.cost.reshape(-1, 4)).any(axis=1)
        for heading, direction in enumerate(DIRECTIONS):
            target = (r + direction[0], c + direction[1])
            edge = (source, target)
            if (not 0 <= target[0] < belief.shape[0] or not 0 <= target[1] < belief.shape[1]
                    or graph.edges[*source, heading]):
                continue
            if belief[target] == 1:
                self.reject(edge, "measured_occupied_center")
                continue
            if belief[source] != 0 or belief[target] != 0 or self.attempted[tuple(sorted(edge))] >= MAX_VIEWS_PER_EDGE:
                continue
            p, q = (self.evidence.xy(cell) for cell in edge)
            dimensions = np.asarray(belief.shape[::-1]) * step
            if np.any(np.minimum(p, q) < radius) or np.any(np.maximum(p, q) > dimensions - radius):
                self.reject(edge, "swept_capsule_outside_map")
                continue
            if graph.safe[target] and np.isfinite(paths.cost[paths.node((*target, heading))]):
                continue
            proof = self.proof(edge)
            if proof["reason"] != MISSING_GROUND:
                if proof["reason"] in ("measured_body_hit_in_padded_capsule", "lower_floor_evidence_in_capsule",
                                       "missing_measured_laser_clearance"):
                    self.reject(edge, proof["reason"], certificate=proof)
                continue
            missing = self.probes(edge)
            if len(missing) != proof["unsupported_xy_probes"]:
                raise RuntimeError("Missing support set differs from the inherited swept-volume certificate")
            if not self.finite_laser(missing):
                self.reject(edge, "missing_finite_laser_over_support_gap", certificate=proof)
                continue
            opportunity = potential_frontier(belief, target, direction, reach, reachable)
            if opportunity:
                offers.append(Offer(edge, heading, missing, proof, opportunity))
            else:
                self.reject(edge, "no_unobserved_cells_visible_beyond_gap")
        return offers

    def camera_at(self, pose):
        frame = self.packet.frame
        twist = rotation(pose[2] - self.pose[2])
        observed = frame.world_from_camera
        transform = np.eye(4)
        displacement = observed[:3, 3] - np.r_[self.evidence.xy(self.pose[:2]), 0.]
        transform[:3, :3] = twist @ observed[:3, :3]
        transform[:3, 3] = twist @ displacement + np.r_[self.evidence.xy(pose[:2]), 0.]
        return transform

    def pixel_support(self, camera, missing):
        frame = self.packet.frame
        intrinsic = frame.intrinsic
        rows, cols = frame.depth_m.shape
        cap = float(self.packet.scan.range_max_m) - RANGE_LIMIT_EPS_M
        offsets = np.asarray(HEIGHT_OFFSETS_M)
        points = np.column_stack((np.repeat(missing, len(offsets), axis=0),
                                   np.tile(self.raw._ground_z + offsets, len(missing))))
        local = (points - camera[:3, 3]) @ camera[:3, :3]
        optical = local[:, 2]
        uv = (local @ intrinsic.T)[:, :2] / np.where(optical[:, None] > 0, optical[:, None], np.inf)
        visible = ((optical > .15) & (optical < cap) & (uv[:, 0] >= .5) & (uv[:, 0] < cols - 1.5)
                   & (uv[:, 1] >= .5) & (uv[:, 1] < rows - 1.5))
        idx = np.flatnonzero(visible)
        if len(idx):
            center = np.floor(uv[idx])
            nearby = center[:, None, :] + np.array(((0, 0), (0, 1), (1, 0), (1, 1)))
            rays = ((np.concatenate((nearby, np.ones((*nearby.shape[:2], 1))), axis=2)
                     @ np.linalg.inv(intrinsic).T) @ camera[:3, :3].T)
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = (points[idx, None, 2] - camera[2, 3]) / rays[..., 2]
                hit = camera[:2, 3] + rays[..., :2] * scale[..., None]
            error = np.linalg.norm(hit - points[idx, None, :2], axis=2)
            visible[idx] = np.any((error <= MAX_FLOOR_SAMPLE_DISTANCE_M) & (scale > 0) & (scale < cap), axis=1)
        return visible.reshape(len(missing), len(offsets))

    def proposals(self, offer, belief, paths, budget):
        source, target = offer.edge
        departure = paths.node((*source, offer.heading))
        to_departure = paths.cost_to(departure)
        # Before observing the gap a reverse passage is still hypothetical.
        # Two half turns + moving back across cost five actions. The actual
        # directed return is recomputed on measured edges before crossing.
        reserve = to_departure + 6 + paths.home_cost[departure]
        achievable = np.flatnonzero((paths.cost > 0) & (paths.cost + reserve <= budget)
                                    & (paths.cost + paths.home_cost <= budget))
        rows = []
        offsets = np.asarray(DIRECTIONS)
        cell_x, cell_y = offer.missing.T / self.evidence.resolution
        floor_cells = np.clip(np.column_stack((belief.shape[0] - 1 - cell_y.astype(int), cell_x.astype(int))),
                              (0, 0), np.asarray(belief.shape) - 1)
        for node in achievable:
            pose = paths.pose(node)
            if pose in self.views:
                continue
            distance = np.linalg.norm(self.evidence.xy(pose[:2])[None, :] - offer.missing, axis=1)
            if distance.min() > self.packet.scan.range_max_m:
                continue
            pixel = self.pixel_support(self.camera_at(pose), offer.missing)
            if not pixel.any():
                continue
            line = free_line(belief, pose[:2], floor_cells)
            pixel &= line[:, None]
            if not pixel.any():
                continue
            fraction = float(pixel.mean())
            radius = int(np.ceil(LOOKAHEAD_M / self.evidence.resolution))
            lo = np.maximum((0, 0), np.asarray(pose[:2]) - radius)
            hi = np.minimum(belief.shape, np.asarray(pose[:2]) + radius + 1)
            potential = np.argwhere(belief[lo[0]:hi[0], lo[1]:hi[1]] < 0) + lo
            delta = potential - pose[:2]
            direction = offsets[pose[2]]
            progress = delta @ direction
            lateral = np.abs(delta @ np.roll(direction, 1))
            visible = potential[(progress > 0) & (lateral <= progress) & (np.linalg.norm(delta, axis=1) <= radius)]
            additional = int(np.count_nonzero(free_line(belief, pose[:2], visible)))
            outward = int(paths.cost[node])
            home = int(paths.home_cost[node])
            projected = fraction * (offer.opportunity + additional)
            rows.append(dict(node=int(node), pose=pose, predicted_supported_fraction=fraction,
                             beyond_gap_unknown_cells=offer.opportunity, visible_unknown_cells=additional,
                             outward_actions=outward, return_actions=home, complete_reserve=int(reserve[node]),
                             score=projected / (outward + home + 1)))
        return sorted(rows, key=lambda row: (-row["score"], -row["predicted_supported_fraction"],
                                             row["outward_actions"], row["pose"]))

    def dispatch(self, belief, graph, paths, budget, *, retry=False):
        if retry:
            current_proof = self.proof(self.offer.edge)
            if current_proof["reason"] != MISSING_GROUND:
                return False
            self.offer = Offer(self.offer.edge, self.offer.heading, self.probes(self.offer.edge),
                               current_proof, self.offer.opportunity)
        offers = [self.offer] if retry else self.find_offers(belief, graph, paths)
        best = None
        for offer in offers:
            candidates = self.proposals(offer, belief, paths, budget)
            if not candidates:
                self.reject(offer.edge, "no_new_reachable_view_with_return",
                            certificate=offer.proof, missing_xy_m=offer.missing.tolist())
                continue
            if best is None or candidates[0]["score"] > best[1][0]["score"]:
                best = offer, candidates
        if best is None:
            return False
        if retry:
            self.event("end", outcome=self.retry_outcome, edge=self.offer.edge,
                       action_cost=self.packet.action_id - self.view_begin)
        self.offer, candidates = best
        self.seq += 1
        self.stage = "view"
        self.spent_views += 1
        self.attempted[tuple(sorted(self.offer.edge))] += 1
        chosen = candidates[0]
        self.view_begin = int(self.packet.action_id)
        self.prior_missing = len(self.offer.missing)
        states, actions = paths.route(chosen["node"])
        self.route = list(zip(actions, states[1:], strict=True))
        self.binding.prepare()
        self.event("dispatch", pose=self.pose, edge=self.offer.edge, heading=self.offer.heading,
                   prior_certificate=self.offer.proof, missing_xy_m=self.offer.missing.tolist(),
                   packet_sha256=self.packet.sha256(), candidates=candidates, view=chosen["pose"],
                   ordered_poses=states, ordered_actions=actions, return_actions=chosen["return_actions"])
        return True

    def crossing_route(self, graph, paths, budget):
        source, target = self.offer.edge
        heading = self.offer.heading
        begin, end = paths.node((*source, heading)), paths.node((*target, heading))
        proof = self.proof(self.offer.edge)
        if (min(begin, end) < 0 or not graph.edges[*source, heading] or not proof["allowed"]
                or not np.isfinite(paths.cost[begin] + paths.home_cost[end])
                or paths.cost[begin] + 1 + paths.home_cost[end] > budget):
            return False
        states, actions = paths.route(begin)
        actions.append("forward")
        states.append((*target, heading))
        self.route = list(zip(actions, states[1:], strict=True))
        self.stage = "cross"
        self.event("replan", edge=self.offer.edge, pose=self.pose, certificate=proof, ordered_poses=states,
                   ordered_actions=actions, return_actions=int(paths.home_cost[end]))
        return True

    def finish(self, outcome, **fields):
        self.event("end", outcome=outcome, edge=self.offer.edge, action_cost=self.packet.action_id - self.view_begin,
                   **fields)
        self.binding.prepare()
        self.offer = self.route = self.pending = None
        self.stage = None

    def next(self):
        state = self.binding.state
        if state["closed"] or state["phase"] == "return":
            return None
        if self.pending is not None or state["pending"] is not None:
            raise RuntimeError("Paid action has no registered observation")
        if self.route is None:
            budget = self.binding.remaining
            more_views = self.spent_views < MAX_VIEWS_PER_EPISODE
            if not budget or (self.stage is None and not more_views):
                return None
            belief = self.binding.belief
            graph = self.evidence.build(belief)
            paths = MeasuredPaths(graph, self.pose, self.binding.anchor)
            if min(paths.current, paths.home) < 0 or not np.isfinite(paths.home_cost[paths.current]):
                if self.offer is not None:
                    self.finish("lost_measured_return")
                return None
            if self.stage == "replan":
                if not self.crossing_route(graph, paths, budget):
                    self.finish("no_full_sweep_route_with_measured_return")
                    return None
            elif self.stage == "retry":
                if not more_views or not self.dispatch(belief, graph, paths, budget, retry=True):
                    self.finish("no_additional_reachable_view")
                    return None
            elif not self.dispatch(belief, graph, paths, budget):
                return None
        action, expected = self.route[0]
        if self.stage == "cross" and len(self.route) == 1:
            self.final_proof = self.proof(self.offer.edge)
            if not self.final_proof["allowed"]:
                self.finish("latest_packet_revoked_full_sweep", certificate=self.final_proof)
                return None
        check = self.binding.assess(action)
        if not check["allowed"] or tuple(check["next_pose"]) != expected:
            self.finish("native_action_veto", assessment=check)
            return None
        if self.stage == "cross" and len(self.route) == 1:
            self.cross_guard = check
        self.pending = self.packet.action_id + 1, expected
        self.binding.authorize(action, check, self.stage)
        return action

    def observe(self):
        fresh, signature = self.record()
        if self.pending is None:
            return
        paid, expected = self.pending
        self.pending = None
        if self.packet.action_id != paid or self.packet.collision or self.pose != expected:
            self.finish("allocated_motion_failed")
            return
        self.route.pop(0)
        if self.stage == "view":
            self.proof(self.offer.edge)
            missing_now = self.probes(self.offer.edge)
            if len(missing_now) < self.prior_missing:
                self.event("support", edge=self.offer.edge, pose=self.pose, packet_sha256=self.packet.sha256(),
                           newly_supported=self.prior_missing - len(missing_now),
                           remaining_missing_xy_m=missing_now.tolist())
                self.prior_missing = len(missing_now)
        if self.route:
            return
        self.route = None
        if self.stage == "cross":
            self.event("cross", edge=self.offer.edge, pose=self.pose, certificate=self.final_proof,
                       measured_view_paid=self.acquired_paid, packet_sha256=self.packet.sha256(),
                       native_assessment=self.cross_guard, action_cost=paid - self.view_begin)
            self.finish("crossed")
            return
        proof = self.proof(self.offer.edge)
        missing_after = self.probes(self.offer.edge)
        # A fresh integrated packet is required. Accumulated observations en
        # route can also contribute and are included in the paid attempt.
        gained = max(0, len(self.offer.missing) - len(missing_after))
        self.event("view", pose=self.pose, edge=self.offer.edge, fresh_packet=fresh,
                   packet_sha256=self.packet.sha256(), scan_ranges_sha256=signature[0],
                   depth_values_sha256=signature[1], prior_certificate=self.offer.proof,
                   certificate=proof, unsupported_before=len(self.offer.missing),
                   unsupported_after=len(missing_after), newly_supported=gained,
                   remaining_missing_xy_m=missing_after.tolist(), action_cost=paid - self.view_begin)
        if proof["allowed"] and fresh and gained > 0:
            self.acquired_paid = paid
            self.stage = "replan"
            self.consecutive_no_gain = 0
            return
        self.consecutive_no_gain = self.consecutive_no_gain + 1 if gained == 0 else 0
        retry = (proof["reason"] == MISSING_GROUND and self.consecutive_no_gain < MAX_VIEWS_WITHOUT_GAIN
                 and self.attempted[tuple(sorted(self.offer.edge))] < MAX_VIEWS_PER_EDGE)
        if retry:
            self.retry_outcome = "partial_evidence_retry" if gained else "no_new_evidence_retry"
            self.stage = "retry"
        else:
            self.finish("new_observation_did_not_certify_gap", certificate=proof)
