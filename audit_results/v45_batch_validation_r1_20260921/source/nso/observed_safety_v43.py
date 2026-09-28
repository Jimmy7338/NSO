"""Shared measured-occupancy guard over a supplied primitive navigation prior.

No World, hidden obstacle, object prototype, or semantic score is consulted.
Only already declared edges can be removed. Unknown space is not free space.
"""
from copy import deepcopy
import hashlib
import json
import math
import re

import numpy as np

from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41, PrimitiveStateV41


def array_sha256_v43(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(f"{value.dtype.str}:{value.shape}:".encode()+value.tobytes()).hexdigest()


def segment_box_distance_squared_v43(start, end, box_min, box_max):
    """Exact 2D segment-to-closed-AABB distances, vectorized over boxes."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    low, high = np.asarray(box_min, float), np.asarray(box_max, float)
    if (start.shape != (2,) or end.shape != (2,) or low.ndim != 2 or low.shape[1:] != (2,)
            or high.shape != low.shape or not all(np.isfinite(v).all() for v in (start, end, low, high))
            or np.any(low > high)):
        raise ValueError("finite 2D segment and ordered Nx2 boxes required")
    delta = end-start
    result = np.minimum(np.sum((start-np.clip(start, low, high))**2, axis=1),
                        np.sum((end-np.clip(end, low, high))**2, axis=1))
    denominator = float(delta @ delta)
    if denominator == 0:
        return result
    corners = np.stack((low, high, np.column_stack((low[:, 0], high[:, 1])),
                        np.column_stack((high[:, 0], low[:, 1]))), axis=1)
    parameter = np.clip(np.sum((corners-start)*delta, axis=2)/denominator, 0., 1.)
    result = np.minimum(result, np.min(np.sum((corners-(start+parameter[:, :, None]*delta))**2, axis=2), axis=1))
    first, last = np.zeros(len(low)), np.ones(len(low))
    valid = np.ones(len(low), bool)
    for axis in range(2):
        if abs(delta[axis]) < 1e-15:
            valid &= (start[axis] >= low[:, axis]) & (start[axis] <= high[:, axis])
        else:
            a, b = (low[:, axis]-start[axis])/delta[axis], (high[:, axis]-start[axis])/delta[axis]
            first = np.maximum(first, np.minimum(a, b))
            last = np.minimum(last, np.maximum(a, b))
    result[valid & (first <= last)] = 0.
    return result


class ObservedSafetyV43:
    """Conservative swept-disk rejection; caller reroutes after every update.

    The caller must explicitly choose ``provided_navigation_prior`` to allow
    motion through unobserved map cells. ``reject_unknown`` also removes edges
    whose footprint touches unknown cells or the outside of the map.
    """
    def __init__(self, graph, *, unknown_space_policy, robot_radius_m=.2,
                 maximum_primitive_edges=4096, maximum_cells_per_query=65536):
        if type(graph) is not PublicPrimitiveGraphV41:
            raise TypeError("validated PublicPrimitiveGraphV41 required")
        if unknown_space_policy not in ("provided_navigation_prior", "reject_unknown"):
            raise ValueError("explicit provided navigation prior or reject_unknown policy required")
        if isinstance(robot_radius_m, bool) or not isinstance(robot_radius_m, (int, float)) or not math.isfinite(robot_radius_m) or robot_radius_m < 0:
            raise ValueError("finite nonnegative robot radius required")
        for name, value in (("maximum_primitive_edges", maximum_primitive_edges), ("maximum_cells_per_query", maximum_cells_per_query)):
            if type(value) is not int or value < 1:
                raise ValueError(name+" must be a positive integer")
        if maximum_primitive_edges > 4096:
            raise ValueError("hard limit is 4096 primitive edges")
        self.graph = graph
        self.robot_radius_m = float(robot_radius_m)
        self.unknown_space_policy = unknown_space_policy
        self.maximum_cells_per_query = maximum_cells_per_query
        self._positions = deepcopy(graph.positions)
        self._graph_sha = graph.input_sha256
        edges = set()
        for node in sorted(graph.positions):
            for heading in range(12):
                try:
                    target = graph.successor(PrimitiveStateV41(node, heading), "forward")
                except ValueError:
                    continue
                edges.add(tuple(sorted((node, target.node))))
        if len(edges) > maximum_primitive_edges:
            raise ValueError("expanded primitive edge count exceeds safety bound")
        self._edges = tuple(sorted(edges))
        self._guard_blocked = {}
        self._last = self._map = self._state = self._discovery = None

    def _validate(self, snapshot, occupancy_arrays, state):
        self.graph.validate_state(state)
        if self.graph.input_sha256 != self._graph_sha or self.graph.positions != self._positions:
            raise ValueError("public graph identity or positions changed")
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != "v42.observed_mapping.v1":
            raise ValueError("V42 measured map snapshot required")
        if snapshot.get("backend_poisoned") is not False or snapshot.get("metric_groundtruth_or_scene_input") is not False or snapshot.get("semantic_or_prototype_fusion") is not False:
            raise ValueError("valid measured-only map required")
        if snapshot.get("coverage_fraction") is not None or snapshot.get("coverage_denominator") is not None:
            raise ValueError("online map must not contain a ground-truth coverage denominator")
        if not isinstance(occupancy_arrays, (tuple, list)) or len(occupancy_arrays) != 2:
            raise ValueError("(belief, observed_this_frame) arrays required")
        belief, observed = map(np.asarray, occupancy_arrays)
        if belief.ndim != 2 or belief.dtype != np.int8 or not 1 <= belief.size <= 4_000_000 or not np.isin(belief, [-1, 0, 1]).all():
            raise ValueError("bounded int8 occupancy with values -1/0/1 required")
        if observed.shape != belief.shape or observed.dtype != np.bool_:
            raise ValueError("aligned boolean current-observation mask required")
        if snapshot.get("shape") != list(belief.shape) or snapshot.get("occupancy_sha256") != array_sha256_v43(belief):
            raise ValueError("map shape or occupancy SHA mismatch")
        counts = (int(np.count_nonzero(belief >= 0)), int(np.count_nonzero(belief == 0)), int(np.count_nonzero(belief == 1)))
        if tuple(snapshot.get(k) for k in ("known_cells", "observed_free_cells", "observed_occupied_cells")) != counts:
            raise ValueError("map count mismatch")
        resolution = snapshot.get("resolution_m")
        if isinstance(resolution, bool) or not isinstance(resolution, (int, float)) or not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("positive finite map resolution required")
        origin = np.asarray(snapshot.get("origin_xy_m"), float)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("finite map origin required")
        if snapshot.get("grid_convention") != "row increases with world y; column increases with world x; origin is lower-left boundary":
            raise ValueError("map XY convention mismatch")
        frames = snapshot.get("frames")
        receipts = snapshot.get("receipts")
        if type(frames) is not int or frames < 1 or not isinstance(receipts, list) or len(receipts) != frames:
            raise ValueError("nonempty one-receipt-per-frame map history required")
        last = receipts[-1]
        if (not isinstance(last, dict) or last.get("paid_step") != frames-1
                or not isinstance(last.get("observation_sha256"), str)
                or not re.fullmatch("[0-9a-f]{64}", last["observation_sha256"])):
            raise ValueError("latest paid observation provenance required")
        yaw = state.heading*math.pi/6
        expected = np.array([[math.sin(yaw), 0., math.cos(yaw), self._positions[state.node][0]],
                             [-math.cos(yaw), 0., math.sin(yaw), self._positions[state.node][1]],
                             [0., -1., 0., self.graph.camera_height_m], [0., 0., 0., 1.]])
        pose = np.asarray(last.get("world_from_camera"), float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose, expected, atol=1e-6, rtol=0):
            raise ValueError("current public state differs from latest mapped sensor pose")
        snapshot_sha = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if self._last is not None:
            if frames < self._last["map_version"]:
                raise ValueError("map version moved backwards")
            if frames == self._last["map_version"] and (snapshot_sha != self._last["mapper_snapshot_sha256"]
                    or array_sha256_v43(observed) != self._last["observed_mask_sha256"] or state != self._state):
                raise ValueError("same map version changed data or current state")
        return dict(belief=belief.copy(), observed=observed.copy(), resolution=float(resolution),
                    origin=origin.copy(), version=frames, observation_sha=last["observation_sha256"],
                    occupancy_sha=snapshot["occupancy_sha256"], snapshot_sha=snapshot_sha,
                    observed_mask_sha=array_sha256_v43(observed))

    def _local_cells(self, state, start, end, radius):
        resolution, origin, shape = state["resolution"], state["origin"], state["belief"].shape
        # One extra boundary cell ensures exact closed-square tangency counts.
        raw_low = (np.minimum(start, end)-radius-origin)/resolution
        raw_high = (np.maximum(start, end)+radius-origin)/resolution
        if not np.isfinite([raw_low, raw_high]).all() or np.max(np.abs([raw_low, raw_high])) > 1e12:
            raise ValueError("footprint coordinates exceed bounded grid-index contract")
        low = np.floor(raw_low).astype(np.int64)-1
        high = np.floor(raw_high).astype(np.int64)
        span = high-low+1
        if np.any(span < 1) or int(span[0])*int(span[1]) > self.maximum_cells_per_query:
            raise ValueError("local footprint query exceeds bounded cell count")
        cols, rows = np.meshgrid(np.arange(low[0], high[0]+1), np.arange(low[1], high[1]+1))
        index = np.column_stack((cols.ravel(), rows.ravel()))
        box_min = origin+index*resolution
        distances = segment_box_distance_squared_v43(start, end, box_min, box_min+resolution)
        touched = distances <= radius**2 + 1e-12
        index = index[touched]
        inside = ((index[:, 0] >= 0) & (index[:, 0] < shape[1]) & (index[:, 1] >= 0) & (index[:, 1] < shape[0]))
        values = np.full(len(index), -1, np.int8)
        values[inside] = state["belief"][index[inside, 1], index[inside, 0]]
        occupied = index[values == 1]
        unknown = index[values == -1]
        return dict(occupied_cells=len(occupied), unknown_cells=len(unknown),
                    outside_cells=int(np.count_nonzero(~inside)),
                    occupied_witness_rc=None if not len(occupied) else occupied[0, ::-1].tolist(),
                    unknown_witness_rc=None if not len(unknown) else unknown[0, ::-1].tolist(),
                    blocked=bool(len(occupied) or (len(unknown) and self.unknown_space_policy == "reject_unknown")))

    def update(self, mapper_snapshot, occupancy_arrays, current_state):
        state = self._validate(mapper_snapshot, occupancy_arrays, current_state)
        if self._last is not None and state["version"] == self._last["map_version"]:
            return deepcopy(self._last)
        position = np.asarray(self._positions[current_state.node])
        footprint = self._local_cells(state, position, position, self.robot_radius_m)
        proposals, unknown_edges = [], 0
        for left, right in self._edges:
            evidence = self._local_cells(state, np.asarray(self._positions[left]), np.asarray(self._positions[right]), self.robot_radius_m)
            unknown_edges += int(evidence["unknown_cells"] > 0)
            if evidence["blocked"] and (left, right) not in self._guard_blocked:
                proposals.append(dict(edge=[left, right], evidence=evidence, map_version=state["version"],
                    occupancy_sha256=state["occupancy_sha"], observation_sha256=state["observation_sha"]))
        # Only now mutate graph. All map/hash/pose/bounds checks have succeeded.
        for item in proposals:
            left, right = item["edge"]
            self.graph.block_observed_edge(left, right)
            self._guard_blocked[(left, right)] = item
        receipt = dict(schema_version="v43.observed_safety.v1", map_version=state["version"],
            paid_step=state["version"]-1, observation_sha256=state["observation_sha"],
            occupancy_sha256=state["occupancy_sha"], mapper_snapshot_sha256=state["snapshot_sha"],
            observed_mask_sha256=state["observed_mask_sha"], public_graph_sha256=self._graph_sha,
            robot_radius_m=self.robot_radius_m, current_state=dict(node=current_state.node, heading=current_state.heading),
            current_footprint_conflict=bool(footprint["occupied_cells"]), current_footprint_blocked=footprint["blocked"],
            current_footprint_evidence=footprint, checked_primitive_edges=len(self._edges),
            newly_blocked_edges=proposals, guard_blocked_edges=list(self._guard_blocked.values()),
            unknown_space_policy=self.unknown_space_policy, edges_touching_unknown=unknown_edges,
            unknown_is_observed_free=False, outside_map_is_unknown=True,
            graph_mutation="remove declared edges only; caller must reroute and recheck return reserve",
            guarantee="observed cell-square swept-disk rejection only; unobserved obstacles may remain",
            semantic_input_used=False, extra_free_actions=0)
        self._map, self._state, self._last = state, current_state, receipt
        self._discovery = None
        return deepcopy(receipt)

    def guard_action(self, action):
        if self._last is None:
            raise ValueError("update safety from the current mapped frame first")
        if action not in ("forward", "left", "right", "observe", "stop", "blocked"):
            raise ValueError("unknown primitive or terminal action")
        allowed, reason = True, "measured_footprint_guard_passed_under_declared_unknown_policy"
        if action in ("stop", "blocked"):
            reason = "terminal_action_has_no_paid_motion_or_capture"
        elif self._last["current_footprint_blocked"]:
            allowed, reason = False, "current_footprint_observed_or_unknown_conflict_halt"
        elif action == "forward":
            try:
                self.graph.successor(self._state, "forward")
            except ValueError:
                allowed, reason = False, "declared_forward_edge_absent_or_blocked"
        return dict(requested_action=action, action=action if allowed else "blocked", allowed=allowed, reason=reason,
                    map_version=self._last["map_version"], occupancy_sha256=self._last["occupancy_sha256"],
                    observation_sha256=self._last["observation_sha256"], extra_free_actions=0)

    def discovery_utilities(self, states, *, radius_m=2.):
        if self._map is None:
            raise ValueError("update safety from the current mapped frame first")
        if isinstance(radius_m, bool) or not isinstance(radius_m, (int, float)) or not math.isfinite(radius_m) or radius_m <= 0:
            raise ValueError("positive finite discovery radius required")
        states = tuple(states)
        if len(states) > 256:
            raise ValueError("bounded discovery candidate list required")
        state, rows = self._map, []
        cache, utilities = {}, {}
        for candidate in states:
            self.graph.validate_state(candidate)
            if candidate.node not in cache:
                center = np.asarray(self._positions[candidate.node])
                resolution, origin, shape = state["resolution"], state["origin"], state["belief"].shape
                lower = np.maximum(0, np.floor((center-radius_m-origin)/resolution).astype(np.int64))
                upper = np.minimum(np.array([shape[1]-1, shape[0]-1]), np.floor((center+radius_m-origin)/resolution).astype(np.int64))
                span = np.maximum(0, upper-lower+1)
                if int(span[0])*int(span[1]) > self.maximum_cells_per_query:
                    raise ValueError("local discovery query exceeds bounded cell count")
                cols, rr = np.meshgrid(np.arange(lower[0], upper[0]+1), np.arange(lower[1], upper[1]+1))
                positions = origin+(np.stack((cols, rr), axis=-1)+.5)*resolution
                inside = np.sum((positions-center)**2, axis=-1) <= radius_m**2
                count = int(np.count_nonzero(inside & (state["belief"][rr, cols] == -1)))
                cache[candidate.node] = count*resolution**2
            utilities[candidate] = cache[candidate.node]
            rows.append(dict(node=candidate.node, heading=candidate.heading, unknown_cell_area_m2=cache[candidate.node]))
        self._discovery = dict(schema_version="v43.shared_discovery_proxy.v1", map_version=state["version"],
            occupancy_sha256=state["occupancy_sha"], observation_sha256=state["observation_sha"], radius_m=float(radius_m),
            candidate_scores=rows, semantic_input_used=False,
            interpretation="unknown-cell area in a planar disk; occlusion is unmodeled; proxy, not forecast of measured gain",
            outside_map_area_included=False, orientation_independent_360_degree_scan_proxy=True)
        return utilities

    def discovery_receipt(self):
        return deepcopy(self._discovery)
