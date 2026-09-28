"""Analytic occupancy/geometry fixtures; no DEV World or rollout claims."""
from copy import deepcopy
import hashlib
import math
import unittest

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.primitive_navigation_v41 import (PublicPrimitiveGraphV41, PrimitiveStateV41,
                                         ReturnAwarePrimitiveRouterV41)
from nso.observed_safety_v43 import (ObservedSafetyV43, array_sha256_v43,
                                    segment_box_distance_squared_v43)


def graph(diagonal=False):
    endpoint = [.25*math.cos(math.pi/6), .125] if diagonal else [.25, 0.]
    return PublicPrimitiveGraphV41(dict(schema_version="v41.public_navigation.v1",
        source_kind="provided_navigation_prior", nodes={"home": [0., 0.], "goal": endpoint},
        edges=[["home", "goal"]]))


def pose(g, state):
    yaw = state.heading*math.pi/6
    return np.array([[math.sin(yaw), 0., math.cos(yaw), g.positions[state.node][0]],
                     [-math.cos(yaw), 0., math.sin(yaw), g.positions[state.node][1]],
                     [0., -1., 0., .9], [0., 0., 0., 1.]])


def fixture(g, state=None, *, occupied=(), fill=0, shape=(200, 200), resolution=.01,
            origin=(-1., -1.), frames=1):
    """Hash-bound analytic occupancy contract, not an actual mapper run."""
    state = state or PrimitiveStateV41("home", 0)
    belief = np.full(shape, fill, np.int8)
    for row, col in occupied:
        belief[row, col] = 1
    observed = np.zeros(shape, bool)
    free, obstacle = int(np.count_nonzero(belief == 0)), int(np.count_nonzero(belief == 1))
    snapshot = dict(schema_version="v42.observed_mapping.v1", backend_poisoned=False,
        metric_groundtruth_or_scene_input=False, semantic_or_prototype_fusion=False,
        coverage_fraction=None, coverage_denominator=None, shape=list(shape),
        occupancy_sha256=array_sha256_v43(belief), known_cells=free+obstacle,
        observed_free_cells=free, observed_occupied_cells=obstacle, resolution_m=resolution,
        origin_xy_m=list(origin), frames=frames,
        grid_convention="row increases with world y; column increases with world x; origin is lower-left boundary",
        receipts=[dict(paid_step=step, observation_sha256=hashlib.sha256(f"analytic_{step}".encode()).hexdigest(),
                       world_from_camera=pose(g, state).tolist()) for step in range(frames)])
    return snapshot, (belief, observed)


class ObservedSafetyV43Tests(unittest.TestCase):
    def test_segment_middle_intersection_and_closed_box_tangency(self):
        result = segment_box_distance_squared_v43([0., 0.], [1., 0.],
            [[.4, -.1], [.4, .2], [.4, .2001]], [[.6, .1], [.6, .3], [.6, .3]])
        np.testing.assert_allclose(result, [0., .04, .2001**2], atol=1e-15)

    def test_diagonal_corner_distance_not_bounding_box_collision(self):
        result = segment_box_distance_squared_v43([0., 0.], [1., 1.], [[.45, .6]], [[.55, .7]])
        self.assertAlmostEqual(result[0], (.6-.55)**2/2)
        self.assertGreater(result[0], 0.)

    def test_midpoint_obstacle_blocks_edge_although_endpoint_disks_are_clear(self):
        g = graph()
        safety = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior")
        snapshot, arrays = fixture(g, occupied=[(119, 110)])
        receipt = safety.update(snapshot, arrays, PrimitiveStateV41("home", 0))
        self.assertFalse(receipt["current_footprint_conflict"])
        self.assertEqual(len(receipt["newly_blocked_edges"]), 1)
        self.assertFalse(safety.guard_action("forward")["allowed"])
        self.assertTrue(safety.guard_action("left")["allowed"])
        with self.assertRaises(ValueError):
            g.successor(PrimitiveStateV41("home", 0), "forward")

    def test_exact_tangent_cell_square_is_collision(self):
        g = graph()
        safety = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior")
        snapshot, arrays = fixture(g, occupied=[(120, 110)])
        receipt = safety.update(snapshot, arrays, PrimitiveStateV41("home", 0))
        self.assertEqual(len(receipt["newly_blocked_edges"]), 1)
        self.assertFalse(receipt["current_footprint_conflict"])

    def test_thirty_degree_sweep_rejects_aabb_false_positive(self):
        g = graph(diagonal=True)
        safety = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior", robot_radius_m=.02)
        state = PrimitiveStateV41("home", 1)
        snapshot, arrays = fixture(g, state, occupied=[(112, 112)])
        self.assertEqual(safety.update(snapshot, arrays, state)["newly_blocked_edges"], [])
        self.assertTrue(safety.guard_action("forward")["allowed"])
        snapshot, arrays = fixture(g, state, occupied=[(104, 110)], frames=2)
        self.assertEqual(len(safety.update(snapshot, arrays, state)["newly_blocked_edges"]), 1)

    def test_unknown_space_requires_explicit_prior_policy(self):
        with self.assertRaises(TypeError):
            ObservedSafetyV43(graph())
        g = graph()
        state = PrimitiveStateV41("home", 0)
        snapshot, arrays = fixture(g, fill=-1)
        allowed = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior")
        receipt = allowed.update(snapshot, arrays, state)
        self.assertTrue(allowed.guard_action("forward")["allowed"])
        self.assertGreater(receipt["edges_touching_unknown"], 0)
        self.assertFalse(receipt["unknown_is_observed_free"])
        g2 = graph()
        rejected = ObservedSafetyV43(g2, unknown_space_policy="reject_unknown")
        receipt2 = rejected.update(snapshot, arrays, state)
        self.assertFalse(rejected.guard_action("forward")["allowed"])
        self.assertTrue(receipt2["current_footprint_blocked"])
        self.assertFalse(receipt2["current_footprint_conflict"])

    def test_outside_map_footprint_is_unknown(self):
        g = graph()
        safety = ObservedSafetyV43(g, unknown_space_policy="reject_unknown")
        snapshot, arrays = fixture(g, shape=(10, 10), origin=(0., 0.))
        receipt = safety.update(snapshot, arrays, PrimitiveStateV41("home", 0))
        self.assertGreater(receipt["current_footprint_evidence"]["outside_cells"], 0)
        self.assertFalse(safety.guard_action("left")["allowed"])

    def test_current_footprint_conflict_stops_rotation_and_paid_capture(self):
        g = graph()
        safety = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior")
        snapshot, arrays = fixture(g, occupied=[(100, 100)])
        receipt = safety.update(snapshot, arrays, PrimitiveStateV41("home", 0))
        self.assertTrue(receipt["current_footprint_conflict"])
        for action in ("forward", "left", "right", "observe"):
            guarded = safety.guard_action(action)
            self.assertEqual(guarded["action"], "blocked")
            self.assertFalse(guarded["allowed"])
            self.assertEqual(guarded["extra_free_actions"], 0)
        self.assertTrue(safety.guard_action("stop")["allowed"])
        self.assertEqual(arrays[0][100, 100], 1)

    def test_hash_mismatch_rejects_before_any_edge_removal(self):
        g = graph()
        state = PrimitiveStateV41("home", 0)
        safety = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior")
        snapshot, arrays = fixture(g, occupied=[(119, 110)])
        snapshot["occupancy_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "SHA"):
            safety.update(snapshot, arrays, state)
        self.assertEqual(g.successor(state, "forward").node, "goal")
        with self.assertRaises(ValueError):
            safety.guard_action("forward")

    def test_current_state_must_match_latest_map_receipt_pose(self):
        g = graph()
        safety = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior")
        snapshot, arrays = fixture(g, occupied=[(119, 110)])
        with self.assertRaisesRegex(ValueError, "latest mapped sensor pose"):
            safety.update(snapshot, arrays, PrimitiveStateV41("goal", 0))
        self.assertEqual(g.successor(PrimitiveStateV41("home", 0), "forward").node, "goal")

    def test_version_replay_is_idempotent_but_changed_mask_or_old_version_rejects(self):
        g = graph()
        state = PrimitiveStateV41("home", 0)
        safety = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior")
        snapshot, arrays = fixture(g, frames=2)
        first = safety.update(snapshot, arrays, state)
        self.assertEqual(first, safety.update(snapshot, arrays, state))
        arrays[1][0, 0] = True
        with self.assertRaisesRegex(ValueError, "same map version"):
            safety.update(snapshot, arrays, state)
        old, old_arrays = fixture(g, frames=1)
        with self.assertRaisesRegex(ValueError, "backwards"):
            safety.update(old, old_arrays, state)

    def test_discovery_area_is_shared_unknown_grid_proxy_with_explicit_limits(self):
        g = graph()
        state = PrimitiveStateV41("home", 0)
        safety = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior")
        snapshot, arrays = fixture(g, fill=-1)
        safety.update(snapshot, arrays, state)
        views = [state, PrimitiveStateV41("home", 6), PrimitiveStateV41("goal", 0)]
        scores = safety.discovery_utilities(views, radius_m=.3)
        self.assertAlmostEqual(scores[views[0]], scores[views[1]])
        self.assertGreater(scores[state], 0.)
        receipt = safety.discovery_receipt()
        self.assertIn("occlusion is unmodeled", receipt["interpretation"])
        self.assertFalse(receipt["semantic_input_used"])
        snapshot2, arrays2 = fixture(g, fill=0, frames=2)
        safety.update(snapshot2, arrays2, state)
        self.assertEqual(safety.discovery_utilities(views, radius_m=.3)[state], 0.)

    def test_guard_blocks_return_and_router_reports_no_feasible_return_without_extra_actions(self):
        g = graph()
        home, goal = PrimitiveStateV41("home", 0), PrimitiveStateV41("goal", 0)
        safety = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior", robot_radius_m=.05)
        router = ReturnAwarePrimitiveRouterV41(g, home=home, budget=30)
        def packet(step, state):
            return PaidRGBDObservationV40(f"analytic_{step}", step, np.zeros((1, 1, 3), np.uint8),
                np.zeros((1, 1)), np.eye(3), pose(g, state))
        router.accept(packet(0, home))
        self.assertEqual(router.choose({goal: 1.})["action"], "forward")
        router.accept(packet(1, goal))
        snapshot, arrays = fixture(g, goal, occupied=[(100, 110)], frames=2)
        receipt = safety.update(snapshot, arrays, goal)
        self.assertFalse(receipt["current_footprint_conflict"])
        result = router.choose({})
        self.assertEqual(result["action"], "blocked")
        self.assertEqual(result["reason"], "no_budget_feasible_return")
        self.assertEqual(router.step, 1)
        self.assertIsNone(router.pending)

    def test_local_query_cap_fails_before_edge_mutation(self):
        g = graph()
        safety = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior")
        snapshot, arrays = fixture(g, resolution=.0001)
        with self.assertRaisesRegex(ValueError, "bounded cell count"):
            safety.update(snapshot, arrays, PrimitiveStateV41("home", 0))
        self.assertEqual(g.successor(PrimitiveStateV41("home", 0), "forward").node, "goal")

    def test_shared_input_produces_identical_guard_and_no_mutable_alias(self):
        left, right = graph(), graph()
        a = ObservedSafetyV43(left, unknown_space_policy="provided_navigation_prior")
        b = ObservedSafetyV43(right, unknown_space_policy="provided_navigation_prior")
        state = PrimitiveStateV41("home", 0)
        snapshot, arrays = fixture(left, fill=-1, occupied=[(119, 110)])
        self.assertEqual(a.update(snapshot, arrays, state), b.update(snapshot, arrays, state))
        before = a.discovery_utilities([state], radius_m=.3)
        arrays[0][:] = 0
        snapshot["receipts"][-1]["observation_sha256"] = "0"*64
        self.assertEqual(before, a.discovery_utilities([state], radius_m=.3))

    def test_map_count_and_graph_position_mutation_rejected(self):
        g = graph()
        safety = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior")
        state = PrimitiveStateV41("home", 0)
        snapshot, arrays = fixture(g)
        invalid = deepcopy(snapshot)
        invalid["known_cells"] += 1
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            safety.update(invalid, arrays, state)
        g.positions["goal"] = (.5, 0.)
        with self.assertRaisesRegex(ValueError, "positions changed"):
            safety.update(snapshot, arrays, state)

    def test_actual_mapper_scan_endpoint_drives_guard_with_packet_provenance(self):
        from nso.observed_mapper_v42 import ObservedMapperV42
        from utils.rgbd_contract import PlanarScan
        g = graph()
        state = PrimitiveStateV41("home", 0)
        safety = ObservedSafetyV43(g, unknown_space_policy="provided_navigation_prior")
        mapper = ObservedMapperV42(shape=(200, 200), resolution_m=.01, origin_xy_m=(-1., -1.))
        observation = PaidRGBDObservationV40("analytic_paid_scan", 0, np.zeros((1, 1, 3), np.uint8),
            np.zeros((1, 1)), np.eye(3), pose(g, state))
        laser = np.eye(4)
        laser[2, 3] = .3
        measured_scan = PlanarScan(0., np.array([math.hypot(.125, .19)]),
            math.atan2(.19, .125), .01, 8., laser)
        mapper.update(observation, measured_scan)
        receipt = safety.update(mapper.snapshot(), mapper.occupancy_arrays(), state)
        self.assertEqual(receipt["observation_sha256"], observation.sha256())
        self.assertEqual(receipt["occupancy_sha256"], mapper.snapshot()["occupancy_sha256"])
        self.assertFalse(receipt["current_footprint_conflict"])
        self.assertEqual(len(receipt["newly_blocked_edges"]), 1)
        self.assertEqual(mapper.snapshot()["tsdf_integration_count"], 0)


if __name__ == "__main__":
    unittest.main()
