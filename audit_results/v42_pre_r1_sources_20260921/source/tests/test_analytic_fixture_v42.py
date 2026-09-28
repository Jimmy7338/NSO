"""Analytic packet checks only; no DEV World or performance comparison."""
import math
import unittest

import numpy as np

from nso.analytic_fixture_v42 import (ANALYTIC_MARKER_COLOR_V42,
    analytic_forward_sequence_v42, analytic_plane_observation_v42)
from nso.observed_instances_v41 import ObservedInstancesV41
from nso.observed_residual_v41 import ObservedResidualV41, observed_points_v41
from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41, PrimitiveStateV41


class AnalyticFixtureV42Tests(unittest.TestCase):
    def test_forward_changes_actual_pose_and_axial_depth_of_same_world_planes(self):
        first, second = analytic_forward_sequence_v42()
        np.testing.assert_allclose(second.world_from_camera[:3, 3]-first.world_from_camera[:3, 3], [.25, 0., 0.])
        for obs in (first, second):
            valid = np.flatnonzero(obs.depth_m)
            measured = observed_points_v41(obs, valid)
            marker = np.all(obs.rgb.reshape(-1, 3)[valid] == ANALYTIC_MARKER_COLOR_V42, axis=1)
            np.testing.assert_allclose(measured[marker, 0], 1.83)
            np.testing.assert_allclose(measured[~marker, 0], 1.85)
        self.assertNotEqual(first.sha256(), second.sha256())
        self.assertFalse(np.array_equal(first.depth_m, second.depth_m))
        self.assertEqual((first.paid_step, second.paid_step), (0, 1))

    def test_two_observations_activate_semantics_with_shared_geometry_and_residual(self):
        geometries, losses, active = {}, {}, {}
        for mode in ('G', 'S'):
            ledger = ObservedInstancesV41(palette={'cabinet': ANALYTIC_MARKER_COLOR_V42},
                structure_names=('planar', 'recessed', 'louvered', 'open_frame'),
                class_structure_prior={'cabinet': [.4, .3, .2, .1]}, mode=mode)
            caller = ObservedResidualV41()
            losses[mode], active[mode] = [], []
            for obs in analytic_forward_sequence_v42():
                association = ledger.observe(obs)
                self.assertEqual(len(association['accepted']), 1)
                self.assertTrue(association['accepted'][0]['geometry_feedback_eligible'])
                evidence = caller.observe(obs, association['accepted'])['results'][0]
                self.assertTrue(evidence['accepted'])
                self.assertTrue(ledger.apply_geometry_feedback(evidence['instance_id'],
                    frame_id=obs.frame_id, observation_sha256=obs.sha256(),
                    log_likelihoods=evidence['log_likelihoods'])['applied'])
                losses[mode].append(evidence['log_likelihoods'])
                active[mode].append(ledger.snapshot()['instances'][0]['semantic_conditioning_used'])
            geometries[mode] = ledger.geometry_snapshot()
        self.assertEqual(losses['G'], losses['S'])
        self.assertEqual(geometries['G'], geometries['S'])
        self.assertEqual(active['G'], [False, False])
        self.assertEqual(active['S'], [False, True])

    def test_fixture_states_match_public_primitive_motion(self):
        graph = PublicPrimitiveGraphV41(dict(schema_version='v41.public_navigation.v1',
            source_kind='provided_navigation_prior', nodes={'home': [.75, .75], 'next': [1., .75]},
            edges=[['home', 'next']]))
        first, second = analytic_forward_sequence_v42()
        start = graph.state_from_observation(first)
        self.assertEqual(start, PrimitiveStateV41('home', 0))
        self.assertEqual(graph.state_from_observation(second), graph.successor(start, 'forward'))

    def test_rotation_recomputes_ray_intersection_not_just_pose_metadata(self):
        first = analytic_plane_observation_v42(0)
        turned = analytic_plane_observation_v42(1, (.75, .75, math.pi/6))
        self.assertFalse(np.array_equal(first.depth_m, turned.depth_m))
        measured = observed_points_v41(turned, np.flatnonzero(turned.depth_m))
        self.assertTrue(np.all(np.min(np.abs(measured[:, 0, None]-[1.83, 1.85]), axis=1) < 1e-12))
        self.assertGreater(np.ptp(turned.depth_m[turned.depth_m > 0]), .1)

    def test_invalid_pose_and_step_rejected(self):
        for pose in ([0, 0], [np.nan, 0, 0]):
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                analytic_plane_observation_v42(0, pose)
        for step in (-1, True, 1.5):
            with self.subTest(step=step), self.assertRaises(ValueError):
                analytic_plane_observation_v42(step)


if __name__ == '__main__':
    unittest.main()
