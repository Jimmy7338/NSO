"""Public-template forecast tests without a World or future RGB-D render."""
import ast
from copy import deepcopy
import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np

import nso.prototype_observation_channel as module
from nso.semantic_reliability import SemanticReliabilityBelief, discrete_observation_evi
from nso.surface_evaluation_v40 import CandidateViewV40


def plane_receipt():
    # Analytic public plane, not an actual scene's object position or identity.
    return dict(plane_fit=dict(accepted=True, anchor_world_m=[-.41, -.42, .8],
                              outward_normal_world=[0., -1., 0.]),
                conflict=False, source_frames=[])


def candidate(y=-2., yaw=np.pi/2., view_id='public_candidate'):
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.])
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((np.cross(forward, [0., 0., 1.]), [0., 0., -1.], forward))
    transform[:3, 3] = [0., y, .9]
    intrinsic = np.array([[48., 0., 47.5], [0., 48., 35.5], [0., 0., 1.]])
    return CandidateViewV40(intrinsic, transform, 96, 72, view_id=view_id)


class PrototypeObservationChannelTests(unittest.TestCase):
    def assert_uniform(self, receipt):
        np.testing.assert_array_equal(receipt['channel'], np.full((4, 4), .25))
        self.assertTrue(receipt['fallback'])

    def test_visible_distinct_structures_give_row_stochastic_channel(self):
        receipt = module.build_channel(plane_receipt(), candidate())
        channel = np.array(receipt['channel'])
        self.assertEqual(channel.shape, (4, 4))
        np.testing.assert_allclose(channel.sum(axis=1), 1., rtol=0., atol=1e-12)
        self.assertTrue(np.all(channel > 0))
        self.assertGreater(receipt['distinguishing_ray_count'], 0)
        self.assertGreater(float(np.max(np.ptp(channel, axis=0))), .01)
        self.assertFalse(receipt['observation_channel_is_calibrated'])
        self.assertFalse(receipt['ground_truth_used'])
        self.assertFalse(receipt['actual_future_sensor_rendered'])
        json.dumps(receipt, allow_nan=False)

    def test_no_plane_conflict_and_repeated_view_are_uninformative(self):
        self.assert_uniform(module.build_channel(None, candidate()))
        for field, value in (('plane_fit', None), ('conflict', True)):
            receipt = plane_receipt()
            receipt[field] = value
            self.assert_uniform(module.build_channel(receipt, candidate()))
        self.assert_uniform(module.build_channel(plane_receipt(), candidate(), repeated_view=True))

    def test_behind_camera_and_beyond_range_are_uninformative(self):
        self.assert_uniform(module.build_channel(plane_receipt(), candidate(yaw=-np.pi/2.)))
        self.assert_uniform(module.build_channel(plane_receipt(), candidate(y=-10.)))

    def test_label_or_prior_cannot_enter_api_and_view_name_does_not_change_channel(self):
        baseline = module.build_channel(plane_receipt(), candidate())
        renamed = module.build_channel(plane_receipt(), candidate(view_id='S_or_G_or_Bayes'))
        self.assertEqual(baseline['channel'], renamed['channel'])
        for field in ('class_label', 'structure_probabilities', 'true_structure', 'future_depth'):
            changed = plane_receipt()
            changed[field] = 'forbidden'
            with self.assertRaises(ValueError):
                module.build_channel(changed, candidate())
            changed = plane_receipt()
            changed['plane_fit'][field] = 'forbidden'
            with self.assertRaises(ValueError):
                module.build_channel(changed, candidate())

    def test_geometry_only_input_is_immutable_and_deterministic(self):
        source = plane_receipt()
        saved = deepcopy(source)
        first = module.build_channel(source, candidate())
        second = module.build_channel(source, candidate())
        self.assertEqual(first, second)
        self.assertEqual(source, saved)
        first['channel'][0][0] = 999
        self.assertNotEqual(first, module.build_channel(source, candidate()))

    def test_missing_returns_not_counted_as_distinguishing_geometry(self):
        # Three hypothetical prototypes have identical finite depths, while
        # the fourth has no return. Absence alone must supply no information.
        predictions = np.ones((4, 3, 48))
        predictions[3] = np.inf
        with patch.object(module, '_predictions', return_value=predictions):
            receipt = module.build_channel(plane_receipt(), candidate())
        self.assertFalse(receipt['missing_return_is_empty_space_evidence'])
        self.assertEqual(receipt['distinguishing_ray_count'], 0)
        self.assert_uniform(receipt)

    def test_no_world_or_simulator_dependency(self):
        syntax = ast.parse(inspect.getsource(module))
        forbidden = {'open3d', 'pybullet', 'habitat_sim'}
        for node in ast.walk(syntax):
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.split('.')[0] in forbidden for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or '').split('.')[0], forbidden)
                self.assertNotIn('World', [alias.name for alias in node.names])

    def test_invalid_pose_contract_rejected(self):
        receipt = plane_receipt()
        receipt['plane_fit']['outward_normal_world'] = [0., -2., 0.]
        with self.assertRaises(ValueError):
            module.build_channel(receipt, candidate())
        with self.assertRaises(TypeError):
            module.build_channel(plane_receipt(), dict(world_from_camera=np.eye(4)))

    def test_same_forecast_channel_supports_geometry_bayes_and_shared_evi(self):
        forecast = module.build_channel(plane_receipt(), candidate())
        for sharing, label in ((False, None), (False, 'cabinet'), (True, 'cabinet')):
            belief = SemanticReliabilityBelief(geometry_prior=[.25]*4,
                class_structure_priors={'cabinet': [.1, .7, .1, .1]},
                share_across_instances=sharing)
            belief.register('observed', label)
            result = discrete_observation_evi(belief, 'observed', forecast['channel'],
                                              {'observed': np.eye(4)})
            self.assertGreaterEqual(result['evi'], 0.)
            self.assertAlmostEqual(sum(row['probability'] for row in result['branches']), 1.)
            json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
