"""Mechanism tests for measured-surface visibility, independent of task scores."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import open3d as o3d

from nso.observed_precision_response_v19 import ObservedPrecisionResponseV19


def fixture(*, front_y=7.9, semantic_vote=1., point_count=1):
    config = SimpleNamespace(resolution_m=.2, camera_height_m=.8, max_depth_m=5.,
        width_px=96, height_px=72, fov_deg=90., stereo_model='iid_025px',
        stereo_reference_fx_px=480., stereo_baseline_m=.12)
    mesh = o3d.geometry.TriangleMesh.create_box(1., .08, 1.).translate((3.6, front_y, .3))
    quality = {i:dict(point=np.asarray([4.1 + .2*(i/max(1, point_count-1)), 7.9, .8]),
                      best_range=4., n=1, label=2 if semantic_vote > 0 else 3) for i in range(point_count)}
    mapper = SimpleNamespace(config=config, shape=(40, 40), quality=quality, mesh=lambda:mesh)
    assets = [dict(marked_points=1, class_vote=semantic_vote, observed_low=[3.6, 7.9, .3],
                   observed_high=[4.6, 7.98, 1.3])]
    return mapper, assets


class ObservedPrecisionResponseV19Tests(unittest.TestCase):
    def test_near_visible_front_improves_without_any_hidden_aperture_input(self):
        mapper, assets = fixture()
        response = ObservedPrecisionResponseV19(mapper, assets)
        near = response.score_route(dict(outbound_states=[(6, 20, 0), (5, 20, 0)]))
        far = response.score_route(dict(outbound_states=[(21, 20, 0), (20, 20, 0)]))
        self.assertGreater(near['per_asset_expected_precision_change'][0], .4)
        self.assertGreater(near['per_asset_expected_precision_change'][0], far['per_asset_expected_precision_change'][0])
        self.assertFalse(near['hidden_aperture_used'])
        self.assertEqual(near['per_asset_predicted_measurements'], [1])

    def test_back_facing_outside_fov_range_or_known_occluded_points_give_no_gain(self):
        mapper, assets = fixture()
        response = ObservedPrecisionResponseV19(mapper, assets)
        for pose in [(5, 20, 2), (5, 35, 0), (35, 20, 0)]:
            result = response.score_route(dict(outbound_states=[(5, 20, 1), pose]))
            self.assertEqual(result['per_asset_expected_precision_change'], [0.])
            self.assertEqual(result['per_asset_predicted_measurements'], [0])
        occluded_mapper, occluded_assets = fixture(front_y=7.2)
        blocked = ObservedPrecisionResponseV19(occluded_mapper, occluded_assets).score_route(
            dict(outbound_states=[(6, 20, 0), (5, 20, 0)]))
        self.assertEqual(blocked['per_asset_expected_precision_change'], [0.])
        self.assertEqual(blocked['known_occluded_point_frames'], 1)

    def test_paid_intermediate_views_count_even_when_endpoint_faces_away(self):
        mapper, assets = fixture()
        response = ObservedPrecisionResponseV19(mapper, assets)
        endpoint_away = response.score_route(dict(outbound_states=[(5, 20, 1), (5, 20, 2)]))
        observed_en_route = response.score_route(dict(outbound_states=[(6, 20, 0), (5, 20, 0), (5, 20, 1), (5, 20, 2)]))
        self.assertEqual(endpoint_away['per_asset_expected_precision_change'], [0.])
        self.assertGreater(observed_en_route['per_asset_expected_precision_change'][0], .4)
        self.assertEqual(observed_en_route['paid_future_observations'], 3)
        self.assertEqual(observed_en_route['per_asset_predicted_measurements'], [1])
        unchanged = response.score_route(dict(outbound_states=[(5, 20, 0)]))
        self.assertEqual(unchanged['per_asset_expected_precision_change'], [0.])

    def test_noise_tolerance_semantics_and_support_cap_do_not_change_contract(self):
        mapper, assets = fixture(front_y=7.82, point_count=180)
        before = deepcopy(mapper.quality)
        route = dict(outbound_states=[(6, 20, 0), (5, 20, 0)])
        result = ObservedPrecisionResponseV19(mapper, assets).score_route(route)
        self.assertEqual(result['per_asset_support_counts'], [128])
        self.assertEqual(result['known_occluded_point_frames'], 0)
        self.assertGreater(result['per_asset_expected_precision_change'][0], .4)
        mapper2, assets2 = fixture(front_y=7.82, semantic_vote=-1., point_count=180)
        self.assertEqual(result, ObservedPrecisionResponseV19(mapper2, assets2).score_route(route))
        for key in before:
            np.testing.assert_array_equal(before[key]['point'], mapper.quality[key]['point'])
            self.assertEqual(before[key]['n'], mapper.quality[key]['n'])
        mapper.config.stereo_model = 'ideal'
        ideal = ObservedPrecisionResponseV19(mapper, assets).score_route(route)
        self.assertEqual(ideal['per_asset_expected_precision_change'], [0.])


if __name__ == '__main__':
    unittest.main()
