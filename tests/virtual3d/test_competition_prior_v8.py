"""Prior regressions on sensor-shaped fixtures; no world or future outcomes.

Grouping and the shared feature block are fixed test inputs. Aperture geometry,
full-prefix subtraction, prior algebra and scoring use the actual implementation.
"""
from types import SimpleNamespace

import numpy as np
import unittest
from unittest.mock import patch

from env.virtual3d import camera_pose
import nso.competition_candidates_v8 as candidates
import nso.competition_prior_v8 as prior


def make_fixture():
    config = SimpleNamespace(fov_deg=90., max_depth_m=4., height_px=72,
                             width_px=96, resolution_m=.2, camera_height_m=.8)
    asset = {'rear_boundary_xy': np.array([4.1, 5.1]),
             'back_axis': np.array([0., 1.]), 'front_axis': np.array([0., -1.]),
             'side_axis': np.array([-1., 0.]),
             'observed_low': np.array([3.5, 3.9, .35]),
             'observed_high': np.array([4.7, 5.1, 1.6]),
             'measured_width_m': 1.2, 'measured_depth_m': 1.2,
             'measured_height_m': 1.25, 'class_vote': -1., 'marked_points': 10}
    retained = camera_pose((12, 20), 2, config, 40)
    omitted = camera_pose((10, 20), 2, config, 40)
    # A paid turn away and back repeats the last actual prefix view.
    route = {'candidate_id': 0, 'cost': 2,
             'states': [[10, 20, 2], [10, 20, 1], [10, 20, 2]],
             'actions': ['left', 'right']}
    mapper = SimpleNamespace(config=config, shape=(40, 40),
                             keyframes=[SimpleNamespace(world_from_camera=retained)],
                             quality_evidence=lambda: {'point': np.zeros((10, 3))})
    block = np.zeros((1, 12)); block[0, :4] = [.125, .25, .4, .5]
    return SimpleNamespace(config=config, asset=asset, retained=retained,
                           omitted=omitted, mapper=mapper, route=route,
                           common=(.125 + .25) * 4.**2 + .4 * 10 * .15**2, block=block)


class CompetitionPriorTests(unittest.TestCase):
    def setUp(self):
        self.fixture = make_fixture()
        group = patch.object(prior, 'measured_assets', lambda _: [self.fixture.asset])
        feature = patch.object(prior, 'response_features', lambda *_: ({'N': self.fixture.block.copy()}, {}))
        group.start(); feature.start()
        self.addCleanup(group.stop); self.addCleanup(feature.stop)

    def test_complete_prefix_prevents_reward_for_nonretained_observed_pose(self):
        f = self.fixture
        # This later view is omitted by the mapper's 0.45 m / cosine<0.8 rule.
        assert np.linalg.norm(f.omitted[:3, 3] - f.retained[:3, 3]) < .45
        assert f.omitted[:3, 2] @ f.retained[:3, 2] > .8
        first = candidates.aperture_support(f.asset, f.retained, f.config)
        last = candidates.aperture_support(f.asset, f.omitted, f.config)
        # Establish a material failure under decimated history, not a zero fixture.
        assert np.maximum(last - first, 0).mean() > .5
        incomplete = prior.score_routes(f.mapper, [f.route], [f.retained])
        assert incomplete['audit'][0]['aperture_factors'][0] > .5
        complete = prior.score_routes(f.mapper, [f.route], [f.retained, f.omitted])
        np.testing.assert_array_equal(complete['audit'][0]['aperture_factors'], [0.])
        for mode in ('N', 'G', 'O', 'S', 'X', 'M'):
            np.testing.assert_array_equal(complete['scores'][mode], [f.common])


    def test_history_is_explicit_and_duplicate_poses_do_not_add_support(self):
        f = self.fixture
        poses = [f.retained, f.omitted]
        original = prior.score_routes(f.mapper, [f.route], poses)
        # No implicit fallback to the decimated mapper history is permitted.
        f.mapper.keyframes = []
        changed = prior.score_routes(f.mapper, [f.route], poses * 3)
        assert changed == original


    def test_marked_common_and_category_swap_and_missing_fallback(self):
        f = self.fixture
        baseline = prior.score_routes(f.mapper, [f.route], [f.retained])
        assert baseline['scores']['G'] == baseline['scores']['O'] == baseline['scores']['M']
        assert baseline['scores']['S'][0] > baseline['scores']['G'][0] > baseline['scores']['X'][0]
        f.asset['class_vote'] = 1.
        swapped = prior.score_routes(f.mapper, [f.route], [f.retained])
        assert swapped['scores']['S'] == baseline['scores']['X']
        assert swapped['scores']['X'] == baseline['scores']['S']
        for mode in ('N', 'G', 'O', 'M'):
            assert swapped['scores'][mode] == baseline['scores'][mode]
        f.asset.update(class_vote=0., marked_points=0)
        missing = prior.score_routes(f.mapper, [f.route], [f.retained])
        assert missing['scores']['S'] == missing['scores']['G'] == baseline['scores']['G']
        assert missing['scores']['X'] == missing['scores']['M'] == baseline['scores']['G']
        # O is intentionally invalid as a matched support comparator outside the
        # protocol's two-marked-group gate; absence cannot prove an S/O advantage.
        assert missing['scores']['O'] == missing['scores']['N']


    def test_omitted_history_argument_is_not_silently_replaced(self):
        with self.assertRaisesRegex(TypeError, 'prefix_camera_poses'):
            prior.score_routes(self.fixture.mapper, [self.fixture.route])


    def test_invalid_history_rejected_before_scoring(self):
        def forbidden(_):
            raise AssertionError('invalid history reached grouping/scoring')
        with patch.object(prior, 'measured_assets', forbidden):
            for poses in (None, [], np.eye(4), np.full((1, 4, 4), np.nan),
                          np.full((1, 4, 4), np.inf), np.zeros((2, 3, 4))):
                with self.subTest(shape=np.shape(poses)):
                    with self.assertRaisesRegex(ValueError, 'actual prefix camera poses'):
                        prior.score_routes(self.fixture.mapper, [self.fixture.route], poses)


if __name__ == '__main__':
    unittest.main()
