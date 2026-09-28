"""Pure analytic exposure contracts: no World, trajectory, TSDF or evaluation."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from env.development_sensor_v41 import runtime_counts_v41
from nso.analytic_fixture_v42 import analytic_forward_sequence_v42
from nso.controller_article_v1 import ArticleObservedResidualV1, ArticleViewPredictorV1
from nso.surface_evaluation_v40 import CandidateViewV40
from nso.view_quality_article_exposure_v3 import (
    ArticleExposureViewPredictorV3, nominal_visible_mask)
from nso.view_quality_v42 import _digest, _sample_union
from tests.test_view_quality_v42 import ledger, public_view


def history(view, step=0):
    camera = dict(intrinsic=view.intrinsic.tolist(),
        world_from_camera=view.world_from_camera.tolist(),
        width=view.width, height=view.height)
    return dict(**camera, camera_geometry_sha256=_digest(camera), paid_step=step,
                frame_id=f'analytic-camera-{step}', observation_sha256='a'*64)


def analytic_box_predictors(box):
    """A finite quadrature box fixture, never a generated study scene."""
    base = ArticleViewPredictorV1(residual=ArticleObservedResidualV1())
    new = ArticleExposureViewPredictorV3(residual=ArticleObservedResidualV1())
    boxes = np.asarray([box], dtype=float)
    for predictor in (base, new):
        predictor.bank = SimpleNamespace(candidates=[('planar', 1., boxes, np.zeros(3))],
                                         sha256='analytic-single-box')
        predictor._samples = [_sample_union(boxes, .08)]
    state = dict(plane=dict(outward_normal_world=[0., -1., 0.], anchor_world_m=[0., 0., 0.]))
    # A disjoint finite measured-support fixture leaves the nominal box unseen.
    support = np.asarray([[10., 10., 10.]])
    return base, new, state, support


class ArticleExposureTests(unittest.TestCase):
    def setUp(self):
        self.counts = runtime_counts_v41()

    def tearDown(self):
        self.assertEqual(runtime_counts_v41(), self.counts)

    def test_distinct_yaw_fully_overlapping_nominal_exposure_is_zero(self):
        old, new, state, support = analytic_box_predictors([2., 2.2, -.2, .2, .7, 1.1])
        prior = public_view('old', x=0., y=0.)
        target = public_view('new-yaw', x=0., y=0., yaw=np.pi/6)
        new._paid_cameras = [history(prior)]
        before = old._areas(state, support, target)[0]
        after = new._areas(state, support, target)[0]
        self.assertGreater(before['new_surface_area_m2'], 0.)
        self.assertEqual(after['new_surface_area_m2'], 0.)
        self.assertEqual(after['excluded_previously_exposed_unseen_area_m2'],
                         before['new_surface_area_m2'])
        self.assertTrue(after['exposure_exclusion_is_not_measured_surface'])

    def test_rotating_into_previously_unseen_fov_retains_area(self):
        old, new, state, support = analytic_box_predictors([2., 2.2, 1.8, 2., .7, 1.1])
        prior = public_view('old', x=0., y=0., yaw=-np.pi/6)
        target = public_view('new-fov', x=0., y=0., yaw=np.pi/6)
        new._paid_cameras = [history(prior)]
        before = old._areas(state, support, target)[0]
        after = new._areas(state, support, target)[0]
        self.assertGreater(before['new_surface_area_m2'], 0.)
        self.assertEqual(after['new_surface_area_m2'], before['new_surface_area_m2'])
        self.assertEqual(after['excluded_previously_exposed_unseen_area_m2'], 0.)

    def test_rotation_into_valid_axial_range_retains_area(self):
        old, new, state, support = analytic_box_predictors([4.1, 4.2, -.1, .1, .8, 1.])
        prior = public_view('outside-far', x=0., y=0.)
        target = public_view('inside-far', x=0., y=0., yaw=np.pi/6)
        new._paid_cameras = [history(prior)]
        self.assertEqual(old._areas(state, support, prior)[0]['visible_area_m2'], 0.)
        before = old._areas(state, support, target)[0]['new_surface_area_m2']
        self.assertGreater(before, 0.)
        self.assertEqual(new._areas(state, support, target)[0]['new_surface_area_m2'], before)

    def test_translation_does_not_inherit_other_center_exposure(self):
        old, new, state, support = analytic_box_predictors([2., 2.2, -.2, .2, .7, 1.1])
        prior = public_view('old', x=0., y=0.)
        target = public_view('translated', x=.25, y=0.)
        new._paid_cameras = [history(prior)]
        before = old._areas(state, support, target)[0]
        after = new._areas(state, support, target)[0]
        self.assertEqual(after['same_center_distinct_paid_views'], 0)
        for name, value in before.items():
            self.assertEqual(after[name], value)

    def test_no_history_matches_frozen_areas_and_preserves_support(self):
        old, new, state, support = analytic_box_predictors([2., 2.2, -.2, .2, .7, 1.1])
        view = public_view('target', x=0., y=0.)
        state_before, support_before = deepcopy(state), support.copy()
        before, after = old._areas(state, support, view)[0], new._areas(state, support, view)[0]
        for name, value in before.items():
            self.assertEqual(after[name], value)
        np.testing.assert_array_equal(support, support_before)
        self.assertEqual(state, state_before)

    def test_union_is_order_invariant_and_does_not_count_duplicate_frames(self):
        _, a, state, support = analytic_box_predictors([2., 2.2, -.2, .2, .7, 1.1])
        _, b, _, _ = analytic_box_predictors([2., 2.2, -.2, .2, .7, 1.1])
        first, second = public_view('v0', x=0., y=0.), public_view('v1', x=0., y=0., yaw=np.pi/6)
        a._paid_cameras = [history(first), history(second, 1), history(first, 2)]
        b._paid_cameras = [history(second, 1), history(first)]
        target = public_view('target', x=0., y=0., yaw=-np.pi/6)
        ar, br = a._areas(state, support, target)[0], b._areas(state, support, target)[0]
        self.assertEqual(ar, br)
        self.assertEqual(ar['same_center_distinct_paid_views'], 2)

    def test_incompatible_intrinsics_or_image_shape_do_not_discount(self):
        old, new, state, support = analytic_box_predictors([2., 2.2, -.2, .2, .7, 1.1])
        prior = public_view('old', x=0., y=0.)
        target = public_view('new-yaw', x=0., y=0., yaw=np.pi/6)
        k = prior.intrinsic.copy(); k[0, 0] += 1.
        changed = [CandidateViewV40(k, prior.world_from_camera, 96, 72),
                   CandidateViewV40(prior.intrinsic, prior.world_from_camera, 95, 72)]
        before = old._areas(state, support, target)[0]['new_surface_area_m2']
        for view in changed:
            new._paid_cameras = [history(view)]
            row = new._areas(state, support, target)[0]
            self.assertEqual(row['same_center_distinct_paid_views'], 0)
            self.assertEqual(row['new_surface_area_m2'], before)

    def test_center_tolerance_is_absolute_and_does_not_round_translations(self):
        _, new, state, support = analytic_box_predictors([2., 2.2, -.2, .2, .7, 1.1])
        prior = public_view('old', x=0., y=0.)
        new._paid_cameras = [history(prior)]
        for x, expected in [(0.5e-9, 1), (2e-9, 0), (.25, 0)]:
            view = public_view('candidate', x=x, y=0., yaw=np.pi/6)
            self.assertEqual(len(new._same_center_history(view)), expected)

    def test_nonempty_history_preserves_measurement_and_area_partition(self):
        old, new, state, support = analytic_box_predictors([2., 2.2, -.2, .2, .7, 1.1])
        prior = public_view('old', x=0., y=0.)
        new._paid_cameras = [history(prior)]
        original_state, original_support = deepcopy(state), support.copy()
        original_history = deepcopy(new._paid_cameras)
        target = public_view('new-yaw', x=0., y=0., yaw=np.pi/6)
        before, after = old._areas(state, support, target)[0], new._areas(state, support, target)[0]
        self.assertEqual(after['visible_area_m2'], before['visible_area_m2'])
        self.assertEqual(after['already_observed_area_m2'], before['already_observed_area_m2'])
        self.assertAlmostEqual(after['new_surface_area_before_exposure_m2'],
            after['new_surface_area_m2']+after['excluded_previously_exposed_unseen_area_m2'])
        self.assertAlmostEqual(after['visible_area_m2'], after['already_observed_area_m2']
            +after['new_surface_area_m2']+after['excluded_previously_exposed_unseen_area_m2'])
        self.assertEqual(state, original_state)
        np.testing.assert_array_equal(support, original_support)
        self.assertEqual(new._paid_cameras, original_history)

    def test_prototype_self_occlusion_does_not_mark_hidden_rear_wall_exposed(self):
        boxes = np.asarray([[2., 2.2, -.2, .2, .7, 1.1],
                            [2.5, 2.7, -.1, .1, .8, 1.]])
        points = np.asarray([[2., 0., .9], [2.5, 0., .9]])
        normals = np.asarray([[-1., 0., 0.], [-1., 0., 0.]])
        v = public_view('front', x=0., y=0.)
        got = nominal_visible_mask(points, normals, boxes, np.eye(3), np.zeros(3),
            v.intrinsic, v.world_from_camera, v.width, v.height)
        np.testing.assert_array_equal(got, [True, False])

    def test_observation_and_uncertain_plane_fallback_are_unchanged(self):
        old = ArticleViewPredictorV1(residual=ArticleObservedResidualV1())
        new = ArticleExposureViewPredictorV3(residual=ArticleObservedResidualV1())
        shared = ledger(mode='G')
        packets = analytic_forward_sequence_v42()
        hashes = [p.sha256() for p in packets]
        for p in packets:
            associations = shared.observe(p)['accepted']
            self.assertEqual(old.observe(p, associations), new.observe(p, associations))
        self.assertEqual(old.observed_plane_receipts(), new.observed_plane_receipts())
        for key in old._instances:
            np.testing.assert_array_equal(list(old._instances[key]['support'].values()),
                                          list(new._instances[key]['support'].values()))
        snapshot = shared.snapshot()['instances'][0]
        ambiguous = deepcopy(snapshot); ambiguous['association_uncertain'] = True
        view = public_view('unpaid', y=1.25)
        out = new.forecast(ambiguous, [view])
        self.assertEqual(out['candidates'][0]['expected_new_surface_area_m2'], 0.)
        self.assertEqual(out['candidates'][0]['fallback_reason'], 'association_uncertain')
        new._instances[snapshot['instance_id']]['plane'] = None
        out = new.forecast(snapshot, [view])
        self.assertEqual(out['candidates'][0]['fallback_reason'], 'no_reliable_observed_label_plane')
        self.assertFalse(out['actual_depth_and_tsdf_modified'])
        self.assertEqual([p.sha256() for p in packets], hashes)


if __name__ == '__main__':
    unittest.main()
