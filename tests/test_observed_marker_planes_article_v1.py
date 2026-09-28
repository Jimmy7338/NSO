"""Paid-pixel synthetic fixtures and a read-only, sealed 26-point replay."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_v41 import support_digest
from nso.observed_marker_planes_article_v1 import ObservedMarkerPlanesArticleV1
from nso.observed_residual_v41 import fit_marker_plane_v41, observed_points_v41


def fixture(step=0, *, sparse=True, origin_x=0., anchor_x=0., noise=0., crop=False):
    shape = (72, 96)
    intrinsic = np.array([[80., 0., 47.5], [0., 80., 35.5], [0., 0., 1.]])
    transform = np.eye(4)
    transform[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    transform[:3, 3] = [origin_x, -1.5, .8]
    yy, xx = np.indices(shape)
    x, z = origin_x+1.08*(xx-47.5)/80., .8-1.08*(yy-35.5)/80.
    mask = (abs(x-anchor_x) <= .16) & (abs(z-.8) <= .15)
    if crop:
        mask &= abs(x-anchor_x) < .035
    rows, columns = np.where(mask)
    if sparse:
        ys = np.unique(rows)[np.linspace(0, len(np.unique(rows))-1, 4, dtype=int)]
        xs = np.unique(columns)[np.linspace(0, len(np.unique(columns))-1, 3, dtype=int)]
        mask &= np.isin(yy, ys) & np.isin(xx, xs)
    pixels = np.flatnonzero(mask)
    depth = np.zeros(shape)
    depth.ravel()[pixels] = 1.08+np.random.default_rng(100+step).normal(0., noise, len(pixels))
    rgb = np.full(shape+(3,), 165, np.uint8); rgb[mask] = [40, 100, 220]
    obs = PaidRGBDObservationV40('marker-'+str(step), step, rgb, depth, intrinsic, transform)
    return obs, association(obs, pixels)


def association(obs, pixels, key='instance_test'):
    points = observed_points_v41(obs, pixels)
    return dict(instance_id=key, frame_id=obs.frame_id, paid_step=obs.paid_step,
        observation_sha256=obs.sha256(), pixel_indices=np.asarray(pixels).tolist(),
        marker_pixel_indices=np.asarray(pixels).tolist(), points_world_m=points.tolist(),
        support_sha256=support_digest(points), geometry_feedback_eligible=True)


class MarkerPlaneArticleTests(unittest.TestCase):
    def test_legacy_success_retains_its_geometric_values(self):
        obs, item = fixture(sparse=False)
        expected = fit_marker_plane_v41(obs, item['marker_pixel_indices'])
        self.assertTrue(expected['accepted'], expected)
        store = ObservedMarkerPlanesArticleV1()
        store.observe(obs, [item])
        actual = store.plane('instance_test')
        for key in expected:
            self.assertEqual(actual[key], expected[key])
        self.assertEqual(actual['plane_source'], 'paid_single_frame_legacy')

    def test_sparse_distinct_paid_views_can_resolve_plane(self):
        store = ObservedMarkerPlanesArticleV1()
        for step, x in enumerate([-.2, 0., .2]):
            obs, item = fixture(step, origin_x=x, noise=.006)
            self.assertEqual(len(item['marker_pixel_indices']), 12)
            row = store.observe(obs, [item])['results'][0]
            self.assertFalse(row['single_frame_fit']['accepted'])
        plane = store.plane('instance_test')
        self.assertIsNotNone(plane, row)
        self.assertEqual(plane['plane_source'], 'paid_multiview_marker')
        self.assertFalse(plane['estimated_from_current_pixels'])
        self.assertFalse(plane['confidence_calibrated'])
        self.assertEqual(plane['frame_count'], 3)
        self.assertLessEqual(plane['anchor_uncertainty_m'], .06)
        self.assertLessEqual(plane['yaw_uncertainty_deg'], 12.)
        np.testing.assert_allclose(plane['anchor_world_m'], [0., -.42, .8], atol=.025)
        self.assertGreater(np.dot(plane['outward_normal_world'], [0., -1., 0.]), .98)
        for source in plane['source_frames']:
            for name in ('observation_sha256', 'payload_sha256', 'depth_sha256',
                         'intrinsic_sha256', 'world_from_camera_sha256', 'marker_mask_sha256'):
                self.assertEqual(len(source[name]), 64)

    def test_same_pose_with_new_noise_is_not_an_independent_view(self):
        store = ObservedMarkerPlanesArticleV1()
        for step in range(3):
            obs, item = fixture(step, noise=.003)
            row = store.observe(obs, [item])['results'][0]
        self.assertEqual(row['reason'], 'near_repeat_view_excluded')
        self.assertEqual(len(store.receipt('instance_test')['source_frames']), 1)
        self.assertIsNone(store.plane('instance_test'))

    def test_relabelled_identical_payload_is_rejected(self):
        store = ObservedMarkerPlanesArticleV1()
        obs, item = fixture()
        store.observe(obs, [item])
        copy = PaidRGBDObservationV40('renamed', 1, obs.rgb, obs.depth_m,
                                      obs.intrinsic, obs.world_from_camera)
        row = store.observe(copy, [association(copy, np.array(item['pixel_indices']))])['results'][0]
        self.assertTrue(row['duplicate_measurement'])
        self.assertFalse(row['cache_added'])
        with self.assertRaises(ValueError):
            store.observe(copy, [])

    def test_partial_label_keeps_centre_ambiguity(self):
        store = ObservedMarkerPlanesArticleV1()
        for step, x in enumerate([-.2, 0., .2]):
            obs, item = fixture(step, origin_x=x, crop=True)
            row = store.observe(obs, [item])['results'][0]
        self.assertIsNone(store.plane('instance_test'), row)
        self.assertIn(row['reason'], ('marker_parameter_uncertainty_too_large',
                                     'leave_one_view_fit_unidentifiable'))

    def test_scatter_inconsistent_with_noise_does_not_manufacture_a_plane(self):
        store = ObservedMarkerPlanesArticleV1()
        for step, x in enumerate([-.2, 0., .2]):
            obs, item = fixture(step, origin_x=x, noise=.09)
            row = store.observe(obs, [item])['results'][0]
        self.assertIsNone(store.plane('instance_test'))
        self.assertEqual(row['reason'], 'marker_inconsistent_with_declared_depth_noise')

    def test_overlapping_instance_masks_are_not_unique_associations(self):
        obs, item = fixture()
        second = deepcopy(item); second['instance_id'] = 'other'
        with self.assertRaises(ValueError):
            ObservedMarkerPlanesArticleV1().observe(obs, [item, second])

    def test_private_fields_hash_and_measured_points_are_validated(self):
        obs, item = fixture()
        for mutation in ('private', 'hash', 'points'):
            bad = deepcopy(item)
            if mutation == 'private': bad['actual_scene_pose'] = [0., 0., 0.]
            if mutation == 'hash': bad['observation_sha256'] = '0'*64
            if mutation == 'points': bad['points_world_m'][0][0] += .1
            store = ObservedMarkerPlanesArticleV1()
            with self.assertRaises(ValueError): store.observe(obs, [bad])
            self.assertTrue(store.observe(obs, [item])['results'][0]['cache_added'])

    def test_uncertain_association_never_initializes(self):
        obs, item = fixture(sparse=False); item['association_uncertain'] = True
        store = ObservedMarkerPlanesArticleV1()
        row = store.observe(obs, [item])['results'][0]
        self.assertFalse(row['cache_added'])
        self.assertIsNone(store.plane('instance_test'))

    def test_finite_point_and_frame_capacity(self):
        store = ObservedMarkerPlanesArticleV1(maximum_points=24, maximum_paid_frames=3)
        for step, x in enumerate([-.2, 0., .2]):
            obs, item = fixture(step, origin_x=x)
            row = store.observe(obs, [item])['results'][0]
        self.assertEqual(row['reason'], 'marker_cache_capacity_reached')
        self.assertIsNone(store.plane('instance_test'))
        obs, item = fixture(3, origin_x=.4)
        with self.assertRaises(ValueError): store.observe(obs, [item])

    def test_conflicting_reliable_planes_invalidate_cache(self):
        store = ObservedMarkerPlanesArticleV1()
        for step, x in enumerate([0., .24]):
            obs, item = fixture(step, sparse=False, anchor_x=x)
            row = store.observe(obs, [item])['results'][0]
        self.assertEqual(row['reason'], 'conflicting_paid_marker_planes')
        self.assertTrue(store.receipt('instance_test')['conflict'])
        self.assertIsNone(store.plane('instance_test'))

    def test_returned_receipt_cannot_mutate_internal_plane(self):
        obs, item = fixture(sparse=False)
        store = ObservedMarkerPlanesArticleV1(); store.observe(obs, [item])
        plane = store.plane('instance_test'); plane['anchor_world_m'][0] = 999.
        self.assertLess(abs(store.plane('instance_test')['anchor_world_m'][0]), .05)

    def test_sealed_p01_nominal_instance2_twenty_six_points(self):
        root = Path(__file__).resolve().parents[1]/'audit_results/semantic_development_acquisition_20260923/episodes/core_P01_nom_S_b120_lexicographic'
        if not (root/'artifact_manifest.json').exists():
            self.skipTest('sealed historical paid packets are not present in this checkout')
        manifest = json.loads((root/'artifact_manifest.json').read_text())
        store = ObservedMarkerPlanesArticleV1()
        for step in (54, 55, 56):
            packet_name, record_name = f'packets/{step:03d}_rgbd.npz', f'steps/{step:03d}.json.gz'
            for name in (packet_name, record_name):
                self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),
                                 manifest['files'][name]['sha256'])
            with np.load(root/packet_name, allow_pickle=False) as arrays:
                values = {key: arrays[key] for key in PaidRGBDObservationV40.__dataclass_fields__}
            for key in ('frame_id', 'paid_step'): values[key] = values[key].item()
            obs = PaidRGBDObservationV40.from_mapping(values)
            with gzip.open(root/record_name, 'rt') as stream:
                rows = json.load(stream)['controller_evidence']['association']['accepted']
            item = next(r for r in rows if r['instance_id'] == 'instance_0002')
            row = store.observe(obs, [item])['results'][0]
        fit = row['current_plane_fit']
        self.assertEqual(fit['valid_depth_pixels'], 26)
        self.assertEqual(fit['frame_count'], 3)
        self.assertEqual(fit['reason'], 'marker_parameter_uncertainty_too_large')
        self.assertTrue(fit['anchor_uncertainty_m'] > .06 or fit['yaw_uncertainty_deg'] > 12.)
        self.assertIsNone(store.plane('instance_0002'))


if __name__ == '__main__':
    unittest.main()
