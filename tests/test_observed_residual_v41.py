"""Analytic observed-pixel fixtures only: no World, rollout or reconstruction."""
from copy import deepcopy
import unittest

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_v41 import ObservedInstancesV41, support_digest
from nso.observed_residual_v41 import (ObservedResidualV41, build_prototype_bank_v41,
    fit_marker_plane_v41, observed_points_v41, pixel_mask_sha256_v41, _ray_box_depths)


COLOR = (40, 100, 220)
NAMES = ("planar", "recessed", "louvered", "open_frame")


def fixture(step=0, *, kind="planar", marker=True, origin_x=-.41):
    """Independent closed-form planes at y=-.42,-.4 or .32, all measured.

    The interior-only recessed patch is a residual unit fixture, not a claim
    that the association frontend can bridge its depth discontinuity.
    """
    shape = (72, 96)
    k = np.array([[48., 0., 47.5], [0., 48., 35.5], [0., 0., 1.]])
    transform = np.eye(4)
    transform[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    transform[:3, 3] = [origin_x, -1.5, .8]
    yy, xx = np.indices(shape)
    ux, uz = (xx-47.5)/48., -(yy-35.5)/48.
    depth = np.zeros(shape, np.float64)
    rgb = np.full(shape+(3,), 165, np.uint8)
    d = 1.1 if kind == "planar" else 1.82
    xworld, zworld = origin_x+d*ux, .8+d*uz
    support = (xworld >= (-.6 if kind == "planar" else -.15)) & (xworld <= .45) & (zworld >= .2) & (zworld <= 1.4)
    depth[support] = d
    marker_mask = ((origin_x+1.08*ux >= -.57) & (origin_x+1.08*ux <= -.25)
                   & (.8+1.08*uz >= .65) & (.8+1.08*uz <= .95))
    if marker:
        rgb[marker_mask] = COLOR
        depth[marker_mask] = 1.08
    else:
        marker_mask[:] = False
    observation = PaidRGBDObservationV40(f"analytic-{step}-{kind}", step, rgb, depth, k, transform)
    return observation, np.flatnonzero(depth), np.flatnonzero(marker_mask)


def association(observation, pixels, marker_pixels=(), instance="fixture_instance"):
    points = observed_points_v41(observation, np.asarray(pixels, np.int64))
    return dict(instance_id=instance, frame_id=observation.frame_id, paid_step=observation.paid_step,
        observation_sha256=observation.sha256(), pixel_indices=np.asarray(pixels).tolist(),
        marker_pixel_indices=np.asarray(marker_pixels).tolist(), points_world_m=points.tolist(),
        support_sha256=support_digest(points), geometry_feedback_eligible=True)


class ObservedResidualTests(unittest.TestCase):
    def test_shared_bank_is_fixed_and_hashes_generic_source(self):
        a, b = build_prototype_bank_v41(), build_prototype_bank_v41()
        self.assertEqual(a.sha256, b.sha256)
        self.assertEqual(len(a.candidates), 12)
        self.assertFalse(a.receipt()["actual_scene_parameters_used"])
        self.assertEqual(a.receipt()["nominal_dimensions_m"], [1.2, .8, 1.6])
        self.assertEqual(len(a.receipt()["source_function_sha256"]), 64)
        with self.assertRaises(ValueError):
            a.candidates[0][2][0, 0] = 999.

    def test_plane_uses_measured_label_surface_not_object_centre(self):
        obs, _, marker = fixture()
        fitted = fit_marker_plane_v41(obs, marker)
        self.assertTrue(fitted["accepted"], fitted)
        np.testing.assert_allclose(fitted["anchor_world_m"], [-.41, -.42, .8], atol=.012)
        np.testing.assert_allclose(fitted["outward_normal_world"], [0, -1, 0], atol=1e-12)
        self.assertFalse(fitted["object_centre_known"])
        self.assertGreater(fitted["anchor_uncertainty_m"], 0.)
        self.assertEqual(fitted["observation_sha256"], obs.sha256())

    def test_partial_marker_is_rejected(self):
        obs, _, marker = fixture()
        marker = marker[marker % 96 <= 46]
        fitted = fit_marker_plane_v41(obs, marker)
        self.assertFalse(fitted["accepted"])
        self.assertEqual(fitted["reason"], "marker_extent_incomplete_or_wrong_mount")

    def test_horizontal_mount_is_rejected(self):
        obs, _, marker = fixture()
        transform = np.eye(4)
        changed = PaidRGBDObservationV40("horizontal", 0, obs.rgb, obs.depth_m, obs.intrinsic, transform)
        fitted = fit_marker_plane_v41(changed, marker)
        self.assertFalse(fitted["accepted"])
        self.assertEqual(fitted["reason"], "marker_not_upright")

    def test_nonplanar_marker_is_rejected(self):
        obs, _, marker = fixture()
        depth = obs.depth_m.copy()
        depth.ravel()[marker[::2]] += .12
        changed = PaidRGBDObservationV40("nonplanar", 0, obs.rgb, depth, obs.intrinsic, obs.world_from_camera)
        self.assertFalse(fit_marker_plane_v41(changed, marker)["accepted"])

    def test_current_front_plane_residual_prefers_planar(self):
        obs, pixels, marker = fixture()
        result = ObservedResidualV41().observe(obs, [association(obs, pixels, marker)])["results"][0]
        self.assertTrue(result["accepted"], result)
        self.assertEqual(np.argmin(result["losses_m"]), 0)
        self.assertLess(result["losses_m"][0], .012)
        self.assertGreater(result["losses_m"][1], .10)
        self.assertEqual(result["log_likelihoods"][0], 0.)
        self.assertEqual(len(set(result["valid_sample_counts"])), 1)
        self.assertFalse(result["calibrated_likelihood"])

    def test_actual_changed_depth_changes_structure_evidence(self):
        caller = ObservedResidualV41()
        obs, pixels, marker = fixture()
        first = caller.observe(obs, [association(obs, pixels, marker)])["results"][0]
        obs2, pixels2, _ = fixture(1, kind="recessed", marker=False)
        second = caller.observe(obs2, [association(obs2, pixels2)])["results"][0]
        self.assertTrue(second["accepted"], second)
        self.assertEqual(np.argmin(second["losses_m"]), 1)
        self.assertLess(second["losses_m"][1], .012)
        self.assertGreater(second["losses_m"][0], .2)
        self.assertNotEqual(first["log_likelihoods"], second["log_likelihoods"])
        self.assertEqual(second["plane_fit"]["observation_sha256"], obs.sha256())
        self.assertEqual(second["observation_sha256"], obs2.sha256())

    def test_marker_free_without_observed_anchor_is_rejected(self):
        obs, pixels, _ = fixture(marker=False)
        result = ObservedResidualV41().observe(obs, [association(obs, pixels)])["results"][0]
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "no_reliable_observed_label_plane")

    def test_missing_depth_is_ignored_not_negative_surface_evidence(self):
        obs, pixels, marker = fixture()
        support = pixels[~np.isin(pixels, marker)]
        for selected in (support[:100], support[:500]):
            caller = ObservedResidualV41()
            caller.observe(obs, [association(obs, pixels, marker)])
            depth = np.zeros_like(obs.depth_m)
            changed = PaidRGBDObservationV40("all-missing", 1, obs.rgb, depth, obs.intrinsic, obs.world_from_camera)
            result = caller.observe(changed, [association(changed, selected)])["results"][0]
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "insufficient_current_valid_depth")
            self.assertNotIn("log_likelihoods", result)

    def test_missing_pixels_do_not_dilute_measured_residual(self):
        obs, pixels, marker = fixture()
        caller = ObservedResidualV41(maximum_sample_pixels=6912)
        caller.observe(obs, [association(obs, pixels, marker)])
        obs2, px2, _ = fixture(1, kind="recessed", marker=False)
        zeros = np.flatnonzero(obs2.depth_m == 0)[:300]
        with_missing = np.unique(np.r_[px2, zeros])
        got = caller.observe(obs2, [association(obs2, with_missing)])["results"][0]
        self.assertEqual(got["missing_or_out_of_range_ignored_count"], 300)
        self.assertEqual(got["valid_sample_counts"], [len(px2)]*4)
        self.assertLess(got["losses_m"][1], .012)

    def test_packet_point_support_and_frame_hashes_are_enforced(self):
        obs, pixels, marker = fixture()
        original = association(obs, pixels, marker)
        for mutation in ("packet", "points", "support", "frame", "private"):
            changed = deepcopy(original)
            if mutation == "packet": changed["observation_sha256"] = "0"*64
            if mutation == "points": changed["points_world_m"][0][0] += .2
            if mutation == "support": changed["support_sha256"] = "0"*64
            if mutation == "frame": changed["frame_id"] = "other"
            if mutation == "private": changed["world_aabb_m"] = [[0]*3, [1]*3]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                ObservedResidualV41().observe(obs, [changed])

    def test_marker_mask_cannot_escape_current_association(self):
        obs, pixels, marker = fixture()
        changed = association(obs, pixels[~np.isin(pixels, marker)], marker)
        with self.assertRaises(ValueError): ObservedResidualV41().observe(obs, [changed])

    def test_repeated_packet_is_rejected(self):
        obs, pixels, marker = fixture()
        caller = ObservedResidualV41()
        caller.observe(obs, [association(obs, pixels, marker)])
        with self.assertRaises(ValueError): caller.observe(obs, [association(obs, pixels, marker)])

    def test_ray_depth_is_axial_not_radial_at_fov_edge(self):
        boxes = np.array([[-10, 10, -10, 10, 3.8, 3.9]], dtype=float)
        depth = _ray_box_depths(np.zeros(3), np.array([[.8, .5, 1.], [0., 0., 1.]]), boxes)
        np.testing.assert_allclose(depth, [3.8, 3.8])
        self.assertGreater(3.8*np.linalg.norm([.8, .5, 1.]), 4.)

    def test_ray_parallel_inside_outside_and_near_clip(self):
        box = np.array([[-1, 1, -1, 1, -.1, .5]])
        depth = _ray_box_depths(np.zeros(3), np.array([[0., 0., 1.], [0., 0., -1.]]), box)
        np.testing.assert_allclose(depth, [.5, .1])
        depth = _ray_box_depths(np.array([2., 0., 0.]), np.array([[0., 0., 1.]]), box)
        self.assertTrue(np.isinf(depth[0]))

    def test_near_clipping_inside_overlapping_union_never_exposes_internal_face(self):
        boxes = np.array([[-1, 1, -1, 1, 0., .5], [-1, 1, -1, 1, .4, 2.]])
        depth = _ray_box_depths(np.zeros(3), np.array([[0., 0., 1.]]), boxes)
        np.testing.assert_allclose(depth, [2.])
        np.testing.assert_allclose(_ray_box_depths(np.zeros(3), np.array([[0., 0., 1.]]), boxes[::-1]), [2.])

    def test_masks_hash_shape_and_actual_pixel_membership(self):
        self.assertNotEqual(pixel_mask_sha256_v41([1], (2, 2)), pixel_mask_sha256_v41([2], (2, 2)))
        self.assertNotEqual(pixel_mask_sha256_v41([1], (2, 2)), pixel_mask_sha256_v41([1], (1, 4)))

    def test_real_frontend_current_residual_feedback_integration_and_gs_equality(self):
        observed = fixture()[0]
        geometries, scores = [], []
        for mode in ("G", "S"):
            ledger = ObservedInstancesV41(palette={"cabinet": COLOR}, structure_names=NAMES,
                class_structure_prior={"cabinet": [.4, .3, .2, .1]}, mode=mode)
            associated = ledger.observe(observed)
            self.assertEqual(len(associated["accepted"]), 1, associated["rejected"])
            evidence = ObservedResidualV41().observe(observed, associated["accepted"])["results"][0]
            self.assertTrue(evidence["accepted"], evidence)
            feedback = ledger.apply_geometry_feedback(evidence["instance_id"], frame_id=evidence["frame_id"],
                observation_sha256=evidence["observation_sha256"], log_likelihoods=evidence["log_likelihoods"])
            self.assertTrue(feedback["applied"])
            geometries.append(ledger.geometry_snapshot())
            scores.append(evidence["log_likelihoods"])
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(geometries[0], geometries[1])

    def test_real_frontend_marker_free_measured_support_uses_prior_observed_plane(self):
        ledger = ObservedInstancesV41(palette={"cabinet": COLOR}, structure_names=NAMES,
            class_structure_prior={"cabinet": [.4, .3, .2, .1]})
        caller = ObservedResidualV41()
        first = fixture()[0]
        initial = caller.observe(first, ledger.observe(first)["accepted"])["results"][0]
        second = fixture(1, origin_x=-.21)[0]
        # Hide only semantic color; the measured label plate and all surfaces
        # retain their depth. This is a paid changed camera pose, not replay.
        gray = np.full_like(second.rgb, 165)
        second = PaidRGBDObservationV40(second.frame_id, second.paid_step, gray, second.depth_m,
                                       second.intrinsic, second.world_from_camera)
        association_report = ledger.observe(second)
        self.assertEqual(len(association_report["accepted"]), 1, association_report["rejected"])
        self.assertEqual(association_report["accepted"][0]["marker_pixel_indices"], [])
        evidence = caller.observe(second, association_report["accepted"])["results"][0]
        self.assertTrue(evidence["accepted"], evidence)
        self.assertEqual(evidence["instance_id"], initial["instance_id"])
        self.assertEqual(evidence["plane_fit"]["observation_sha256"], first.sha256())
        self.assertEqual(evidence["observation_sha256"], second.sha256())
        feedback = ledger.apply_geometry_feedback(evidence["instance_id"], frame_id=evidence["frame_id"],
            observation_sha256=evidence["observation_sha256"], log_likelihoods=evidence["log_likelihoods"])
        self.assertTrue(feedback["applied"])


if __name__ == "__main__":
    unittest.main()
