import unittest

import numpy as np

from nso.semantic_gain_v12 import (SetFeatureConfig, candidate_set_features,
                                   required_candidate_capacity)


def fixture(count=5):
    audit = {
        "common_proxy_per_action": .2,
        "aperture_factors": np.linspace(.1, .9, count).tolist(),
        "generic_area_proxies": np.linspace(2., 5., count).tolist(),
        "prefix_aperture_support": np.linspace(.05, .45, count).tolist(),
        "potential_proxies_before_cost": {"G": 9.},
    }
    assets = [
        {"class_vote": -1. if i % 2 else 1., "marked_points": 5 + i,
         "support_points": 100 + i, "support_m2_proxy": 1. + .3 * i}
        for i in range(count)
    ]
    return audit, assets


class SemanticGainV12Test(unittest.TestCase):
    def test_candidate_capacity_keeps_both_roles_for_every_asset(self):
        self.assertEqual(required_candidate_capacity(1), 7)
        self.assertEqual(required_candidate_capacity(12), 29)
        self.assertEqual(required_candidate_capacity(24, coverage_slots=6), 55)

    def test_supports_variable_asset_counts_with_fixed_dimension(self):
        for count in (1, 2, 5, 12, 24):
            audit, assets = fixture(count)
            semantic, geometry, metadata = candidate_set_features(
                audit, {"cost": 17, "asset_index": count - 1}, assets)
            self.assertEqual(semantic.shape, (20,))
            self.assertEqual(geometry.shape, (20,))
            self.assertEqual(metadata["asset_count"], count)

    def test_joint_asset_permutation_and_target_remap_is_exactly_invariant(self):
        audit, assets = fixture(7)
        target = 4
        semantic, geometry, _ = candidate_set_features(
            audit, {"cost": 23, "asset_index": target}, assets)
        permutation = np.asarray([5, 1, 4, 0, 6, 2, 3])
        permuted_audit = {**audit}
        for key in ("aperture_factors", "generic_area_proxies", "prefix_aperture_support"):
            permuted_audit[key] = np.asarray(audit[key])[permutation].tolist()
        remapped_target = int(np.flatnonzero(permutation == target)[0])
        permuted_assets = [assets[i] for i in permutation]
        permuted_semantic, permuted_geometry, _ = candidate_set_features(
            permuted_audit, {"cost": 23, "asset_index": remapped_target}, permuted_assets)
        np.testing.assert_array_equal(permuted_semantic, semantic)
        np.testing.assert_array_equal(permuted_geometry, geometry)

    def test_geometry_is_label_invariant_and_full_vote_swap_negates_semantic_tail(self):
        audit, assets = fixture(6)
        semantic, geometry, _ = candidate_set_features(
            audit, {"cost": 19, "asset_index": 3}, assets)
        swapped_assets = [{**asset, "class_vote": -asset["class_vote"]} for asset in assets]
        swapped, swapped_geometry, _ = candidate_set_features(
            audit, {"cost": 19, "asset_index": 3}, swapped_assets)
        np.testing.assert_array_equal(swapped_geometry, geometry)
        np.testing.assert_array_equal(swapped[:12], semantic[:12])
        np.testing.assert_allclose(swapped[12:], -semantic[12:], rtol=0, atol=1e-15)

    def test_coverage_candidate_has_no_order_dependent_target(self):
        audit, assets = fixture(5)
        semantic, geometry, _ = candidate_set_features(
            audit, {"cost": 11, "asset_index": None}, assets)
        permutation = np.asarray([4, 2, 0, 3, 1])
        permuted = {**audit, **{
            key: np.asarray(audit[key])[permutation].tolist()
            for key in ("aperture_factors", "generic_area_proxies", "prefix_aperture_support")}}
        semantic2, geometry2, _ = candidate_set_features(
            permuted, {"cost": 11, "asset_index": None}, [assets[i] for i in permutation])
        np.testing.assert_allclose(semantic2, semantic, rtol=0, atol=1e-15)
        np.testing.assert_allclose(geometry2, geometry, rtol=0, atol=1e-15)

    def test_rejects_out_of_range_count_and_target(self):
        audit, assets = fixture(2)
        with self.assertRaisesRegex(ValueError, "outside"):
            candidate_set_features(audit, {"cost": 3, "asset_index": 2}, assets)
        audit25, assets25 = fixture(25)
        with self.assertRaisesRegex(ValueError, "maximum_assets"):
            candidate_set_features(audit25, {"cost": 3, "asset_index": 0}, assets25,
                                   config=SetFeatureConfig(maximum_assets=24))


if __name__ == "__main__":
    unittest.main()
