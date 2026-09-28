import unittest

import numpy as np

from nso.semantic_gain_v11 import (ConditionalGainResidual, FrozenSemanticGainEnsemble,
                                   TinyMLP, TinyMLPConfig, candidate_features)


class SemanticGainV11Test(unittest.TestCase):
    def test_geometry_is_label_invariant_and_semantics_swap_sign(self):
        audit = {"common_proxy_per_action": .1, "aperture_factors": [.8, .2],
                 "generic_area_proxies": [4., 3.], "potential_proxies_before_cost": {"G": 5.}}
        candidate = {"cost": 10, "asset_index": 0}
        assets = [{"class_vote": -1., "marked_points": 10, "support_points": 100, "support_m2_proxy": 2.},
                  {"class_vote": 1., "marked_points": 10, "support_points": 100, "support_m2_proxy": 3.}]
        semantic, geometry, _ = candidate_features(audit, candidate, assets)
        swapped, swapped_geometry, _ = candidate_features(audit, candidate, [
            {**assets[0], "class_vote": 1.}, {**assets[1], "class_vote": -1.}])
        np.testing.assert_array_equal(geometry, swapped_geometry)
        np.testing.assert_allclose(semantic[:8], swapped[:8])
        np.testing.assert_allclose(semantic[8:], -swapped[8:])

    def test_tiny_mlp_actually_optimizes_and_predicts(self):
        x = np.arange(72, dtype=float).reshape(6, 12) / 72
        y = np.asarray([-1., -.4, -.1, .2, .6, 1.])
        model = TinyMLP(TinyMLPConfig(epochs=200, l2=1e-4)).fit(x, y, seed=7)
        self.assertEqual(model.parameter_count, 57)
        self.assertTrue(model.diagnostics["loss_decreased"])
        prediction = model.predict(x)
        self.assertEqual(prediction.shape, (6,))
        self.assertTrue(np.isfinite(prediction).all())
        restored = TinyMLP.from_dict(model.to_dict())
        np.testing.assert_array_equal(restored.predict(x), prediction)

    def test_conditional_feedback_is_keyed_and_disabled_mode_only_records(self):
        asset = {"aabb_center": [1.01, 2.01, .8], "class_vote": -1.}
        key = ConditionalGainResidual.key(asset, 3)
        outcome = {"predicted_cells": 10, "realized_predicted_cells": 2}
        enabled = ConditionalGainResidual(feedback_enabled=True)
        event = enabled.observe(key, outcome, action_id=1)
        self.assertTrue(event["feedback_consumed_for_planning"])
        self.assertLess(enabled.posterior(key), .5)
        self.assertEqual(enabled.semantic_scale(key), 1.)
        enabled.observe(key, outcome, action_id=2)
        self.assertLess(enabled.semantic_scale(key), 1.)
        disabled = ConditionalGainResidual(feedback_enabled=False)
        event = disabled.observe(key, outcome, action_id=1)
        self.assertFalse(event["feedback_consumed_for_planning"])
        self.assertEqual(disabled.posterior(key), .5)
        self.assertEqual(disabled.snapshot()["observed_actions"], 1)


if __name__ == "__main__":
    unittest.main()
