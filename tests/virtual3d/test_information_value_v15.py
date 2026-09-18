import unittest
import numpy as np
from nso.information_value_v15 import observed_group_information


class ObservedGroupInformationTest(unittest.TestCase):
    def test_hidden_world_identity_cannot_supply_unobserved_semantics(self):
        r = observed_group_information([[2, 0], [0, 2]], ['same', 'same'])
        self.assertEqual(r['information_value'], 0.)
        self.assertEqual(r['latent_world_oracle'], 2.)
        self.assertEqual(r['observed_semantic_oracle'], 1.)

    def test_observed_classes_can_resolve_opposite_best_actions(self):
        r = observed_group_information([[2, 0], [0, 2]], ['a', 'b'])
        self.assertEqual(r['information_value'], 1.)
        self.assertEqual(r['relative_information_value'], 1.)

    def test_group_probability_tracks_number_of_worlds(self):
        r = observed_group_information([[4, 0], [0, 2], [0, 2]], ['a', 'b', 'b'])
        self.assertAlmostEqual(r['observed_semantic_oracle'], 8 / 3)
        self.assertAlmostEqual(r['information_value'], 4 / 3)

    def test_world_candidate_and_group_name_permutations_preserve_value(self):
        values = np.array([[4., 0., 1.], [0., 2., 1.], [1., 3., 0.]])
        a = observed_group_information(values, ['x', 'y', 'y'])
        b = observed_group_information(values[[2, 0, 1]][:, [1, 2, 0]], ['a', 'b', 'a'])
        for key in ('geometry_blind_oracle', 'observed_semantic_oracle', 'information_value'):
            self.assertAlmostEqual(a[key], b[key])

    def test_nonpositive_baseline_does_not_yield_relative_pass(self):
        r = observed_group_information([[-1, -2], [-2, -1]], ['a', 'b'])
        self.assertIsNone(r['relative_information_value'])
        self.assertEqual(r['information_value'], .5)

    def test_invalid_or_missing_inputs_fail_closed(self):
        for values, groups in [([], []), ([[np.nan]], ['a']), ([[1]], []),
                               ([[1]], [1]), ([[1]], ['']), ([1, 2], ['a', 'b'])]:
            with self.subTest(values=values, groups=groups), self.assertRaises(ValueError):
                observed_group_information(values, groups)


if __name__ == '__main__':
    unittest.main()
