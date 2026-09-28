"""Pure inference/decision tests; no World, renderer, mapper or evaluation."""
import json
import unittest

import numpy as np

from nso.semantic_reliability import SemanticReliabilityBelief, discrete_observation_evi


def model(shared=True):
    return SemanticReliabilityBelief(geometry_prior=[.5, .5],
        class_structure_priors={'cabinet': [.9, .1], 'rack': [.2, .8]},
        share_across_instances=shared)


class SemanticReliabilityTests(unittest.TestCase):
    def test_single_instance_is_fixed_mixture_bayes(self):
        shared, fixed = model(), model(False)
        evidence = np.log([.15, .85])
        expected = np.array([.7, .3]) * np.exp(evidence)
        expected /= expected.sum()
        for belief in (shared, fixed):
            belief.register('a', 'cabinet')
            belief.replace_log_evidence('a', evidence)
            receipt = belief.posterior('a')
            np.testing.assert_allclose(receipt['structure_probabilities'], expected)
            self.assertEqual(receipt['reliability_weights_leave_one_out'], [.5, .5])
        self.assertEqual(shared.posterior('a'), fixed.posterior('a'))

    def test_peer_geometry_transfers_without_self_double_count(self):
        belief = model()
        belief.register('a', 'cabinet')
        belief.register('b', 'cabinet')
        belief.replace_log_evidence('a', np.log([.01, .99]))
        # A's own geometry cannot attenuate A's prior a second time.
        self.assertEqual(belief.posterior('a')['active_structure_prior'], [.7, .3])
        peer = belief.posterior('b')
        rho = .108 / (.5 + .108)
        self.assertAlmostEqual(peer['rho'], rho)
        self.assertAlmostEqual(peer['structure_probabilities'][0], .5 + .4*rho)
        self.assertLess(peer['structure_probabilities'][0], .7)
        self.assertFalse(peer['score_is_calibrated'])

    def test_replacement_idempotence_and_constant_offset_invariance(self):
        belief = model()
        for key in ('a', 'b'):
            belief.register(key, 'cabinet')
        belief.replace_log_evidence('a', [-4., 0.])
        first = belief.snapshot()
        belief.replace_log_evidence('a', [-4., 0.])
        self.assertEqual(first, belief.snapshot())
        belief.replace_log_evidence('a', [96., 100.])
        self.assertEqual(first, belief.snapshot())
        belief.replace_log_evidence('a', [0., 0.])
        self.assertEqual(belief.posterior('b')['rho'], .5)

    def test_class_isolation_unknown_class_and_withdrawn_qualification(self):
        belief = model()
        belief.register('a', 'cabinet')
        belief.register('b', 'rack')
        belief.register('c', None)
        baseline = belief.posterior('b')
        belief.replace_log_evidence('a', [-10., 0.])
        self.assertEqual(baseline, belief.posterior('b'))
        self.assertEqual(belief.posterior('c')['structure_probabilities'], [.5, .5])
        belief.set_class('b', 'cabinet')
        self.assertLess(belief.posterior('b')['rho'], .5)
        belief.set_class('a', None)
        self.assertEqual(belief.posterior('b')['rho'], .5)

    def test_no_sharing_is_independent_of_all_peer_evidence(self):
        belief = model(False)
        for key in ('a', 'b'):
            belief.register(key, 'cabinet')
        baseline = belief.posterior('b')
        belief.replace_log_evidence('a', [-20., 0.])
        self.assertEqual(baseline, belief.posterior('b'))

    def test_instance_registration_order_is_irrelevant(self):
        a, b = model(), model()
        for belief, keys in ((a, ('one', 'two', 'three')), (b, ('three', 'two', 'one'))):
            for index, key in enumerate(keys):
                belief.register(key, 'cabinet')
            belief.replace_log_evidence('one', [-3., 0.])
            belief.replace_log_evidence('two', [0., -2.])
        self.assertEqual(a.snapshot(), b.snapshot())

    def test_joint_enumeration_verifies_leave_one_out_and_branch_tower_law(self):
        belief = model()
        for key in ('a', 'b'):
            belief.register(key, 'cabinet')
        belief.replace_log_evidence('a', [-1., 0.])
        belief.replace_log_evidence('b', [0., -.6])
        priors = np.array([[.5, .5], [.9, .1]])
        joint = sum(.5 * np.outer(prior*np.exp([-1., 0.]), prior*np.exp([0., -.6]))
                    for prior in priors)
        joint /= joint.sum()
        np.testing.assert_allclose(belief.posterior('a')['structure_probabilities'], joint.sum(axis=1))
        np.testing.assert_allclose(belief.posterior('b')['structure_probabilities'], joint.sum(axis=0))
        channel = np.array([[.85, .15], [.2, .8]])
        weighted_peer = np.zeros(2)
        outcome_probabilities = joint.sum(axis=1) @ channel
        for outcome, probability in enumerate(outcome_probabilities):
            branch = belief.observation_branch('a', channel[:, outcome])
            expected = (joint*channel[:, outcome, None]).sum(axis=0)/probability
            actual = branch.posterior('b')['structure_probabilities']
            np.testing.assert_allclose(actual, expected)
            weighted_peer += probability*np.asarray(actual)
        np.testing.assert_allclose(weighted_peer, joint.sum(axis=0))

    def test_branch_isolation_zero_probabilities_and_json_receipts(self):
        belief = model()
        belief.register('a', 'cabinet')
        prior = belief.snapshot()
        branch = belief.observation_branch('a', [1., 0.])
        self.assertEqual(branch.posterior('a')['structure_probabilities'], [1., 0.])
        self.assertEqual(prior, belief.snapshot())
        json.dumps(branch.snapshot(), allow_nan=False)
        with self.assertRaises(ValueError):
            branch.observation_branch('a', [0., 1.])

    def test_four_structures_three_instances_match_joint_enumeration(self):
        geometry = np.array([.4, .3, .2, .1])
        semantic = np.array([.1, .1, .2, .6])
        belief = SemanticReliabilityBelief(geometry_prior=geometry,
                                          class_structure_priors={'machine': semantic})
        likelihoods = np.array([[.1, .5, .7, .2], [.8, .2, .1, .4], [.3, .4, .6, .2]])
        for index, key in enumerate(('a', 'b', 'c')):
            belief.register(key, 'machine')
            belief.replace_log_evidence(key, np.log(likelihoods[index]))
        joint = np.zeros((4, 4, 4))
        for prior in (geometry, semantic):
            factors = prior[None, :] * likelihoods
            joint += .5 * (factors[0, :, None, None] * factors[1, None, :, None]
                           * factors[2, None, None, :])
        joint /= joint.sum()
        for index, key in enumerate(('a', 'b', 'c')):
            other_axes = tuple(axis for axis in range(3) if axis != index)
            np.testing.assert_allclose(belief.posterior(key)['structure_probabilities'],
                                       joint.sum(axis=other_axes))

    def test_uninformative_channel_has_zero_evi(self):
        belief = model()
        for key in ('a', 'b'):
            belief.register(key, 'cabinet')
        receipt = discrete_observation_evi(belief, 'a', [[.2, .8], [.2, .8]],
            {'a': [[2., 0.], [0., 2.]], 'b': [[.5, 1.], [1., .5]]})
        self.assertEqual(receipt['evi'], 0.)

    def test_shared_diagnosis_can_change_another_instance_action(self):
        # A itself has zero terminal value. Information is useful only because
        # it changes the preferred inspection action at same-class instance B.
        receipts = []
        for sharing in (False, True):
            belief = model(sharing)
            for key in ('a', 'b'):
                belief.register(key, 'cabinet')
            receipts.append(discrete_observation_evi(belief, 'a', [[1., 0.], [0., 1.]],
                {'b': [[1., 0.], [.65, .65]]}))
        self.assertEqual(receipts[0]['evi'], 0.)
        self.assertGreater(receipts[1]['evi'], 0.)
        self.assertEqual({row['best_action'] for row in receipts[1]['branches']}, {0, 1})

    def test_actions_are_joint_not_separately_maximized(self):
        belief = model(False)
        belief.register('a')
        belief.register('b')
        receipt = discrete_observation_evi(belief, 'a', [[1.], [1.]],
            {'a': [[1., 1.], [0., 0.]], 'b': [[0., 0.], [1., 1.]]})
        self.assertEqual(receipt['best_utility_before_observation'], 1.)
        self.assertEqual(receipt['evi'], 0.)

    def test_local_geometry_evi_available_in_all_modes(self):
        for sharing in (False, True):
            belief = model(sharing)
            belief.register('a')
            receipt = discrete_observation_evi(belief, 'a', [[1., 0.], [0., 1.]],
                {'a': np.eye(2)})
            self.assertAlmostEqual(receipt['evi'], .5)

    def test_invalid_or_undiscovered_inputs_rejected(self):
        belief = model()
        belief.register('a')
        with self.assertRaises(ValueError):
            belief.replace_log_evidence('a', [0., float('nan')])
        with self.assertRaises(ValueError):
            discrete_observation_evi(belief, 'a', [[.5, .6], [.5, .5]], {'a': np.eye(2)})
        with self.assertRaises(ValueError):
            discrete_observation_evi(belief, 'a', np.eye(2), {'future_unseen': np.eye(2)})
        with self.assertRaises(ValueError):
            belief.register('b', 'future_unseen_class')


if __name__ == '__main__':
    unittest.main()
