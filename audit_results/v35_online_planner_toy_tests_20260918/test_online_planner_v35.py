"""Independent tiny-tree oracle for V35's declared one-observation lookahead.

These tests never create a world, query a sensor, instantiate a mapper or
evaluate a reconstruction. The oracle enumerates the policy tree directly;
it does not call the production planner's recursion or Bayesian helper.
"""
import unittest

from nso.online_planner_v35 import ForecastBeliefPlannerV35


NEG = -1e30


class ToyModel:
    def __init__(self, edges, observed_masks, rewards, *, anchor=0,
                 required_mask=0):
        self.poses = tuple((index, 0, 0) for index in range(len(edges)))
        self.edges = tuple(tuple(row) for row in edges)
        self.anchor = anchor
        self.observed_masks = tuple(observed_masks)
        self.rewards = dict(rewards)
        self.required_mask = required_mask

    def terminal(self, mask):
        return dict(joint=float(self.rewards.get(mask, 0.)),
                    feasible=(mask & self.required_mask) == self.required_mask)


def exhaustive_value(models, informative_nodes, node, remaining, masks,
                     probability0, reliability, information_available=True,
                     excluded=()):
    """Enumerate every action and both soft observations, without memoization."""
    terms = [model.terminal(mask) for model, mask in zip(models, masks)]
    stop = NEG
    if node == models[0].anchor and all(term['feasible'] for term in terms):
        stop = probability0 * terms[0]['joint'] + (1-probability0) * terms[1]['joint']
    best = stop
    if remaining:
        for _, following in models[0].edges[node]:
            next_masks = tuple(mask | model.observed_masks[following]
                               for model, mask in zip(models, masks))
            if (information_available and following in informative_nodes
                    and following not in excluded):
                branches = []
                for likelihood0, likelihood1 in ((reliability, 1-reliability),
                                                  (1-reliability, reliability)):
                    mass = probability0*likelihood0 + (1-probability0)*likelihood1
                    if mass == 0:
                        continue
                    posterior0 = probability0*likelihood0/mass
                    branch = exhaustive_value(models, informative_nodes, following,
                        remaining-1, next_masks, posterior0, reliability, False, excluded)
                    branches.append((mass, branch))
                # All positive-probability branches must satisfy the same
                # return and both-template coverage constraint.
                candidate = (sum(mass*value for mass, value in branches)
                             if all(value > NEG/2 for _, value in branches) else NEG)
            else:
                candidate = exhaustive_value(models, informative_nodes, following,
                    remaining-1, next_masks, probability0, reliability,
                    information_available, excluded)
            best = max(best, candidate)
    return best


def probe_models():
    # Three paid actions: reach a diagnostic pose, choose an asset side, return.
    # Reaching the diagnostic pose gives no surface reward by itself.
    edges = ((('forward', 1),),
             (('left', 2), ('right', 3)),
             (('forward', 0),), (('forward', 0),))
    masks = (0, 0, 1, 2)
    return (ToyModel(edges, masks, {0: 0., 1: 1., 2: .2, 3: 1.}),
            ToyModel(edges, masks, {0: 0., 1: .2, 2: 1., 3: 1.}))


class OnlinePlannerV35Tests(unittest.TestCase):
    def compare(self, models, informative, node, remaining, masks, probability,
                reliability=.99, feedback=True, excluded=()):
        planner = ForecastBeliefPlannerV35(models, informative,
            reliability=reliability, maximum_states=100000)
        result = planner.select(node, remaining, masks, probability,
            geometry_feedback=feedback, excluded_information_nodes=excluded)
        expected = exhaustive_value(models, set(informative), node, remaining,
            masks, probability, reliability, feedback, set(excluded))
        self.assertGreater(expected, NEG/2)
        self.assertAlmostEqual(result['expected_proxy'], expected, places=11)
        return result

    def test_soft_information_matches_explicit_policy_tree(self):
        models = probe_models()
        # The reliability, prior and budget grids exercise weighted branching,
        # including uninformative observations and degenerate model priors.
        for remaining in (2, 3, 4):
            for reliability in (.5, .8, .99):
                for probability in (0., .2, .5, .8, 1.):
                    with self.subTest(remaining=remaining, r=reliability, p=probability):
                        self.compare(models, {1}, 0, remaining, (0, 0),
                                     probability, reliability)

    def test_forecast_is_soft_not_perfect_truth(self):
        result = self.compare(probe_models(), {1}, 0, 3, (0, 0), .5)
        self.assertAlmostEqual(result['expected_proxy'], .992, places=12)
        self.assertLess(result['expected_proxy'], 1.)
        self.assertEqual(result['action'], 'forward')

    def test_no_feedback_does_not_branch(self):
        models = probe_models()
        result = self.compare(models, {1}, 0, 3, (0, 0), .5, feedback=False)
        self.assertAlmostEqual(result['expected_proxy'], .6, places=12)
        no_information = self.compare(models, set(), 0, 3, (0, 0), .5)
        self.assertEqual(result['action'], no_information['action'])
        self.assertEqual(result['expected_proxy'], no_information['expected_proxy'])

    def test_lookahead_forecasts_only_one_observation(self):
        # Two consecutive diagnostic poses precede the side choice. The
        # declared approximation may forecast one observation, not both.
        edges = ((('forward', 1),), (('forward', 2),),
                 (('left', 3), ('right', 4)),
                 (('forward', 0),), (('forward', 0),))
        masks = (0, 0, 0, 1, 2)
        models = (ToyModel(edges, masks, {0: 0., 1: 1., 2: .2, 3: 1.}),
                  ToyModel(edges, masks, {0: 0., 1: .2, 2: 1., 3: 1.}))
        result = self.compare(models, {1, 2}, 0, 4, (0, 0), .6, reliability=.8)
        self.assertAlmostEqual(result['expected_proxy'], .84, places=12)

    def test_paid_diagnostic_pose_not_forecast_again(self):
        result = self.compare(probe_models(), {1}, 0, 3, (0, 0), .5, excluded={1})
        self.assertAlmostEqual(result['expected_proxy'], .6, places=12)

    def test_current_pose_does_not_give_free_information(self):
        result = self.compare(probe_models(), {1}, 1, 2, (0, 0), .5)
        self.assertAlmostEqual(result['expected_proxy'], .6, places=12)

    def test_actual_posterior_changes_exploitation_action(self):
        models = probe_models()
        left = self.compare(models, {1}, 1, 2, (0, 0), .9)
        right = self.compare(models, {1}, 1, 2, (0, 0), .1)
        self.assertEqual(left['action'], 'left')
        self.assertEqual(right['action'], 'right')

    def test_no_class_same_public_state_same_decision(self):
        models = probe_models()
        geometric = self.compare(models, {1}, 0, 3, (0, 0), .5)
        unknown_class = self.compare(models, {1}, 0, 3, (0, 0), .5)
        self.assertEqual(geometric['action'], unknown_class['action'])
        self.assertEqual(geometric['expected_proxy'], unknown_class['expected_proxy'])

    def test_outbound_reward_cannot_waive_return_budget(self):
        # The attractive side requires two paid actions including return.
        edges = ((('left', 1),), (('right', 0),))
        models = tuple(ToyModel(edges, (0, 1), {0: .1, 1: 1.}) for _ in range(2))
        one = self.compare(models, set(), 0, 1, (0, 0), .5)
        two = self.compare(models, set(), 0, 2, (0, 0), .5)
        self.assertIsNone(one['action'])
        self.assertAlmostEqual(one['expected_proxy'], .1)
        self.assertEqual(two['action'], 'left')
        self.assertAlmostEqual(two['expected_proxy'], 1.)

    def test_both_hypotheses_must_pass_coverage_even_if_one_prior_zero(self):
        # Taking either side alone is attractive but does not satisfy both
        # hypothetical coverage constraints. Four actions visit both sides.
        edges = ((('left', 1), ('right', 2)),
                 (('forward', 0),), (('forward', 0),))
        models = (ToyModel(edges, (0, 1, 2), {0: .1, 1: 1., 2: .5, 3: .8}, required_mask=1),
                  ToyModel(edges, (0, 1, 2), {0: .1, 1: .5, 2: 1., 3: .8}, required_mask=2))
        result = self.compare(models, set(), 0, 4, (0, 0), 1.)
        self.assertAlmostEqual(result['expected_proxy'], .8)

    def test_repeated_pose_exposure_has_no_new_proxy_reward(self):
        edges = ((('left', 1),), (('right', 0),))
        models = tuple(ToyModel(edges, (0, 1), {0: 0., 1: .7}) for _ in range(2))
        two = self.compare(models, set(), 0, 2, (0, 0), .5)
        four = self.compare(models, set(), 0, 4, (0, 0), .5)
        self.assertAlmostEqual(two['expected_proxy'], .7)
        self.assertEqual(two['expected_proxy'], four['expected_proxy'])


if __name__ == '__main__':
    unittest.main()
