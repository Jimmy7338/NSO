"""Independent finite-history contracts; no scene, raycast, mapper, or Q calls.

Toy observations are abstract bit evidence, not physical surface measurements.
The reference enumerator retains the complete action/observation history and
enumerates all feasible outcome vectors; it uses neither solver caches nor its
Bellman values/action reconstruction.
"""
import itertools
import unittest
from dataclasses import dataclass

from nso.finite_belief_solver_v33 import (
    NEG, ExactCoverageBeliefSolverV33, SolverLimitV33,
)


@dataclass
class ToyModel:
    poses: tuple
    edges: tuple
    signatures: tuple
    observed_masks: tuple
    coverage_weights: tuple
    surface_weights: tuple
    anchor: int = 0

    def terminal(self, mask):
        coverage = sum(w for bit, w in enumerate(self.coverage_weights)
                       if mask & (1 << bit))
        surface = sum(w for bit, w in enumerate(self.surface_weights)
                      if mask & (1 << bit))
        if coverage > 1 + 1e-12 or surface > 1 + 1e-12:
            raise AssertionError('invalid toy evidence weights')
        return dict(feasible=coverage >= .8 - 1e-12, coverage=coverage,
                    surface=surface, joint=coverage * surface)


def model(edges, coverage, surface, signatures=None, masks=None, poses=None):
    size = len(edges)
    return ToyModel(
        poses=tuple(poses or [(i, 0, 0) for i in range(size)]),
        edges=tuple(tuple(row) for row in edges),
        signatures=tuple(signatures or [('ordinary', i) for i in range(size)]),
        observed_masks=tuple(masks or [1 << i for i in range(size)]),
        coverage_weights=tuple(coverage), surface_weights=tuple(surface))


def exhaustive_history_values(models, initial_masks, budget, hypotheses=(0, 1)):
    """All qualified terminal utility vectors for observation-history policies.

    Each recursive call groups worlds by their *entire* current observation
    history. At a shared history only one common action is legal. At distinct
    histories independently enumerated continuations form a Cartesian product.
    An infeasible world is never omitted or assigned zero utility.
    """
    anchor = models[0].anchor
    initial = tuple((h, anchor, initial_masks[h],
                     (models[h].signatures[anchor],)) for h in hypotheses)

    def enumerate_policies(worlds, actions):
        groups = {}
        for state in worlds:
            groups.setdefault(state[3], []).append(state)
        if len(groups) > 1:
            options = [enumerate_policies(tuple(group), actions)
                       for group in groups.values()]
            return {tuple(sorted(itertools.chain.from_iterable(branches)))
                    for branches in itertools.product(*options)}
        outcomes = set()
        if all(node == anchor and models[h].terminal(mask)['feasible']
               for h, node, mask, _ in worlds):
            outcomes.add(tuple((h, models[h].terminal(mask)['joint'])
                               for h, _, mask, _ in worlds))
        if len(actions) == budget:
            return outcomes
        common_node = worlds[0][1]
        if any(node != common_node for _, node, _, _ in worlds):
            raise AssertionError('shared action history with inconsistent pose')
        for action, following in models[0].edges[common_node]:
            after = tuple((h, following,
                           mask | models[h].observed_masks[following],
                           history + ((action, models[h].signatures[following]),))
                          for h, _, mask, history in worlds)
            outcomes.update(enumerate_policies(after, actions + (action,)))
        return outcomes

    return enumerate_policies(initial, ())


def best_history_value(models, initial_masks, budget, hypotheses=(0, 1)):
    values = exhaustive_history_values(models, initial_masks, budget, hypotheses)
    return max((sum(value for _, value in row) / len(row) for row in values),
               default=NEG)


def two_arm(coverage=(.8, 0., .2), surface=(.1, .8, .1)):
    edges = ((('A', 1), ('B', 2)), (('home', 0),), (('home', 0),))
    a = model(edges, coverage, surface)
    b = model(edges, coverage, surface)
    return (a, b), (1, 1)


def probe_then_choose(revealing=True):
    edges = ((('A', 1), ('B', 2), ('probe', 3)),
             (('home', 0),), (('home', 0),), (('A', 1), ('B', 2)))
    signatures = [('ordinary', i) for i in range(4)]
    other = list(signatures)
    if revealing:
        other[3] = ('different', 3)
    return ((model(edges, (.8, 0., 0., 0.), (0., .9, .1, 0.), signatures),
             model(edges, (.8, 0., 0., 0.), (0., .1, .9, 0.), other)), (1, 1))


class FiniteCoverageBeliefV33Tests(unittest.TestCase):
    def test_coverage_constraint_rejects_larger_surface_option(self):
        models, initial = two_arm((.7, 0., .3), (.1, .5, .4))
        solver = ExactCoverageBeliefSolverV33(models, initial)
        for h in (0, 1):
            witness = solver.witness(h, 2)
            self.assertEqual(witness['suffix_actions'], ['B', 'home'])
            self.assertTrue(witness['terminal']['feasible'])
            self.assertEqual(witness['terminal']['coverage'], 1.)
            self.assertEqual(witness['terminal']['surface'], .5)

    def test_joint_objective_can_prefer_lower_surface(self):
        models, initial = two_arm((.8, 0., .2), (.1, .45, .44))
        solver = ExactCoverageBeliefSolverV33(models, initial)
        # A gives F=.55, C=.8, J=.44; B gives F=.54, C=1, J=.54.
        witness = solver.witness(0, 2)
        self.assertEqual(witness['suffix_actions'], ['B', 'home'])
        self.assertAlmostEqual(witness['terminal']['joint'], .54)

    def test_expectation_is_of_products_not_product_of_means(self):
        models = (model(((),), (.8,), (1.,)),
                  model(((),), (1.,), (.2,)))
        solver = ExactCoverageBeliefSolverV33(models, (1, 1))
        value = solver.uncertain(0, 0, 1, 1)
        self.assertAlmostEqual(value, .5)
        self.assertNotAlmostEqual(value, ((.8 + 1) / 2) * ((1 + .2) / 2))

    def test_stop_is_legal_at_initial_qualified_exact_anchor(self):
        models, initial = two_arm((1., 0., 0.), (1., 0., 0.))
        solver = ExactCoverageBeliefSolverV33(models, initial)
        for budget in (0, 4):
            witness = solver.witness(0, budget)
            self.assertEqual(witness['suffix_actions'], [])
            self.assertEqual(witness['suffix_paid_actions'], 0)
            self.assertEqual(witness['suffix_nodes'], [0])

    def test_probe_then_redirect_at_first_observation(self):
        models, initial = probe_then_choose()
        solver = ExactCoverageBeliefSolverV33(models, initial)
        self.assertAlmostEqual(solver.uncertain(0, 3, *initial), .72)
        for h, selected in ((0, 'A'), (1, 'B')):
            witness = solver.witness(h, 3)
            self.assertEqual(witness['suffix_actions'], ['probe', selected, 'home'])
            self.assertEqual(witness['first_information_suffix_action'], 1)
            self.assertEqual(witness['decision_sources'],
                             ['geometry_belief', 'revealed_geometry', 'revealed_geometry'])
            self.assertEqual(witness['suffix_paid_actions'], 3)
        hidden_models, _ = probe_then_choose(False)
        hidden = ExactCoverageBeliefSolverV33(hidden_models, initial)
        self.assertAlmostEqual(hidden.uncertain(0, 3, *initial), .4)

    def test_same_structure_carries_zero_information_value(self):
        models, initial = two_arm()
        solver = ExactCoverageBeliefSolverV33(models, initial)
        for budget in range(5):
            g = solver.uncertain(0, budget, *initial)
            s = sum(solver.known(h, 0, budget, initial[h]) for h in (0, 1)) / 2
            self.assertEqual(g, s)
            self.assertEqual(solver.witness(0, budget)['suffix_actions'],
                             solver.witness(1, budget)['suffix_actions'])

    def test_both_worlds_must_be_coverage_qualified(self):
        edges = ((('A', 1), ('B', 2)), (('home', 0),), (('home', 0),))
        models = (model(edges, (0., 1., 0.), (0., 1., 0.)),
                  model(edges, (0., 0., 1.), (0., 0., 1.)))
        solver = ExactCoverageBeliefSolverV33(models, (1, 1))
        self.assertEqual(solver.known(0, 0, 2, 1), 1.)
        self.assertEqual(solver.known(1, 0, 2, 1), 1.)
        self.assertEqual(solver.uncertain(0, 2, 1, 1), NEG)
        for h in (0, 1):
            with self.assertRaisesRegex(ValueError, 'no coverage-qualified'):
                solver.witness(h, 2)

    def test_same_position_wrong_heading_is_not_return(self):
        edges = ((('right', 1),), (('left', 0), ('forward', 2)),
                 (('back', 1),))
        poses = ((0, 0, 0), (0, 0, 1), (1, 0, 1))
        m = model(edges, (.8, 0., 0.), (0., 0., 1.), poses=poses)
        solver = ExactCoverageBeliefSolverV33((m, m), (1, 1))
        self.assertEqual(solver.uncertain(0, 3, 1, 1), 0.)
        witness = solver.witness(0, 4)
        self.assertEqual(witness['suffix_actions'], ['right', 'forward', 'back', 'left'])
        self.assertEqual(witness['suffix_poses'][-2], [0, 0, 1])
        self.assertEqual(witness['suffix_poses'][-1], [0, 0, 0])
        self.assertAlmostEqual(witness['terminal']['joint'], .8)

    def test_swapped_prior_is_corrected_by_observation(self):
        edges = ((('A', 1), ('B', 2)), (('home', 0),),
                 (('A', 1), ('D', 3), ('home', 0)), (('home', 0),))
        sig0 = tuple(('same', i) for i in range(4))
        sig1 = list(sig0); sig1[2] = ('revealing', 2)
        models = (model(edges, (1., 0., 0., 0.), (0., .9, .05, .05), sig0),
                  model(edges, (1., 0., 0., 0.), (0., 0., .9, .1), sig1))
        solver = ExactCoverageBeliefSolverV33(models, (1, 1))
        witness = solver.witness(0, 3, 'swapped_class_with_correction', hint=1)
        self.assertEqual(witness['suffix_actions'], ['B', 'A', 'home'])
        self.assertEqual(witness['first_information_suffix_action'], 1)
        self.assertEqual(witness['decision_sources'][0], 'class_hypothesis')
        self.assertEqual(witness['decision_sources'][1], 'revealed_geometry')
        self.assertAlmostEqual(witness['terminal']['joint'], .95)

    def test_initial_revelation_is_available_to_g_without_paid_probe(self):
        models, initial = probe_then_choose(False)
        models[1].signatures = (('revealing_initial',),) + models[1].signatures[1:]
        solver = ExactCoverageBeliefSolverV33(models, initial)
        self.assertAlmostEqual(solver.uncertain(0, 2, *initial), .72)
        for h, arm in ((0, 'A'), (1, 'B')):
            witness = solver.witness(h, 2)
            self.assertEqual(witness['suffix_actions'], [arm, 'home'])
            self.assertEqual(witness['first_information_suffix_action'], 0)
            self.assertEqual(witness['decision_sources'], ['revealed_geometry'] * 2)

    def test_information_persists_after_return_to_identical_view(self):
        # Arm trips cost3 each, probe costs2: within5 one may probe+choose,
        # but cannot visit both arms. This isolates remembered information.
        edges = ((('A', 1), ('B', 2), ('probe', 3)),
                 (('homebound', 4),), (('homebound', 4),),
                 (('home', 0),), (('home', 0),))
        sig0 = tuple(('same', i) for i in range(5))
        sig1 = list(sig0); sig1[3] = ('revealing', 3)
        models = (model(edges, (.8, 0., 0., 0., 0.), (0., .9, .1, 0., 0.), sig0),
                  model(edges, (.8, 0., 0., 0., 0.), (0., .1, .9, 0., 0.), sig1))
        initial = (1, 1)
        solver = ExactCoverageBeliefSolverV33(models, initial)
        self.assertAlmostEqual(solver.uncertain(0, 5, *initial),
                               best_history_value(models, initial, 5))
        for h, arm in ((0, 'A'), (1, 'B')):
            witness = solver.witness(h, 5)
            self.assertEqual(witness['suffix_actions'],
                             ['probe', 'home', arm, 'homebound', 'home'])
            self.assertEqual(witness['decision_sources'][2], 'revealed_geometry')
            self.assertEqual(witness['first_information_suffix_action'], 1)

    def test_coverage_history_is_part_of_state_even_without_surface_change(self):
        m = model(((),), (.7, .3), (1., 0.), masks=(1,))
        solver = ExactCoverageBeliefSolverV33((m, m), (1, 1))
        self.assertEqual(solver.uncertain(0, 0, 1, 1), NEG)
        self.assertEqual(solver.uncertain(0, 0, 3, 3), 1.)
        self.assertEqual(solver.known(0, 0, 0, 1), NEG)
        self.assertEqual(solver.known(0, 0, 0, 3), 1.)

    def test_swapped_control_coverage_failure_is_explicit_not_primary_success(self):
        models = (model(((),), (.7,), (1.,)),
                  model(((),), (.8,), (1.,)))
        solver = ExactCoverageBeliefSolverV33(models, (1, 1))
        witness = solver.witness(0, 0, 'swapped_class_with_correction', hint=1)
        self.assertFalse(witness['actual_coverage_qualified'])
        self.assertFalse(witness['terminal']['feasible'])
        self.assertTrue(witness['returned_exact_pose'])
        self.assertTrue(witness['budget_compliant'])
        with self.assertRaisesRegex(ValueError, 'no coverage-qualified'):
            solver.witness(0, 0, 'G')

    def test_exact_values_match_exhaustive_history_policy_sets(self):
        fixtures = (two_arm(), two_arm((.7, 0., .3), (.1, .5, .4)),
                    probe_then_choose(), probe_then_choose(False))
        compared = 0
        for models, initial in fixtures:
            solver = ExactCoverageBeliefSolverV33(models, initial)
            for budget in range(5):
                with self.subTest(budget=budget, signatures=models[1].signatures):
                    expected = best_history_value(models, initial, budget)
                    self.assertAlmostEqual(solver.uncertain(0, budget, *initial), expected)
                    if expected > NEG / 2:
                        witnesses = [solver.witness(h, budget) for h in (0, 1)]
                        self.assertAlmostEqual(sum(w['terminal']['joint'] for w in witnesses) / 2,
                                               expected)
                    for h in (0, 1):
                        self.assertAlmostEqual(solver.known(h, 0, budget, initial[h]),
                                               best_history_value(models, initial, budget, (h,)))
                    compared += 3
        self.assertEqual(compared, 60)

    def test_state_limit_raises_without_certified_value_and_caches_are_local(self):
        models, initial = two_arm()
        limited = ExactCoverageBeliefSolverV33(models, initial, maximum_states=1)
        with self.assertRaises(SolverLimitV33):
            limited.uncertain(0, 2, *initial)
        other = ExactCoverageBeliefSolverV33(models, initial, maximum_states=100)
        self.assertEqual(other.states, 0)
        self.assertEqual(other.known.cache_info().currsize, 0)
        self.assertGreater(other.uncertain(0, 2, *initial), NEG / 2)
        self.assertGreater(other.uncertain.cache_info().currsize, 0)
        other.clear()
        self.assertEqual(other.uncertain.cache_info().currsize, 0)
        self.assertEqual(other.known.cache_info().currsize, 0)

    def test_shared_action_graph_and_memo_limit_are_required(self):
        models, initial = two_arm()
        models[1].poses = ((0, 0, 1),) + models[1].poses[1:]
        with self.assertRaises(ValueError):
            ExactCoverageBeliefSolverV33(models, initial)
        models, initial = two_arm()
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                ExactCoverageBeliefSolverV33(models, initial, maximum_states=bad)

    def test_witness_rejects_invalid_indices_budgets_and_class_hints(self):
        models, initial = two_arm()
        solver = ExactCoverageBeliefSolverV33(models, initial)
        for actual in (-1, 2, True, .0, None):
            with self.assertRaises(ValueError):
                solver.witness(actual, 2)
        for budget in (-1, True, .5, float('inf'), None):
            with self.assertRaises(ValueError):
                solver.witness(0, budget)
        for policy in ('G', 'class_oracle'):
            with self.assertRaises(ValueError):
                solver.witness(0, 2, policy, hint=1)
        for hint in (None, 0, True, 1.0, 2):
            with self.assertRaises(ValueError):
                solver.witness(0, 2, 'swapped_class_with_correction', hint=hint)
        with self.assertRaises(ValueError):
            solver.witness(0, 2, 'unregistered_policy')


if __name__ == '__main__':
    unittest.main()
