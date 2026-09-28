"""Bounded analytic controller wiring; no development World or benchmark run."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import numpy as np

from nso.analytic_fixture_v42 import analytic_plane_observation_v42
from nso.controller_v43 import ANSControllerV43
from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_mapper_v42 import ObservedMapperV42
from nso.primitive_navigation_v41 import PrimitiveStateV41, PublicPrimitiveGraphV41


def graph():
    return PublicPrimitiveGraphV41(dict(schema_version='v41.public_navigation.v1',
        source_kind='provided_navigation_prior',
        nodes={'home': [.75, .75], 'advance': [1., .75], 'north': [1., 1.25], 'south': [1., .25]},
        edges=[['home', 'advance'], ['advance', 'north'], ['advance', 'south']]))


def controller(public=None, **kwargs):
    values = dict(home=PrimitiveStateV41('home', 0), budget=48,
        palette={'cabinet': [40, 100, 220]},
        structure_names=['planar', 'recessed', 'louvered', 'open_frame'],
        class_structure_prior={'cabinet': [.1, .7, .1, .1]})
    values.update(kwargs)
    return ANSControllerV43(graph() if public is None else public, **values)


def mapper():
    return ObservedMapperV42(shape=(40, 40), resolution_m=.1, origin_xy_m=(-.5, -.5))


def packet(step=0, pose=(.75, .75, 0.), *, measured_plane=False):
    measured = analytic_plane_observation_v42(step, pose)
    if measured_plane:
        return measured
    return PaidRGBDObservationV40('empty-v43-'+str(step), step, np.zeros_like(measured.rgb),
        np.zeros_like(measured.depth_m), measured.intrinsic, measured.world_from_camera)


def after_action(public, state, action, step, *, measured_plane=False):
    new = public.successor(state, action)
    x, y = public.positions[new.node]
    return new, packet(step, (x, y, new.heading*math.pi/6), measured_plane=measured_plane)


def two_observed_planes():
    empty = packet()
    yy, xx = np.indices(empty.depth_m.shape)
    depth = np.zeros_like(empty.depth_m)
    rgb = np.full_like(empty.rgb, 165)
    for center_y in (.3, 1.2):
        measured_y = .75-1.1*(xx-47.5)/48.
        measured_z = .9-1.1*(yy-35.5)/48.
        patch = (np.abs(measured_y-center_y) <= .19) & (measured_z >= .6) & (measured_z <= 1.2)
        depth[patch] = 1.1
        marker_y = .75-1.08*(xx-47.5)/48.
        marker_z = .9-1.08*(yy-35.5)/48.
        marker = (np.abs(marker_y-center_y) <= .16) & (marker_z >= .75) & (marker_z <= 1.05)
        depth[marker] = 1.08
        rgb[marker] = (40, 100, 220)
    return PaidRGBDObservationV40('two-separated-analytic-label-planes', 0,
        rgb, depth, empty.intrinsic, empty.world_from_camera)


class ControllerV43Tests(unittest.TestCase):
    def test_action_zero_explores_without_semantics_or_forced_prefix(self):
        c, m, obs = controller(), mapper(), packet()
        m.update(obs); accepted = c.accept(obs, m); choice = c.choose()
        self.assertEqual(accepted['association']['accepted'], [])
        self.assertEqual(choice['action'], 'forward')
        self.assertEqual(choice['forced_prefix_actions'], 0)
        self.assertTrue(any(r['discovery_unknown_area_m2'] > 0 for r in choice['candidate_utilities']))
        self.assertTrue(all(r['inspection_expected_new_area_m2'] == 0 for r in choice['candidate_utilities']))
        self.assertFalse(choice['actual_future_sensor_rendered'])
        self.assertFalse(choice['ground_truth_scene_input'])

    def test_s_and_g_share_geometry_candidates_and_structure_area_components(self):
        public = graph()
        objects = {name: controller(public, mode=name, discovery_weight=100., inspection_weight=0.) for name in ('G', 'S')}
        maps = {name: mapper() for name in objects}
        state, obs = PrimitiveStateV41('home', 0), packet(measured_plane=True)
        initial = {}
        for name, c in objects.items():
            maps[name].update(obs); c.accept(obs, maps[name]); initial[name] = c.choose()
        self.assertEqual(initial['G']['action'], initial['S']['action'])
        state, obs = after_action(public, state, initial['G']['action'], 1, measured_plane=True)
        decisions = {}
        for name, c in objects.items():
            maps[name].update(obs); c.accept(obs, maps[name]); decisions[name] = c.choose()
        self.assertEqual(objects['G'].snapshot()['geometry'], objects['S'].snapshot()['geometry'])
        self.assertEqual(decisions['G']['candidate_pool'], decisions['S']['candidate_pool'])
        self.assertEqual(decisions['G']['instance_candidate_allocations'], decisions['S']['instance_candidate_allocations'])
        self.assertTrue(objects['S'].snapshot()['observed_instances']['instances'][0]['semantic_conditioning_used'])
        self.assertFalse(objects['G'].snapshot()['observed_instances']['instances'][0]['semantic_conditioning_used'])
        self.assertNotEqual(decisions['G']['forecasts'][0]['structure_probabilities'],
                            decisions['S']['forecasts'][0]['structure_probabilities'])
        for g, s in zip(decisions['G']['forecasts'][0]['candidates'], decisions['S']['forecasts'][0]['candidates']):
            self.assertEqual(g['components'], s['components'])
            self.assertEqual(g['structure_new_surface_area_m2'], s['structure_new_surface_area_m2'])

    def test_candidate_allocation_is_bounded_and_no_class_score_enters_subset(self):
        c, m, obs = controller(), mapper(), packet(measured_plane=True)
        m.update(obs); c.accept(obs, m); choice = c.choose()
        self.assertLessEqual(len(choice['candidate_utilities']), 32)
        self.assertTrue(all(len(row['candidates']) <= 8 for row in choice['instance_candidate_allocations']))
        instance = c.snapshot()['observed_instances']['instances'][0]
        changed = deepcopy(instance)
        changed['structure_probabilities'] = [1., 0., 0., 0.]
        changed['observed_class'] = 'arbitrary_other_class'
        states = [PrimitiveStateV41(row['node'], row['heading']) for row in choice['candidate_utilities']]
        views = [c._candidate_view(s) for s in states]
        first = [s for s, _ in c._instance_candidate_order(instance, states, views)]
        second = [s for s, _ in c._instance_candidate_order(changed, states, views)]
        self.assertEqual(first, second)

    def test_cross_instance_utility_sums_only_assigned_view_predictions(self):
        c, m, obs = controller(), mapper(), two_observed_planes()
        m.update(obs); accepted = c.accept(obs, m); choice = c.choose()
        self.assertEqual(len(accepted['association']['accepted']), 2)
        self.assertEqual(len(choice['forecasts']), 2)
        expected = {f"{row['node']}:{row['heading']}": 0. for row in choice['candidate_utilities']}
        shared = {}
        for forecast in choice['forecasts']:
            self.assertLessEqual(len(forecast['candidates']), 8)
            for row in forecast['candidates']:
                expected[row['view_id']] += row['expected_new_surface_area_m2']
                shared[row['view_id']] = shared.get(row['view_id'], 0)+1
        self.assertTrue(any(count == 2 for count in shared.values()))
        for row in choice['candidate_utilities']:
            self.assertAlmostEqual(row['inspection_expected_new_area_m2'], expected[f"{row['node']}:{row['heading']}"])

    def test_same_xy_turns_cannot_claim_new_full_circle_discovery(self):
        c, m, obs = controller(), mapper(), packet()
        m.update(obs); c.accept(obs, m); choice = c.choose()
        home = [row for row in choice['candidate_utilities'] if row['node'] == 'home']
        self.assertTrue(home)
        self.assertTrue(all(r['unmasked_discovery_unknown_area_m2'] > 0 for r in home))
        self.assertTrue(all(r['discovery_unknown_area_m2'] == 0 for r in home))

    def test_no_progress_watchdog_returns_exact_home_position_and_heading(self):
        public, c, m = graph(), controller(no_progress_patience=1), mapper()
        state = PrimitiveStateV41('home', 0)
        obs = packet(); m.update(obs); c.accept(obs, m)
        actions = []
        for step in range(1, 49):
            choice = c.choose()
            if choice['action'] == 'stop':
                break
            self.assertNotEqual(choice['action'], 'blocked')
            actions.append(choice['action'])
            state, obs = after_action(public, state, choice['action'], step)
            m.update(obs); c.accept(obs, m)
            self.assertLessEqual(c.snapshot()['paid_step'], 48)
        else:
            self.fail('controller did not terminate within declared paid budget')
        self.assertEqual(actions[0], 'forward')
        self.assertEqual(state, PrimitiveStateV41('home', 0))
        self.assertTrue(c.snapshot()['return_latched'])
        self.assertTrue(c.snapshot()['terminal'])

    def test_budget_guard_stops_without_expending_an_unreturnable_action(self):
        c, m, obs = controller(budget=1), mapper(), packet()
        m.update(obs); c.accept(obs, m)
        choice = c.choose()
        self.assertEqual(choice['action'], 'stop')
        self.assertTrue(all(not row['feasible'] for row in choice['routing']['candidates']))
        self.assertEqual(c.snapshot()['paid_step'], 0)

    def test_one_choice_and_one_frame_per_transaction(self):
        c, m, obs = controller(), mapper(), packet()
        m.update(obs); c.accept(obs, m); c.choose()
        with self.assertRaises(ValueError): c.choose()
        with self.assertRaises(ValueError): c.accept(obs, m)

    def test_mapper_must_be_current_and_contain_the_same_packet_history(self):
        c, m = controller(), mapper()
        with self.assertRaises(ValueError): c.accept(packet(), m)
        first = packet(); m.update(first); c.accept(first, m); choice = c.choose()
        _, second = after_action(graph(), PrimitiveStateV41('home', 0), choice['action'], 1)
        other = mapper()
        altered = PaidRGBDObservationV40('wrong-history', 0, first.rgb, first.depth_m, first.intrinsic, first.world_from_camera)
        other.update(altered); other.update(second)
        with self.assertRaises(ValueError): c.accept(second, other)
        self.assertEqual(c.snapshot()['paid_step'], 0)

    def test_post_router_failure_poisoning_prevents_partial_history_retry(self):
        c, m, obs = controller(), mapper(), packet()
        m.update(obs)
        with patch.object(c._residual, 'observe', side_effect=RuntimeError('injected residual failure')):
            with self.assertRaises(RuntimeError): c.accept(obs, m)
        self.assertTrue(c.snapshot()['poisoned'])
        with self.assertRaises(RuntimeError): c.choose()
        with self.assertRaises(RuntimeError): c.accept(obs, m)

    def test_external_graph_and_returned_receipt_mutation_cannot_change_controller(self):
        public = graph(); c = controller(public)
        public.block_observed_edge('home', 'advance')
        m, obs = mapper(), packet(); m.update(obs)
        result = c.accept(obs, m)
        result['routing']['paid_step'] = 999
        snapshot = c.snapshot(); snapshot['configuration']['budget'] = 999
        choice = c.choose()
        self.assertEqual(choice['action'], 'forward')
        choice['action'] = 'stop'
        self.assertEqual(c.snapshot()['pending_action'], 'forward')
        self.assertEqual(c.snapshot()['configuration']['budget'], 48)

    def test_mixture_weights_and_uncertainty_diagnostic_are_explicit(self):
        c, m, obs = controller(discovery_weight=.5, inspection_weight=2., uncertainty_penalty=.25), mapper(), packet(measured_plane=True)
        m.update(obs); c.accept(obs, m); choice = c.choose()
        for row in choice['candidate_utilities']:
            expected = .5*row['discovery_unknown_area_m2']+2.*max(0., row['inspection_expected_new_area_m2']-.25*row['inspection_area_std_m2'])
            self.assertAlmostEqual(row['combined_design_utility'], expected)
        self.assertFalse(choice['learned_uncertainty'])
        self.assertFalse(choice['calibrated_quality'])
        self.assertEqual(set(choice['modules']), {'OV_SDF', 'STGHP', 'RPN_UQ', 'IGCR'})


if __name__ == '__main__':
    unittest.main()
