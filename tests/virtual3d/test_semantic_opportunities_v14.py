from copy import deepcopy
import unittest
from types import SimpleNamespace

import numpy as np

from nso.semantic_opportunities_v14 import (
    bounded_residual, candidate_capacity, route_instance_features, observed_descriptors,
)
from env.virtual3d import camera_pose
from tests.virtual3d.test_hierarchical_options_v10 import (
    MeasuredMapperFixture, bfs_cost, known_safe_direct, transition,
)


def fixture(count=2):
    mapper = MeasuredMapperFixture()
    routes = []
    for cid, distance in enumerate((2, 4)):
        outward = ['forward'] * distance
        back = ['right', 'right'] + ['forward'] * distance + ['right', 'right']
        states = [(5, 3, 1)]
        for action in outward + back:
            states.append(transition(states[-1], action))
        routes.append(dict(candidate_id=cid, states=[list(p) for p in states],
                           outbound_cost=len(outward), return_cost=len(back),
                           cost=len(outward + back), outbound_actions=outward,
                           return_actions=back, actions=outward + back))
    assets = [dict(observed_low=[1.8 + .4 * i, .7, .3], observed_high=[2. + .4 * i, 1.5, 1.3],
                   aabb_center=[1.9 + .4 * i, 1.1, .8], front_axis=[1., 0.], back_axis=[-1., 0.],
                   side_axis=[0., -1.], rear_boundary_xy=[1.8 + .4 * i, 1.1], measured_width_m=.8)
              for i in range(count)]
    records = [dict(asset_index=i, class_vote=1. if i % 2 == 0 else -1., semantic_confidence=.8,
                    direction_unobserved_fractions=[1., .5, 0., 1., .8, .4, .3, 1.],
                    inverse_sqrt_support_uncertainty_proxy=.2) for i in range(count)]
    descriptors = [dict(candidate_id=c['candidate_id'], observation_horizon='outbound_only_v14',
                        remaining_budget=30, total_budget=30, outbound_cost=c['outbound_cost'],
                        return_cost=c['return_cost'], roundtrip_cost=c['cost'],
                        unknown_allocated_grid_fraction=0., expected_new_camera_cells=0,
                        expected_new_radar_cells=0, camera_yield_posterior=.5, radar_yield_posterior=.5,
                        observed_assets=deepcopy(records), observed_aperture_audit=dict(
                            aperture_factors=[0.] * count, prefix_aperture_support=[0.] * count))
                   for c in routes]
    return mapper, routes, assets, descriptors


class SemanticOpportunitiesV14Test(unittest.TestCase):
    def test_observed_adapter_excludes_return_and_detaches_candidate_annotations(self):
        mapper, routes, assets, _ = fixture()
        for a in assets:
            a.update(class_vote=1., support_points=10, marked_points=10, bits=np.zeros(10, np.uint8))
        seen_states = []
        def masks(mapper, states, attempted):
            seen_states.append(states)
            return np.zeros(mapper.shape, bool), np.zeros(mapper.shape, bool)
        gain = SimpleNamespace(route_masks=masks, posterior_mean=lambda channel: .5,
                               attempted_camera_mask=lambda mapper, history: np.zeros(mapper.shape, bool))
        ledger = SimpleNamespace(remaining_budget=30, total_budget=30,
            planning_camera_poses=lambda: [camera_pose((5, 3), 1, mapper.config, mapper.shape[0])])
        runtime = SimpleNamespace(components=SimpleNamespace(_cpu_backend=SimpleNamespace(
            scenes=[dict(mapper=mapper, ledger=ledger, assets=assets, gain=gain)])))
        _, descriptors = observed_descriptors(runtime, routes)
        for states, route in zip(seen_states, routes):
            self.assertEqual(states, route['states'][1:route['outbound_cost'] + 1])
        descriptors[0]['observed_assets'][0]['class_vote'] = -1.
        self.assertEqual(descriptors[1]['observed_assets'][0]['class_vote'], 1.)
        self.assertEqual(assets[0]['class_vote'], 1.)

    def test_zero_to_24_instances_and_no_confidence_fallback(self):
        for n in (0, 2, 8, 12, 24):
            rows = route_instance_features(*fixture(n), confidence_scale=0.)
            for row in rows:
                self.assertEqual(len(row['geometry']), 26)
                self.assertEqual(len(row['semantic']), 26)
                self.assertEqual(row['semantic'][18:], [0.] * 8)
                self.assertEqual(bounded_residual(.31, 100., row['residual_confidence'], .1), .31)
            self.assertEqual(candidate_capacity(n), 5 + 2 * n)
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                candidate_capacity(invalid)

    def test_category_changes_semantics_at_zero_immediate_aperture_only(self):
        data = fixture()
        normal = route_instance_features(*data)
        swapped = deepcopy(data[3])
        for d in swapped:
            for a in d['observed_assets']:
                a['class_vote'] *= -1
        changed = route_instance_features(*data[:3], swapped)
        for a, b in zip(normal, changed):
            np.testing.assert_array_equal(a['geometry'], b['geometry'])
            np.testing.assert_allclose(a['semantic'][18:], -np.asarray(b['semantic'][18:]))
            self.assertEqual(a['second_view_witnesses'], b['second_view_witnesses'])
        self.assertNotEqual(normal[0]['semantic'][18:], normal[1]['semantic'][18:])

    def test_instance_permutation_and_candidate_permutation_invariance(self):
        mapper, routes, assets, desc = fixture(8)
        baseline = route_instance_features(mapper, routes, assets, desc)
        perm = [4, 1, 7, 3, 0, 6, 5, 2]
        swapped = deepcopy(desc)
        for d in swapped:
            d['observed_assets'] = [dict(d['observed_assets'][i], asset_index=j) for j, i in enumerate(perm)]
        actual = route_instance_features(mapper, routes[::-1], [assets[i] for i in perm], swapped[::-1])
        for expected, row in zip(baseline, actual[::-1]):
            np.testing.assert_array_equal(expected['geometry'], row['geometry'])
            np.testing.assert_array_equal(expected['semantic'], row['semantic'])

    def test_second_view_reservation_matches_independent_bfs(self):
        data = fixture()
        rows = route_instance_features(*data)
        safe = known_safe_direct(data[0])
        count = 0
        for row, route in zip(rows, data[1]):
            end = tuple(route['states'][route['outbound_cost']])
            home = tuple(route['states'][-1])
            for witness in row['second_view_witnesses']:
                if witness is None:
                    continue
                count += 1
                other = data[1][witness['candidate_id']]
                view = tuple(other['states'][other['outbound_cost']])
                extra, back = bfs_cost(safe, end, view), bfs_cost(safe, view, home)
                self.assertEqual(witness['known_safe_extra_cost'], extra)
                self.assertEqual(witness['reserved_return_cost'], back)
                self.assertEqual(witness['total_cost'], route['outbound_cost'] + extra + back)
                self.assertLessEqual(witness['total_cost'], 30)
        self.assertGreater(count, 0)

    def test_invalid_route_and_uncommitted_descriptor_are_rejected(self):
        data = fixture()
        data[1][0]['states'][1] = [5, 9, 1]
        with self.assertRaises(ValueError):
            route_instance_features(*data)
        data = fixture()
        data[3][0]['observation_horizon'] = 'full_roundtrip'
        with self.assertRaisesRegex(ValueError, 'outbound-only'):
            route_instance_features(*data)
        data = fixture()
        data[0].belief[5, 4] = -1
        with self.assertRaisesRegex(ValueError, 'known-safe'):
            route_instance_features(*data)

    def test_unused_ids_and_truth_fields_do_not_enter_features(self):
        data = fixture()
        baseline = route_instance_features(*data)
        for asset in data[2]:
            asset.update(true_category=999, future_return=1e9, parent='LEAK')
        for d in data[3]:
            d.update(before={'f1': 1e9}, future_return=-1e9, parent='LEAK')
        self.assertEqual(baseline, route_instance_features(*data))


if __name__ == '__main__':
    unittest.main()
