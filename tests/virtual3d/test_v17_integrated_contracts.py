from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import open3d as o3d
from tests.virtual3d.test_semantic_opportunities_v14 import fixture
from tests.virtual3d.test_staged_approach_runtime_v16 import fixture as runtime_fixture
from nso.semantic_opportunities_v17 import route_instance_features
from nso.observed_option_runtime_v17 import ObservedOptionRuntimeV17


class V17ContractTests(unittest.TestCase):
    def data(self):
        mapper, routes, assets, descriptors = fixture()
        mapper.mesh = lambda: o3d.geometry.TriangleMesh()
        for d in descriptors:
            d['observation_horizon'] = 'outbound_only_v17_observed_mesh'
        return mapper, routes, assets, descriptors

    def test_observed_wall_removes_second_view_witness_without_erasing_proximity(self):
        data = self.data()
        baseline = route_instance_features(*data)
        self.assertTrue(any(w is not None for r in baseline for w in r['second_view_witnesses']))
        mesh = o3d.geometry.TriangleMesh.create_box(.05, 3., 2.).translate([1.6, 0., 0.])
        data[0].mesh = lambda: mesh
        masked = route_instance_features(*data)
        self.assertTrue(all(w is None for r in masked for w in r['second_view_witnesses']))
        for a, b in zip(baseline, masked):
            self.assertEqual(a['geometry'][:13], b['geometry'][:13])
            self.assertEqual(len(b['semantic']), len(b['geometry']))
            self.assertEqual(len(b['geometry']), 26)

    def test_labels_change_only_signed_features_and_zero_confidence_clears_tail(self):
        data = self.data(); baseline = route_instance_features(*data)
        changed = deepcopy(data[3])
        for d in changed:
            for a in d['observed_assets']: a['class_vote'] *= -1
        flipped = route_instance_features(*data[:3], changed)
        zero = route_instance_features(*data, confidence_scale=0.)
        for a, b, z in zip(baseline, flipped, zero):
            self.assertEqual(a['geometry'], b['geometry'])
            self.assertEqual(a['geometry'], z['geometry'])
            np.testing.assert_array_equal(a['semantic'][18:], -np.asarray(b['semantic'][18:]))
            self.assertEqual(z['semantic'][18:], [0.] * 8)
            self.assertEqual(z['residual_confidence'], 0.)

    def runtime(self):
        original, _ = runtime_fixture()
        runtime = ObservedOptionRuntimeV17(original.components, 1, (10, 10))
        runtime.states = original.states
        runtime.components._cpu_backend = SimpleNamespace(scenes=[dict(region_continuation={'first': 1})])
        return runtime

    def test_pending_observation_cannot_be_discarded_to_cancel_commitment(self):
        runtime = self.runtime(); runtime.states[0]['pending'] = {'action': 'forward'}
        with self.assertRaises(RuntimeError): runtime.begin_return()
        with self.assertRaises(RuntimeError): runtime.install_two_stage_candidate({}, 'unused')
        self.assertIsNotNone(runtime.components._cpu_backend.scenes[0]['region_continuation'])

    def test_explicit_return_cancels_pending_second_stage(self):
        runtime = self.runtime(); runtime.begin_return()
        self.assertIsNone(runtime.components._cpu_backend.scenes[0]['region_continuation'])
        self.assertEqual(runtime.states[0]['active_actions'], [])
        self.assertEqual(runtime.states[0]['phase'], 'return')


if __name__ == '__main__': unittest.main()
