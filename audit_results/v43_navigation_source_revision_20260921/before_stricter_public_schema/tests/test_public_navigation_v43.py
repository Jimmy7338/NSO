"""Static blueprint and public-loader tests; no simulator or sensor execution."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from nso.primitive_navigation_v41 import PrimitiveStateV41, PublicPrimitiveGraphV41
from nso.public_navigation_v43 import (bind_public_navigation_v43, compile_blueprint_graph_v43,
    load_public_navigation_bundle_v43, segment_intersects_rectangle_v43, select_candidate_states_v43)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'audit_results/v40_p1_development_geometry_20260920'
BUNDLES = ROOT/'audit_results/v43_public_navigation_20260921'
PROTOCOL = ROOT/'configs/virtual3d/v40_scene_protocol_20260920.json'


def workspace(size=(3.4, 3.4)):
    return dict(schema_version='v40.development_workspace.v1', room_inner_bounds_xy_m=[[0., 0.], list(size)],
                start_position_world_m=[.75, .75, .9], start_yaw_deg=0.)


def short_graph():
    return PublicPrimitiveGraphV41(dict(schema_version='v41.public_navigation.v1',
        source_kind='provided_navigation_prior', nodes={'home': [.75, .75], 'end': [1.75, .75]},
        edges=[['home', 'end']]))


class PublicNavigationV43Tests(unittest.TestCase):
    def test_continuous_thin_wall_blocks_edges_with_free_endpoints(self):
        graph, certificate = compile_blueprint_graph_v43(workspace(), [[1.21, 1.22, -.5, 4.]])
        self.assertTrue(all(point[0] == .75 for point in graph['nodes'].values()))
        self.assertGreater(certificate['safe_lattice_nodes'], certificate['reachable_nodes'])
        self.assertTrue(segment_intersects_rectangle_v43([.75, .75], [1.75, .75], [1.21, 1.22, -.5, 4.]))
        self.assertFalse(segment_intersects_rectangle_v43([.75, .75], [.75, 1.75], [1.21, 1.22, -.5, 4.]))

    def test_closed_touching_collision_and_robot_radius_are_conservative(self):
        self.assertTrue(segment_intersects_rectangle_v43([0., 0.], [1., 1.], [1., 2., 1., 2.]))
        # The centre is outside the solid, but its .2m robot envelope overlaps.
        with self.assertRaisesRegex(ValueError, 'start blocked'):
            compile_blueprint_graph_v43(workspace(), [[.9, 1., .7, .8]])

    def test_deterministic_graph_and_no_private_surface_fields(self):
        obstacles = [[1.6, 2., 1.6, 2.], [2.5, 2.6, .5, .6]]
        one, certificate = compile_blueprint_graph_v43(workspace(), obstacles)
        two, _ = compile_blueprint_graph_v43(workspace(), list(reversed(obstacles)))
        self.assertEqual(one, two)
        self.assertEqual(set(one), {'schema_version', 'source_kind', 'nodes', 'edges'})
        self.assertFalse(certificate['collision_envelopes_exported'])
        self.assertFalse(certificate['assumptions']['unknown_environment_exploration'])
        self.assertNotIn('world_aabb_m', json.dumps([one, certificate]))

    def test_fixed_start_and_node_cap_cannot_be_silently_repaired(self):
        changed = workspace(); changed['start_position_world_m'][0] += .25
        with self.assertRaisesRegex(ValueError, 'fixed'):
            compile_blueprint_graph_v43(changed, [])
        with self.assertRaisesRegex(ValueError, '256'):
            compile_blueprint_graph_v43(workspace((50., 50.)), [])

    def test_all_six_public_bundles_bind_same_frozen_parameters(self):
        protocol = json.loads(PROTOCOL.read_text())
        for family in 'ABCDEF':
            parent = f'DEV_{family}_00'
            with self.subTest(parent=parent):
                got = load_public_navigation_bundle_v43(BUNDLES/parent, ASSETS/parent)
                resolved = deepcopy(got['public_spec'])
                resolved['navigation'] = protocol['public_defaults']['navigation']
                self.assertEqual(resolved, protocol['public_defaults'])
                self.assertLessEqual(len(got['graph'].original_nodes), 256)
                self.assertEqual(got['home_state'], PrimitiveStateV41('home', 0))
                self.assertEqual(got['workspace']['navigation_graph_status'], 'pending')
                self.assertEqual(got['public_spec']['navigation']['status'], 'materialized')

    def test_runtime_loader_never_opens_private_metadata(self):
        original = Path.open; opened = []
        def checked(path, *args, **kwargs):
            opened.append(str(path))
            if 'evaluation_private' in path.parts or 'renderer_private' in path.parts:
                raise AssertionError('runtime read of private asset')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', checked):
            load_public_navigation_bundle_v43(BUNDLES/'DEV_A_00', ASSETS/'DEV_A_00')
        self.assertTrue(opened)
        self.assertTrue(all('instances.json' not in name and 'geometry.npz' not in name for name in opened))

    def test_byte_tampering_is_rejected_before_graph_use(self):
        with tempfile.TemporaryDirectory(prefix='nso-v43-nav-test-') as temporary:
            target = Path(temporary)/'DEV_A_00'; shutil.copytree(BUNDLES/'DEV_A_00', target)
            with (target/'graph.json').open('ab') as stream:
                stream.write(b' ')
            with self.assertRaisesRegex(ValueError, 'bytes/hash'):
                load_public_navigation_bundle_v43(target, ASSETS/'DEV_A_00')

    def test_sidecar_private_extension_and_parameter_drift_rejected(self):
        sidecar = json.loads((BUNDLES/'DEV_A_00/sidecar.json').read_text())
        graph = json.loads((BUNDLES/'DEV_A_00/graph.json').read_text())
        spec = json.loads((ASSETS/'DEV_A_00/public_planner_spec.json').read_text())
        protocol = json.loads(PROTOCOL.read_text())
        original = deepcopy(spec)
        bind_public_navigation_v43(spec, sidecar, graph, protocol)
        self.assertEqual(spec, original)
        changed = deepcopy(sidecar); changed['actual_target_structure'] = 'planar'
        with self.assertRaisesRegex(ValueError, 'whitelist'):
            bind_public_navigation_v43(spec, changed, graph, protocol)
        changed = deepcopy(spec); changed['task']['max_actions'] += 1
        with self.assertRaisesRegex(ValueError, 'protocol defaults'):
            bind_public_navigation_v43(changed, sidecar, graph, protocol)

    def test_alternative_protocol_bytes_rejected(self):
        with tempfile.TemporaryDirectory(prefix='nso-v43-protocol-test-') as temporary:
            path = Path(temporary)/'protocol.json'; path.write_bytes(PROTOCOL.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError, 'frozen V40'):
                load_public_navigation_bundle_v43(BUNDLES/'DEV_A_00', ASSETS/'DEV_A_00', path)

    def test_candidate_pool_excludes_paid_views_matches_real_route_cost_and_quota(self):
        graph = short_graph(); start = PrimitiveStateV41('home', 0)
        chosen, receipt = select_candidate_states_v43(graph, start, [start])
        self.assertNotIn(start, chosen)
        self.assertLessEqual(len(chosen), 32)
        self.assertLessEqual(max(Counter(s.node for s in chosen).values()), 4)
        for state, row in zip(chosen, receipt['candidates']):
            self.assertEqual(graph.route(start, state).cost, row['outbound_primitive_cost'])
        self.assertEqual((chosen, receipt), select_candidate_states_v43(graph, start, [start]))

    def test_candidate_pool_retains_eight_spatial_choices_when_available(self):
        graph = load_public_navigation_bundle_v43(BUNDLES/'DEV_A_00', ASSETS/'DEV_A_00')['graph']
        start = PrimitiveStateV41('home', 0)
        chosen, _ = select_candidate_states_v43(graph, start, [start])
        self.assertEqual(len(chosen), 32)
        self.assertGreaterEqual(len({s.node for s in chosen}), 8)
        self.assertLessEqual(max(Counter(s.node for s in chosen).values()), 4)

    def test_current_intermediate_node_is_eligible_and_blocked_edge_not_crossed(self):
        graph = short_graph(); home = PrimitiveStateV41('home', 0)
        intermediate = graph.successor(home, 'forward')
        chosen, _ = select_candidate_states_v43(graph, intermediate, [home, intermediate])
        self.assertTrue(any(s.node == intermediate.node for s in chosen))
        graph.block_observed_edge(home.node, intermediate.node)
        chosen, _ = select_candidate_states_v43(graph, home, [home])
        self.assertTrue(all(s.node == 'home' for s in chosen))
        for limit in (0, 33, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                select_candidate_states_v43(graph, home, limit=limit)


if __name__ == '__main__':
    unittest.main()
