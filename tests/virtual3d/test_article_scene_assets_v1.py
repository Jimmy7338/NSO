"""Static article asset contracts. No sensors, worlds, TSDF or policies run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nso.article_scene_assets_v1 import (
    DEFAULT_DESIGN, MAX_BUNDLE_BYTES, PUBLIC_FILES, ROOT, SCENE_IDS,
    build_article_scene_assets, load_design, load_private_article_scene,
    load_public_article_scene, navigation_blueprint,
)
from nso.primitive_navigation_v41 import PrimitiveStateV41
from nso.public_navigation_v43 import compile_blueprint_graph_v43, segment_intersects_rectangle_v43
from nso.scene_contract_v40 import canonical_json_bytes


class ArticleSceneAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='article_assets_static_')
        cls.bundle = Path(cls.temporary.name)/'assets'
        cls.built = build_article_scene_assets(cls.bundle)
        cls.pin = cls.built['manifest_sha256']
        cls.design = load_design()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_nine_independent_layouts_split_before_any_policy_results(self):
        self.assertEqual(len(SCENE_IDS), 9)
        self.assertEqual(len(self.built['development_scene_ids']), 3)
        self.assertEqual(len(self.built['test_scene_ids']), 6)
        old = json.loads((ROOT/'configs/virtual3d/semantic_scene_matrix_draft_20260923.json').read_text())
        old_signatures = {canonical_json_bytes([x['room_size_m'], x['facility_xy_yaw_deg'], x['occluder_boxes_m']])
                          for x in old['parents']}
        signatures = {canonical_json_bytes([x['room_size_m'], x['facility_xy_yaw_deg'], x['occluder_boxes_m']])
                      for x in self.design['layouts']}
        self.assertEqual(len(signatures), 9)
        self.assertFalse(signatures & old_signatures)
        self.assertLess(self.built['bytes_written'], MAX_BUNDLE_BYTES)
        self.assertEqual(self.built['worlds_created'], 0)

    def test_public_loader_succeeds_with_all_private_files_absent(self):
        with tempfile.TemporaryDirectory(prefix='article_public_only_') as folder:
            root = Path(folder)
            shutil.copy2(self.bundle/'manifest.json', root/'manifest.json')
            directory = root/SCENE_IDS[0]
            directory.mkdir()
            for name in PUBLIC_FILES:
                shutil.copy2(self.bundle/SCENE_IDS[0]/name, directory/name)
            opened = []
            original = Path.read_bytes
            def traced(path):
                opened.append(path.relative_to(root).as_posix())
                return original(path)
            with patch.object(Path, 'read_bytes', traced):
                result = load_public_article_scene(directory, expected_manifest_sha256=self.pin)
            self.assertEqual(set(opened), {'manifest.json'} | {SCENE_IDS[0]+'/'+p for p in PUBLIC_FILES})
            self.assertEqual(result['home_state'], PrimitiveStateV41('home', 0))
            self.assertEqual(result['asset_manifest_sha256'], self.pin)
            self.assertEqual(result['public_spec']['task']['max_actions'], 160)

    def test_mesh_ownership_and_readonly_private_arrays(self):
        for identity in SCENE_IDS:
            private = load_private_article_scene(self.bundle/identity, expected_manifest_sha256=self.pin)
            self.assertEqual(set(private['arrays']), {'vertices', 'triangles', 'triangle_instance_id', 'triangle_rgb'})
            self.assertEqual(set(private['arrays']['triangle_instance_id']), {-1, 0, 1, 2, 3})
            self.assertTrue(all(not a.flags.writeable for a in private['arrays'].values()))
            self.assertEqual(len(private['metadata']['private_instances']), 4)
            self.assertEqual(len(private['markers']), 4)
            self.assertEqual(private['public_workspace']['start_position_world_m'], [.75, .75, .9])

    def test_all_graph_nodes_and_edges_clear_full_structure_envelopes(self):
        for layout in self.design['layouts']:
            loaded = load_public_article_scene(self.bundle/layout['scene_id'], expected_manifest_sha256=self.pin)
            _, _, rectangles = navigation_blueprint(layout)
            inflated = np.asarray(rectangles) + [-.2, .2, -.2, .2]
            nodes = loaded['graph_spec']['nodes']
            for position in nodes.values():
                self.assertFalse(any(segment_intersects_rectangle_v43(position, position, r) for r in inflated))
            for a, b in loaded['graph_spec']['edges']:
                self.assertFalse(any(segment_intersects_rectangle_v43(nodes[a], nodes[b], r) for r in inflated))
            certificate = loaded['certificate']
            self.assertEqual(certificate['reachable_nodes'], certificate['safe_lattice_nodes'])
            graph = loaded['graph']
            # Primitive expansion must remain within certified coarse corridors.
            for state_name, position in graph.positions.items():
                self.assertFalse(any(segment_intersects_rectangle_v43(position, position, r) for r in inflated))

    def test_navigation_is_invariant_to_hidden_class_and_structure_assignment(self):
        for layout in self.design['layouts']:
            changed = deepcopy(layout)
            changed['true_classes'] = list(reversed(changed['true_classes']))
            changed['structure_quantiles'] = [1.-q for q in changed['structure_quantiles']]
            first = navigation_blueprint(layout)[2]
            second = navigation_blueprint(changed)[2]
            self.assertEqual(first, second)
            workspace = json.loads((self.bundle/layout['scene_id']/'public_workspace.json').read_text())
            first_graph, _ = compile_blueprint_graph_v43(workspace, first)
            second_graph, _ = compile_blueprint_graph_v43(workspace, second)
            self.assertEqual(first_graph, second_graph)

    def test_static_frontal_route_witnesses_and_cost_accounting(self):
        for identity in SCENE_IDS:
            loaded = load_public_article_scene(self.bundle/identity, expected_manifest_sha256=self.pin)
            graph, home = loaded['graph'], loaded['home_state']
            audit = json.loads((self.bundle/identity/'evaluation_private/static_geometry_audit.json').read_text())
            self.assertFalse(audit['sensor_pixels_or_depth_fit_verified'])
            self.assertFalse(audit['exported_to_policy'])
            self.assertEqual(len(audit['per_facility']), 4)
            for item in audit['per_facility']:
                witness = item['witnesses'][0]
                state = PrimitiveStateV41(witness['node'], witness['heading'])
                self.assertEqual(graph.route(home, state).cost, witness['outbound_paid_actions'])
                self.assertEqual(graph.route(state, home).cost, witness['full_pose_return_paid_actions'])
                self.assertEqual(witness['one_view_return_paid_actions'],
                    witness['outbound_paid_actions'] + 1 + witness['full_pose_return_paid_actions'])
                self.assertLessEqual(witness['one_view_return_paid_actions'], 160)

    def test_external_pin_identity_and_corruption_are_rejected(self):
        with self.assertRaises(ValueError):
            load_public_article_scene(self.bundle/SCENE_IDS[0], expected_manifest_sha256='0'*64)
        with self.assertRaises(ValueError):
            load_public_article_scene(self.bundle/'SEM_P00', expected_manifest_sha256=self.pin)
        with self.assertRaises(FileExistsError):
            build_article_scene_assets(self.bundle)
        with tempfile.TemporaryDirectory(prefix='article_corrupt_') as folder:
            copy = Path(folder)/'copy'
            shutil.copytree(self.bundle, copy)
            directory = copy/SCENE_IDS[0]
            path = directory/'renderer_private/markers.json'
            path.write_text(path.read_text()+' ')
            # A private corruption cannot alter the public load boundary.
            load_public_article_scene(directory, expected_manifest_sha256=self.pin)
            with self.assertRaises(ValueError):
                load_private_article_scene(directory, expected_manifest_sha256=self.pin)

    def test_pinned_extra_truth_field_cannot_pass_public_schema(self):
        with tempfile.TemporaryDirectory(prefix='article_public_schema_') as folder:
            copy = Path(folder)/'copy'
            shutil.copytree(self.bundle, copy)
            name = SCENE_IDS[0]+'/public_workspace.json'
            value = json.loads((copy/name).read_text())
            value['private_instances'] = [{'structure': 'planar'}]
            data = canonical_json_bytes(value)
            (copy/name).write_bytes(data)
            manifest = json.loads((copy/'manifest.json').read_text())
            manifest['artifact_sha256'][name] = hashlib.sha256(data).hexdigest()
            manifest_data = canonical_json_bytes(manifest)
            (copy/'manifest.json').write_bytes(manifest_data)
            pin = hashlib.sha256(manifest_data).hexdigest()
            with self.assertRaises(ValueError):
                load_public_article_scene(copy/SCENE_IDS[0], expected_manifest_sha256=pin)


if __name__ == '__main__':
    unittest.main()
