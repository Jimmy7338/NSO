"""Static scene/relationship/navigation boundaries; no World or sensor capture."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nso.scene_contract_v40 import canonical_json_bytes
from nso.semantic_scene_assets import (
    CONDITIONS, PARENTS, asset_id, build_assets, file_sha256, load_design,
    load_semantic_scene_asset,
)
from nso.semantic_scene_navigation import build_public_navigation, load_semantic_navigation_bundle


class SemanticSceneAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='nso-semantic-static-')
        cls.root = Path(cls.temporary.name)
        cls.assets, cls.navigation = cls.root/'assets', cls.root/'navigation'
        cls.asset_result = build_assets(cls.assets)
        cls.navigation_result = build_public_navigation(cls.assets, cls.navigation,
            expected_asset_manifest_sha256=cls.asset_result['manifest_sha256'])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def asset(self, parent, condition):
        return load_semantic_scene_asset(self.assets/asset_id(parent, condition),
            expected_manifest_sha256=self.asset_result['manifest_sha256'])

    def nav(self, parent, condition, pin=None):
        return load_semantic_navigation_bundle(self.navigation/asset_id(parent, condition),
            expected_manifest_sha256=pin or self.navigation_result['manifest_sha256'])

    def test_complete_static_bundle_has_bound_closed_meshes_and_no_world(self):
        self.assertEqual(self.asset_result['assets'], 24)
        self.assertEqual(self.navigation_result['assets'], 24)
        self.assertEqual(self.asset_result['worlds_created'], 0)
        self.assertFalse(self.asset_result['primary_matrix_started'])
        for parent in PARENTS:
            for condition in CONDITIONS:
                loaded = self.asset(parent, condition)
                self.assertEqual(loaded['metadata']['public_workspace'], loaded['public_workspace'])
                self.assertEqual(len(loaded['markers']), 4)
                self.assertEqual(set(loaded['arrays']),
                    {'vertices', 'triangles', 'triangle_instance_id', 'triangle_rgb'})
                self.assertTrue(all(not array.flags.writeable for array in loaded['arrays'].values()))
                self.assertEqual(set(loaded['arrays']['triangle_instance_id']), {-1, 0, 1, 2, 3})

    def test_wrong_cues_preserve_geometry_and_shift_preserves_categories(self):
        for parent in PARENTS:
            nominal = self.asset(parent, 'nominal_relationship')
            wrong = self.asset(parent, 'persistent_recognition_error')
            shifted = self.asset(parent, 'structure_relationship_shift')
            for name in nominal['arrays']:
                np.testing.assert_array_equal(nominal['arrays'][name], wrong['arrays'][name])
            self.assertNotEqual([m['rgb'] for m in nominal['markers']], [m['rgb'] for m in wrong['markers']])
            self.assertEqual([m['rgb'] for m in nominal['markers']], [m['rgb'] for m in shifted['markers']])
            for ordinary, corrupted in zip(nominal['markers'], wrong['markers']):
                for key in ('center_world_m', 'normal_world', 'width_m', 'height_m'):
                    self.assertEqual(ordinary[key], corrupted[key])

    def test_actual_uninformative_block_balances_structure_without_changing_prior(self):
        for parent in PARENTS:
            nominal = self.asset(parent, 'nominal_relationship')
            neutral = self.asset(parent, 'uninformative_relationship')
            self.assertEqual(nominal['public_spec'], neutral['public_spec'])
            objects = neutral['metadata']['private_instances']
            categories = sorted({item['category'] for item in objects})
            self.assertEqual(len(categories), 2)
            self.assertEqual(Counter(item['structure'] for item in objects if item['category'] == categories[0]),
                             Counter(item['structure'] for item in objects if item['category'] == categories[1]))
            self.assertTrue(all(item['category'] == item['observed_category'] for item in objects))

    def test_navigation_is_connected_and_identical_for_all_paired_hidden_conditions(self):
        graph_hashes = []
        for parent in PARENTS:
            bundles = [self.nav(parent, condition) for condition in CONDITIONS]
            first = bundles[0]
            graph_hashes.append(first['graph'].input_sha256)
            self.assertEqual(first['certificate']['safe_lattice_nodes'], first['certificate']['reachable_nodes'])
            self.assertLessEqual(first['certificate']['declared_lattice_nodes'], 256)
            for other in bundles[1:]:
                self.assertEqual(first['graph_spec'], other['graph_spec'])
                self.assertEqual(first['public_spec'], other['public_spec'])
                self.assertEqual(first['workspace'], other['workspace'])
        self.assertEqual(len(set(graph_hashes)), 6)

    def test_public_loader_works_with_private_assets_unavailable(self):
        hidden = self.root/'private_assets_offline'
        self.assets.rename(hidden)
        try:
            loaded = self.nav('SEM_P00', 'nominal_relationship')
            self.assertNotIn('private_instances', loaded)
            self.assertNotIn('metadata', loaded)
            self.assertFalse(loaded['workspace']['instance_truth_included'])
        finally:
            hidden.rename(self.assets)

    def test_pin_mismatch_and_changed_asset_bytes_are_rejected(self):
        directory = self.assets/asset_id('SEM_P00', 'nominal_relationship')
        with self.assertRaisesRegex(ValueError, 'external pin'):
            load_semantic_scene_asset(directory, expected_manifest_sha256='0'*64)
        with self.assertRaisesRegex(ValueError, 'external pin'):
            self.nav('SEM_P00', 'nominal_relationship', pin='0'*64)
        path = directory/'public_workspace.json'
        original = path.read_bytes()
        try:
            path.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                self.asset('SEM_P00', 'nominal_relationship')
        finally:
            path.write_bytes(original)

    def test_public_workspace_rejects_truth_even_under_a_new_valid_pin(self):
        identity = asset_id('SEM_P00', 'nominal_relationship')
        path, manifest_path = self.navigation/identity/'public_workspace.json', self.navigation/'manifest.json'
        original, manifest_bytes = path.read_bytes(), manifest_path.read_bytes()
        try:
            value = json.loads(original); value['private_instance_positions'] = [[1., 2., 0.]]
            path.write_bytes(canonical_json_bytes(value))
            manifest = json.loads(manifest_bytes)
            manifest['artifact_sha256'][identity+'/public_workspace.json'] = file_sha256(path)
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, 'whitelist'):
                self.nav('SEM_P00', 'nominal_relationship', pin=file_sha256(manifest_path))
        finally:
            path.write_bytes(original); manifest_path.write_bytes(manifest_bytes)

    def test_invalid_design_and_existing_output_are_rejected(self):
        with self.assertRaises(FileExistsError):
            build_assets(self.assets)
        with self.assertRaises(FileExistsError):
            build_public_navigation(self.assets, self.navigation,
                expected_asset_manifest_sha256=self.asset_result['manifest_sha256'])
        bad = deepcopy(load_design())
        bad['parents'][0]['fixed_assignments']['uninformative_relationship']['true_structures'][0] = 'open_frame'
        path = self.root/'bad_design.json'; path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'balance structure'):
            load_design(path)


if __name__ == '__main__':
    unittest.main()
