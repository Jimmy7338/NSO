"""Administrative and geometry-only fixtures; no World/render/DP/TSDF call."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import importlib.metadata
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_final_local_layout_validation_20260929 as batch


@dataclass(frozen=True)
class FakeConfig:
    width_m: float = 999.
    height_m: float = 777.
    max_steps: int = 1


class LayoutContractTests(unittest.TestCase):
    def setUp(self):
        self.config = batch.read(batch.CONFIG)

    def test_exact_fixed_matrix_and_no_automatic_retry(self):
        batch.validate_config(self.config)
        self.assertEqual(len(self.config['cases']), 8)
        self.assertFalse(self.config['automatic_retries'])
        self.assertEqual(self.config['unallocated_reserve_tasks'], 8)

    def test_missing_or_duplicate_slot_is_rejected(self):
        for rows in (self.config['cases'][:-1], [self.config['cases'][0]] * 8):
            bad = deepcopy(self.config); bad['cases'] = rows
            with self.assertRaises(ValueError): batch.validate_config(bad)

    def test_budget_and_information_intervention_are_fixed(self):
        for key, value in [('budget', 54), ('noise_seed', 1), ('modes', ['G', 'S', 'X']),
                           ('automatic_retries', True), ('new_category_claim', True)]:
            bad = deepcopy(self.config); bad[key] = value
            with self.assertRaises(ValueError): batch.validate_config(bad)

    def test_two_actual_background_topologies_not_rotated_copies(self):
        parents = batch.parents(self.config)
        base = next(p for p in batch.read(batch.SCENE)['parents'] if p['id'] == 'P00')
        removed = []
        for parent in parents.values():
            self.assertEqual(parent['hypotheses'], base['hypotheses'])
            self.assertEqual(parent['device_frame'], base['device_frame'])
            self.assertEqual(parent['prefix_actions'], base['prefix_actions'])
            removed.append(set(map(tuple, base['nav_cells'])) - set(map(tuple, parent['nav_cells'])))
        self.assertEqual(removed, [{(3, 3)}, {(0, 4)}])

    def test_safe_complete_graph_prefix_and_return(self):
        report = batch.static_review(self.config)
        self.assertEqual([r['safe_centres'] for r in report['layouts']], [23, 23])
        self.assertEqual([r['public_poses'] for r in report['layouts']], [92, 92])
        self.assertEqual([r['maximum_shortest_return_actions'] for r in report['layouts']], [16, 14])
        for row in report['layouts']:
            self.assertTrue(row['both_hypotheses_same_graph'])
            self.assertTrue(row['common_prefix_returned'])
            self.assertTrue(row['all_poses_returnable'])
        self.assertEqual(report['new_online_tasks'], 0)

    def test_background_stays_outside_unchanged_padded_roi(self):
        report = batch.static_review(self.config)
        np.testing.assert_allclose([r['background_roi_separation_m'][0] for r in report['layouts']], [.15, .20])
        bad = deepcopy(self.config)
        bad['layouts'][0]['background_boxes'] = [[2.0, 2.2, 2.4, 2.6, 0., 1.]]
        with self.assertRaises(ValueError): batch.static_review(bad)

    def test_nonfinite_geometry_is_rejected_without_render(self):
        bad = deepcopy(self.config)
        bad['layouts'][0]['background_boxes'][0][0] = float('nan')
        with self.assertRaises(ValueError): batch.parents(bad)

    def test_adapter_replaces_stale_navigation_transform_and_caches(self):
        # A plain namespace exercises constructor-state adaptation. No actual
        # InformationPixelWorldV34 object is instantiated or queried.
        for parent in batch.parents(self.config).values():
            old = SimpleNamespace(config=FakeConfig(), hypothesis=0,
                _nav_cells=frozenset({(999, 999)}), shift=np.array([88., 88., 0.]),
                _floor_cache={'stale': True}, step_count=7, collisions=4,
                last_clean_depth=np.ones((2, 2)), counts={'worlds': 1})
            old.pose_to_cell = lambda pose: old.transform.world_to_cell(np.asarray(pose[:2]) + old.shift[:2])
            original_parent = deepcopy(parent)
            result = batch.configure_world(old, parent, 42, 350918)
            self.assertIs(result, old)
            self.assertEqual(old._nav_cells, frozenset(map(tuple, parent['nav_cells'])))
            np.testing.assert_array_equal(old.shift, [3.5, .5, 0.])
            self.assertEqual(old.shape, (25, 35))
            self.assertEqual((old.config.width_m, old.config.height_m, old.config.max_steps), (7., 5., 42))
            self.assertEqual(old._pose, (0, 0, 0)); self.assertEqual(old.start, old.position)
            self.assertEqual((old.heading, old.step_count, old.collisions), (0, 0, 0))
            self.assertIsNone(old._floor_cache); self.assertIsNone(old.last_clean_depth)
            self.assertEqual(old.noise_seed, 350918)
            self.assertEqual(len(old._reference_boxes), 6)
            self.assertEqual(len(old.solid_boxes), 7); self.assertEqual(len(old.boxes), 8)
            self.assertEqual(old.primitives.shape, (8, 6))
            self.assertEqual(parent, original_parent)
            self.assertIsNot(old.parent, parent)

    def test_protected_original_science_unchanged(self):
        for name, expected in self.config['protected_scientific_sha256'].items():
            self.assertEqual(batch.sha(batch.ROOT / name), expected, name)

    def test_started_case_cannot_be_overwritten_or_retried(self):
        with tempfile.TemporaryDirectory() as path:
            out = Path(path); (out / 'cases' / self.config['cases'][0]['id']).mkdir(parents=True)
            with patch.object(batch, 'OUT', out), patch.object(batch, 'verify_sources', return_value=self.config):
                with self.assertRaises(FileExistsError): batch.run_case(0)
                with self.assertRaises(RuntimeError): batch.run_all()

    def test_unallocated_reserve_has_no_runnable_index(self):
        with patch.object(batch, 'verify_sources', return_value=self.config):
            for index in (-1, 8, 15, None):
                with self.assertRaises(ValueError): batch.run_case(index)

    def test_bundled_package_version_without_dist_info(self):
        with patch('importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError), \
             patch('importlib.import_module', return_value=SimpleNamespace(__version__='bundled-test')):
            result=batch.runtime_metadata()
        self.assertEqual(result['versions'],dict(numpy='bundled-test',scipy='bundled-test',open3d='bundled-test'))
        self.assertIn('cpu_models',result);self.assertIn('logical_cpu_count',result)

    def test_runtime_preflight_failure_does_not_claim_case_slot(self):
        with tempfile.TemporaryDirectory() as path:
            out=Path(path)
            with patch.object(batch,'OUT',out),patch.object(batch,'verify_sources',return_value=self.config), \
                 patch.object(batch,'runtime_metadata',side_effect=RuntimeError('missing runtime')):
                with self.assertRaises(RuntimeError):batch.run_case(0)
            self.assertFalse((out/'cases').exists())


if __name__ == '__main__':
    unittest.main()
