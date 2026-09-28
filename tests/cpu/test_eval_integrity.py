"""CPU regression checks; no Habitat, scene assets, or downloaded model weights."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from utils.eval_protocol import (TRAIN_FLAGS, enforce_eval_mode, freeze_models,
                                 model_fingerprints, verify_frozen)
from utils.paper_eval import (coverage_metrics, GoalMetrics, PaperMetricsTracker,
                             aggregate_episode_metrics)
from scripts.audit_assets import inspect_asset
from scripts.summarize_eval_results import summarize, _read_matrix

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class MetricsTests(unittest.TestCase):
    def test_physical_area_and_resolution(self):
        observed = np.ones((10, 10))
        result = coverage_metrics(observed, observed, 5)
        self.assertAlmostEqual(result['explored_area_m2'], .25)
        self.assertEqual(result['coverage_ratio'], 1)
        self.assertAlmostEqual(coverage_metrics(observed, observed, 10)['explored_area_m2'], 1)

    def test_unexplorable_and_empty(self):
        reference = np.zeros((10, 10)); reference[:5] = 1
        result = coverage_metrics(np.ones((10, 10)), reference, 5)
        self.assertEqual(result['observed_free_cells'], 50)
        self.assertEqual(result['coverage_ratio'], 1)
        self.assertIsNone(coverage_metrics(reference, np.zeros_like(reference), 5)['coverage_ratio'])
        with self.assertRaises(ValueError):
            coverage_metrics(reference, np.zeros((2, 2)), 5)


    def test_count_once_and_keep_zero_and_missing(self):
        t = PaperMetricsTracker()
        t.update_step(info={'coverage_ratio': 0, 'explored_area_m2': 0,
                            'semantic_reward_sample': 0})
        t.record_loop()
        result = t.snapshot()
        self.assertEqual(result.step_count, 1)
        self.assertEqual(result.loop_closure_count, 1)
        self.assertEqual(result.coverage_ratio, 0)
        self.assertEqual(result.semantic_reward_mean, 0)
        self.assertIsNone(result.trajectory_drift_rmse_cm)
        stats = aggregate_episode_metrics([result])
        self.assertEqual(stats['coverage_ratio_count'], 1)
        self.assertEqual(stats['trajectory_drift_rmse_cm_count'], 0)
        self.assertIsNone(stats['trajectory_drift_rmse_cm'])
        t.reset_episode()
        self.assertEqual(t.snapshot().step_count, 0)

    def test_pose_increment_is_not_ate(self):
        t = PaperMetricsTracker()
        t.update_step(pose_err=[.03, .04, np.pi / 2])
        result = t.snapshot()
        self.assertAlmostEqual(result.odometry_step_translation_rmse_cm, 5)
        self.assertAlmostEqual(result.odometry_step_rotation_rmse_deg, 90)
        self.assertIsNone(result.trajectory_drift_rmse_cm)

    def test_goal_budget_and_repeated_replans(self):
        g = GoalMetrics(budget=3, radius_cells=1)
        g.start((10, 20))
        for position in [(0, 0), (10, 20), (0, 0)]:
            g.mark_unreachable(True)
            g.advance(position)
        self.assertEqual(g.attempts, 1)
        self.assertEqual(g.unreachable, 1)
        self.assertEqual(g.completed, 1)
        self.assertEqual(g.successes, 1)  # reached before budget endpoint
        g.advance((10, 20))
        self.assertEqual(g.completed, 1)
        g.start((100, 100)); g.advance((0, 0), terminal=True)
        self.assertEqual(g.completed, 2)
        self.assertEqual(g.successes, 1)


class FreezeTests(unittest.TestCase):
    def test_eval_disables_all_training_flags(self):
        for initial in ({}, dict.fromkeys(TRAIN_FLAGS, True),
                        {name: index % 2 == 0 for index, name in enumerate(TRAIN_FLAGS)}):
            with self.subTest(initial=initial):
                args = SimpleNamespace(eval=True, **initial)
                self.assertIs(enforce_eval_mode(args), args)
                self.assertTrue(all(getattr(args, flag) is False for flag in TRAIN_FLAGS))

    def test_training_configuration_is_preserved_outside_eval(self):
        flags = {name: index % 2 == 0 for index, name in enumerate(TRAIN_FLAGS)}
        args = SimpleNamespace(eval=False, **flags)
        self.assertIs(enforce_eval_mode(args), args)
        self.assertEqual({name: getattr(args, name) for name in TRAIN_FLAGS}, flags)

    def test_parameters_buffers_frozen_and_mutation_detected(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
        models = {'test': model}
        freeze_models(models)
        before = model_fingerprints(models)
        for _ in range(5):
            with torch.no_grad():
                model(torch.randn(2, 4))
        self.assertEqual(before, verify_frozen(models, before))
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        model[1].running_mean.add_(1)
        with self.assertRaises(RuntimeError):
            verify_frozen(models, before)

    def test_mc_dropout_does_not_train_parameters(self):
        module = load_file('test_rpn_uq', 'nso/reachability_uq.py')
        model = module.ReachabilityHeadUQ(in_channels=2, hidden=8, t_mc=3)
        freeze_models({'rpn': model})
        before = model_fingerprints({'rpn': model})
        mu, var = model.predict_with_uncertainty(torch.randn(1, 2, 8, 8))
        self.assertTrue(torch.isfinite(mu).all())
        self.assertGreater(var.max().item(), 0)
        self.assertEqual(before, verify_frozen({'rpn': model}, before))


class AuditTests(unittest.TestCase):
    def test_lfs_pointer_is_not_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); p = root / 'model_best.slam'
            p.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:' + 'a' * 64 + '\nsize 10000\n')
            result = inspect_asset(p, root)
            self.assertEqual(result['status'], 'lfs_pointer_not_materialized')
            self.assertEqual(result['expected_size_bytes'], 10000)
            self.assertNotEqual(result['size_bytes'], 10000)

    def test_multiline_matrix_records_not_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'matrix.txt'; p.write_text('[0 .1\n .3]\n[0\n .4 .5]\n')
            self.assertEqual(_read_matrix(p), [[0, .1, .3], [0, .4, .5]])
            p.write_text('[0 .1')
            with self.assertRaises(ValueError):
                _read_matrix(p)

    def test_real_legacy_audit_does_not_claim_physical_area(self):
        folder = ROOT / 'eval_results/paper_fast'
        result = summarize(folder / 'train.log', folder, 'historical')
        self.assertEqual(result['episode_count'], 80)
        self.assertEqual(result['logged_config']['num_processes'], 4)
        self.assertAlmostEqual(result['coverage_ratio_episode_final']['mean'], .79545914125)
        self.assertFalse(result['evaluation_integrity_verified'])
        self.assertIsNone(result['explored_area_m2_episode_final']['mean'])
        self.assertGreater(result['legacy_unvalidated']['logged_drift_cm_not_ate']['count'], 0)
        self.assertEqual(result['legacy_unvalidated']['online_coverage_increment']['min'], 0)

    def test_new_schema_complete_and_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = {'metrics_schema_version': 2, 'run_id': 'abc', 'status': 'complete',
                    'eval_frozen_verified': True, 'model_fingerprints_before': {'x': 'hash'},
                    'model_fingerprints_after': {'x': 'hash'},
                    'config': {'num_processes': 1, 'num_episodes': 1, 'max_episode_length': 3}}
            (root / 'run_metadata.json').write_text(json.dumps(meta))
            record = {'metrics_schema_version': 2, 'run_id': 'abc', 'env_index': 0,
                      'episode_index': 0, 'status': 'complete', 'step_count': 3,
                      'coverage_ratio': 0, 'explored_area_m2': 0}
            (root / 'episodes.jsonl').write_text(json.dumps(record) + '\n')
            result = summarize(root / 'missing.log', root, 'test')
            self.assertTrue(result['evaluation_integrity_verified'])
            self.assertEqual(result['coverage_ratio_episode_final']['mean'], 0)
            self.assertIsNone(result['trajectory_drift_rmse_cm']['mean'])
            meta['model_fingerprints_after']['x'] = 'changed'
            (root / 'run_metadata.json').write_text(json.dumps(meta))
            self.assertFalse(summarize(root / 'missing', root, 'test')['evaluation_integrity_verified'])


if __name__ == '__main__':
    unittest.main()
