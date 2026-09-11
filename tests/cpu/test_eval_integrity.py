"""CPU regression checks; no Habitat, scene assets, or downloaded model weights."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

from arguments import get_args
from utils.eval_protocol import TRAIN_FLAGS, freeze_models, model_fingerprints, verify_frozen
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

    def test_reward_cannot_change_metrics(self):
        # Execute the actual environment reward implementation with only its
        # required fields; this isolates it from Habitat rendering imports.
        tree = ast.parse((ROOT / 'env/habitat/exploration_env.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Exploration_Env')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_global_reward')
        namespace = {'coverage_metrics': coverage_metrics}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<env reward>', 'exec'), namespace)
        args = SimpleNamespace(map_resolution=5, semantic_reward_coeff=1, structural_reward_coeff=1,
                               frontier_reward_coeff=1, use_semantic=True, paper_rewards=1,
                               intrinsic_reward_coeff=.05)
        obj = SimpleNamespace(args=args, explored_map=np.ones((10, 10)),
                              explorable_map=np.ones((10, 10)), prev_explored_area=0,
                              semantic_bonus_acc=100, structural_bonus_acc=200,
                              frontier_bonus_acc=300, timestep=1, _last_intrinsic_val=0)
        total, delta, _, area_reward = namespace['get_global_reward'](obj)
        self.assertAlmostEqual(total, 600.005)
        self.assertAlmostEqual(area_reward, .005)
        self.assertEqual(delta, 1)
        physical = coverage_metrics(obj.explored_map, obj.explorable_map, 5)
        tracker = PaperMetricsTracker()
        tracker.update_step(info=dict(physical, exp_reward=total, exp_ratio=delta))
        self.assertAlmostEqual(tracker.snapshot().explored_area_m2, .25)
        self.assertEqual(namespace['get_global_reward'](obj)[1], 0)

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
    def test_eval_overrides_presets_and_explicit_train_flags(self):
        for extra in ([], ['--paper_mode'], ['--paper_mode', '--train_global', '1', '--train_semantic']):
            with patch.object(sys, 'argv', ['main.py', '--no_cuda', '--eval', '1'] + extra):
                args = get_args()
            self.assertTrue(all(not getattr(args, flag) for flag in TRAIN_FLAGS))
        with patch.object(sys, 'argv', ['main.py', '--no_cuda', '--paper_mode']):
            args = get_args()
        self.assertTrue(args.train_goal_reachability)
        self.assertTrue(args.train_global)

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


class AutoResetTests(unittest.TestCase):
    def test_habitat2_terminal_info_survives_reset(self):
        module = load_file('test_h2_compat', 'env/habitat2/vector_env_compat.py')
        terminal = {'episode_id': 'old', 'coverage_ratio': .9, 'exp_reward': 777}
        reset = {'episode_id': 'new', 'coverage_ratio': .1, 'exp_reward': None}
        obs, _, done, info = module._split_step_result(((np.zeros((3, 2, 2)), reset), 0, True, terminal))
        self.assertTrue(done)
        self.assertEqual(info['episode_id'], 'new')
        self.assertEqual(info['terminal_info']['coverage_ratio'], .9)
        terminal['coverage_ratio'] = 0
        self.assertEqual(info['terminal_info']['coverage_ratio'], .9)

    def test_sync_reset_preserves_terminal_and_respects_no_autoreset(self):
        stub = ModuleType('habitat.core.env'); stub.Env = stub.RLEnv = object
        with patch.dict(sys.modules, {'habitat.core.env': stub}):
            module = load_file('test_sync', 'env/habitat/sync_vector_env.py')
        class FakeEnv:
            observation_space = action_space = None
            def __init__(self):
                self.info = {}; self.resets = 0
            def reset(self):
                self.resets += 1
                self.info.clear(); self.info.update(coverage_ratio=0, episode_id=f'ep{self.resets}')
                return np.zeros((2, 2)), self.info
            def step(self, action):
                self.info.update(coverage_ratio=1, explored_area_m2=.25)
                return np.ones((2, 2)), 99, True, self.info
        for autoreset in (True, False):
            vec = module.SyncVectorEnv(lambda: FakeEnv(), [()], auto_reset_done=autoreset)
            vec.reset()
            _, _, _, infos = vec.step([0])
            terminal = infos[0].get('terminal_info', infos[0])
            self.assertEqual(terminal['coverage_ratio'], 1)
            self.assertEqual(terminal['episode_id'], 'ep1')
            self.assertEqual(vec.envs[0].resets, 2 if autoreset else 1)
            vec.close()


class MainRecordingTests(unittest.TestCase):
    def test_actual_main_recording_final_step_and_autoreset(self):
        # Run the real recording block with terminal/reset packets, without
        # importing main.py's rendering and model dependencies.
        tree = ast.parse((ROOT / 'main.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        step_loop = next(n for n in ast.walk(main) if isinstance(n, ast.For)
                         and isinstance(n.target, ast.Name) and n.target.id == 'step')
        def assigns(node, name):
            return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        start = next(i for i, n in enumerate(step_loop.body) if assigns(n, 'step_infos'))
        end = next(i for i, n in enumerate(step_loop.body) if assigns(n, 'l_masks'))
        block = compile(ast.Module(body=step_loop.body[start:end], type_ignores=[]), '<main recorder>', 'exec')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            args = SimpleNamespace(eval=True, num_local_steps=2, max_episode_length=3)
            arrays = np.full((1, 2, 2), np.nan)
            ns = {'np': np, 'json': json, 'args': args, 'run_id': 'run',
                  'eval_output_dir': folder, 'METRICS_SCHEMA_VERSION': 2,
                  'paper_metrics': [PaperMetricsTracker()],
                  'explored_area_log': arrays.copy(), 'explored_ratio_log': arrays.copy()}
            for episode in range(2):
                for step in range(3):
                    terminal = step == 2
                    packet = {'coverage_ratio': (step + 1) / 10, 'explored_area_m2': step + 1,
                              'episode_id': str(episode), 'scene_id': 'scene',
                              'exp_reward': 10000, 'pose_err': [0, 0, 0]}
                    info = {'coverage_ratio': 0, 'episode_id': 'next', 'terminal_info': packet} if terminal else packet
                    ns.update(step=step, ep_num=episode, eval_g_step=step // 2 + 1,
                              infos=[info], done=[terminal])
                    exec(block, ns)
            records = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
            self.assertEqual([r['episode_id'] for r in records], ['0', '1'])
            self.assertEqual([r['step_count'] for r in records], [3, 3])
            self.assertEqual([r['explored_area_m2'] for r in records], [3, 3])
            np.testing.assert_allclose(ns['explored_ratio_log'][0], [[.2, .3], [.2, .3]])
            self.assertEqual(ns['paper_metrics'][0].snapshot().step_count, 0)
            ns.update(step=0, done=[True])
            with self.assertRaises(RuntimeError):
                exec(block, ns)


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
