"""Sensor-only geometric counterexamples and original-gain negative controls."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import sys
import unittest
from unittest.mock import patch
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
B = ROOT.parent / 'dual_template_exploration_20261119'
SOURCE = ROOT.parent / 'reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source'
sys.path[:0] = [str(SOURCE), str(B)]
spec = importlib.util.spec_from_file_location('repair_under_test', ROOT / 'method/occlusion_repair.py')
repair = importlib.util.module_from_spec(spec); spec.loader.exec_module(repair)
from nso.semantic_completion_v3 import ObjectCompletionModel
from utils.reconstruction_metrics import ray_scene


def frame(depth=3., *, oblique=True):
    pose = np.eye(4)
    if oblique:
        a = -np.pi / 4
        pose[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        pose[:3, 3] = [1, 0, 0]
    return SimpleNamespace(world_from_camera=pose,
        intrinsic=np.array([[30., 0, 2.], [0, 30., 2.], [0, 0, 1.]]),
        depth_m=np.full((5, 5), depth, dtype=float))


def panel(z):
    return o3d.geometry.TriangleMesh.create_box(.8, .8, .02).translate((-.4, -.4, z))


def fake_model(zs=(1.,), history=None):
    mesh = o3d.geometry.TriangleMesh()
    for z in zs: mesh += panel(z)
    config = SimpleNamespace(fov_deg=90., max_depth_m=8., height_px=90, width_px=120)
    return SimpleNamespace(points=np.array([[0., 0., 2.]]), weights=np.array([.3]),
        hypothesis_ids=np.array([0]), hypothesis_rays=[ray_scene(mesh)],
        mapper=SimpleNamespace(config=config, shape=(40, 50), keyframes=[frame()] if history is None else history),
        audit=dict(history_rejection=True), semantic=True)


def query(model, enabled, real=None):
    with patch('env.virtual3d.camera_pose', return_value=np.eye(4)):
        return repair.trace_gain(model, (0, 0), 0, real, repair_enabled=enabled)


class Tests(unittest.TestCase):
    def test_source_counterexample_free_board_blocks_unobserved_target(self):
        m = fake_model()
        status = repair.classify_history(np.array([[0, 0, 1.], [0, 0, 2.]]), m.mapper.keyframes)
        np.testing.assert_array_equal(status['strict_free'], [True, False])
        old, _ = query(m, False); new, stats = query(m, True)
        np.testing.assert_array_equal(old, [0.]); np.testing.assert_array_equal(new, [.3])
        self.assertEqual(stats['recovered_points'], 1)
    def test_measured_surface_remains_opaque(self):
        m = fake_model(history=[frame(np.sqrt(2))])
        new, stats = query(m, True)
        np.testing.assert_array_equal(new, [0.]); self.assertGreater(stats['measured_surface_intersections'], 0)
    def test_surface_evidence_vetoes_conflicting_free_frame(self):
        m = fake_model(history=[frame(), frame(np.sqrt(2))])
        new, stats = query(m, True)
        np.testing.assert_array_equal(new, [0.]); self.assertGreater(stats['history_conflicts_vetoed'], 0)
    def test_unknown_second_occluder_not_skipped(self):
        m = fake_model(zs=(1., 1.5))
        new, stats = query(m, True)
        np.testing.assert_array_equal(new, [0.])
        self.assertGreater(stats['strict_free_intersections'], 0); self.assertGreater(stats['unresolved_intersections'], 0)
    def test_multiple_free_occluders_may_be_skipped(self):
        new, stats = query(fake_model(zs=(1., 1.04)), True)
        np.testing.assert_array_equal(new, [.3]); self.assertGreaterEqual(stats['strict_free_intersections'], 4)
    def test_invalid_missing_and_outside_depth_never_free(self):
        for depth in (0., float('nan'), float('inf')):
            with self.subTest(depth=depth): np.testing.assert_array_equal(query(fake_model(history=[frame(depth)]), True)[0], [0.])
        c = repair.classify_history([[10., 0, 1.], [0., 0, .15], [0., 0, -1.]], [frame(3., oblique=False)])
        self.assertFalse(c['strict_free'].any())
    def test_strict_tolerance_boundary(self):
        f = frame(1.1, oblique=False)
        points = np.array([[0, 0, 1.], [0, 0, 1.-1e-6], [0, 0, 1.+1e-6]])
        c = repair.classify_history(points, [f])
        np.testing.assert_array_equal(c['strict_free'], [False, True, False])
        self.assertTrue(c['measured_surface'][0])
    def test_real_TSDF_occlusion_unchanged(self):
        new, stats = query(fake_model(), True, real=ray_scene(panel(.5)))
        np.testing.assert_array_equal(new, [0.]); self.assertEqual(stats['stages']['measured_TSDF']['points'], 0)
    def test_independent_hypotheses_not_merged(self):
        m = fake_model(history=[])
        m.points = np.array([[0., 0., 2.], [.1, 0., 2.]])
        m.weights = np.array([.3, .4]); m.hypothesis_ids = np.array([0, 1])
        m.hypothesis_rays.append(ray_scene(panel(2.)))
        np.testing.assert_array_equal(query(m, True)[0], [0., .4])
    def test_off_equals_original_on_real_source_models_and_install_isolated(self):
        spec = importlib.util.spec_from_file_location('B_test_fixture', B / 'method/test_adapter.py')
        oldtest = importlib.util.module_from_spec(spec); spec.loader.exec_module(oldtest)
        oldtest.Tests.setUpClass(); fixture = oldtest.Tests()
        models = [oldtest.FullTemplateModel(fixture.mapper(4), semantic=s, history_rejection=h)
                  for s in (False, True) for h in (False, True)]
        original_sha = hashlib.sha256((B / 'method/model.py').read_bytes()).hexdigest()
        states = [(15, 10, 1), (15, 30, 3), (5, 20, 2), (25, 20, 0)]
        for m in models:
            real = ray_scene(m.mapper.mesh())
            before = (m.points.copy(), m.weights.copy(), m.hypothesis_ids.copy())
            for state in states:
                expected = ObjectCompletionModel.gain(m, state[:2], state[2], real)
                off, _ = repair.trace_gain(m, state[:2], state[2], real, False)
                np.testing.assert_array_equal(off, expected)
            for a, b in zip(before, (m.points, m.weights, m.hypothesis_ids)): np.testing.assert_array_equal(a, b)
        repair.install(oldtest.FullTemplateModel, enabled=False)
        m = models[0]
        for state in states:
            np.testing.assert_array_equal(m.gain(state[:2], state[2], None), ObjectCompletionModel.gain(m, state[:2], state[2], None))
        self.assertEqual(repair.diagnostics(m)['gain_calls'], 4)
        with self.assertRaises(RuntimeError): repair.install(oldtest.FullTemplateModel, enabled=True)
        json.dumps(repair.diagnostics())
        self.assertEqual(hashlib.sha256((B / 'method/model.py').read_bytes()).hexdigest(), original_sha)


if __name__ == '__main__': unittest.main(verbosity=2)
