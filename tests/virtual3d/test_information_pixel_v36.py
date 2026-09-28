"""V36 adapter admission with mock construction and array-only noise tests.

No real World, renderer, sensor packet, controller, DP, mapper, TSDF or Q runs.
P01 checks read already-saved public paired predictions and shared graph only.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from env.facility_documentation_v19 import stereo_depth_v19
from env.information_pixel_v34 import InformationPixelWorldV34, sensor_counts_v34
from env.information_pixel_v36 import (CONFIRMATION_NOISE_SEED_V36,
                                     InformationPixelWorldV36)
from nso.observation_belief_v35 import PublicTemplatesV35


ROOT = Path(__file__).resolve().parents[2]
RECEIPT_V36 = {'array_noise_function_calls': 0, 'mock_parent_constructor_calls': 0}
FROZEN_INPUTS_V36 = (
    'audit_results/v35_online_development_20260918/manifest.json',
    'docs/research/V36_FROZEN_CONFIRMATION_PLAN_20260918.md',
    'audit_results/v33_direction_information_r1_20260917/P01_geometry.json',
    'audit_results/v34_pixel_information_20260918/P01_h0_pixels.npz',
    'audit_results/v34_pixel_information_20260918/P01_h1_pixels.npz',
)


def _noise(clean, **kwargs):
    RECEIPT_V36['array_noise_function_calls'] += 1
    return stereo_depth_v19(clean, model='iid_025px', noise_seed=kwargs.pop(
        'noise_seed', CONFIRMATION_NOISE_SEED_V36), max_depth_m=4., **kwargs)


class InformationPixelV36Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = dict(sensor_counts_v34())

    @classmethod
    def tearDownClass(cls):
        after = dict(sensor_counts_v34())
        RECEIPT_V36['sensor_counter_delta'] = {k: after[k]-cls.before[k] for k in after}
        if any(RECEIPT_V36['sensor_counter_delta'].values()):
            raise AssertionError('array-only admission unexpectedly created physical observations')

    def test_constructor_forwards_unchanged_arguments_and_only_replaces_seed(self):
        calls = []
        config = object()

        def fake_parent_init(obj, parent_id, hypothesis, episode_id, noise_model='clean'):
            RECEIPT_V36['mock_parent_constructor_calls'] += 1
            calls.append((parent_id, hypothesis, episode_id, noise_model))
            obj.noise_seed = 1901
            obj.config = config

        with patch.object(InformationPixelWorldV34, '__init__', fake_parent_init):
            for parent in ('P00', 'P01'):
                for hypothesis in (0, 1):
                    obj = InformationPixelWorldV36(parent, hypothesis, 'opaque', 'iid_025px')
                    self.assertEqual(vars(obj), {'noise_seed': 350918, 'config': config})
                    self.assertEqual(calls[-1], (parent, hypothesis, 'opaque', 'iid_025px'))
        self.assertEqual(len(calls), 4)

    def test_constructor_cannot_render_and_exposes_no_seed_parameter(self):
        self.assertEqual(inspect.signature(InformationPixelWorldV36),
                         inspect.signature(InformationPixelWorldV34))
        self.assertEqual(CONFIRMATION_NOISE_SEED_V36, 350918)
        self.assertEqual({k for k in vars(InformationPixelWorldV36) if not k.startswith('__')}, set())
        parent_tree = ast.parse(textwrap.dedent(inspect.getsource(InformationPixelWorldV34.__init__)))
        calls = {n.func.attr for n in ast.walk(parent_tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls & {'_packet', 'packet', 'packet_at', 'step', 'render', 'update'})
        adapter_tree = ast.parse(textwrap.dedent(inspect.getsource(InformationPixelWorldV36.__init__)))
        assigns = [n for n in ast.walk(adapter_tree) if isinstance(n, ast.Assign)]
        self.assertEqual(len(assigns), 1)
        self.assertEqual(ast.unparse(assigns[0].targets[0]), 'self.noise_seed')

    def test_all_sensor_camera_geometry_execution_and_truth_methods_are_inherited(self):
        for name, value in vars(InformationPixelWorldV34).items():
            if name != '__init__' and (callable(value) or isinstance(value, property)):
                self.assertIs(getattr(InformationPixelWorldV36, name),
                              getattr(InformationPixelWorldV34, name))

    def test_same_fixed_seed_reproducible_on_synthetic_array(self):
        clean = np.linspace(.5, 3.5, 48, dtype=np.float32).reshape(6, 8)
        first = _noise(clean, parent_index=1, step=7)
        second = _noise(clean, parent_index=1, step=7)
        np.testing.assert_array_equal(first, second)
        RECEIPT_V36['same_seed_array_sha256'] = hashlib.sha256(first.tobytes()).hexdigest()

    def test_fixed_holdout_differs_from_existing_development_seed(self):
        clean = np.full((6, 8), 2., np.float32)
        new = _noise(clean, parent_index=0, step=18)
        old = _noise(clean, parent_index=0, step=18, noise_seed=1901)
        self.assertFalse(np.array_equal(new, old))
        RECEIPT_V36['different_seed_changed_pixel_count'] = int(np.count_nonzero(new != old))

    def test_parent_step_seed_derivation_matches_original_formula(self):
        clean = np.full((4, 5), 2., np.float32)
        outputs = []
        for parent_index in (0, 1):
            for step in (0, 18):
                got = _noise(clean, parent_index=parent_index, step=step)
                rng = np.random.default_rng(np.random.SeedSequence([parent_index, step, 350918, 73919]))
                # NumPy resolves scalar / float32-array operands at float32
                # before writing to the original function's float64 out array.
                # Preserve that declared input rounding in this reference.
                initial_disparity = float(np.float32(480.*.12) / np.float32(2.))
                expected = (57.599999999999994 / (np.full(clean.shape, initial_disparity)
                            + rng.normal(0., .25, clean.shape))).astype(np.float32)
                np.testing.assert_array_equal(got, expected)
                outputs.append(got)
        self.assertEqual(len({value.tobytes() for value in outputs}), 4)
        RECEIPT_V36['seed_fields'] = ['parent_index', 'action_step', 'noise_seed', 73919]

    def test_invalid_zero_and_far_pixels_stay_missing(self):
        got = _noise(np.array([[0., .1, 4.1, 2.]], np.float32), parent_index=0, step=0)
        np.testing.assert_array_equal(got[0, :3], [0., 0., 0.])
        self.assertGreater(float(got[0, 3]), .15)

    def test_class_never_enters_inherited_noise_call(self):
        self.assertFalse(set(inspect.signature(stereo_depth_v19).parameters) &
                         {'hypothesis', 'class_id', 'rgb', 'semantic', 'episode_id'})
        tree = ast.parse(textwrap.dedent(inspect.getsource(InformationPixelWorldV34._packet)))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'stereo_depth_v19']
        self.assertEqual(len(calls), 1)
        self.assertEqual({k.arg: ast.unparse(k.value) for k in calls[0].keywords}, {
            'model': 'noise_model', 'parent_index': 'int(self.parent_id[-1])',
            'step': 'step', 'noise_seed': 'self.noise_seed', 'max_depth_m': '4.0'})

    def test_old_frozen_sensor_scene_policy_mapper_and_metric_hashes_unchanged(self):
        frozen = json.loads((ROOT/FROZEN_INPUTS_V36[0]).read_text())['source_sha256']
        paths = ('env/information_pixel_v34.py', 'env/facility_documentation_v19.py',
                 'configs/virtual3d/v33_direction_scene_r1_20260917.json',
                 'nso/observation_belief_v35.py', 'nso/online_planner_v35.py',
                 'nso/cpu_four_modules_v35.py', 'nso/observed_runtime_mapper_v10.py',
                 'nso/surface_measurement_v34.py')
        for path in paths:
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), frozen[path])
        RECEIPT_V36['unchanged_frozen_sha256'] = {p: frozen[p] for p in paths}

    def test_P01_saved_template_calibration_graph_and_input_isolation(self):
        # Do not select oracle witnesses, measured Q or policy routes in this JSON.
        tables = json.loads((ROOT/FROZEN_INPUTS_V36[2]).read_text())['tables']
        for field in ('poses', 'edges', 'anchor', 'prefix_nodes', 'floor_cells'):
            self.assertEqual(tables[0][field], tables[1][field])
        poses = tables[0]['poses']
        templates = PublicTemplatesV35.from_saved(ROOT/'audit_results/v34_pixel_information_20260918', 'P01', poses)
        self.assertEqual(templates.depth.shape, (2, 112, 72, 96))
        self.assertEqual(templates.ranges.shape, (2, 112, 180))
        self.assertEqual(templates._forecast, {})
        self.assertFalse(templates.provenance['template_rgb_used'])
        for index, successors in enumerate(tables[0]['edges']):
            x, y, heading = poses[index]
            for action, next_node in successors:
                self.assertIn(action, ('forward', 'left', 'right'))
                dx, dy = ((0, 1), (1, 0), (0, -1), (-1, 0))[heading]
                expected = ([x+dx, y+dy, heading] if action == 'forward' else
                            [x, y, (heading+(-1 if action == 'left' else 1)) % 4])
                self.assertEqual(poses[next_node], expected)
        prefix = tables[0]['prefix_nodes']
        self.assertEqual(len(prefix), 19)
        self.assertEqual(prefix[0], tables[0]['anchor'])
        self.assertEqual(prefix[-1], tables[0]['anchor'])
        for current, following in zip(prefix, prefix[1:]):
            self.assertIn(following, [n for _, n in tables[0]['edges'][current]])
        with np.load(ROOT/FROZEN_INPUTS_V36[3], allow_pickle=False) as data:
            np.testing.assert_array_equal(data['intrinsic'], np.tile(
                [[48., 0., 47.5], [0., 48., 35.5], [0., 0., 1.]], (112, 1, 1)))
            np.testing.assert_allclose(data['scan_calibration'], np.tile(
                [-np.pi, 2*np.pi/180, 8.], (112, 1)), rtol=0., atol=0.)
            np.testing.assert_array_equal(data['camera_pose'][:, 2, 3], np.full(112, .9))
            np.testing.assert_array_equal(data['laser_pose'][:, 2, 3], np.full(112, .25))
        RECEIPT_V36['P01_saved_format_passed'] = True
        RECEIPT_V36['P01_public_nodes'] = len(poses)
        RECEIPT_V36['P01_saved_public_fields'] = templates.provenance['loaded_fields']
        RECEIPT_V36['P01_legal_edges'] = sum(map(len, tables[0]['edges']))
        RECEIPT_V36['controller_or_DP_calls'] = 0


if __name__ == '__main__':
    unittest.main()
