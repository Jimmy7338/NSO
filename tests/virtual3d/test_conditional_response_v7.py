import copy
import json
import unittest

import numpy as np

from nso.conditional_response_v7 import (
    AbsoluteAreaCalibrator, CapacityLimitError, GroupedResponseRidge,
    HistoryFeatures, ResponseModelBank, _solve,
)


class GroupedRidgeTests(unittest.TestCase):
    def test_history_mean_loss_matches_analytic_solution_and_duplication(self):
        x = {'a': np.array([[-1.], [1.]]), 'b': np.array([[-1.], [1.]])}
        y = {k: 2 * v for k, v in x.items()}
        coefficient, report = _solve(x, y, 1., True)
        np.testing.assert_allclose(coefficient, [[4. / 3.]])
        self.assertAlmostEqual(report['effective_df_per_target'], 2. / 3.)
        repeated = {**x, 'b': np.repeat(x['b'], 7, axis=0)}
        repeated_y = {**y, 'b': np.repeat(y['b'], 7, axis=0)}
        other, _ = _solve(repeated, repeated_y, 1., True)
        np.testing.assert_allclose(other, coefficient, atol=1e-14)

    def test_tiny_physical_features_are_not_amplified_by_standardization(self):
        x = {str(i): np.array([[-1e-10], [1e-10]]) for i in range(4)}
        y = {k: np.array([[-1.], [1.]]) for k in x}
        coefficient, _ = _solve(x, y, .1, True)
        self.assertLess(abs(coefficient[0, 0]), 1e-7)
        self.assertLess(abs((x['0'] @ coefficient)).max(), 1e-16)

    def test_parent_grouping_history_offsets_and_no_test_targets(self):
        x, y, context = {}, {}, {}
        for parent in range(4):
            for world in range(2):
                h = f'p{parent}/w{world}'
                x[h] = np.array([[-1., 0.], [0., 0.], [1., 0.]]) + parent
                y[h] = np.array([-1., 0., 1.]) + 100 * world + 1000 * parent
                context[h] = f'parent{parent}'
        model = GroupedResponseRidge.fit(x, y, context)
        for alpha in model.cv:
            self.assertEqual(len(alpha['folds']), 4)
            for fold in alpha['folds']:
                self.assertNotIn(fold['held_out_context'], fold['train_contexts'])
                self.assertEqual(len(fold['validation_histories']), 2)
                self.assertTrue(all(context[k] != fold['held_out_context'] for k in fold['train_histories']))
        shifted = GroupedResponseRidge.fit(x, {k: v + i * 50 for i, (k, v) in enumerate(y.items())}, context)
        np.testing.assert_allclose(model.coefficient, shifted.coefficient, atol=1e-12)
        prediction = model.predict(x['p0/w0'])
        self.assertLess(prediction[0, 0], 0)
        self.assertGreater(prediction[-1, 0], 0)
        np.testing.assert_allclose(prediction, model.predict(x['p0/w0'] + 100), atol=1e-12)
        with self.assertRaises(TypeError):
            model.predict(x['p0/w0'], y['p0/w0'])
        restored = GroupedResponseRidge.from_dict(json.loads(json.dumps(model.to_dict())))
        np.testing.assert_array_equal(prediction, restored.predict(x['p0/w0']))

    def test_tied_cv_selects_strongest_regularization(self):
        x = {str(i): np.array([[-1.], [1.]]) for i in range(4)}
        model = GroupedResponseRidge.fit(x, {k: np.zeros(2) for k in x}, {k: k for k in x})
        self.assertEqual(model.alpha, 10.)

    def test_uncentered_mode_is_explicit_and_keeps_no_learned_intercept(self):
        x = {str(i): np.ones((2, 1)) for i in range(4)}
        y = {k: np.full(2, 2.) for k in x}
        model = GroupedResponseRidge.fit(x, y, {k: k for k in x}, center_history=False)
        self.assertEqual(model.alpha, .1)
        np.testing.assert_allclose(model.predict([[1.]]), [[8. / 4.1]])
        np.testing.assert_array_equal(model.predict([[0.]]), [[0.]])

    def test_df_gate_does_not_silently_replace_cv_winner(self):
        basis = np.r_[np.eye(7), -np.eye(7)]
        x = {str(i): basis for i in range(8)}
        y = {k: basis.sum(axis=1) for k in x}
        with self.assertRaises(CapacityLimitError) as raised:
            GroupedResponseRidge.fit(x, y, {k: k for k in x})
        model = raised.exception.models['model']
        self.assertEqual(model.alpha, .1)
        self.assertFalse(model.capacity_ok)
        self.assertGreater(model.diagnostics['effective_df_per_target'], 4)
        _, stronger = _solve(x, {k: a[:, None] for k, a in y.items()}, 1., True)
        self.assertLess(stronger['effective_df_per_target'], 4)

    def test_bad_ids_and_nonfinite_arrays_rejected(self):
        with self.assertRaises(ValueError):
            GroupedResponseRidge.fit({'a': [[1]]}, {'a': [1]}, {'a': 'one'})
        with self.assertRaises(ValueError):
            GroupedResponseRidge.fit({'a': [[1]], 'b': [[np.inf]]}, {'a': [1], 'b': [1]}, {'a': 'a', 'b': 'b'})
        with self.assertRaises(ValueError):
            GroupedResponseRidge.fit({'a': [[1]], 'b': [[1]]}, {'a': [1]}, {'a': 'a', 'b': 'b'})


def bank_data():
    histories, targets = [], {}
    t = np.array([-1., -.5, .5, 1.])
    for context in range(8):
        common = np.c_[t, np.zeros((4, 3))]
        obj = np.c_[t ** 2, np.zeros((4, 3))]
        signed = (1 if context % 2 else -1) * obj
        zeros = np.zeros_like(obj)
        features = {
            'G': np.c_[common, obj, zeros], 'O': np.c_[common, obj * .5, zeros],
            'S': np.c_[common, obj * .5, signed], 'N': np.c_[common, zeros, zeros],
            'G_capacity': np.c_[common, obj, np.c_[t ** 3, np.zeros((4, 3))]],
        }
        features['X'] = np.c_[common, obj * .5, -signed]
        features['M'] = features['G'].copy()
        history = HistoryFeatures(str(context), f'parent{context}', features)
        histories.append(history)
        targets[history.history_id] = np.c_[2 * t + signed[:, 0], -.01 * t]
    return histories, targets


class ModelBankTests(unittest.TestCase):
    def test_independent_models_and_frozen_test_interventions(self):
        histories, targets = bank_data()
        bank = ResponseModelBank.fit(histories, targets)
        self.assertEqual(set(bank.models), {'G', 'O', 'S', 'N', 'G_capacity'})
        self.assertFalse(np.array_equal(bank.models['G'].coefficient, bank.models['S'].coefficient))
        self.assertFalse(np.array_equal(bank.models['O'].coefficient, bank.models['S'].coefficient))
        h = histories[1]
        prediction = bank.predict(h)
        np.testing.assert_array_equal(prediction['G'], prediction['M'])
        np.testing.assert_array_equal(prediction['X'], bank.models['S'].predict(h.features['X']))
        np.testing.assert_array_equal(prediction['O_shared'], bank.models['S'].predict(h.features['O']))
        self.assertGreater(abs(prediction['X'] - prediction['S']).max(), .01)
        restored = ResponseModelBank.from_dict(json.loads(json.dumps(bank.to_dict())))
        self.assertEqual(bank.select(h, [1, 1, 1, 1]), restored.select(h, [1, 1, 1, 1]))

    def test_actual_missing_features_and_shared_geometry_are_enforced(self):
        histories, targets = bank_data()
        bank = ResponseModelBank.fit(histories, targets)
        h = copy.deepcopy(histories[0])
        h.features['M'][0, 4] += .2
        with self.assertRaisesRegex(ValueError, 'missing-label'):
            bank.predict(h)
        h = copy.deepcopy(histories[0]); del h.features['M']
        with self.assertRaisesRegex(ValueError, 'M features'):
            bank.predict(h)
        h = copy.deepcopy(histories[0]); h.features['S'][0, 4] += .2
        with self.assertRaisesRegex(ValueError, 'objectness'):
            bank.predict(h)
        h = copy.deepcopy(histories[0]); h.features['X'][0, 0] += .2
        with self.assertRaisesRegex(ValueError, 'geometry'):
            bank.predict(h)


class AbsoluteCalibrationTests(unittest.TestCase):
    def test_exact_clipped_affine_area_example(self):
        r, costs, area = [-1., 0., 1.], [1., 2., 3.], [0., 2., 9.]
        model = AbsoluteAreaCalibrator.fit(r, costs, area, ['h'] * 3, slope_max=10., intercept_bounds=(-5., 5.))
        self.assertAlmostEqual(model.slope, 2.)
        self.assertAlmostEqual(model.intercept, 1.)
        np.testing.assert_allclose(model.predict_area(r, costs), area, atol=1e-8)
        np.testing.assert_array_equal(model.uncalibrated_area(r, costs), [0., 0., 3.])
        self.assertEqual(model.diagnostics['regions_considered'], 4)
        restored = AbsoluteAreaCalibrator.from_dict(json.loads(json.dumps(model.to_dict())))
        np.testing.assert_array_equal(restored.predict_area(r, costs), model.predict_area(r, costs))

    def test_global_region_fit_beats_or_equals_finite_grid(self):
        random = np.random.default_rng(981)
        for trial in range(5):
            r = random.uniform(-1.5, 1.5, 7)
            costs = random.integers(1, 6, 7)
            area = random.uniform(0., 4., 7)
            ids = np.array(['a'] * 2 + ['b'] * 5)
            model = AbsoluteAreaCalibrator.fit(r, costs, area, ids, slope_max=2., intercept_bounds=(-1., 1.))
            weights = np.array([.25] * 2 + [.1] * 5)
            a, b = np.meshgrid(np.linspace(0, 2, 51), np.linspace(-1, 1, 51))
            grid = costs * np.maximum(0., a[..., None] * r + b[..., None])
            grid_min = float(np.min(np.sum(weights * abs(grid - area), axis=-1)))
            self.assertLessEqual(model.diagnostics['history_equal_area_mae'], grid_min + 1e-8)
            self.assertGreaterEqual(model.slope, 0)
            self.assertLessEqual(model.slope, 2)
            self.assertGreaterEqual(model.intercept, -1)
            self.assertLessEqual(model.intercept, 1)

    def test_equal_history_weight_survives_repeating_one_history(self):
        r = np.array([-1., 0., 1., -1., 0., 1.])
        cost = np.array([1., 2., 3., 1., 2., 3.])
        area = np.array([0., 2., 9., 0., 1., 6.])
        ids = np.array(['a'] * 3 + ['b'] * 3)
        model = AbsoluteAreaCalibrator.fit(r, cost, area, ids, slope_max=5., intercept_bounds=(-3., 3.))
        repeat = np.r_[np.arange(3), np.tile(np.arange(3, 6), 9)]
        other = AbsoluteAreaCalibrator.fit(r[repeat], cost[repeat], area[repeat], ids[repeat], slope_max=5., intercept_bounds=(-3., 3.))
        self.assertAlmostEqual(model.diagnostics['history_equal_area_mae'], other.diagnostics['history_equal_area_mae'])
        np.testing.assert_allclose(model.predict_area(r, cost), other.predict_area(r, cost), atol=1e-8)

    def test_zero_labels_have_deterministic_zero_slope_and_intercept(self):
        args = ([-2., -1., 1.], [1., 2., 1.], [0., 0., 0.], ['a', 'a', 'b'])
        first = AbsoluteAreaCalibrator.fit(*args, slope_max=100., intercept_bounds=(-16., 16.))
        second = AbsoluteAreaCalibrator.fit(*args, slope_max=100., intercept_bounds=(-16., 16.))
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual((first.slope, first.intercept), (0., 0.))
        np.testing.assert_array_equal(first.predict_area([-100., 100.], [2., 2.]), [0., 0.])

    def test_repeated_rates_zero_slope_and_intercept_boundary(self):
        model = AbsoluteAreaCalibrator.fit([0., 0., 0.], [1., 2., 3.], [3., 6., 9.], ['a'] * 3,
                                           slope_max=0., intercept_bounds=(-1., 2.))
        self.assertEqual((model.slope, model.intercept), (0., 2.))
        self.assertEqual(model.diagnostics['regions_considered'], 2)
        np.testing.assert_array_equal(model.predict_area([-10., 0., 10.], [1., 2., 3.]), [2., 4., 6.])

    def test_calibration_cannot_change_bank_ranking(self):
        histories, targets = bank_data()
        bank = ResponseModelBank.fit(histories, targets)
        h = histories[1]; before = bank.select(h, [1., 2., 2., 1.])
        r = bank.predict(h)['S'][:, 0]
        calibrated = AbsoluteAreaCalibrator.fit(r, [1., 2., 2., 1.], [0., 0., 0., 0.], ['c'] * 4,
                                               slope_max=100., intercept_bounds=(-16., 16.))
        self.assertFalse(calibrated.predict_area(r, [1., 2., 2., 1.]).any())
        self.assertEqual(before, bank.select(h, [1., 2., 2., 1.]))
        with self.assertRaises(TypeError):
            bank.select(h, [1., 2., 2., 1.], calibrated)

    def test_physical_and_numerical_input_failures(self):
        for rates, costs, areas in (([np.nan], [1], [0]), ([0], [0], [0]), ([0], [1], [-1])):
            with self.assertRaises(ValueError):
                AbsoluteAreaCalibrator.fit(rates, costs, areas, ['a'], slope_max=1, intercept_bounds=(-1, 1))
        with self.assertRaises(ValueError):
            AbsoluteAreaCalibrator.fit([0], [1], [0], ['a'], slope_max=np.inf, intercept_bounds=(-1, 1))
        with self.assertRaises(ValueError):
            AbsoluteAreaCalibrator.fit([0], [1], [0], 'a', slope_max=1, intercept_bounds=(-1, 1))


if __name__ == '__main__':
    unittest.main()
