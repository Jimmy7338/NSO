"""Finite, development-only conditional response fitting on physical features.

Ranking and absolute-area calibration are deliberately separate APIs. Nothing
here reads sensors, semantic ground truth, worlds, files, or held-out outcomes.
Feature units/scales, split roles, and semantic interventions belong to the
caller's frozen feature contract. No training standard deviation is applied.
"""
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import linprog


ALPHAS = (.1, 1., 10.)
CV_TIE_TOLERANCE = 1e-12


def _matrix(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or min(result.shape) < 1 or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a nonempty finite matrix')
    return result


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a nonempty finite vector')
    return result


@dataclass
class HistoryFeatures:
    history_id: str
    context_id: str
    features: Mapping[str, np.ndarray]


class CapacityLimitError(ValueError):
    """The CV-selected model failed the predeclared development capacity gate.

    ``models`` retains all fitted results for honest diagnostics/archiving. Do
    not silently select another alpha or proceed to held-out evaluation.
    """
    def __init__(self, models):
        self.models = models
        failed = [name for name, model in models.items() if not model.capacity_ok]
        super().__init__('selected ridge effective df exceeds T contexts / 2: ' + ', '.join(failed))


def _solve(histories, targets, alpha, center_history):
    design, response = [], []
    for history, x in histories.items():
        y = targets[history]
        if center_history:
            x, y = x - x.mean(axis=0), y - y.mean(axis=0)
        design.append(x / np.sqrt(len(x)))
        response.append(y / np.sqrt(len(y)))
    design, response = np.concatenate(design), np.concatenate(response)
    u, singular, vt = np.linalg.svd(design, full_matrices=False)
    coefficient = (vt.T * (singular / (singular ** 2 + alpha))) @ u.T @ response
    if not np.isfinite(coefficient).all():
        raise ValueError('nonfinite fit; check declared physical feature scales')
    diagnostics = {
        'rows': len(design), 'histories': len(histories), 'columns': design.shape[1],
        'design_rank': int(np.linalg.matrix_rank(design)),
        'singular_values': singular.tolist(),
        'effective_df_per_target': float(np.sum(singular ** 2 / (singular ** 2 + alpha))),
        'active_columns': int(np.count_nonzero(np.linalg.norm(design, axis=0))),
    }
    return coefficient, diagnostics


@dataclass
class GroupedResponseRidge:
    coefficient: np.ndarray
    alpha: float
    center_history: bool
    diagnostics: dict
    cv: list

    @property
    def capacity_ok(self):
        return self.diagnostics['capacity_ok']

    @classmethod
    def fit(cls, x_by_history, targets_by_history, context_by_history, *,
            center_history=True, enforce_capacity=True):
        """Select alpha once by leave-one-parent-context-out relative MSE.

        Training loss is sum_h mean_a ||y_ha - X_ha beta||² + alpha||beta||²,
        with optional per-history centering and **no learned scale**. Selection
        uses target column 0 only, averages validation MSE equally by history,
        and chooses stronger regularization within the fixed numerical tie
        tolerance. Other target columns are fitted independently.

        Alpha is selected before testing df <= number of T parent contexts / 2.
        A failed gate never causes a hidden second hyperparameter selection.
        """
        keys = list(x_by_history)
        if not keys or any(not isinstance(k, str) or not k for k in keys):
            raise ValueError('nonempty string history IDs required')
        keys.sort()
        if set(keys) != set(targets_by_history) or set(keys) != set(context_by_history):
            raise ValueError('feature, target, and context history IDs must agree exactly')
        x = {k: _matrix(x_by_history[k], 'features') for k in keys}
        y = {}
        for k in keys:
            raw = np.asarray(targets_by_history[k], dtype=float)
            y[k] = _matrix(raw[:, None] if raw.ndim == 1 else raw, 'targets')
            if len(y[k]) != len(x[k]):
                raise ValueError('target rows do not match candidate rows')
        if len({a.shape[1] for a in x.values()}) != 1 or len({a.shape[1] for a in y.values()}) != 1:
            raise ValueError('feature and target column counts must be consistent')
        if any(not isinstance(context_by_history[k], str) or not context_by_history[k] for k in keys):
            raise ValueError('nonempty string parent context IDs required')
        contexts = sorted(set(context_by_history.values()))
        if len(contexts) < 2:
            raise ValueError('grouped CV requires at least two independent parent contexts')
        cv = []
        for alpha in ALPHAS:
            folds, history_losses = [], []
            for held in contexts:
                train = {k: x[k] for k in keys if context_by_history[k] != held}
                validation = [k for k in keys if context_by_history[k] == held]
                coefficient, report = _solve(train, y, alpha, center_history)
                losses = {}
                for k in validation:
                    vx = x[k] - x[k].mean(axis=0) if center_history else x[k]
                    vy = y[k] - y[k].mean(axis=0) if center_history else y[k]
                    loss = float(np.mean(((vx @ coefficient)[:, 0] - vy[:, 0]) ** 2))
                    if not np.isfinite(loss):
                        raise ValueError('nonfinite CV loss')
                    losses[k] = loss
                    history_losses.append(loss)
                folds.append({'held_out_context': held, 'train_histories': list(train),
                    'train_contexts': sorted({context_by_history[k] for k in train}),
                    'validation_histories': validation, 'history_mse': losses, 'design': report})
            cv.append({'alpha': alpha, 'mean_history_validation_mse': float(np.mean(history_losses)), 'folds': folds})
        minimum = min(row['mean_history_validation_mse'] for row in cv)
        tolerance = CV_TIE_TOLERANCE * max(1., abs(minimum))
        alpha = max(row['alpha'] for row in cv if row['mean_history_validation_mse'] <= minimum + tolerance)
        coefficient, report = _solve(x, y, alpha, center_history)
        report.update(parent_contexts=contexts, parent_context_count=len(contexts),
            effective_df_limit=len(contexts) / 2.,
            capacity_ok=report['effective_df_per_target'] <= len(contexts) / 2. + 1e-12,
            scale_policy='caller-declared physical scales; no data-dependent scaling',
            loss='sum_history(mean_candidates(squared_error)) + alpha * squared_coefficient_norm',
            cv_primary_target_column=0, cv_tie_tolerance=tolerance,
            zero_rank=report['design_rank'] == 0)
        model = cls(coefficient, alpha, bool(center_history), report, cv)
        if enforce_capacity and not model.capacity_ok:
            raise CapacityLimitError({'model': model})
        return model

    def predict(self, x):
        """Signed relative predictions; one complete candidate history per call.

        No test target/intercept is accepted. When centering is enabled, calling
        with a single candidate necessarily returns zero; pass the whole pool.
        """
        x = _matrix(x, 'prediction features')
        if x.shape[1] != len(self.coefficient):
            raise ValueError('prediction feature columns do not match frozen model')
        if self.center_history:
            x = x - x.mean(axis=0)
        result = x @ self.coefficient
        if not np.isfinite(result).all():
            raise ValueError('nonfinite prediction')
        return result

    def to_dict(self):
        return {'coefficient': self.coefficient.tolist(), 'alpha': self.alpha,
                'center_history': self.center_history, 'diagnostics': self.diagnostics, 'cv': self.cv}

    @classmethod
    def from_dict(cls, data):
        coefficient = _matrix(data['coefficient'], 'coefficient')
        if data['alpha'] not in ALPHAS:
            raise ValueError('alpha outside frozen candidate set')
        return cls(coefficient, float(data['alpha']), bool(data['center_history']), data['diagnostics'], data['cv'])


def _validate_shared(features, common_columns, object_columns, require_missing=False):
    arrays = {k: _matrix(v, k) for k, v in features.items()}
    if not {'G', 'O', 'S', 'N'} <= set(arrays):
        raise ValueError('G/O/S/N feature matrices are required')
    if len({a.shape for a in arrays.values()}) != 1:
        raise ValueError('all channel matrices must have the same candidate rows and padded columns')
    if common_columns < 1 or object_columns < 0 or common_columns + object_columns > arrays['G'].shape[1]:
        raise ValueError('invalid shared feature block widths')
    for name, x in arrays.items():
        if not np.array_equal(x[:, :common_columns], arrays['G'][:, :common_columns]):
            raise ValueError(f'{name} changes common geometry features')
    stop = common_columns + object_columns
    if not np.array_equal(arrays['O'][:, :stop], arrays['S'][:, :stop]):
        raise ValueError('O and S must share their measured objectness block')
    if 'X' in arrays and not np.array_equal(arrays['X'][:, :stop], arrays['S'][:, :stop]):
        raise ValueError('X may change only class interaction inputs')
    if arrays['N'][:, common_columns:].any():
        raise ValueError('N must contain only the common route block')
    if require_missing and 'M' not in arrays:
        raise ValueError('actual missing-label M features must be supplied for fallback')
    if 'M' in arrays and not np.array_equal(arrays['M'], arrays['G']):
        raise ValueError('actual missing-label features must equal G before explicit fallback')
    return arrays


@dataclass
class ResponseModelBank:
    models: dict
    common_columns: int
    object_columns: int

    @classmethod
    def fit(cls, histories: Sequence[HistoryFeatures], targets_by_history, *,
            fit_names=('G', 'O', 'S', 'N', 'G_capacity'), common_columns=4,
            object_columns=4, center_history=True, enforce_capacity=True):
        """Fit strong channels independently on exactly the same T histories.

        G may have different objectness from O/S. Additional fixed geometric
        interactions belong in G_capacity. X/M/shared interventions are never
        trained as independent models by this API.
        """
        histories = list(histories)
        if not histories or len({h.history_id for h in histories}) != len(histories):
            raise ValueError('training histories must have unique IDs')
        if not {'G', 'O', 'S', 'N'} <= set(fit_names) or set(fit_names) - {'G', 'O', 'S', 'N', 'G_capacity'}:
            raise ValueError('fit only independent G/O/S/N and optional G_capacity')
        if len(set(fit_names)) != len(fit_names):
            raise ValueError('duplicate fit names')
        matrices = {h.history_id: _validate_shared(h.features, common_columns, object_columns) for h in histories}
        if any(set(fit_names) - set(arrays) for arrays in matrices.values()):
            raise ValueError('a requested training channel is missing')
        contexts = {h.history_id: h.context_id for h in histories}
        models = {name: GroupedResponseRidge.fit({k: v[name] for k, v in matrices.items()},
                  targets_by_history, contexts, center_history=center_history, enforce_capacity=False)
                  for name in fit_names}
        if enforce_capacity and any(not m.capacity_ok for m in models.values()):
            raise CapacityLimitError(models)
        return cls(models, common_columns, object_columns)

    def predict(self, history: HistoryFeatures):
        """Return signed scores, including explicit test-only interventions.

        Supplying M is mandatory: dropping labels is not assumed to equal G.
        X is returned when its actual swapped-input matrix is supplied.
        """
        arrays = _validate_shared(history.features, self.common_columns, self.object_columns, require_missing=True)
        result = {name: model.predict(arrays[name]) for name, model in self.models.items()}
        result['M'] = self.models['G'].predict(arrays['M'])
        if 'X' in arrays:
            result['X'] = self.models['S'].predict(arrays['X'])
        for name in ('G', 'O'):
            result[name + '_shared'] = self.models['S'].predict(arrays[name])
        return result

    def select(self, history: HistoryFeatures, costs):
        """Select only with signed primary scores; calibration cannot enter."""
        costs = _vector(costs, 'costs')
        predictions = self.predict(history)
        if np.any(costs <= 0) or len(costs) != len(predictions['G']):
            raise ValueError('positive candidate costs must align with predictions')
        return {name: min(range(len(costs)), key=lambda i: (-values[i, 0], costs[i], i))
                for name, values in predictions.items()}

    def to_dict(self):
        return {'models': {k: v.to_dict() for k, v in self.models.items()},
                'common_columns': self.common_columns, 'object_columns': self.object_columns}

    @classmethod
    def from_dict(cls, data):
        return cls({k: GroupedResponseRidge.from_dict(v) for k, v in data['models'].items()},
                   int(data['common_columns']), int(data['object_columns']))


@dataclass
class AbsoluteAreaCalibrator:
    """C-only bounded global clipped-affine area-L1 fit; never for ranking.

    Since a>=0, the nonpositive part of a*r+b is a prefix of sorted unique r.
    Enumerating its n_unique+1 possible cut positions partitions the bounded
    (a,b) domain into linear regions. On each region, clipped area MAE is an
    ordinary linear program. Their minimum is therefore a global minimum over
    the declared bounds, up to recorded HiGHS numerical tolerances. This proof
    would not justify an arbitrary local fit or fitting unclipped L1 instead.
    """
    slope: float
    intercept: float
    diagnostics: dict

    @classmethod
    def fit(cls, relative_rate, costs, actual_area, history_ids, *, slope_max, intercept_bounds):
        r, length, area = (_vector(v, n) for v, n in ((relative_rate, 'relative_rate'), (costs, 'costs'), (actual_area, 'actual_area')))
        ids = np.asarray(history_ids)
        if ids.ndim != 1 or len({len(r), len(length), len(area), len(ids)}) != 1:
            raise ValueError('calibration arrays must align')
        if np.any(length <= 0) or np.any(area < 0):
            raise ValueError('costs must be positive and true unique areas nonnegative')
        if any(not isinstance(h, (str, np.str_)) or not h for h in ids):
            raise ValueError('nonempty string calibration history IDs required')
        if len(intercept_bounds) != 2:
            raise ValueError('two intercept bounds required')
        lower, upper = map(float, intercept_bounds)
        slope_max = float(slope_max)
        if not np.isfinite([lower, upper, slope_max]).all() or lower > upper or slope_max < 0:
            raise ValueError('finite ordered bounds and nonnegative slope_max required')
        histories = sorted(set(ids))
        weights = np.array([1. / (len(histories) * np.count_nonzero(ids == h)) for h in ids])
        unique = np.unique(r)
        candidates, regions = [], []
        tolerance = 1e-9
        for split in range(len(unique) + 1):
            active = r >= unique[split] if split < len(unique) else np.zeros(len(r), bool)
            index = np.flatnonzero(active)
            n = len(index)
            objective = np.r_[0., 0., weights[index]]
            constraints, rhs = [], []
            if split:
                row = np.zeros(n + 2); row[:2] = [unique[split - 1], 1.]
                constraints.append(row); rhs.append(0.)
            if split < len(unique):
                row = np.zeros(n + 2); row[:2] = [-unique[split], -1.]
                constraints.append(row); rhs.append(0.)
            for j, i in enumerate(index):
                row = np.zeros(n + 2); row[:2] = [length[i] * r[i], length[i]]; row[j + 2] = -1.
                constraints.append(row); rhs.append(area[i])
                row = row.copy(); row[:2] *= -1
                constraints.append(row); rhs.append(-area[i])
            fit = linprog(objective, A_ub=np.array(constraints), b_ub=np.array(rhs),
                bounds=[(0., slope_max), (lower, upper)] + [(0., None)] * n,
                method='highs', options={'primal_feasibility_tolerance': tolerance, 'dual_feasibility_tolerance': tolerance})
            if fit.status == 2:
                regions.append({'inactive_unique_count': split, 'feasible': False})
                continue
            if not fit.success or not np.isfinite(fit.x).all():
                raise ValueError(f'calibration LP failed in region {split}: {fit.message}')
            a = float(np.clip(fit.x[0], 0., slope_max)); b = float(np.clip(fit.x[1], lower, upper))
            value = float(np.dot(weights, np.abs(length * np.maximum(0., a * r + b) - area)))
            lp_value = float(fit.fun + np.dot(weights[~active], area[~active]))
            if abs(value - lp_value) > 1e-7 * max(1., abs(value), abs(lp_value)):
                raise ValueError('calibration LP region disagrees with clipped physical output')
            regions.append({'inactive_unique_count': split, 'feasible': True, 'objective': value, 'lp_objective': lp_value})
            candidates.append((value, a, b, split))
        if not candidates:
            raise ValueError('no feasible calibration activation region')
        # Include the deterministic constant weighted-median fit. This gives a
        # natural a=0 tie candidate without a regularization term or new search.
        order = np.argsort(area / length, kind='stable')
        constant_weights = weights[order] * length[order]
        median = (area / length)[order[np.searchsorted(np.cumsum(constant_weights), constant_weights.sum() / 2., side='left')]]
        constant_b = float(np.clip(median, lower, upper))
        candidates.append((float(np.dot(weights, np.abs(length * max(0., constant_b) - area))), 0., constant_b, -1))
        minimum = min(c[0] for c in candidates)
        tie_tolerance = 1e-10 * max(1., abs(minimum))
        chosen = min((c for c in candidates if c[0] <= minimum + tie_tolerance), key=lambda c: (c[1], abs(c[2]), c[2], c[3]))
        value, a, b, split = chosen
        report = {'objective': 'mean_history(mean_candidates(abs(cost * max(0, a * signed_r + b) - true_area)))',
            'history_count': len(histories), 'rows': len(r), 'slope_bounds': [0., slope_max],
            'intercept_bounds': [lower, upper], 'regions_considered': len(unique) + 1,
            'regions': regions, 'chosen_region': split, 'history_equal_area_mae': value,
            'best_region_objective': minimum, 'numerical_tie_tolerance': tie_tolerance,
            'uncalibrated_history_equal_area_mae': float(np.dot(weights, np.abs(cls.uncalibrated_area(r, length) - area))),
            'solver': 'scipy.optimize.linprog/highs', 'scipy_version': scipy.__version__,
            'solver_primal_dual_tolerance': tolerance,
            'optimality': 'global over declared bounded domain up to LP and tie numerical tolerances',
            'ranking_must_use_original_signed_response': True}
        return cls(a, b, report)

    @staticmethod
    def uncalibrated_area(relative_rate, costs):
        r, length = _vector(relative_rate, 'relative_rate'), _vector(costs, 'costs')
        if len(r) != len(length) or np.any(length <= 0):
            raise ValueError('positive costs must align with rates')
        return length * np.maximum(0., r)

    def predict_area(self, relative_rate, costs):
        r, length = _vector(relative_rate, 'relative_rate'), _vector(costs, 'costs')
        if len(r) != len(length) or np.any(length <= 0):
            raise ValueError('positive costs must align with rates')
        with np.errstate(over='ignore', invalid='ignore'):
            result = length * np.maximum(0., self.slope * r + self.intercept)
        if not np.isfinite(result).all():
            raise ValueError('nonfinite calibrated area; check declared physical units')
        return result

    def to_dict(self):
        return {'slope': self.slope, 'intercept': self.intercept, 'diagnostics': self.diagnostics}

    @classmethod
    def from_dict(cls, data):
        if not np.isfinite([data['slope'], data['intercept']]).all() or data['slope'] < 0:
            raise ValueError('invalid absolute-area calibration parameters')
        return cls(float(data['slope']), float(data['intercept']), data['diagnostics'])
