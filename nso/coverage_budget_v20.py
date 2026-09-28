"""Observed-only two-dimensional coverage accounting and budget prediction.

No world, evaluator, class label or ground-truth reachable mask is accepted.
The public task mask is an ROI, not a true free-space mask. Safety and possible
connectivity are model-dependent planning proxies, never motion permission or
a guarantee that the evaluator's 80 percent coverage requirement will be met.
"""
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
from numbers import Integral

import numpy as np
from scipy.ndimage import label

from utils.grid_geometry import inflated_obstacles


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return int(value)


def _belief(value):
    array = np.asarray(value)
    if array.ndim != 2 or not array.size or not np.isin(array, [-1, 0, 1]).all():
        raise ValueError('nonempty unknown/free/occupied belief required')
    return array


def _mask(value, shape, name):
    array = np.asarray(value)
    if array.shape != tuple(shape) or array.dtype != np.bool_:
        raise ValueError(name + ' must be a Boolean mask matching the belief')
    return array


def _roi(task_mask, shape):
    return np.ones(shape, bool) if task_mask is None else _mask(task_mask, shape, 'task ROI')


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def union_unknown_mask(belief, *sensor_masks, task_mask=None):
    """A cell earns at most one 2D proposal, only while actually unknown."""
    belief = _belief(belief); union = np.zeros(belief.shape, bool)
    for mask in sensor_masks:
        union |= _mask(mask, belief.shape, 'sensor proposal')
    return union & (belief == -1) & _roi(task_mask, belief.shape)


def _component(allowed, anchor):
    result = np.zeros_like(allowed)
    if allowed[anchor]:
        groups, _ = label(allowed)
        result = groups == groups[anchor]
    return result


def coverage_sets(belief, anchor, radius_cells, task_mask=None, confirmed_occupied=None, target=.8):
    """Return S/P masks and audit from sensor evidence, without hidden floor.

    With no independent occupied confirmation, P retains the whole possible
    ROI. A repeated stale occupied belief must not be used as confirmation.
    """
    belief = _belief(belief); roi = _roi(task_mask, belief.shape)
    if len(anchor) not in (2, 3):
        raise ValueError('anchor must contain row and column, optionally heading')
    anchor = tuple(_integer(value, 'anchor coordinate') for value in anchor[:2])
    if not all(value < limit for value, limit in zip(anchor, belief.shape)):
        raise ValueError('anchor outside task grid')
    if not np.isfinite(radius_cells) or radius_cells < 0 or not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError('valid robot radius and target fraction required')
    confirmed = np.zeros(belief.shape, bool) if confirmed_occupied is None else _mask(confirmed_occupied, belief.shape, 'confirmed occupied')
    if np.any(confirmed & (belief != 1)):
        raise ValueError('confirmed occupied evidence contradicts current belief')
    safe = _component(roi & ~inflated_obstacles((belief != 0) | ~roi, radius_cells), anchor)
    possible = _component(roi & ~inflated_obstacles(confirmed | ~roi, radius_cells), anchor)
    if np.any(safe & ~possible):
        raise AssertionError('observed safe set is not inside possible set')
    n_safe, n_possible = int(safe.sum()), int(possible.sum())
    status = 'ok' if n_possible and safe[anchor] else 'anchor_not_observed_safe' if n_possible else 'anchor_not_possible'
    denominator = n_possible if n_possible else None
    audit = dict(status=status, safe_cells=n_safe, possible_cells=n_possible, S_count=n_safe, P_count=n_possible,
        C_plan=n_safe / denominator if denominator else None,
        deficit_cells=max(0, int(math.ceil(target * n_possible)) - n_safe) if status == 'ok' else None,
        target_fraction=float(target), task_cells=int(roi.sum()), confirmed_occupied_cells=int((confirmed & roi).sum()),
        safe_sha256=_digest(safe), possible_sha256=_digest(possible), task_roi_sha256=_digest(roi),
        possible_source='public ROI minus independently confirmed observed obstacles',
        evaluation_truth_used=False, class_used=False, true_coverage_guaranteed=False)
    return dict(safe=safe, possible=possible, audit=audit)


def scan_hit_mask(scan, shape, resolution_m):
    """Unique in-grid obstacle endpoints from one actual PlanarScan or dict."""
    shape = tuple(_integer(x, 'shape', 1) for x in shape)
    if len(shape) != 2 or not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError('two-dimensional shape and positive resolution required')
    result = np.zeros(shape, bool)
    if scan is None:
        return result
    get = (lambda name: scan[name]) if isinstance(scan, Mapping) else (lambda name: getattr(scan, name))
    ranges = np.asarray(get('ranges_m'), float)
    pose = np.asarray(get('world_from_laser'), float)
    angle_min, angle_step, range_max = (float(get(k)) for k in ('angle_min_rad', 'angle_increment_rad', 'range_max_m'))
    if ranges.ndim != 1 or pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.isfinite([angle_min, angle_step, range_max]).all() or range_max <= 0:
        raise ValueError('invalid observed planar scan')
    ids = np.flatnonzero(np.isfinite(ranges) & (ranges > 0) & (ranges < range_max - 1e-3))
    if not len(ids):
        return result
    angles = angle_min + ids * angle_step
    local = np.column_stack([np.cos(angles) * ranges[ids], np.sin(angles) * ranges[ids], np.zeros(len(ids))])
    points = local @ pose[:3, :3].T + pose[:3, 3]
    cols = np.floor(points[:, 0] / resolution_m).astype(int)
    rows = shape[0] - 1 - np.floor(points[:, 1] / resolution_m).astype(int)
    inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    result[rows[inside], cols[inside]] = True
    return result


def assess_coverage_budget(deficit_cells, remaining_budget, outbound_cost, return_cost,
                           predicted_gain_cells, conservative_rate, reserve_actions=5):
    """Model budget gate; unavailable/zero rates never invent completion time."""
    budget = _integer(remaining_budget, 'remaining budget')
    outbound = _integer(outbound_cost, 'outbound cost'); returning = _integer(return_cost, 'return cost')
    margin = _integer(reserve_actions, 'reserve actions')
    if not np.isfinite(predicted_gain_cells) or predicted_gain_cells < 0:
        raise ValueError('predicted coverage gain must be finite and nonnegative')
    if deficit_cells is not None and (not np.isfinite(deficit_cells) or deficit_cells < 0):
        raise ValueError('coverage deficit must be finite and nonnegative')
    if conservative_rate is not None and (not np.isfinite(conservative_rate) or conservative_rate < 0):
        raise ValueError('conservative rate must be finite and nonnegative or None')
    remainder = max(0., float(deficit_cells) - predicted_gain_cells) if deficit_cells is not None else None
    rate_available = conservative_rate is not None and conservative_rate > 0
    reserve = 0 if remainder == 0 else int(math.ceil(remainder / conservative_rate)) if remainder is not None and rate_available else None
    required = outbound + returning + margin + reserve if reserve is not None else None
    if outbound + returning > budget:
        reason = 'route_and_return_exceed_budget'
    elif deficit_cells is None:
        reason = 'coverage_progress_unavailable'
    elif reserve is None:
        reason = 'coverage_rate_unavailable'
    elif required > budget:
        reason = 'insufficient_predicted_coverage_reserve'
    else:
        reason = 'model_budget_passed'
    slack = budget - outbound - returning - margin
    pressure = remainder / (conservative_rate * slack) if remainder is not None and rate_available and slack > 0 else 0. if remainder == 0 else None
    return dict(allowed=reason == 'model_budget_passed', reason=reason, deficit_cells=deficit_cells,
        predicted_gain_cells=float(predicted_gain_cells), residual_deficit_cells=remainder,
        remaining_budget=budget, outbound_cost=outbound, return_cost=returning, reserve_actions=margin,
        conservative_rate=None if conservative_rate is None else float(conservative_rate),
        coverage_reserve_actions=reserve, total_required_actions=required,
        remaining_after_outbound_return_and_margin=slack, pressure_after_route=pressure,
        movement_authorized=False, true_coverage_guaranteed=False, evaluation_truth_used=False, class_used=False)


class ObservedCoverageLedgerV20:
    def __init__(self, shape, anchor, radius_cells, task_mask=None, target=.8,
                 prefix_actions=5, history_prefixes=6, rate_discount=.5,
                 gain_discount=.5, reserve_actions=5):
        self.shape = tuple(_integer(x, 'shape', 1) for x in shape)
        if len(self.shape) != 2:
            raise ValueError('two-dimensional task shape required')
        self.anchor = tuple(anchor); self.radius_cells = float(radius_cells)
        self.task_mask = _roi(task_mask, self.shape).copy(); self.target = float(target)
        self.prefix_actions = _integer(prefix_actions, 'prefix actions', 1)
        self.history_prefixes = _integer(history_prefixes, 'history prefixes', 1)
        self.reserve_actions = _integer(reserve_actions, 'reserve actions')
        if not np.isfinite([rate_discount, gain_discount]).all() or not 0 < rate_discount <= 1 or not 0 < gain_discount <= 1:
            raise ValueError('discounts must be in (0,1]')
        self.rate_discount, self.gain_discount = float(rate_discount), float(gain_discount)
        self.radar_hit_counts = np.zeros(self.shape, np.uint16)
        self.history = deque(maxlen=self.history_prefixes); self.pending_prefix = None
        self.events = []; self.action_id = None; self.belief = None; self.state = None
        self.union_success = self.union_failure = 1  # smoothing, not a confidence guarantee
        self.scan_observations = 0

    def _hits(self, hits):
        if hits is not None:
            mask = _mask(hits, self.shape, 'actual scan hits') & self.task_mask
            self.radar_hit_counts[mask] = np.minimum(self.radar_hit_counts[mask].astype(np.uint32) + 1, 65535)
            self.scan_observations += 1

    def _state(self, belief):
        confirmed = (self.radar_hit_counts >= 2) & (belief == 1)
        return coverage_sets(belief, self.anchor, self.radius_cells, self.task_mask, confirmed, self.target)

    def start(self, belief, action_id=0, radar_hits=None):
        if self.action_id is not None:
            raise RuntimeError('coverage ledger already started')
        action_id = _integer(action_id, 'initial action id')
        value = _belief(belief)
        if value.shape != self.shape:
            raise ValueError('belief does not match declared task shape')
        self._hits(radar_hits)
        self.belief = value.copy(); self.action_id = action_id; self.state = self._state(self.belief)
        return self.snapshot()

    def flush_prefix(self):
        """Finalize a shorter actual prefix on replan/termination; no fake steps."""
        if self.pending_prefix is not None:
            row = deepcopy(self.pending_prefix)
            row['credited_safe_gain'] = max(0, row['safe_added'] - row['safe_lost'])
            if row['coverage_intent']:
                self.history.append(row)
            self.pending_prefix = None
            return row
        return None

    def observe(self, belief, action_id, coverage_intent, predicted_mask=None, radar_hits=None):
        if self.action_id is None:
            raise RuntimeError('start coverage ledger before observing paid actions')
        action_id = _integer(action_id, 'paid action id')
        if action_id != self.action_id + 1:
            raise ValueError('each paid action must be observed exactly once in order')
        if type(coverage_intent) is not bool:
            raise ValueError('coverage intention must be Boolean')
        value = _belief(belief)
        if value.shape != self.shape:
            raise ValueError('belief does not match declared task shape')
        proposed = None if predicted_mask is None else union_unknown_mask(self.belief, predicted_mask, task_mask=self.task_mask)
        self._hits(radar_hits); state = self._state(value)
        actual = (self.belief == -1) & (value != -1) & self.task_mask
        safe_added = int(np.count_nonzero(state['safe'] & ~self.state['safe']))
        safe_lost = int(np.count_nonzero(self.state['safe'] & ~state['safe']))
        success = int(np.count_nonzero(actual & proposed)) if proposed is not None else None
        failure = int(np.count_nonzero(proposed & ~actual)) if proposed is not None else None
        if proposed is not None:
            self.union_success += success; self.union_failure += failure
        if self.pending_prefix is not None and self.pending_prefix['coverage_intent'] != coverage_intent:
            self.flush_prefix()
        if self.pending_prefix is None:
            self.pending_prefix = dict(first_action_id=action_id, last_action_id=action_id, coverage_intent=coverage_intent,
                paid_actions=0, actual_new_known=0, safe_added=0, safe_lost=0)
        self.pending_prefix.update(last_action_id=action_id,
            paid_actions=self.pending_prefix['paid_actions'] + 1,
            actual_new_known=self.pending_prefix['actual_new_known'] + int(actual.sum()),
            safe_added=self.pending_prefix['safe_added'] + safe_added, safe_lost=self.pending_prefix['safe_lost'] + safe_lost)
        event = dict(action_id=action_id, coverage_intent=coverage_intent,
            actual_new_known_cells=int(actual.sum()), actual_new_known_sha256=_digest(actual),
            known_lost_cells=int(np.count_nonzero((self.belief != -1) & (value == -1) & self.task_mask)),
            safe_added_cells=safe_added, safe_lost_cells=safe_lost,
            possible_cells_before=self.state['audit']['possible_cells'], possible_cells_after=state['audit']['possible_cells'],
            predicted_union_cells=int(proposed.sum()) if proposed is not None else None,
            realized_predicted_cells=success, unrealized_predicted_cells=failure,
            unpredicted_realized_cells=int(np.count_nonzero(actual & ~proposed)) if proposed is not None else None,
            actual_new_known_counts_shared_cell_once=True, camera_seen_only_credit=False)
        if self.pending_prefix['paid_actions'] == self.prefix_actions:
            event['closed_prefix'] = self.flush_prefix()
        self.action_id = action_id; self.belief = value.copy(); self.state = state
        self.events.append(deepcopy(event))
        return event

    def _rate(self):
        rows = list(self.history)
        if self.pending_prefix is not None and self.pending_prefix['coverage_intent']:
            row = deepcopy(self.pending_prefix)
            row['credited_safe_gain'] = max(0, row['safe_added'] - row['safe_lost'])
            rows = (rows + [row])[-self.history_prefixes:]
        cost = sum(row['paid_actions'] for row in rows)
        # Loss in a later prefix offsets earlier gains; clipping each prefix
        # separately would overestimate progress when the observed map flaps.
        gain = max(0, sum(row['safe_added'] - row['safe_lost'] for row in rows))
        return (self.rate_discount * gain / cost if cost else None), rows

    def snapshot(self, remaining_budget=None, current_return_cost=None):
        if self.state is None:
            raise RuntimeError('coverage ledger has no actual observation')
        rate, rows = self._rate(); audit = deepcopy(self.state['audit'])
        pressure = None
        if audit['deficit_cells'] == 0:
            pressure = 0.
        elif audit['deficit_cells'] is not None and rate is not None and rate > 0 and remaining_budget is not None and current_return_cost is not None:
            slack = remaining_budget - current_return_cost - self.reserve_actions
            pressure = audit['deficit_cells'] / (rate * slack) if slack > 0 else None
        audit.update(action_id=self.action_id, conservative_rate=rate,
            planning_coverage=audit['C_plan'],
            rate_status='available' if rate is not None and rate > 0 else 'zero_realized_rate' if rate == 0 else 'no_paid_coverage_prefix',
            rate_discount=self.rate_discount, gain_discount=self.gain_discount, coverage_pressure=pressure,
            coverage_prefixes=deepcopy(rows), pending_prefix=deepcopy(self.pending_prefix),
            prefix_actions=self.prefix_actions, history_prefixes=self.history_prefixes,
            observed_actions=len(self.events), scan_observations=self.scan_observations,
            radar_confirmation_count=2, confirmed_hit_cells=int((self.radar_hit_counts >= 2).sum()),
            confirmed_hit_current_belief_conflicts=int(np.count_nonzero((self.radar_hit_counts >= 2) & (self.belief != 1))),
            union_yield_mean=self.union_success / (self.union_success + self.union_failure),
            union_yield=self.union_success / (self.union_success + self.union_failure),
            union_success=self.union_success, union_failure=self.union_failure,
            rate_is_confidence_bound=False, safe_map_permission_unchanged=True)
        return audit

    def predict_gain(self, predicted_mask):
        if self.state is None:
            raise RuntimeError('coverage prediction requires actual initial evidence')
        proposed = union_unknown_mask(self.belief, predicted_mask, task_mask=self.task_mask)
        imagined = self.belief.copy(); imagined[proposed] = 0
        hypothetical = self._state(imagined)
        optimistic_gain = int(np.count_nonzero(hypothetical['safe'] & ~self.state['safe']))
        yield_mean = self.union_success / (self.union_success + self.union_failure)
        return dict(predicted_union_cells=int(proposed.sum()), predicted_union_sha256=_digest(proposed),
            optimistic_safe_gain_cells=optimistic_gain, discounted_safe_gain_cells=optimistic_gain * yield_mean * self.gain_discount,
            union_yield_mean=yield_mean, gain_discount=self.gain_discount,
            future_unknown_free_assumption='temporary prediction only; never fused or authorized',
            actual_map_changed=False, class_used=False, evaluation_truth_used=False)

    def assess_route(self, route, predicted_mask, remaining_budget):
        prediction = self.predict_gain(predicted_mask); progress = self.snapshot()
        result = assess_coverage_budget(progress['deficit_cells'], remaining_budget,
            route['outbound_cost'], route['return_cost'], prediction['discounted_safe_gain_cells'],
            progress['conservative_rate'], self.reserve_actions)
        return dict(result, prediction=prediction, coverage_status=progress['status'], C_plan=progress['C_plan'])
