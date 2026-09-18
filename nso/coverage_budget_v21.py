"""Fixed public-ROI mapping progress, independent of navigation clearance.

This heuristic counts both observed free and occupied cells. Its denominator
may include unobservable closed interiors. It is not true reachable-floor
coverage, a certified bound, or permission to move. No world/evaluator input
is accepted. Navigation S/P is retained only as an observed diagnostic.
"""
from copy import deepcopy
import math

import numpy as np

from nso import coverage_budget_v20 as previous
from nso.coverage_budget_v20 import assess_coverage_budget, scan_hit_mask, union_unknown_mask


class ObservedCoverageLedgerV21(previous.ObservedCoverageLedgerV20):
    """The V20 paid-observation API with all budget terms in known ROI cells."""

    def __init__(self, shape, anchor, radius_cells, task_mask=None, target=.8,
                 prefix_actions=5, history_prefixes=6, rate_discount=.5,
                 gain_discount=.5, reserve_actions=5):
        super().__init__(shape, anchor, radius_cells, task_mask, target,
                         prefix_actions, history_prefixes, rate_discount,
                         gain_discount, reserve_actions)
        if not self.task_mask.any():
            raise ValueError('fixed task ROI must contain at least one cell')
        if not np.isfinite(self.target) or not 0 < self.target <= 1:
            raise ValueError('target fraction must be finite and in (0,1]')
        # The caller's mask was copied by V20; later caller edits cannot change
        # the public task denominator. This class never mutates its own mask.
        self.task_mask.setflags(write=False)
        self.task_cells = int(self.task_mask.sum())

    def _state(self, belief):
        navigation = super()._state(belief)
        known = (belief != -1) & self.task_mask
        count = int(known.sum()); target_count = int(math.ceil(self.target * self.task_cells))
        audit = deepcopy(navigation['audit'])
        audit.update(status='ok', C_plan=count / self.task_cells,
            deficit_cells=max(0, target_count - count), known_cells=count,
            unknown_cells=self.task_cells - count, target_known_cells=target_count,
            known_free_cells=int(np.count_nonzero((belief == 0) & self.task_mask)),
            known_occupied_cells=int(np.count_nonzero((belief == 1) & self.task_mask)),
            known_sha256=previous._digest(known),
            navigation_diagnostics=deepcopy(navigation['audit']),
            navigation_counts_are_diagnostic_only=True,
            coverage_proxy='fixed_task_roi_known_fraction',
            coverage_budget_unit='known_task_roi_cells', denominator_fixed=True,
            occupied_cells_count_as_known=True, closed_interiors_may_be_unobservable=True,
            proxy_is_true_reachable_coverage=False, proxy_is_certified_bound=False,
            target_is_public_heuristic=True, navigation_used_for_budget=False)
        return dict(safe=navigation['safe'], possible=navigation['possible'],
                    known=known, audit=audit)

    def flush_prefix(self):
        """Close even a short paid prefix, retaining negative net known change."""
        if self.pending_prefix is None:
            return None
        row = deepcopy(self.pending_prefix)
        row['net_known_gain'] = row['actual_new_known'] - row['known_lost']
        row['credited_known_gain'] = max(0, row['net_known_gain'])
        if row['coverage_intent']:
            self.history.append(row)
        self.pending_prefix = None
        return row

    def observe(self, belief, action_id, coverage_intent, predicted_mask=None, radar_hits=None):
        if self.action_id is None:
            raise RuntimeError('start coverage ledger before observing paid actions')
        action_id = previous._integer(action_id, 'paid action id')
        if action_id != self.action_id + 1:
            raise ValueError('each paid action must be observed exactly once in order')
        if type(coverage_intent) is not bool:
            raise ValueError('coverage intention must be Boolean')
        value = previous._belief(belief)
        if value.shape != self.shape:
            raise ValueError('belief does not match declared task shape')
        proposed = None if predicted_mask is None else union_unknown_mask(
            self.belief, predicted_mask, task_mask=self.task_mask)
        self._hits(radar_hits)
        state = self._state(value)
        actual = (self.belief == -1) & (value != -1) & self.task_mask
        lost = (self.belief != -1) & (value == -1) & self.task_mask
        added_count, lost_count = int(actual.sum()), int(lost.sum())
        safe_added = int(np.count_nonzero(state['safe'] & ~self.state['safe']))
        safe_lost = int(np.count_nonzero(self.state['safe'] & ~state['safe']))
        success = int(np.count_nonzero(actual & proposed)) if proposed is not None else None
        failure = int(np.count_nonzero(proposed & ~actual)) if proposed is not None else None
        if proposed is not None:
            self.union_success += success
            self.union_failure += failure
        if self.pending_prefix is not None and self.pending_prefix['coverage_intent'] != coverage_intent:
            self.flush_prefix()
        if self.pending_prefix is None:
            self.pending_prefix = dict(first_action_id=action_id, last_action_id=action_id,
                coverage_intent=coverage_intent, paid_actions=0, actual_new_known=0,
                known_lost=0, safe_added=0, safe_lost=0)
        self.pending_prefix.update(last_action_id=action_id,
            paid_actions=self.pending_prefix['paid_actions'] + 1,
            actual_new_known=self.pending_prefix['actual_new_known'] + added_count,
            known_lost=self.pending_prefix['known_lost'] + lost_count,
            safe_added=self.pending_prefix['safe_added'] + safe_added,
            safe_lost=self.pending_prefix['safe_lost'] + safe_lost)
        event = dict(action_id=action_id, coverage_intent=coverage_intent,
            actual_new_known_cells=added_count, actual_new_known_sha256=previous._digest(actual),
            known_lost_cells=lost_count, known_lost_sha256=previous._digest(lost),
            net_known_gain_cells=added_count - lost_count,
            known_cells_before=self.state['audit']['known_cells'], known_cells_after=state['audit']['known_cells'],
            safe_added_cells=safe_added, safe_lost_cells=safe_lost,
            possible_cells_before=self.state['audit']['possible_cells'], possible_cells_after=state['audit']['possible_cells'],
            navigation_counts_are_diagnostic_only=True, coverage_budget_unit='known_task_roi_cells',
            predicted_union_cells=int(proposed.sum()) if proposed is not None else None,
            realized_predicted_cells=success, unrealized_predicted_cells=failure,
            unpredicted_realized_cells=int(np.count_nonzero(actual & ~proposed)) if proposed is not None else None,
            actual_new_known_counts_shared_cell_once=True, camera_seen_only_credit=False)
        if self.pending_prefix['paid_actions'] == self.prefix_actions:
            event['closed_prefix'] = self.flush_prefix()
        self.action_id = action_id
        self.belief = value.copy()
        self.state = state
        self.events.append(deepcopy(event))
        return event

    def _rate(self):
        rows = list(self.history)
        if self.pending_prefix is not None and self.pending_prefix['coverage_intent']:
            row = deepcopy(self.pending_prefix)
            row['net_known_gain'] = row['actual_new_known'] - row['known_lost']
            row['credited_known_gain'] = max(0, row['net_known_gain'])
            rows = (rows + [row])[-self.history_prefixes:]
        cost = sum(row['paid_actions'] for row in rows)
        # Clip once over the whole history window: later loss offsets earlier
        # gains. Turns and zero-gain travel retain their actual action costs.
        gain = max(0, sum(row['actual_new_known'] - row['known_lost'] for row in rows))
        return (self.rate_discount * gain / cost if cost else None), rows

    def snapshot(self, remaining_budget=None, current_return_cost=None):
        audit = super().snapshot(remaining_budget, current_return_cost)
        audit.update(rate_unit='net_known_task_roi_cells_per_paid_action',
            rate_uses_navigation_safe_gain=False,
            coverage_intent_affects_only_rate_history=True,
            rate_history_paid_actions=sum(row['paid_actions'] for row in audit['coverage_prefixes']),
            rate_history_net_known_gain=sum(row['actual_new_known'] - row['known_lost']
                                            for row in audit['coverage_prefixes']))
        return audit

    def predict_gain(self, predicted_mask):
        if self.state is None:
            raise RuntimeError('coverage prediction requires actual initial evidence')
        proposed = union_unknown_mask(self.belief, predicted_mask, task_mask=self.task_mask)
        count = int(proposed.sum())
        yield_mean = self.union_success / (self.union_success + self.union_failure)
        discounted = count * yield_mean * self.gain_discount
        return dict(predicted_union_cells=count, predicted_union_sha256=previous._digest(proposed),
            optimistic_known_gain_cells=count, discounted_known_gain_cells=discounted,
            predicted_gain_cells=discounted, union_yield_mean=yield_mean,
            gain_discount=self.gain_discount, coverage_budget_unit='known_task_roi_cells',
            navigation_used_for_prediction=False, future_free_assumption_used=False,
            actual_map_changed=False, class_used=False, evaluation_truth_used=False,
            true_coverage_guaranteed=False)

    def assess_route(self, route, predicted_mask, remaining_budget):
        prediction = self.predict_gain(predicted_mask)
        progress = self.snapshot()
        result = assess_coverage_budget(progress['deficit_cells'], remaining_budget,
            route['outbound_cost'], route['return_cost'], prediction['discounted_known_gain_cells'],
            progress['conservative_rate'], self.reserve_actions)
        return dict(result, prediction=prediction, coverage_status=progress['status'],
            C_plan=progress['C_plan'], planning_coverage=progress['planning_coverage'],
            coverage_proxy=progress['coverage_proxy'], coverage_budget_unit='known_task_roi_cells',
            navigation_used_for_budget=False, proxy_is_certified_bound=False)
