"""V27-A: unchanged V26 planning, independent full measured feedback support.

The arrival/old-cue/visibility/range/sector gates and >=4, /16 rule are retained.
Only the support extraction changes; no whole-option reward, GT or learned head.
"""
import numpy as np
from nso.observed_planner_v26 import ObservedPlannerV26, _xy, _sector
from nso.observed_feedback_v27 import validate_feedback_pair_v27, local_support_v27


class ObservedPlannerV27(ObservedPlannerV26):
    def observe_transition(self, before, after, selected=None, cues=(), *,
                           support_before=None, support_after=None):
        # Guard sequence and semantic applicability are copied from frozen V26.
        if after.action_id != before.action_id + 1:
            raise ValueError('feedback requires exactly one actually observed paid transition')
        if self.feedback_action is not None and after.action_id <= self.feedback_action:
            raise ValueError('feedback cannot count a paid observation twice')
        validate_feedback_pair_v27(before, after, support_before, support_after)
        old_sample = {p.key: p for p in before.patches}
        sample_improved = [p for p in after.patches if p.key in old_sample
            and (bool(p.bits & ~old_sample[p.key].bits)
                 or p.best_range < old_sample[p.key].best_range-.01)]
        reached = selected is not None and tuple(selected['pose']) == (*after.position, after.heading)
        updates = []
        if reached:
            predicted = {row['cue_id']: row for row in selected.get('semantic_evidence', ())
                         if row['hypothesis_gain'] > 0}
            # CRITICAL: this is the original sampled AFTER state, never full support.
            _, visible_now = self._semantic_gain(after, selected['pose'], cues, {})
            visible_ids = {row['cue_id'] for row in visible_now}
            for cue in cues:
                if (cue.action_id > before.action_id or cue.cue_id not in predicted
                        or cue.cue_id not in visible_ids):
                    continue
                center = np.asarray(cue.center[:2])
                delta = _xy(after, (*after.position, after.heading)) - center
                if not .5 <= np.linalg.norm(delta) <= 2.5:
                    continue
                key = cue.cue_id, _sector(delta)
                if key[1] != predicted[cue.cue_id]['sector']:
                    continue
                counts = local_support_v27(support_before, support_after, center)
                row = dict(cue_id=cue.cue_id, sector=key[1], **counts)
                if counts['comparable_patches'] < 4:
                    row['status'] = 'unavailable_insufficient_common_measured_support'
                elif self.mode != 'S':
                    row['status'] = 'disabled_no_directional_feedback'
                else:
                    observed = min(1., counts['improved_common_patches']/16.)
                    count, total = self.feedback.get(key, (0, 0.))
                    self.feedback[key] = (count+1, total+observed)
                    row.update(status='updated', observed_yield_proxy=observed,
                        feedback_count_before=count, feedback_count_after=count+1,
                        feedback_factor_before=(1.+total)/(1.+count),
                        feedback_factor_after=(1.+total+observed)/(2.+count))
                updates.append(row)
        self.feedback_action = after.action_id
        event = dict(action_id=after.action_id, before_action_id=before.action_id,
            option_id=None if selected is None else selected.get('feedback_option_id'),
            known_cell_delta=int(np.count_nonzero(after.belief != -1)-np.count_nonzero(before.belief != -1)),
            sampled_key_entries=sum(p.key not in old_sample for p in after.patches),
            improved_common_patches=len(sample_improved), selected_endpoint_reached=reached,
            transition_scope='endpoint_last_paid_step' if reached else 'not_at_selected_endpoint',
            directional_updates=updates, evaluation_Q_used=False,
            support_before_sha256=support_before.support_sha256,
            support_after_sha256=support_after.support_sha256,
            feedback_support='full_observed_whitelist_separate_from_256_planning_sample',
            proxy_caveat='adjacent measured support /16; not calibrated outline improvement')
        self._record('IGCR', 'measured_transition_feedback', event)
        return event
