"""V27-A analytic support and runtime contracts; no World or TSDF integration.

Handwritten packets and the existing NON-FUSING unit mapper are reused as test
fixtures only. No saved experiment packet, mesh or evaluation is loaded.
"""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

from nso.observed_feedback_v27 import (make_feedback_support_v27, capture_feedback_support_v27,
    local_support_v27)
from nso.observed_planner_v26 import ObservedPlannerV26
from nso.observed_planner_v27 import ObservedPlannerV27
from nso.observed_runtime_v27 import ObservedANSRuntimeV27
from nso.observed_state_v26 import GeometryPatchV26, geometry_state_v26
from tests.virtual3d.test_observed_planner_v26 import state_for, cue_for
from tests.virtual3d.test_observed_runtime_v26 import NonFusingMapper, SHAPE, START, config, packet


def patches(n=6, *, bits=1, distance=3., first=0, point=(4.3, 4.3, .8)):
    return tuple(GeometryPatchV26((first+i, 1, 1), point, (0., -1., 0.), 1, bits, distance, 0.)
                 for i in range(n))


def pair(before_patches, after_patches, *, planning_patches=()):
    before = replace(state_for(position=(31, 21)), patches=planning_patches)
    after = replace(before, action_id=1, position=(30, 21), remaining_budget=99)
    return (before, after,
        make_feedback_support_v27(0, before.geometry_sha256, before_patches),
        make_feedback_support_v27(1, after.geometry_sha256, after_patches))


def selected_for(planner, before, after, cue):
    _, evidence = planner._semantic_gain(before, (*after.position, after.heading), (cue,), {})
    assert evidence and evidence[0]['hypothesis_gain'] > 0
    return dict(pose=[*after.position, after.heading], semantic_evidence=evidence, feedback_option_id=1)


def feedback(planner, values, cue=None):
    before, after, old, new = values
    cue = cue or cue_for()
    return planner.observe_transition(before, after, selected_for(planner, before, after, cue),
        (cue,), support_before=old, support_after=new)


class GeometryOnlyRow(dict):
    def __getitem__(self, key):
        if key not in ('point', 'normal', 'n', 'bits', 'best_range', 'residual'):
            raise AssertionError('A category/owner/nongeometry field was read')
        return super().__getitem__(key)


class MeasuredFixtureMapper(NonFusingMapper):
    def __init__(self, shape, sensor_config):
        super().__init__(shape, sensor_config)
        self.quality = {(i, 1, 1): GeometryOnlyRow(point=(1.+(i % 20)*.2, 1.+(i//20)*.2, .8),
            normal=(0., -1., 0.), n=1, bits=1, best_range=3., residual=0., label='forbidden')
            for i in range(400)}

    def update(self, frame, scan):
        super().update(frame, scan)
        if self.frames > 1:
            for row in self.quality.values():
                row['bits'] = 5


class ObservedFeedbackV27Tests(unittest.TestCase):
    def runtime(self, mode='S', replan_interval=5):
        with patch('nso.observed_runtime_v26.ObservedRuntimeMapperV10', MeasuredFixtureMapper):
            return ObservedANSRuntimeV27(SHAPE, config(), 20, mode, replan_interval)

    def issue(self, runtime, pose, *, semantic=True):
        evidence = []
        if semantic:
            _, evidence = runtime.planner._semantic_gain(runtime.state, pose, runtime.cues, {})
        selected = dict(pose=list(pose), group='analytic_paid_option', semantic_evidence=evidence)
        with patch.object(runtime.planner, 'plan', return_value=dict(selected=selected,
                candidates=[selected], audit={'analytic_fixture': True})):
            return runtime.next_action()

    def test_remote_growth_and_class_swap_do_not_erase_local_measured_support(self):
        old = patches(); improved = patches(bits=5, distance=2.)
        base = pair(old, improved)
        grown = pair(old, improved+patches(1000, first=1000, point=(7., 7., .8)))
        first = feedback(ObservedPlannerV27('S'), base)['directional_updates'][0]
        second = feedback(ObservedPlannerV27('S'), grown)['directional_updates'][0]
        for field in ('comparable_patches', 'improved_common_patches', 'observed_yield_proxy'):
            self.assertEqual(first[field], second[field])
        self.assertEqual(first['comparable_patches'], 6)
        self.assertEqual(first['improved_common_patches'], 6)
        self.assertEqual(second['new_keys'], 1000)
        swapped = feedback(ObservedPlannerV27('S'), grown, cue_for(class_id=2))['directional_updates'][0]
        for field in ('comparable_patches', 'improved_common_patches', 'observed_yield_proxy'):
            self.assertEqual(swapped[field], second[field])
        # Concrete regression: identical measured geometry, disjoint planning samples.
        before, after, old_full, new_full = pair(old, improved, planning_patches=old)
        after = replace(after, patches=patches(first=900))
        legacy = ObservedPlannerV26('S')
        selected = selected_for(legacy, before, after, cue_for())
        rejected = legacy.observe_transition(before, after, selected, (cue_for(),))
        self.assertEqual(rejected['directional_updates'][0]['comparable_patches'], 0)
        self.assertEqual(rejected['directional_updates'][0]['status'],
                         'unavailable_insufficient_common_measured_support')
        accepted = ObservedPlannerV27('S').observe_transition(before, after, selected, (cue_for(),),
            support_before=old_full, support_after=new_full)
        self.assertEqual(accepted['directional_updates'][0]['status'], 'updated')
        self.assertEqual(accepted['directional_updates'][0]['comparable_patches'], 6)

    def test_new_keys_and_three_common_are_unavailable_but_measured_zero_is_valid(self):
        for count in (0, 3):
            planner = ObservedPlannerV27('S')
            values = pair(patches(count), patches(count, bits=5)+patches(100, first=100))
            row = feedback(planner, values)['directional_updates'][0]
            self.assertEqual(row['status'], 'unavailable_insufficient_common_measured_support')
            self.assertEqual(row['comparable_patches'], count)
            self.assertEqual(planner.feedback, {})
        planner = ObservedPlannerV27('S')
        row = feedback(planner, pair(patches(4), patches(4)))['directional_updates'][0]
        self.assertEqual(row['status'], 'updated')
        self.assertEqual(row['observed_yield_proxy'], 0.)
        self.assertEqual(row['feedback_count_after'], 1)
        self.assertEqual(row['feedback_factor_after'], .5)

    def test_after_point_radius_old_cue_threshold_and_union_are_exact(self):
        old = patches(1, point=(5.81, 4.3, .8))
        after = patches(1, point=(5.79, 4.3, .8), bits=5, distance=2.)
        values = pair(old, after)
        row = local_support_v27(values[2], values[3], cue_for().center)
        self.assertEqual(row['comparable_patches'], 1)  # old point was outside radius
        self.assertEqual(row['improved_common_patches'], 1)
        self.assertEqual(row['both_improved_patches'], 1)  # no double credit
        self.assertEqual(local_support_v27(values[2], values[3], (0., 0.))['comparable_patches'], 0)
        departed = pair(after, old)
        self.assertEqual(local_support_v27(departed[2], departed[3], cue_for().center)['comparable_patches'], 0)
        old = patches(4, distance=3.)
        for distance, expected in ((3.-.01, 0), (3.-.010001, 4)):
            a, b, x, y = pair(old, patches(4, distance=distance))
            self.assertEqual(local_support_v27(x, y, cue_for().center)['improved_common_patches'], expected)

    def test_capture_is_full_immutable_geometry_and_preserves_256_planning_view(self):
        mapper = MeasuredFixtureMapper(SHAPE, config()); initial = packet()
        mapper.update(initial.frame, initial.scan)
        planning = geometry_state_v26(mapper, initial, (*START, 0), 20)
        support = capture_feedback_support_v27(mapper, initial, (*START, 0), 20, planning)
        self.assertEqual(len(planning.patches), 256); self.assertEqual(len(support.patches), 400)
        self.assertNotEqual(support.support_sha256, planning.geometry_sha256)
        mapper.quality[(0, 1, 1)]['bits'] = 7
        self.assertEqual(support.patches[0].bits, 1)
        self.assertEqual(len(planning.patches), 256)
        with self.assertRaisesRegex(ValueError, 'reproduce'):
            capture_feedback_support_v27(mapper, initial, (*START, 0), 20, planning)

    def test_visibility_still_receives_sampled_state_and_disabled_feedback_never_writes(self):
        values = pair(patches(6), patches(6, bits=5), planning_patches=patches(2))
        before, after, old, new = values
        planner = ObservedPlannerV27('S')
        selected = selected_for(planner, before, after, cue_for())
        seen = []
        real = planner._semantic_gain
        def observe(state, *args):
            seen.append(state)
            return real(state, *args)
        with patch.object(planner, '_semantic_gain', side_effect=observe):
            event = planner.observe_transition(before, after, selected, (cue_for(),),
                support_before=old, support_after=new)
        self.assertEqual(len(seen), 1); self.assertIs(seen[0], after)
        self.assertEqual(len(seen[0].patches), 2)
        self.assertEqual(event['directional_updates'][0]['comparable_patches'], 6)
        disabled = ObservedPlannerV27('S_no_feedback')
        gain_before = disabled._semantic_gain(after, selected['pose'], (cue_for(),), {})
        row = feedback(disabled, values)['directional_updates'][0]
        self.assertEqual(row['status'], 'disabled_no_directional_feedback')
        self.assertEqual(disabled.feedback, {})
        self.assertEqual(disabled._semantic_gain(after, selected['pose'], (cue_for(),), {}), gain_before)
        # Future cues, wrong sectors, no arrival, and known occlusion cannot bypass original gates.
        for kind in ('future_cue', 'wrong_sector', 'not_arrived', 'wall'):
            p = ObservedPlannerV27('S'); a = after; c = cue_for(); s = dict(selected)
            if kind == 'future_cue': c = cue_for(action=1)
            if kind == 'wrong_sector': s['semantic_evidence'] = [dict(e, sector=(e['sector']+1)%8) for e in selected['semantic_evidence']]
            if kind == 'not_arrived': s['pose'] = [29, 21, 0]
            if kind == 'wall':
                belief = after.belief.copy(); belief[27, :] = 1
                a = replace(after, belief=belief, geometry_sha256='blocked-analytic-state')
            n = make_feedback_support_v27(1, a.geometry_sha256, new.patches)
            event = p.observe_transition(before, a, s, (c,), support_before=old, support_after=n)
            self.assertFalse(event['directional_updates']); self.assertFalse(p.feedback)

    def test_original_plans_equal_when_feedback_table_equal_and_g_rejects_cues(self):
        def stable(value):
            if isinstance(value, dict):return {k:stable(v) for k,v in value.items() if k!='planning_seconds'}
            if isinstance(value, list):return [stable(v) for v in value]
            return value
        observed = state_for()
        for mode in ('G', 'S'):
            cues = () if mode=='G' else (cue_for(),)
            self.assertEqual(stable(ObservedPlannerV26(mode).plan(observed, cues)),
                             stable(ObservedPlannerV27(mode).plan(observed, cues)))
        with self.assertRaisesRegex(ValueError, 'must not receive'):
            ObservedPlannerV27('G').plan(observed, (cue_for(),))

    def test_runtime_paid_arrival_then_replan_preserves_authorization_and_single_update(self):
        runtime = self.runtime()
        with self.assertRaisesRegex(ValueError, 'action-zero'):
            runtime.accept(packet(2, action='forward'))
        self.assertEqual(runtime.mapper.frames, 0)
        runtime.accept(packet(marker=True))
        self.assertEqual(len(runtime.state.patches), 256)
        self.assertEqual(len(runtime.feedback_support.patches), 400)
        self.assertFalse(runtime.planner.feedback)
        self.assertEqual(self.issue(runtime, (14, 15, 0)), 'forward')
        for invalid in (packet(1, action='forward', position=START),
                packet(1, action='right', heading=1), packet(2, action='forward', position=(14,15)),
                replace(packet(1, action='forward', position=(14,15)), episode_id='wrong-episode')):
            with self.assertRaises(ValueError):runtime.accept(invalid)
            self.assertEqual(runtime.mapper.frames, 1)
        runtime.accept(packet(1, position=(14,15), action='forward', marker=True))
        event = runtime.planner.calls[-1]
        self.assertEqual(event['option_id'], 1)
        self.assertTrue(event['selected_endpoint_reached'])
        self.assertEqual(event['directional_updates'][0]['status'], 'updated')
        self.assertEqual(runtime.mapper.update_times, [0.,1.])
        self.assertEqual(runtime.feedback_lifecycle[-1]['reason'], 'reached_after_feedback')
        count = dict(runtime.planner.feedback)
        with self.assertRaises(ValueError):runtime.accept(packet(1, position=(14,15), action='forward'))
        self.assertEqual(runtime.planner.feedback, count)
        # A paid turn still advances support exactly once, with no invented whole-option reward.
        self.assertEqual(self.issue(runtime, (14,15,1), semantic=False), 'right')
        runtime.accept(packet(2, position=(14,15), heading=1, action='right'))
        self.assertEqual(runtime.mapper.update_times, [0.,1.,2.])
        self.assertEqual(runtime.planner.calls[-1]['action_id'], 2)
        self.assertFalse(runtime.planner.calls[-1]['directional_updates'])
        self.assertFalse(runtime.summary()['whole_option_reward'])

    def test_replanning_cancellation_collision_and_support_identity_cannot_create_credit(self):
        runtime = self.runtime(replan_interval=1);runtime.accept(packet(marker=True))
        self.assertEqual(self.issue(runtime, (12,15,0)), 'forward')
        runtime.accept(packet(1, position=(14,15), action='forward'))
        self.assertFalse(runtime.planner.feedback)
        self.issue(runtime, (13,15,0))
        closed = [r for r in runtime.feedback_lifecycle if r['event']=='closed']
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]['reason'], 'replanned_after_previous_paid_feedback')
        self.assertFalse(closed[0]['cancellation_reward_added'])
        runtime.accept(packet(2, position=(14,15), action='forward', collision=True))
        self.assertEqual(runtime.mapper.frames, 3)
        self.assertFalse(runtime.planner.feedback)
        self.assertEqual(runtime.planner.calls[-1]['operation'], 'collision_transition_no_gain_calibration')
        self.assertEqual(runtime.feedback_lifecycle[-1]['reason'], 'collision_observed')
        before, after, old, new = pair(patches(), patches(bits=5))
        p = ObservedPlannerV27('S'); selected = selected_for(p,before,after,cue_for())
        for invalid in (replace(new,action_id=2), replace(new,planning_geometry_sha256='wrong-state')):
            with self.assertRaisesRegex(ValueError, 'must match'):
                p.observe_transition(before,after,selected,(cue_for(),),support_before=old,support_after=invalid)
        p.observe_transition(before,after,selected,(cue_for(),),support_before=old,support_after=new)
        with self.assertRaisesRegex(ValueError, 'twice'):
            p.observe_transition(before,after,selected,(cue_for(),),support_before=old,support_after=new)


if __name__ == '__main__':
    unittest.main()
