"""Synthetic ground-analysis invariants; no experiment or quality evaluation."""
from copy import deepcopy
import io
import unittest

import numpy as np

from scripts import analyze_article_ground_comparisons_20260928 as analysis


def metrics(value=.5):
    return dict(C_nav=.8,macro_precision=.6,macro_completeness=.45,macro_f1=value,Q=value,J_nav=.8*value,
        reference_fingerprint='same-reference',per_instance=[dict(instance_id=i,f1=value) for i in range(4)])


def slots():
    result=[]
    for arm in analysis.ARMS:
        for family in ('AISLE','CELL','LOOP'):
            for method in analysis.METHODS:
                slot=dict(scene_id='ART1_'+family+'_DEV',method=method,budget=160,noise_seed=92801)
                result.append(analysis.blank_row(arm,arm+'_'+family+'_'+method,slot,None))
    return result


def trace(*,right=False,different_actions=True,unmatched=False):
    key='other_id' if right else 'first_id'
    data=dict(result=dict(actions=[dict(sensor_action='forward'),dict(sensor_action='right' if right and different_actions else 'left')]),steps={})
    for i in range(3):
        changed=right and i>=1
        options={('direct','target',0,key):dict(score=.3 if changed else .2,expected_gain=.6 if changed else .4,
            evi=0.,total_cost=2,scheduling_priority=0.)}
        belief=dict(prob=[.4,.3,.2,.1] if changed else [.25]*4,rho=.6 if changed else .5,
            observed_class='cabinet',peers=[])
        data['steps'][i]=dict(observation_sha=('right' if right else 'left')+str(i),
            observation_content_sha=('physical-new' if right and different_actions and i==2 else 'physical')+str(i),
            action='turn_right' if right and different_actions and i==1 else 'turn_left' if i==1 else 'forward',
            beliefs={key:belief},forecasts={key:[dict(view_id='v',gain=.1,structure_gain=[.1]*4,
                fallback=False,reason=None,repeated=False)]},options=options,
            selected=dict(target=dict(node='target',heading=0)),replanned=True,reason='normal',remaining=160-i,
            observed_anchors={key:[1. if unmatched else 0.,0.,1.]})
    return data


class GroundComparisonTests(unittest.TestCase):
    def test_all_twentyfour_arm_rows_and_twentyfour_pairs_retained_when_unstarted(self):
        rows=slots();pairs,events=analysis.pair_rows(rows,{})
        self.assertEqual(len(rows),24);self.assertEqual(len(pairs),24)
        self.assertTrue(all(r['quality_status']=='unavailable' for r in pairs));self.assertFalse(events)
        summary=analysis.layout_summary(pairs)
        self.assertEqual(len(summary),8)
        self.assertTrue(all(r['expected_layout_units']==3 and r['available_layout_units']==0 for r in summary.values()))

    def test_missing_running_failed_slots_never_become_zero_metrics(self):
        slot=dict(scene_id='ART1_CELL_DEV',method='G',budget=160,noise_seed=92801)
        for entry,status in [(None,'unstarted'),(dict(status='reserved'),'running_reserved'),
            (dict(status='attempt_failed'),'awaiting_independent_review')]:
            row=analysis.blank_row('GroundOff','cell',slot,entry)
            self.assertEqual(row['status'],status)
            self.assertTrue(all(row[k] is None for k in analysis.METRICS))

    def test_original_failed_motion_complete_can_have_derived_score_without_status_repair(self):
        row=slots()[0];row.update(online_status='attempt_failed',original_end_to_end_qualified=False)
        analysis.apply_metrics(row,metrics(),version=analysis.VERSION,motion=True,mode='new_derived_measurement')
        self.assertEqual(row['online_status'],'attempt_failed');self.assertFalse(row['original_end_to_end_qualified'])
        self.assertTrue(row['motion_completion_verified']);self.assertTrue(row['quality_measurement_available'])
        self.assertEqual(row['J_nav'],.4)

    def test_metric_version_and_reference_mismatch_cannot_make_paired_effect(self):
        rows=slots();chosen=[r for r in rows if r['scene_id']=='ART1_AISLE_DEV' and r['method']=='S']
        for row in chosen:analysis.apply_metrics(row,metrics(),version=analysis.VERSION,motion=True,mode='derived')
        chosen[0]['reference_fingerprint']='different'
        pairs,_=analysis.pair_rows(rows,{})
        pair=next(r for r in pairs if r['comparison']=='GroundOn-GroundOff/S' and r['scene_id']=='ART1_AISLE_DEV')
        self.assertEqual(pair['quality_status'],'incompatible_metric_or_reference');self.assertNotIn('delta_J_nav',pair)
        with self.assertRaises(ValueError):analysis.apply_metrics(chosen[0],metrics(),version='old-unadapted',motion=True,mode='old')

    def test_negative_effect_preserved_and_incomplete_layouts_have_no_pooled_mean(self):
        rows=slots()
        for row in rows:
            if row['scene_id']=='ART1_AISLE_DEV' and row['method']=='S':
                analysis.apply_metrics(row,metrics(.4 if row['arm']=='GroundOn' else .6),version=analysis.VERSION,motion=True,mode='derived')
        pairs,_=analysis.pair_rows(rows,{})
        group=analysis.layout_summary(pairs)['GroundOn-GroundOff/S']
        self.assertEqual(group['available_layout_units'],1)
        self.assertLess(group['paired_layout_deltas'][0]['J_nav'],0)
        self.assertIsNone(group['all_layouts_mean_delta']['J_nav'])

    def test_same_prefix_score_posterior_and_action_witness_ignore_later_trajectory(self):
        result,witness=analysis.compact_comparison(trace(),trace(right=True))
        self.assertEqual(result['first_action_divergence_paid_step'],2)
        self.assertEqual(result['first_observation_divergence_paid_step'],2)
        self.assertEqual(result['common_observation_prefix_frames'],2)
        self.assertEqual(result['first_changed_score_paid_step'],1)
        self.assertEqual(result['first_changed_posterior_paid_step'],1)
        self.assertEqual(result['changed_posterior_instance_steps'],1)
        self.assertTrue(witness['first_action_witness']['same_physical_observation'])
        self.assertEqual(witness['first_action_witness']['decision_paid_step'],1)

    def test_observed_anchor_mapping_handles_renamed_ids_without_using_class(self):
        left=trace();right=trace(right=True,different_actions=False)
        result,_=analysis.compact_comparison(left,right)
        self.assertGreater(result['changed_common_candidate_scores'],0)
        self.assertEqual(result['unmatched_right_instance_prefix_frames'],0)
        right=trace(right=True,different_actions=False,unmatched=True)
        result,_=analysis.compact_comparison(left,right)
        self.assertEqual(result['changed_posterior_instance_steps'],0)
        self.assertEqual(result['numeric_common_candidate_scores'],0)
        self.assertEqual(result['unmatched_right_instance_prefix_frames'],3)

    def test_ambiguous_observed_anchors_are_not_forced_to_match(self):
        self.assertEqual(analysis.observed_identity_pairs({'a':[0,0,0],'b':[.1,0,0]},{'r':[.05,0,0]}),{})

    def test_cross_run_sensor_equality_excludes_only_frame_id(self):
        def npz(**fields):
            out=io.BytesIO();np.savez_compressed(out,**fields);return out.getvalue()
        fields=dict(frame_id=np.array('left'),paid_step=np.array(0),rgb=np.zeros((2,2,3),np.uint8),
            depth_m=np.ones((2,2),np.float32),intrinsic=np.eye(3),world_from_camera=np.eye(4))
        scan=npz(timestamp_s=np.array(0.),ranges_m=np.ones(2),angle_min_rad=np.array(0.),angle_increment_rad=np.array(.1),
            range_max_m=np.array(10.),world_from_laser=np.eye(4))
        original=analysis.old.sensor_content_sha(npz(**fields),scan)
        fields['frame_id']=np.array('right')
        self.assertEqual(original,analysis.old.sensor_content_sha(npz(**fields),scan))
        fields['world_from_camera'][0,3]=.25
        self.assertNotEqual(original,analysis.old.sensor_content_sha(npz(**fields),scan))

    def test_macro_f1_not_harmonic_mean_of_macro_precision_and_recall(self):
        row=slots()[0];value=metrics(.5)
        analysis.apply_metrics(row,value,version=analysis.VERSION,motion=True,mode='derived')
        self.assertEqual(row['F1'],.5)
        value['per_instance'][0]['f1']=.1
        with self.assertRaises(ValueError):analysis.apply_metrics(row,value,version=analysis.VERSION,motion=True,mode='derived')


if __name__=='__main__':unittest.main()
