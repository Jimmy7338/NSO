"""Administrative synthetic checks; no real final summary is generated."""
from copy import deepcopy
import unittest

from scripts import summarize_article36_final_20260928 as s
from tests.test_analyze_article_ground_comparisons_20260928 import metrics


def rows():
    result=[]
    for index,version in enumerate(s.VERSIONS):
        for scene in s.SCENES:
            for method in s.METHODS:
                run=version+'_'+scene+'_'+method;slot=dict(scene_id=scene,method=method,budget=160,noise_seed=92801)
                failure=version=='OriginalV1' and scene=='ART1_CELL_DEV' and method=='G'
                row=s.a.blank_row(version,run,slot,dict(status='attempt_failed' if failure else 'controller_stop',qualified=not failure))
                row.update(version=version,review_passed=True)
                s.a.ground.apply_metrics(row,metrics(.4+index*.1),version=s.a.VERSION,motion=True,mode='new_derived_measurement' if failure else 'synthetic')
                row.update(paid_actions=2,path_length_m=.25,turns=1,explicit_observe_actions=0,observation_frames=3)
                result.append(row)
    return result


class FinalSummaryTests(unittest.TestCase):
    def test_every_version_and_signed_comparison_remains(self):
        values=rows();delta=s.differences(values)
        self.assertEqual(len(values),36);self.assertEqual(len(delta),36)
        self.assertEqual(len([r for r in delta if r['comparison']=='ExposureV3-GroundV2']),12)
        failed=next(r for r in values if r['online_status']=='attempt_failed')
        self.assertFalse(failed['original_end_to_end_qualified']);self.assertTrue(failed['motion_completion_verified'])
        pair=next(r for r in delta if r['right_run']==failed['run_id'])
        self.assertTrue(pair['quality_available']);self.assertFalse(pair['right_original_qualified'])
        self.assertAlmostEqual(pair['delta_J_nav'],.08)

    def test_missing_quality_remains_missing_not_zero(self):
        values=rows();row=next(r for r in values if r['version']=='ExposureV3')
        row.update(quality_measurement_available=False,**{k:None for k in s.METRICS})
        affected=[r for r in s.differences(values) if r['left_run']==row['run_id']]
        self.assertEqual(len(affected),2)
        self.assertTrue(all(not r['quality_available'] and r['delta_J_nav'] is None for r in affected))
        self.assertTrue(all(r['delta_path_length_m']==0 for r in affected))

    def test_metric_reference_mismatch_and_duplicate_rows_rejected(self):
        values=rows();values[-1]['reference_fingerprint']='other'
        with self.assertRaises(ValueError):s.differences(values)
        values=rows();values[-1]=deepcopy(values[0])
        with self.assertRaises(ValueError):s.differences(values)

    def test_partial_or_active_inventory_rejected(self):
        slots={scene+method:dict(scene_id=scene,method=method,budget=160,noise_seed=92801) for scene in s.SCENES for method in s.METHODS}
        entries={run:dict(status='controller_stop') for run in slots};protocol=dict(slots=slots)
        s.complete_inventory(protocol,entries)
        first=next(iter(entries));entries[first]['status']='reserved'
        with self.assertRaises(ValueError):s.complete_inventory(protocol,entries)
        entries.pop(first)
        with self.assertRaises(ValueError):s.complete_inventory(protocol,entries)

    def test_physical_paid_cost_rejects_free_or_unaccounted_frames(self):
        physics=dict(action_counts=dict(forward=1,turn_left=1),paid_actions=2,rgbd_frames=3,
            translation_m=.25,budget=160,collisions=0,returned_xy_and_yaw=True)
        self.assertTrue(s.check_physics(physics))
        for altered in [dict(physics,rgbd_frames=4),dict(physics,paid_actions=3),dict(physics,translation_m=float('nan'))]:
            with self.assertRaises(ValueError):s.check_physics(altered)

    def test_report_tables_stay_six_columns_and_cell_failure_is_marked(self):
        values=rows();delta=s.differences(values)
        summary=dict(version_records={v:12 for v in s.VERSIONS},original_qualified=dict(OriginalV1=11,GroundV2=12,ExposureV3=12),
            motion_complete={v:12 for v in s.VERSIONS},quality_available={v:12 for v in s.VERSIONS})
        sb=[dict(version=v,scene_id=scene,pair=dict(delta_J_nav='0',actions_identical='True',
            first_action_divergence_paid_step='',common_observation_prefix_frames='161',mechanism_status='common_actual_prefix_only'))
            for v in s.VERSIONS for scene in s.SCENES]
        text=s.chinese_report(values,delta,sb,summary,s.DEFAULT_OUTPUT,s.STAGE/'synthetic_full_snapshot')
        for line in text.splitlines():
            if line.startswith('|'):self.assertLessEqual(len(line.split('|'))-2,6)
        self.assertIn('|CELL|G|0.320000†|',text)
        self.assertIn('|OriginalV1|12|11|12|12|',text)
        self.assertIn('本报告不自动判定或启动 main',text)


if __name__=='__main__':unittest.main()
