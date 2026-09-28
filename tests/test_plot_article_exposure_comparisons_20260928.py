"""Only synthetic table loading and display selectors; never render a figure."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts import analyze_article_exposure_comparisons_20260928 as analysis
from scripts import plot_article_exposure_comparisons_20260928 as plot
from tests.test_analyze_article_exposure_comparisons_20260928 import rows


class ExposurePlotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic_exposure_plot_',dir=analysis.ROOT/'tmp')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def snapshot(self,*,unqualified=False):
        inventory=rows();row=inventory[0]
        row.update(paid_actions=2,path_length_m=0.,turns=1,explicit_observe_actions=1,
            review_passed=True,trace_status='sealed_trace_available',original_end_to_end_qualified=not unqualified,
            motion_completion_verified=not unqualified,status='reviewed_unqualified' if unqualified else 'reviewed_qualified')
        trajectory=[dict(run_id=row['run_id'],arm=row['arm'],scene_id=row['scene_id'],method=row['method'],
            original_end_to_end_qualified=row['original_end_to_end_qualified'],motion_completion_verified=row['motion_completion_verified'],
            paid_step=i,x_m=0.,y_m=0.,yaw_rad=0.,cumulative_path_m=0.,actual_sensor_action=action)
            for i,action in enumerate(('observe','turn_left','observe'))]
        analysis.old.write_csv(self.root/'slots.csv',inventory,['arm','run_id','status'])
        analysis.old.write_csv(self.root/'trajectories.csv',trajectory,['run_id','paid_step'])
        analysis.old.write_csv(self.root/'paired_effects.csv',[],['comparison'])
        (self.root/'summary.json').write_text('{}\n');self.reseal()
        return row,trajectory

    def reseal(self):
        manifest=dict(schema='article.exposure_comparison_manifest.v1',files={p.name:dict(bytes=p.stat().st_size,sha256=plot.sha(p))
            for p in self.root.iterdir() if p.name!='manifest.json'})
        (self.root/'manifest.json').write_text(json.dumps(manifest))

    def test_full_inventory_loaded_without_missing_results_becoming_zero(self):
        row,_=self.snapshot();slots,traces=plot.verified_snapshot(self.root)
        self.assertEqual(len(slots),24);self.assertEqual(set(traces),{row['run_id']})
        self.assertEqual(sum(r['status']=='unstarted' for r in slots),23)
        self.assertTrue(all(r['J_nav']=='' for r in slots))

    def test_unqualified_but_reviewed_actual_path_remains_visible(self):
        row,_=self.snapshot(unqualified=True);slots,traces=plot.verified_snapshot(self.root)
        self.assertIn(row['run_id'],traces);self.assertEqual(plot.status_label(slots[0]),'incomplete return')
        self.assertFalse(plot.base.has_metric(slots[0]))

    def test_real_sensor_action_names_and_initial_free_frame(self):
        _,trajectory=self.snapshot();features=plot.path_features(trajectory)
        self.assertEqual(features['turns'],[1]);self.assertEqual(features['observations'],[2])
        self.assertEqual(features['xy'].shape,(3,2))

    def test_tampered_csv_and_changed_cumulative_distance_are_rejected(self):
        _,trajectory=self.snapshot();(self.root/'trajectories.csv').write_text('tamper')
        with self.assertRaises(ValueError):plot.verified_snapshot(self.root)
        trajectory[-1]['cumulative_path_m']=10.
        (self.root/'trajectories.csv').unlink()
        analysis.old.write_csv(self.root/'trajectories.csv',trajectory,['run_id','paid_step']);self.reseal()
        with self.assertRaises(ValueError):plot.verified_snapshot(self.root)

    def test_missing_failure_running_statuses_are_distinct(self):
        for status,expected in [('unstarted','unstarted'),('running_reserved','running'),
            ('failed_attempt_reviewed','failed attempt'),('awaiting_independent_review','awaiting review')]:
            self.assertEqual(plot.status_label(dict(status=status)),expected)


if __name__=='__main__':unittest.main()
