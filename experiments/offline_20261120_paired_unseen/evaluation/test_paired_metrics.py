import unittest
import numpy as np
import paired_metrics as m

def pred(ids=(0,1,2),values=(1.,2.,3.)):
 return [dict(candidate_id=i,predicted_gain_m2=v,common_score=.5,score=v+.5,cost=10+i)for i,v in zip(ids,values)]
class MetricsTests(unittest.TestCase):
 def test_same_reference_gross_lost_net(self):
  r=m.surface_increment([np.inf,0,.02,.2],[0,.2,.02,.3],[.2,.3,.1,.4]);self.assertAlmostEqual(r['gross_new_area_m2'],.2);self.assertAlmostEqual(r['lost_area_m2'],.3);self.assertAlmostEqual(r['net_new_area_m2'],-.1)
 def test_closed_5cm_boundary_and_empty(self):
  r=m.surface_increment([np.inf],[.05],[2.]);self.assertEqual(r['gross_new_area_m2'],2.);self.assertIsNone(m.surface_increment([],[],[])['after_recall'])
 def test_identical_reconstruction_zero(self):
  r=m.surface_increment([0,np.inf,.2],[0,np.inf,.2],[1,1,1]);self.assertEqual(r['net_new_area_m2'],0.)
 def test_probability_weighted_area_conservation(self):
  r=m.surface_increment([.2,.2],[0,0],[.1,.9]);self.assertEqual(r['gross_new_area_m2'],1.)
 def test_null_failure_not_zero(self):
  a={0:dict(status='completed',object_gross_new_m2=.4),1:dict(status='failed'),2:dict(status='completed',object_gross_new_m2=.8)}
  r=m.summarize_condition(pred(values=(1,3,2)),a);self.assertIsNone(r['selected_observed_area_m2']);self.assertIsNone(r['observed_gap_m2']);self.assertFalse(r['candidate_best_fully_observed']);self.assertEqual(r['missing_area_candidates'],[1])
 def test_partial_surface_kept_without_success(self):
  a={0:dict(status='completed',object_gross_new_m2=.4),1:dict(status='partial',object_gross_new_m2=.7),2:dict(status='completed',object_gross_new_m2=.8)}
  r=m.summarize_condition(pred(values=(1,3,2)),a);self.assertEqual(r['selected_observed_area_m2'],.7);self.assertFalse(r['candidate_best_fully_observed']);self.assertEqual(r['selected_status'],'partial')
 def test_shared_actual_best_and_selected_delta(self):
  a={i:dict(status='completed',object_gross_new_m2=v)for i,v in enumerate([.2,.8,.5])};r=m.paired_summary(pred(values=(3,2,1)),pred(values=(1,3,2)),a)
  self.assertAlmostEqual(r['new_minus_old_selected_area_m2'],.6);self.assertEqual(r['new']['observed_gap_m2'],0.);self.assertFalse(r['target_unchanged'])
 def test_unchanged_and_constant_rank(self):
  a={i:dict(status='completed',object_gross_new_m2=.5)for i in range(3)};r=m.paired_summary(pred(),pred(),a);self.assertTrue(r['target_unchanged']);self.assertIsNone(r['old']['template_rank_rho'])
 def test_tie_original_cost_then_ID(self):
  rows=[dict(candidate_id=3,score=1,cost=4),dict(candidate_id=2,score=1,cost=4),dict(candidate_id=0,score=1,cost=5)];self.assertEqual(m.winner(rows),2)
 def test_no_mismatched_candidates_or_common(self):
  with self.assertRaises(ValueError):m.paired_summary(pred(),pred(ids=(2,1,0)),{})
  b=pred();b[0]['common_score']=.7
  with self.assertRaises(ValueError):m.paired_summary(pred(),b,{})
 def test_layout_missing_not_silently_mean_dropped(self):
  a={i:dict(status='completed',object_gross_new_m2=.5)for i in range(3)};x=m.paired_summary(pred(),pred(),a);b={i:dict(status='failed')for i in range(3)};y=m.paired_summary(pred(),pred(),b);r=m.summarize_layout([x,y]);self.assertIsNone(r['old_selected_area_m2']);self.assertEqual(r['states'],2)
 def test_invalid_reference_inputs(self):
  for b,a,w in [([np.nan],[1],[1]),([1],[1],[-1]),([1],[1,2],[1])]:
   with self.assertRaises(ValueError):m.surface_increment(b,a,w)
if __name__=='__main__':unittest.main()
