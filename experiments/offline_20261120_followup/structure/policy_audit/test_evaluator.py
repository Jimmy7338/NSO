import copy,unittest
import numpy as np
import evaluate_batch as e
class TestIndependentPolicyAudit(unittest.TestCase):
 def setUp(self):
  self.models={k:v['model_audit'] for k,v in e.read(e.STUDY/'structure/replay_attempt_001/shelf_left_24/result.json')['arms'].items()}
 def test_frozen_initial_models(self):e.audit_models(self.models)
 def test_wrong_temperature_rejected(self):
  self.models['S']['structural_temperature']=1
  with self.assertRaises(AssertionError):e.audit_models(self.models)
 def test_mixture_tamper_rejected(self):
  o=self.models['S']['objects'][0];o['joint_probability_mass'][0][0]+=.01
  with self.assertRaises(AssertionError):e.audit_models(self.models)
 def test_G_prior_tamper_rejected(self):
  self.models['G']['objects'][0]['shape_prior']=[.9,.1]
  with self.assertRaises(AssertionError):e.audit_models(self.models)
 def test_conditional_geometry_remains_independent_of_T(self):
  o=self.models['S']['objects'][0];a=e.independent_joint(o,1.);b=e.independent_joint(o,16.)
  for h in range(2):
   if a[:,h].sum()>0 and b[:,h].sum()>0:np.testing.assert_allclose(a[:,h]/a[:,h].sum(),b[:,h]/b[:,h].sum(),atol=1e-14,rtol=1e-12)
 def test_signed_comparison_retained(self):
  r=e.read(e.B8/'audit/policy_rollouts/summary.json')['rows']['shelf_left']['S_uncertainty'];w=copy.deepcopy(r)
  w['surface_metrics']['f1_05cm']-=.01;w['path_length_m']+=.2;w['first_coverage_action']['1.0']+=5
  d=e.paired(r,w);self.assertAlmostEqual(d['F1_5cm_delta_pp'],-1.);self.assertAlmostEqual(d['path_length_delta_m'],.2);self.assertEqual(d['first_coverage_action_delta']['1.0'],5)
if __name__=='__main__':unittest.main()
