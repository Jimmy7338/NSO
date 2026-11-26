import copy,unittest
import numpy as np
import evaluate_batch as e
class TestIndependentPolicyAudit(unittest.TestCase):
 def setUp(self):
  self.models={k:v['model_audit'] for k,v in e.read(e.B9/'structure/replay_attempt_001/shelf_left_24/result.json')['arms'].items()}
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


class TestNoveltyAudit(unittest.TestCase):
 def setUp(self):
  path=e.B9/'structure/rollouts/temperature/layouts/shelf_left/runs/G_temperature/attempt_001/template_audit.json'
  calls=e.read(path);self.call=next(c for c in calls if c['model_audits']['G'].get('objects'))
  c=self.call;c['gain_implementation']='B10_native5cm_novelty_on_B9_T16_B8_particles_B2_visibility'
  c['declared_reward_revision']=True;c['shared_G_S_novelty_mask']=True
  c['native_novelty_source_sha256']=e.sha(e.STUDY/'method/measured_novelty.py')
  c['native_novelty_receipts']={}
  for name in ('G','S'):
   r=dict(schema='B10_native_mesh_novelty_v1',tolerance_m=.05,decision_affecting_reward_revision=True,probability_geometry_history_unchanged=True,source_geometry_consumed=False,template_points=c['model_audits'][name]['raw_template_points'],expected_area_before_m2=2.,expected_area_after_m2=1.5,consumed_expected_area_m2=.5,measured_mesh_empty=False)
   c['model_audits'][name]['native_mesh_novelty']=r.copy()
   c['native_novelty_receipts'][name]=dict(r,mask_sha256='shared-test-mask')
 def test_valid_receipt_with_actual_nonempty_joint_model(self):e.audit_call(self.call,'G_native_novelty')
 def test_changed_S_mask_is_rejected(self):
  self.call['native_novelty_receipts']['S']['mask_sha256']='different'
  with self.assertRaises(AssertionError):e.audit_call(self.call,'G_native_novelty')
 def test_inconsistent_consumption_is_rejected(self):
  self.call['native_novelty_receipts']['G']['consumed_expected_area_m2']=.4
  with self.assertRaises(AssertionError):e.audit_call(self.call,'G_native_novelty')
if __name__=='__main__':unittest.main()
