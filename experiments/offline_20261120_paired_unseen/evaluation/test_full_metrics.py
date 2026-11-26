"""Full-loop metric and posterior audit checks, no actor actions."""
import copy,unittest
import numpy as np
from evaluate_full_run import check_models,coverage_from_belief,first_correct_free_actions

def call_fixture():
 q=np.array([.4,.6,1e-18]);energy=np.array([[.002,.003],[.003,.002],[.004,.001]])
 models={}
 for name,sp in [('G',np.array([.5,.5])),('S',np.array([.1,.9]))]:
  a=np.log(q[:,None])-energy/(2*.01**2);maximum=a.max(0);v=np.exp(a-maximum);conditional=v/v.sum(0);z=maximum+np.log(v.sum(0));logit=(z[1]-z[0])/16+np.log(sp[1]/sp[0]);shape=np.exp([-np.logaddexp(0,logit),-np.logaddexp(0,-logit)])
  obj=dict(support_track=1,particles=[1,2,3],prior_mass=q.tolist(),energy_m2=energy.tolist(),fit_input_points_sha256='p',fit_input_normals_sha256='n',fit_input_labels_sha256='l',conditional_geometry=conditional.tolist(),shape_prior=sp.tolist(),joint_probability_mass=(conditional*shape).tolist())
  models[name]=dict(extent_mode='uncertainty',sigma_m=.01,structural_temperature=16.,applied_global_point_cap=None,source_geometry_consumed=False,old_surface_likelihood_reapplied=False,objects=[obj])
 return dict(B11_predictor='visibility',extent_mode='uncertainty',sigma_m=.01,structural_temperature=16.,conditional_mean_or_fine_basis=False,shared_G_S_full_basis=True,original_candidate_provider=True,full_policy_cost_band_applied=False,model_audits=models,candidate_count=3,acquisition_diagnostics=None)

class TestFullMetrics(unittest.TestCase):
 def test_prior_temperature_full_mixture_valid(self):check_models(call_fixture())
 def test_missing_positive_mass_rejected(self):
  c=call_fixture();c['model_audits']['S']['objects'][0]['joint_probability_mass'][0][0]+=.001
  with self.assertRaises(AssertionError):check_models(c)
 def test_shared_fitted_geometry_required(self):
  c=call_fixture();c['model_audits']['S']['objects'][0]['fit_input_points_sha256']='other'
  with self.assertRaises(AssertionError):check_models(c)
 def test_neutral_prior_must_be_neutral(self):
  c=call_fixture();c['model_audits']['G']['objects'][0]['shape_prior']=[.3,.7]
  with self.assertRaises(AssertionError):check_models(c)
 def test_no_conditional_mean_or_cost_gate(self):
  for key in ['conditional_mean_or_fine_basis','full_policy_cost_band_applied']:
   c=call_fixture();c[key]=True
   with self.assertRaises(AssertionError):check_models(c)
 def test_acquisition_without_candidates_can_have_no_engine(self):
  c=call_fixture();c.update(B11_predictor='acquisition',candidate_count=0);check_models(c)
  c['candidate_count']=1
  with self.assertRaises(AssertionError):check_models(c)
 def test_valid_acquisition_contract(self):
  c=call_fixture();c.update(B11_predictor='acquisition',acquisition_diagnostics=dict(original_basis_and_posterior_unchanged=True,all_past_fused=True,no_positive_probability_pruning=True,geometry_collapse=False,source_geometry_read=False,policy_candidates_changed=False));check_models(c)
  c['acquisition_diagnostics']['no_positive_probability_pruning']=False
  with self.assertRaises(AssertionError):check_models(c)
 def test_correct_free_not_merely_known(self):
  b=np.array([[0,1,-1],[0,0,1]]);r=np.array([[1,1,1],[0,0,1]],bool);x=coverage_from_belief(b,r)
  self.assertEqual(x['reference_free_cells'],4);self.assertEqual(x['known_cells'],3);self.assertEqual(x['correctly_free_cells'],1);self.assertEqual(x['false_occupied_cells'],2);self.assertEqual(x['free_correct_fraction'],.25)
 def test_reachable_boolean_required(self):
  with self.assertRaises(AssertionError):coverage_from_belief(np.array([[0]]),np.array([[1]]))
 def test_threshold_first_hit_initial_and_censor(self):
  trace=[dict(step=i+1,free_correct_fraction=x)for i,x in enumerate([.95,.94,.99])]
  self.assertEqual(first_correct_free_actions(.2,trace),{'0.95':1,'1.0':None});self.assertEqual(first_correct_free_actions(1,[]),{'0.95':0,'1.0':0})
 def test_nonconsecutive_trace_rejected(self):
  with self.assertRaises(AssertionError):first_correct_free_actions(0,[dict(step=2,free_correct_fraction=1)])
if __name__=='__main__':unittest.main()
