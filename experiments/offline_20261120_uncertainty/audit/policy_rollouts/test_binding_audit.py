import copy,unittest
from evaluate_batch import STUDY,ARMS,read,sha,audit_call,expected_mode

def actual_call(mode):
    p=STUDY/'replay/shelf_left_24_attempt_001'
    models={n:read(p/(n+'_'+mode+'_predictions.json'))['model_audits'][n]for n in ('G','S')}
    return dict(gain_implementation='B8_conditional_joint_particles_with_B2_visibility_and_active_only_exact_gain',
        model_source_sha256=sha(STUDY/'method/model.py'),selected_scale_sha256=sha(STUDY/'audit/calibration_attempt_001/selected_scale.json'),
        sigma_m=.01,shared_observed_instance_support=True,shared_particle_bank=True,shared_G_S_point_basis=True,
        mean_collapse_likelihood_recomputed=False,model_audits=models,reused_original_gain_function=False,source_gain_new_calls={'G':1,'S':1})

class BindingAuditTests(unittest.TestCase):
    def test_five_modes(self):
        self.assertEqual([expected_mode(a)for a in ARMS],['fixed','fixed','uncertainty','uncertainty','mean_collapse'])
    def test_frozen_nonempty_shared_banks_accepted(self):
        for mode in ('fixed','uncertainty'):
            c=actual_call(mode);self.assertTrue(c['model_audits']['G']['objects']);audit_call(c,'S_'+mode)
    def test_double_likelihood_rejected(self):
        c=actual_call('uncertainty');c['model_audits']['S']['old_surface_likelihood_reapplied']=True
        with self.assertRaises(AssertionError):audit_call(c,'S_uncertainty')
    def test_different_observed_support_rejected(self):
        c=actual_call('fixed');c['model_audits']['S']['objects'][0]['fit_input_points_sha256']='changed'
        with self.assertRaises(AssertionError):audit_call(c,'G_fixed')
    def test_scale_change_rejected(self):
        c=actual_call('fixed');c['sigma_m']=.06
        with self.assertRaises(AssertionError):audit_call(c,'S_fixed')
if __name__=='__main__':unittest.main()
