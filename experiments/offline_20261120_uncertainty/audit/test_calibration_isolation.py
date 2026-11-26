import json,tempfile,unittest
from pathlib import Path
from evaluate_calibration import choose_scale,load_cases,prediction_bank,sha,case_metrics

def write_resource(path,data):
    path.write_text(json.dumps(data));return {'path':str(path),'sha256':sha(path)}

def prediction():
    return dict(schema='B8_joint_depth_particle_distribution_v1',semantic=False,extent_mode='uncertainty',old_surface_likelihood_applied_again=False,source_geometry_consumed=False,unmodeled_tracks=[],tracks=[dict(support_track=1,status={'distribution_calibratable':True},particles=[{'depth_m':.3,'prior_mass':.5},{'depth_m':.8,'prior_mass':.5}],prior_mass=[.5,.5],surface_energy_m2=[[0,0],[0,0]],free_energy_m2=[[0,0],[0,0]],energy_m2=[[0,0],[0,0]],shape_prior=[.5,.5],depth_axis_world=[0,1,0])])

class IsolationTests(unittest.TestCase):
    def test_tie_chooses_broader_scale_before_results(self):
        self.assertEqual(choose_scale([{'sigma_m':.01,'training_equal_object_crps_m':.1},{'sigma_m':.22,'training_equal_object_crps_m':.1}]),.22)
    def test_train_loading_never_opens_heldout_truth(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);pr=write_resource(p/'prediction.json',prediction());tr=write_resource(p/'truth.json',dict(object_id='train_object',object_vertices_world=[[0,0,0],[1,.5,1]],world_rotation=[[1,0,0],[0,1,0],[0,0,1]]))
            ds={'cases':[dict(case_id='train',object_id='train_object',condition='front',split='train',truth=tr),dict(case_id='unread',object_id='heldout_object',condition='front',split='heldout',truth={'path':'/must_not_be_opened','sha256':'forbidden'})]}
            index={'cases':[dict(case_id='train',prediction=pr)]}
            cases,failures,opened=load_cases(ds,index,'train')
            self.assertEqual(len(cases),1);self.assertFalse(failures);self.assertEqual(opened,[tr['path']])
    def test_multiple_tracks_not_selected_by_truth(self):
        p=prediction();p['tracks']*=2
        with self.assertRaises(ValueError):prediction_bank(p)
    def test_failed_case_preserves_denominator_with_fixed_public_prior(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);bad=prediction();bad['tracks']*=2
            pr=write_resource(p/'prediction.json',bad)
            tr=write_resource(p/'truth.json',dict(object_id='object',object_vertices_world=[[0,0,0],[2,.5,1]],world_rotation=[[1,0,0],[0,1,0],[0,0,1]]))
            ds={'cases':[dict(case_id='bad',object_id='object',condition='front',split='train',truth=tr)]}
            cases,failures,_=load_cases(ds,{'cases':[dict(case_id='bad',prediction=pr)]},'train')
            self.assertEqual(len(cases),1);self.assertEqual(len(failures),1)
            self.assertTrue(cases[0]['target']['fallback_axis'])
            self.assertEqual(cases[0]['target']['projected_true_extent_m'],.5)
            a=case_metrics(cases,.01)['posterior'][0];b=case_metrics(cases,.22)['posterior'][0]
            self.assertEqual(a['metrics'],b['metrics'])
            self.assertTrue(a['evaluator_prior_fallback'])
            self.assertAlmostEqual(a['metrics']['depth_probability_mass'][0],1/24)

    def test_likelihood_components_must_match_exported_total(self):
        p=prediction();p['tracks'][0]['energy_m2'][0][0]=.1
        with self.assertRaises(AssertionError):prediction_bank(p)

if __name__=='__main__':unittest.main(verbosity=2)
