"""B6 mechanism tests; synthetic truth belongs only to test fixtures."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib,importlib.util,sys,unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
B=ROOT/'dual_template_exploration_20261119';B5=ROOT/'dual_template_objectfit_20261120'
SOURCE=ROOT/'reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source'
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(B))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
model=load('tested_B6_model',HERE/'model.py');geometry=model._geometry
old=load('frozen_B5_model_for_B6',B5/'method/model.py')
fixtures=load('B6_independent_fixtures',B/'method/test_adapter.py')

def patch_xy(z):
    x,y=np.meshgrid(np.linspace(-.3,.3,5),np.linspace(-.2,.2,4))
    return np.column_stack([x.ravel(),y.ravel(),np.full(x.size,z)])

class OrientationTests(unittest.TestCase):
    def test_non_outer_diagonal_cannot_determine_orientation(self):
        p=np.array([[0,0,.2],[.2,.3,.2],[.2,.3,.4],[.2,.3,.6],[.2,.3,.8],[-.3,.3,.5],[.4,-.1,.6]],float)
        n=np.array([[-1.,0,0]]*5+[[0,0,1]]*2);v=np.tile([-.8,.55,0.],(len(p),1))
        before=geometry.fit_envelope(p,n,[.9,.65,.9],v,outer_orientation=False)
        after=geometry.fit_envelope(p,n,[.9,.65,.9],v,outer_orientation=True)
        self.assertTrue(before['audit']['orientation_observed'])
        self.assertGreater(abs(before['audit']['yaw_rad']),.1)
        self.assertFalse(after['audit']['orientation_observed'])
        self.assertEqual(after['audit']['yaw_rad'],0.)
        self.assertTrue(all(not q['qualified_outer_orientation'] for q in after['audit']['orientation_qualification']))

    def test_observed_outer_face_retains_oblique_orientation(self):
        y,z=np.meshgrid(np.linspace(-.4,.4,9),np.linspace(.2,1.4,10))
        p=np.column_stack([np.zeros(y.size),y.ravel(),z.ravel()]);theta=np.pi/6
        R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        n=np.tile([-1.,0,0],(len(p),1))@R.T;v=n.copy()
        result=geometry.fit_envelope(p@R.T+[3,4,0],n,[.9,.65,1.5],v,outer_orientation=True)
        self.assertAlmostEqual(result['audit']['yaw_rad'],theta,places=8)
        self.assertTrue(result['audit']['orientation_observed'])
        self.assertIsNone(result['audit']['extent_uncertainty'][0]['extent_interval_m'][1])

class LayerTests(unittest.TestCase):
    def test_no_observed_horizontal_patch_keeps_absolute_prior(self):
        p=patch_xy(.5);n=np.tile([1.,0,0],(len(p),1));v=np.tile([0,0,1.],(len(p),1))
        a,aa=geometry.fit_horizontal_layers(p,n,v,1.5,1.5)
        b,bb=geometry.fit_horizontal_layers(p,n,v,1.5,2.2)
        self.assertEqual(a,b);self.assertEqual(aa['unconstrained_prior_ids'],[0,1,2])
        np.testing.assert_allclose([r['lower_m'] for r in a],[.225,.75,1.275])

    def test_collinear_xy_with_large_bounds_is_not_a_horizontal_patch(self):
        x=np.linspace(-.4,.4,12);p=np.column_stack([x,x,np.full(len(x),.5)])
        parts,a=geometry.fit_horizontal_layers(p,np.tile([0,0,1.],(len(p),1)),np.tile([0,0,1.],(len(p),1)),1.5,2.)
        self.assertEqual(a['observed_horizontal_planes'],[]);self.assertEqual(len(parts),3)

    def test_unobserved_camera_side_cannot_place_a_slab(self):
        p=patch_xy(.5)
        _,a=geometry.fit_horizontal_layers(p,np.tile([0,0,1.],(len(p),1)),np.zeros_like(p),1.5,2.)
        self.assertEqual(a['observed_horizontal_planes'],[])

    def test_one_measured_top_adds_layer_without_deleting_unconstrained_priors(self):
        p=patch_xy(.52)
        parts,a=geometry.fit_horizontal_layers(p,np.tile([0,0,-1.],(len(p),1)),np.tile([0,0,1.],(len(p),1)),1.5,2.)
        self.assertEqual(len(parts),4);self.assertEqual(a['unconstrained_prior_ids'],[0,1,2])
        measured=[x for x in parts if x['source_plane_ids']]
        self.assertEqual(len(measured),1);self.assertAlmostEqual(measured[0]['upper_m'],.52)
        self.assertAlmostEqual(measured[0]['upper_m']-measured[0]['lower_m'],.08)

    def test_measured_face_reanchors_only_overlapping_prior(self):
        p=patch_xy(.80)
        parts,a=geometry.fit_horizontal_layers(p,np.tile([0,0,-1.],(len(p),1)),np.tile([0,0,1.],(len(p),1)),1.5,2.)
        self.assertEqual(len(parts),3);self.assertEqual(a['unconstrained_prior_ids'],[0,2])
        self.assertAlmostEqual(parts[1]['lower_m'],.72);self.assertAlmostEqual(parts[1]['upper_m'],.80)

    def test_opposite_observed_faces_identify_thickness_without_burying_faces(self):
        bottom,top=patch_xy(.78),patch_xy(.80);p=np.concatenate([bottom,top])
        n=np.repeat([[0,0,1.],[0,0,-1.]],len(bottom),axis=0)
        v=np.repeat([[0,0,-1.],[0,0,1.]],len(bottom),axis=0)
        parts,a=geometry.fit_horizontal_layers(p,n,v,1.5,2.)
        self.assertEqual(len(a['paired_opposite_faces']),1)
        measured=[x for x in parts if x['source_plane_ids']]
        self.assertEqual(len(measured),1);self.assertAlmostEqual(measured[0]['lower_m'],.78);self.assertAlmostEqual(measured[0]['upper_m'],.80)

    def test_overlapping_observed_intervals_form_one_component(self):
        # Two separately supported, same-side faces generate overlapping prior
        # thickness intervals. Their union prevents duplicate volume/surfaces.
        p=np.concatenate([patch_xy(.50),patch_xy(.56)])
        n=np.tile([0,0,-1.],(len(p),1));v=-n
        parts,a=geometry.fit_horizontal_layers(p,n,v,1.5,2.)
        measured=[x for x in parts if x['source_plane_ids']]
        self.assertEqual(len(measured),1)
        self.assertAlmostEqual(measured[0]['lower_m'],.42);self.assertAlmostEqual(measured[0]['upper_m'],.56)

    def test_unconstrained_prior_outside_box_leg_height_is_retained_and_declared(self):
        p=patch_xy(.5)
        parts,a=geometry.fit_horizontal_layers(p,np.tile([1.,0,0],(len(p),1)),np.zeros_like(p),.9,.778)
        self.assertTrue(a['slab_bounds_exceed_envelope'])
        self.assertAlmostEqual(max(x['upper_m'] for x in parts),.845)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Tests.setUpClass();cls.mapper=fixtures.Tests().mapper(4)
        assigned={k:(1 if r['label'] in (2,3) else 0) for k,r in cls.mapper.quality.items()}
        class Index:
            def memberships_for_keys(self,keys):return np.asarray([assigned[tuple(k)] for k in keys],np.int64)
        cls.mapper.instance_support_index=Index();cls.mapper.asset_grouping_profile='B6_test_observed_support'

    def test_both_disabled_exactly_reproduce_frozen_B5_OF(self):
        before=old.configure(full_observed_support=True,geometry_fit=True)(self.mapper,True)
        after=model.configure()(self.mapper,True)
        np.testing.assert_array_equal(before.points,after.points);np.testing.assert_array_equal(before.weights,after.weights)
        np.testing.assert_array_equal(before.hypothesis_ids,after.hypothesis_ids)
        for state in ((15,10,1),(15,30,3)):
            np.testing.assert_allclose(before.gain(state[:2],state[2],None),after.gain(state[:2],state[2],None),rtol=0,atol=0)

    def test_all_stages_share_G_S_support_and_complete_geometry(self):
        for orientation,layers in ((True,False),(False,True),(True,True)):
            cls=model.configure(outer_orientation=orientation,observed_layers=layers)
            g,s=cls(self.mapper,False),cls(self.mapper,True)
            np.testing.assert_array_equal(g.points,s.points);np.testing.assert_array_equal(g.hypothesis_ids,s.hypothesis_ids)
            for go,so in zip(g.objects,s.objects):
                self.assertEqual(go['fit_input_points_sha256'],so['fit_input_points_sha256'])
                self.assertEqual(go['observed_layers_audit'],so['observed_layers_audit'])
                self.assertEqual(go['object_probability'],so['object_probability'])
            self.assertTrue(all(o['shelf_prior_probability']==.5 for o in g.objects))

    def test_clean_history_basis_and_posterior_are_unchanged_by_consumption(self):
        cls=model.configure(outer_orientation=True,observed_layers=True)
        full=cls(self.mapper,True);nc=cls(self.mapper,True,measured_surface_consumption=False);repeat=cls(self.mapper,True)
        np.testing.assert_array_equal(full.points,nc.points);np.testing.assert_array_equal(full.base_weights,nc.base_weights)
        np.testing.assert_array_equal(full.weights,repeat.weights)
        classified=model._repair.classify_history(nc.points,self.mapper.keyframes)
        self.assertTrue(np.all(nc.weights[classified['strict_free']]==0))
        self.assertTrue(all(o['free_space_log_likelihood_ratio']==0 and o['template_valid_probability_mass']==1 for o in full.objects))

if __name__=='__main__':unittest.main(verbosity=2)
