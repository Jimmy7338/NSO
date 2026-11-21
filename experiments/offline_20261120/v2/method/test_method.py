"""Meaningful model parity and measured-only geometry/evidence tests."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
PARENT=HERE.parents[1]/'dual_template_exploration_20261119'
SOURCE=HERE.parents[1]/'reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source'
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(PARENT))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

model=load('tested_B4_model',HERE/'model.py')
old_tests=load('B4_independent_old_fixtures',PARENT/'method/test_adapter.py')
geometry=model._geometry


def frame(depth):
    return SimpleNamespace(world_from_camera=np.eye(4),intrinsic=np.eye(3),depth_m=np.array([[depth]],float))


class GeometryTests(unittest.TestCase):
    def test_oblique_opposite_faces_fit_measured_extent(self):
        # A 30 degree object with 0.5m measured width, below the original .9m minimum.
        y,z=np.meshgrid(np.linspace(-.4,.4,13),np.linspace(.2,1.75,15))
        points=np.concatenate([np.column_stack([np.full(y.size,x),y.ravel(),z.ravel()]) for x in (-.25,.25)])
        normals=np.concatenate([np.tile([sign,0,0],(y.size,1)) for sign in (1,-1)])
        angle=np.pi/6;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        world=points@R.T+[3,4,0];normals=normals@R.T
        result=geometry.fit_envelope(world,normals,[.9,.65,1.5])
        self.assertAlmostEqual(result['audit']['yaw_rad'],angle)
        self.assertAlmostEqual(result['dims'][0],.5)
        self.assertTrue(result['audit']['extent_uncertainty'][0]['both_faces_observed'])
        self.assertGreater(result['dims'][2],1.5)
        np.testing.assert_allclose(result['center'][:2],[3,4],atol=1e-12)

    def test_ninety_degree_axis_ambiguity_preserves_observed_extent(self):
        x,z=np.meshgrid(np.linspace(-.4,.4,13),np.linspace(.2,1.2,12))
        points=np.concatenate([np.column_stack([3+x.ravel(),np.full(x.size,y),z.ravel()]) for y in (3.75,4.25)])
        normals=np.concatenate([np.tile([0,sign,0],(x.size,1)) for sign in (1,-1)])
        result=geometry.fit_envelope(points,normals,[.9,.65,1.5])
        self.assertAlmostEqual(result['dims'][1],.5)
        self.assertTrue(result['audit']['extent_uncertainty'][1]['both_faces_observed'])
        self.assertAlmostEqual(result['center'][1],4.)

    def test_observed_upper_horizontal_face_not_forced_to_point_nine(self):
        x,y=np.meshgrid(np.linspace(2.5,3.5,12),np.linspace(3.7,4.3,12))
        points=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,.778)])
        normals=np.tile([0.,0.,-1.],(len(points),1))
        result=geometry.fit_envelope(points,normals,[.9,.65,.9])
        self.assertAlmostEqual(result['dims'][2],.778)
        self.assertFalse(result['audit']['vertical_extent']['complete_extent_observed'])
        self.assertIsNone(result['audit']['vertical_extent']['upper_bound_m'])

    def test_partial_face_retains_unknown_extent_and_anchors_visible_face(self):
        y,z=np.meshgrid(np.linspace(-.3,.3,12),np.linspace(.3,1.4,12))
        points=np.column_stack([np.full(y.size,2.),3+y.ravel(),z.ravel()])
        normals=np.tile([1.,0,0],(len(points),1))
        result=geometry.fit_envelope(points,normals,[.9,.65,1.5])
        self.assertAlmostEqual(result['center'][0]-result['dims'][0]/2,2.)
        state=result['audit']['extent_uncertainty'][0]
        self.assertTrue(state['unresolved_extent_uses_prior'])
        self.assertIsNone(state['extent_interval_m'][1])
        self.assertAlmostEqual(result['dims'][0],.9)

    def test_invalid_unknown_depth_has_no_free_evidence(self):
        point=np.array([[0.,0.,1.]])
        for depth in (0.,-1.,np.nan,np.inf):
            fraction,audit=geometry.contradiction_fraction(point,[1.],[frame(depth)],model._repair.classify_history)
            self.assertEqual(fraction,0.)
            self.assertEqual(audit['resolved_area_m2'],0.)
        _,audit=geometry.contradiction_fraction(point,[1.],[frame(.5)],model._repair.classify_history)
        self.assertEqual(audit['resolved_area_m2'],0.)

    def test_measured_surface_veto_and_strict_boundary(self):
        point=np.array([[0.,0.,1.]])
        fraction,_=geometry.contradiction_fraction(point,[1.],[frame(2.),frame(1.)],model._repair.classify_history)
        self.assertEqual(fraction,0.)
        fraction,_=geometry.contradiction_fraction(point,[1.],[frame(1.1)],model._repair.classify_history)
        self.assertEqual(fraction,0.)
        fraction,_=geometry.contradiction_fraction(np.array([[0.,0.,np.nextafter(1.,0.)]]),[1.],[frame(1.1)],model._repair.classify_history)
        self.assertEqual(fraction,1.)

    def test_duplicate_history_does_not_multiply_likelihood(self):
        points=np.array([[0,0,1.],[0,0,2.]])
        a=geometry.contradiction_fraction(points,[1.,2.],[frame(2.)],model._repair.classify_history)
        b=geometry.contradiction_fraction(points,[1.,2.],[frame(2.)]*10,model._repair.classify_history)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a[0],1/3)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_tests.Tests.setUpClass()
        cls.fixture=old_tests.Tests()
        cls.mapper=cls.fixture.mapper(4)

    def test_sampling_only_matches_uncapped_original_active_integral(self):
        old=old_tests.FullTemplateModel(self.mapper,True)
        new=model.FullTemplateModel(self.mapper,True)
        self.assertLessEqual(old.audit['points_after_history'],3500)
        np.testing.assert_array_equal(new.points[new.history_keep],old.points)
        np.testing.assert_allclose(new.weights[new.history_keep],old.weights,rtol=0,atol=0)
        for state in ((15,10,1),(15,30,3),(5,20,2)):
            expected,_=model._repair.trace_gain(old,state[:2],state[2],None,repair_enabled=True)
            actual=new.gain(state[:2],state[2],None)
            self.assertAlmostEqual(float(expected.sum()),float(actual.sum()),places=12)

    def test_all_switches_keep_G_S_common_geometry(self):
        for gf,fl in ((False,False),(True,False),(True,True),(False,True)):
            cls=model.configure(geometry_fit=gf,free_space_likelihood=fl)
            g,s=cls(self.mapper,False),cls(self.mapper,True)
            np.testing.assert_array_equal(g.points,s.points)
            np.testing.assert_array_equal(g.hypothesis_ids,s.hypothesis_ids)
            self.assertEqual(len(g.points),g.audit['raw_template_points'])

    def test_history_ablation_has_fixed_basis_and_keeps_free_rejection(self):
        cls=model.configure(geometry_fit=True,free_space_likelihood=True)
        a,b=cls(self.mapper,True),cls(self.mapper,True,measured_surface_consumption=False)
        np.testing.assert_array_equal(a.points,b.points)
        np.testing.assert_array_equal(a.hypothesis_ids,b.hypothesis_ids)
        np.testing.assert_array_equal(a.base_weights,b.base_weights)
        self.assertGreaterEqual(b.weights.sum(),a.weights.sum())
        state=model._repair.classify_history(b.points,self.mapper.keyframes)
        self.assertTrue(np.all(b.weights[state['strict_free']]==0))
        self.assertTrue(np.all(b.weights[state['measured_surface']]>0))
        self.assertEqual(b.audit['removed_as_measured_or_depth_tolerance'],0)

    def test_free_likelihood_is_independent_of_history_and_uses_same_G_S_evidence(self):
        cls=model.configure(geometry_fit=True,free_space_likelihood=True)
        g,s,nohistory=cls(self.mapper,False),cls(self.mapper,True),cls(self.mapper,True,history_rejection=False)
        for og,os,on in zip(g.objects,s.objects,nohistory.objects):
            self.assertEqual(og['free_space_log_likelihood_ratio'],os['free_space_log_likelihood_ratio'])
            self.assertEqual(os['free_space_log_likelihood_ratio'],on['free_space_log_likelihood_ratio'])
            self.assertEqual(os['shelf_probability'],on['shelf_probability'])
        disabled=model.FullTemplateModel(self.mapper,True,geometry_fit=True,free_space_likelihood=False)
        self.assertTrue(all(o['free_space_log_likelihood_ratio']==0 for o in disabled.objects))

    def test_no_history_all_points_and_no_cap(self):
        a=model.FullTemplateModel(self.mapper,True,history_rejection=False)
        self.assertEqual(len(a.points),a.audit['raw_template_points'])
        np.testing.assert_array_equal(a.weights,a.base_weights)
        self.assertIsNone(a.audit['applied_global_point_cap'])


class ApplicabilityTests(unittest.TestCase):
    def test_all_bad_templates_have_nonzero_unmodeled_mass(self):
        masses,likelihood,valid=model.absolute_applicability(.3,[1.,1.],.3)
        self.assertAlmostEqual(valid,np.exp(-3.125))
        self.assertGreater(1-valid,.95)
        self.assertAlmostEqual(masses.sum(),valid)
        self.assertAlmostEqual(masses[1]/valid,.3)

    def test_no_resolved_evidence_leaves_mass_unchanged(self):
        masses,likelihood,valid=model.absolute_applicability(.9,[0.,0.],.9)
        np.testing.assert_allclose(masses,[.1,.9],rtol=0,atol=1e-15)
        self.assertEqual(valid,1.)

    def test_free_likelihood_applied_once_not_squared(self):
        prior=.6;free=np.array([.8,.1]);L=np.exp(-3.125*free)
        raw=np.array([1-prior,prior])*L
        conditional=raw[1]/raw.sum()
        masses,likelihood,valid=model.absolute_applicability(prior,free,conditional)
        np.testing.assert_allclose(masses,raw,rtol=1e-14)
        self.assertAlmostEqual(valid,float(raw.sum()))

    def test_conditional_clipping_preserved(self):
        clipped=float(1/(1+np.exp(-6)))
        masses,_,valid=model.absolute_applicability(.99,[1.,0.],clipped)
        self.assertAlmostEqual(masses[1]/valid,clipped)

    def test_pooling_keeps_unknown_groups_and_unique_observed_indices(self):
        groups=[(1,np.array([0,1]),0),(2,np.array([2,3]),0),
                (101,np.array([4,5]),1),(102,np.array([6,7]),1)]
        result=model.pool_observed_instances(groups,(10,10))
        self.assertEqual(len(result),3)
        self.assertEqual([x[0]for x in result[:2]],[1,2])
        np.testing.assert_array_equal(result[0][1],[0,1])
        np.testing.assert_array_equal(result[1][1],[2,3])
        np.testing.assert_array_equal(result[2][1],[4,5,6,7])
        self.assertEqual(result[2][2],1)


class PooledModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_tests.Tests.setUpClass();cls.fixture=old_tests.Tests()
        cls.mapper=cls.fixture.mapper(4)
        groups=model.measured_groups(cls.mapper,cls.mapper.quality_evidence(max_points=10000))
        biggest=max(groups,key=lambda x:len(x[1]));half=len(biggest[1])//2
        cls.groups=[g for g in groups if g[0]!=biggest[0]]+[(5001,biggest[1][:half],1),(5002,biggest[1][half:],1)]
        cls.total=len(biggest[1])

    def test_same_tracked_fragments_create_one_model_and_keep_common_basis(self):
        from unittest.mock import patch
        with patch.object(model,'measured_groups',return_value=self.groups):
            C=model.configure(geometry_fit=True,free_space_likelihood=True,instance_pooling=True,applicability=True)
            a,b=C(self.mapper,False),C(self.mapper,True)
        tracked=[o for o in a.objects if o['support_track']==1]
        self.assertEqual(len(tracked),1)
        self.assertEqual(tracked[0]['observed_points'],self.total)
        self.assertEqual(tracked[0]['source_component_count'],2)
        np.testing.assert_array_equal(a.points,b.points)
        np.testing.assert_array_equal(a.hypothesis_ids,b.hypothesis_ids)
        self.assertEqual(a.audit['skipped_pooled_groups'],[])

    def test_applicability_preserves_conditional_posterior_and_history_basis(self):
        from unittest.mock import patch
        with patch.object(model,'measured_groups',return_value=self.groups):
            kwargs=dict(geometry_fit=True,free_space_likelihood=True,instance_pooling=True)
            a=model.FullTemplateModel(self.mapper,True,applicability=False,**kwargs)
            b=model.FullTemplateModel(self.mapper,True,applicability=True,**kwargs)
            c=model.FullTemplateModel(self.mapper,True,applicability=True,**kwargs)
        np.testing.assert_array_equal(a.points,b.points)
        np.testing.assert_array_equal(b.weights,c.weights)
        for oa,ob,oc in zip(a.objects,b.objects,c.objects):
            self.assertEqual(oa['shelf_probability'],ob['shelf_probability'])
            self.assertEqual(ob['template_valid_probability_mass'],oc['template_valid_probability_mass'])
            self.assertLessEqual(ob['template_valid_probability_mass'],1.)
            self.assertGreaterEqual(ob['unmodeled_probability_mass'],0.)


if __name__=='__main__':unittest.main(verbosity=2)
