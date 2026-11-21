"""Synthetic mechanism tests; evaluator geometry never enters model code."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys, importlib.util, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
B=ROOT/'dual_template_exploration_20261119'
SOURCE=ROOT/'reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source'
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(B))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
model=load('tested_B5_model',HERE/'model.py');geometry=model._geometry
old=load('frozen_B4_model',ROOT/'dual_template_instance_20261120/method/model.py')
fixtures=load('B5_independent_fixtures',B/'method/test_adapter.py')

def vertical_face(axis,value,tangent_values=np.linspace(-.4,.4,7),height_values=np.linspace(.2,1.4,9)):
    t,z=np.meshgrid(tangent_values,height_values);p=np.zeros((t.size,3))
    p[:,axis]=value;p[:,1-axis]=t.ravel();p[:,2]=z.ravel();return p

class GeometryTests(unittest.TestCase):
    def test_point_geometry_corrects_biased_normals(self):
        p=vertical_face(0,0.)
        theta=np.pi/6;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        n=np.tile([np.cos(.15),np.sin(.15),0],(len(p),1))@R.T
        views=np.tile([-1.,0,0],(len(p),1))@R.T
        fitted=geometry.fit_envelope(p@R.T+[3,4,0],n,[.9,.65,1.5],views)
        self.assertAlmostEqual(fitted['audit']['yaw_rad'],theta,places=8)
        self.assertAlmostEqual(fitted['dims'][0],.9)
        local_center=fitted['center']@R
        origin=np.array([3,4,0])@R
        self.assertAlmostEqual(local_center[0]-.45,origin[0])
        self.assertIsNone(fitted['audit']['extent_uncertainty'][0]['extent_interval_m'][1])

    def test_camera_side_not_flipped_normal_sets_anchor(self):
        p=vertical_face(0,2.);n=np.tile([-1.,0,0],(len(p),1));view=np.tile([-1.,0,0],(len(p),1))
        fitted=geometry.fit_envelope(p,n,[.9,.65,1.5],view)
        self.assertAlmostEqual(fitted['center'][0]-.5*fitted['dims'][0],2.)

    def test_two_opposite_measured_planes_determine_width(self):
        p=np.concatenate([vertical_face(0,x) for x in (-.25,.25)])
        n=np.tile([1.,0,0],(len(p),1));v=np.repeat([[-1.,0,0],[1.,0,0]],len(p)//2,axis=0)
        fitted=geometry.fit_envelope(p,n,[.9,.65,1.5],v)
        self.assertAlmostEqual(fitted['dims'][0],.5)
        self.assertTrue(fitted['audit']['extent_uncertainty'][0]['both_faces_observed'])

    def test_internal_vertical_plane_is_not_outer_face(self):
        p=np.concatenate([vertical_face(0,x) for x in (-.4,0.,.4)])
        n=np.tile([1.,0,0],(len(p),1));v=np.repeat([[-1.,0,0],[-1.,0,0],[1.,0,0]],len(p)//3,axis=0)
        fitted=geometry.fit_envelope(p,n,[.9,.65,1.5],v)
        self.assertAlmostEqual(fitted['dims'][0],.8)
        inner=[a for a in fitted['audit']['outer_plane_candidates'] if abs(a['coordinate_m'])<.05]
        self.assertTrue(inner);self.assertTrue(all(not a['observed_outer_boundary'] for a in inner))

    def test_single_vertical_line_cannot_identify_yaw(self):
        p=np.column_stack([np.ones(15),np.ones(15),np.linspace(.2,1.8,15)])
        fitted=geometry.fit_envelope(p,np.tile([1.,0,0],(15,1)),[.9,.65,1.5],np.tile([-1.,0,0],(15,1)))
        self.assertFalse(fitted['audit']['orientation_observed'])
        self.assertTrue(all(x['unresolved_extent_uses_prior'] for x in fitted['audit']['extent_uncertainty']))

    def test_degenerate_tls_refinement_cannot_emit_under_supported_plane(self):
        p=vertical_face(0,2.,tangent_values=np.array([-.4,-.2,.2,.4]))
        normals=np.tile([1.,0,0],(len(p),1))
        # An unstable eigensolver orientation loses the measured support. The
        # failed refinement must be discarded rather than median(empty).
        with patch.object(np.linalg,'eigh',return_value=(np.array([0.,1.]),np.array([[0.,1.],[1.,0.]]))):
            planes=geometry._vertical_planes(p,normals)
        self.assertEqual(planes,[])

    def test_internal_horizontal_slab_does_not_establish_top(self):
        x,y=np.meshgrid(np.linspace(-.4,.4,5),np.linspace(-.3,.3,5))
        slab=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,.8)])
        p=np.concatenate([slab,vertical_face(0,.4,height_values=np.linspace(.2,1.9,10))])
        n=np.concatenate([np.tile([0.,0,-1],(len(slab),1)),np.tile([-1.,0,0],(len(p)-len(slab),1))])
        v=-n;fit=geometry.fit_envelope(p,n,[.9,.65,1.5],v)
        self.assertIsNone(fit['audit']['vertical_extent']['upper_horizontal_face_candidate_m'])
        self.assertGreater(fit['dims'][2],1.9)

    def test_observed_top_with_2d_support_not_forced_to_prior_height(self):
        x,y=np.meshgrid(np.linspace(-.4,.4,5),np.linspace(-.3,.3,5))
        p=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,.778)])
        fit=geometry.fit_envelope(p,np.tile([0.,0,-1],(len(p),1)),[.9,.65,.9],np.tile([0.,0,1],(len(p),1)))
        self.assertAlmostEqual(fit['dims'][2],.778)
        self.assertFalse(fit['audit']['vertical_extent']['complete_extent_observed'])

    def test_invalid_unknown_depth_cannot_orient_plane(self):
        p=np.array([[0.,0.,1.]])
        for depth in (0.,-1.,np.nan,np.inf,.5):
            frame=SimpleNamespace(world_from_camera=np.eye(4),intrinsic=np.eye(3),depth_m=np.array([[depth]]))
            views,count=geometry.observed_view_directions(p,[frame])
            self.assertEqual(count[0],0);np.testing.assert_array_equal(views,0.)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Tests.setUpClass();fixture=fixtures.Tests();cls.mapper=fixture.mapper(4)
        assigned={k:(1 if row['label'] in (2,3) else 0) for k,row in cls.mapper.quality.items()}
        class Index:
            def memberships_for_keys(self,keys):return np.asarray([assigned[tuple(k)] for k in keys],np.int64)
        cls.mapper.instance_support_index=Index();cls.mapper.asset_grouping_profile='test_observed_tracks'

    def test_object_support_exactly_preserves_positive_track_templates(self):
        frozen=old.configure(instance_pooling=True)(self.mapper,True)
        new=model.configure()(self.mapper,True)
        self.assertTrue(new.objects)
        self.assertTrue(all(o['support_track']>0 for o in new.objects))
        old_indices=[i for i,o in enumerate(frozen.objects) if o['support_track']>0]
        mask=np.isin(frozen.hypothesis_ids//2,old_indices)
        np.testing.assert_array_equal(new.points,frozen.points[mask])
        np.testing.assert_array_equal(new.weights,frozen.weights[mask])
        for state in ((15,10,1),(15,30,3)):
            np.testing.assert_allclose(new.gain(state[:2],state[2],None),frozen.gain(state[:2],state[2],None)[mask],rtol=0,atol=0)

    def test_all_stages_have_same_G_S_geometry_and_input_hashes(self):
        for full,gf in ((False,False),(True,False),(True,True)):
            cls=model.configure(full_observed_support=full,geometry_fit=gf)
            g,s=cls(self.mapper,False),cls(self.mapper,True)
            np.testing.assert_array_equal(g.points,s.points);np.testing.assert_array_equal(g.hypothesis_ids,s.hypothesis_ids)
            self.assertEqual([o['fit_input_points_sha256'] for o in g.objects],[o['fit_input_points_sha256'] for o in s.objects])
            self.assertTrue(all(o['shelf_prior_probability']==.5 for o in g.objects))

    def test_fitting_changes_no_observed_input_and_history_basis_is_fixed(self):
        old_shape=model.configure(full_observed_support=True)(self.mapper,True)
        fitted=model.configure(full_observed_support=True,geometry_fit=True)(self.mapper,True)
        self.assertEqual([o['fit_input_points_sha256'] for o in old_shape.objects],[o['fit_input_points_sha256'] for o in fitted.objects])
        nc=model.configure(full_observed_support=True,geometry_fit=True)(self.mapper,True,measured_surface_consumption=False)
        np.testing.assert_array_equal(fitted.points,nc.points);np.testing.assert_array_equal(fitted.base_weights,nc.base_weights)
        classified=model._repair.classify_history(nc.points,self.mapper.keyframes)
        self.assertTrue(np.all(nc.weights[classified['strict_free']]==0))

    def test_wall_geometry_remains_in_mapper_when_templates_excluded(self):
        before=np.asarray(self.mapper.mesh().vertices).copy();count=len(self.mapper.quality)
        fitted=model.configure()(self.mapper,True)
        self.assertTrue(fitted.audit['excluded_untracked_template_groups'])
        self.assertEqual(len(self.mapper.quality),count)
        np.testing.assert_array_equal(before,np.asarray(self.mapper.mesh().vertices))

    def test_no_free_likelihood_or_Z_and_no_recursive_posterior(self):
        a=model.configure(full_observed_support=True,geometry_fit=True)(self.mapper,True)
        b=model.configure(full_observed_support=True,geometry_fit=True)(self.mapper,True)
        np.testing.assert_array_equal(a.weights,b.weights)
        self.assertTrue(all(o['free_space_log_likelihood_ratio']==0 and o['template_valid_probability_mass']==1 for o in a.objects))
        with self.assertRaises(ValueError):model.configure(free_space_likelihood=True)(self.mapper,True)

    def test_generic_acquired_object_keeps_same_objectness_G_S(self):
        saved={k:row['label'] for k,row in self.mapper.quality.items()}
        try:
            for row in self.mapper.quality.values():row['label']=1
            g,s=model.configure()(self.mapper,False),model.configure()(self.mapper,True)
            self.assertTrue(g.objects)
            np.testing.assert_array_equal(g.points,s.points)
            np.testing.assert_array_equal(g.weights,s.weights)
            self.assertTrue(all(o['object_probability']==1. for o in s.objects))
        finally:
            for k,row in self.mapper.quality.items():row['label']=saved[k]

if __name__=='__main__':unittest.main(verbosity=2)
