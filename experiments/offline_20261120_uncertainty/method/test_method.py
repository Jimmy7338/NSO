"""B8 tests: synthetic mechanism fixtures only; no calibration/scene outcomes."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import copy,importlib.util,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path('/home/ouc/nso_robot/analyses');HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source'
sys.path.insert(0,str(SOURCE))
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load('tested_B8',HERE/'model.py')
b7tests=load('B8_B7_source_fixtures',ROOT/'dual_template_depth_20261120/method/test_profile.py')

def object_fixture(anchor='low'):
    x,z=np.meshgrid(np.linspace(-.3,.3,5),np.linspace(.2,.8,5));cloud=np.column_stack([x.ravel(),np.zeros(x.size),z.ravel()])
    if anchor=='none':cloud[::2,1]=.2
    obj=dict(center=np.array([0,.325,.5]),dims=np.array([.9,.65,1.]),rotation=np.eye(3),support_track=1,
        fit_input_points_sha256=m.digest(cloud),fit_input_normals_sha256='fixture',fit_input_labels_sha256='fixture',
        geometry_fit_audit={'extent_uncertainty':[{},dict(low_face_coordinate_m=0. if anchor=='low' else None,high_face_coordinate_m=None)]},
        observed_layers_audit={'final_intervals':[dict(lower_m=a,upper_m=a+.08) for a in (.15,.5,.85)]})
    pose=np.eye(4);pose[:3,:3]=np.array([[1,0,0],[0,0,1],[0,-1,0]]);pose[:3,3]=[0,-1,.5]
    frame=SimpleNamespace(world_from_camera=pose,intrinsic=np.array([[25,0,10],[0,25,10],[0,0,1.]]),depth_m=np.ones((21,21)))
    quality={ (i,0,0):dict(point=p,normal=np.array([0,-1,0.]),bits=1<<2,label=2) for i,p in enumerate(cloud)}
    index=SimpleNamespace(memberships_for_keys=lambda keys:np.ones(len(keys),np.int64))
    mapper=SimpleNamespace(quality=quality,instance_support_index=index,keyframes=[frame],shape=(50,50),
        config=SimpleNamespace(fov_deg=90,max_depth_m=5,width_px=120,height_px=90))
    base=SimpleNamespace(objects=[obj],audit={key:True for key in ('geometry_fit','outer_orientation','observed_layers','full_observed_support')})
    return obj,cloud,mapper,base

class PriorPosteriorTests(unittest.TestCase):
    def test_uniform_mass_uses_cell_width_not_node_count(self):
        d,w,a=m.uniform_nodes(.437)
        self.assertEqual(d[0],.437);self.assertEqual(d[-1],1.4);self.assertAlmostEqual(w.sum(),1.)
        self.assertAlmostEqual(w[0],(.5-.437)/2/(1.4-.437))
        self.assertAlmostEqual(float(np.dot(d,w)),(.437+1.4)/2,places=3)
    def test_joint_mass_and_shared_evidence(self):
        e=np.array([[0,.002],[.001,0.],[.003,.001]]);q=np.array([.2,.5,.3])
        g=m.joint_posterior(e,q,[.5,.5],.06);s=m.joint_posterior(e,q,[.1,.9],.06)
        self.assertAlmostEqual(g.sum(),1.);self.assertAlmostEqual(s.sum(),1.)
        np.testing.assert_allclose(s/g/(s[0,0]/g[0,0]),np.tile([1,9],(3,1)),rtol=1e-12)
        np.testing.assert_allclose(m.joint_posterior(e,q,[.1,.9],.06,False),q[:,None]*[.1,.9])
    def test_invalid_energy_and_temperature_raise_instead_of_nan(self):
        for value in (np.nan,np.inf,-np.inf):
            with self.assertRaisesRegex(ValueError,'finite observation'):m.joint_posterior([[value,0]],[1],[.5,.5],.06)
        for sigma in (0,-1,np.nan,np.inf):
            with self.assertRaisesRegex(ValueError,'sigma'):m.joint_posterior([[0,0]],[1],[.5,.5],sigma)
    def test_duplicating_particle_with_split_mass_preserves_expectation(self):
        e=np.array([[.001,.002],[.004,.003]]);q=np.array([.4,.6]);reward=np.array([[2.,1.],[4.,6.]])
        a=m.joint_posterior(e,q,[.5,.5],.06)
        b=m.joint_posterior(np.repeat(e,2,axis=0),np.repeat(q/2,2),[.5,.5],.06)
        self.assertAlmostEqual(float((a*reward).sum()),float((b*np.repeat(reward,2,axis=0)).sum()),places=12)
    def test_no_anchor_retains_position_nodes_and_contains_support(self):
        obj,cloud,_,_=object_fixture('none');p,status=m.geometric_particles(obj,cloud)
        self.assertEqual(len(p)%3,0);self.assertFalse(status['out_of_support'])
        for r in p:self.assertLessEqual(r['lower_m'],cloud[:,1].min()+1e-12);self.assertGreaterEqual(r['upper_m'],cloud[:,1].max()-1e-12)
        self.assertAlmostEqual(sum(r['prior_mass'] for r in p),1.)
    def test_two_faces_and_out_of_prior_support_are_explicit(self):
        obj,cloud,_,_=object_fixture();obj['geometry_fit_audit']['extent_uncertainty'][1]['high_face_coordinate_m']=.41
        p,s=m.geometric_particles(obj,cloud);self.assertEqual(len(p),1);self.assertAlmostEqual(p[0]['depth_m'],.41)
        obj['geometry_fit_audit']['extent_uncertainty'][1]['high_face_coordinate_m']=None
        cloud[-1,1]=1.6;p,s=m.geometric_particles(obj,cloud)
        self.assertEqual(len(p),1);self.assertTrue(s['out_of_support']);self.assertFalse(s['distribution_calibratable']);self.assertEqual(p[0]['depth_m'],1.6)
    def test_fixed_extent_retains_same_unanchored_location_nodes(self):
        obj,cloud,_,_=object_fixture('none');p,s=m.geometric_particles(obj,cloud,'fixed')
        self.assertEqual([r['depth_m'] for r in p],[.65]*3);self.assertEqual([r['allocation'] for r in p],[0,.5,1.])
        np.testing.assert_allclose([r['prior_mass'] for r in p],[.25,.5,.25])

class BankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obj,cls.cloud,cls.mapper,cls.base=object_fixture();cls.bank=m.build_particle_bank(cls.mapper,cls.base)
    def test_front_only_preserves_depth_prior(self):
        row=self.bank.distribution(.06)['tracks'][0]
        np.testing.assert_allclose(row['probability_mass'],row['prior_mass'],atol=2e-10,rtol=0)
        self.assertGreater(np.count_nonzero(np.asarray(row['probability_mass'])>.02),8)
    def test_G_S_share_materialized_basis_and_no_history_basis(self):
        g=m.FullTemplateModel(self.mapper,False,sigma_m=.06,bank=self.bank)
        s=m.FullTemplateModel(self.mapper,True,sigma_m=.06,bank=self.bank)
        nc=m.FullTemplateModel(self.mapper,True,sigma_m=.06,bank=self.bank,measured_surface_consumption=False)
        self.assertIs(g.points,s.points);self.assertIs(g.hypothesis_ids,s.hypothesis_ids)
        np.testing.assert_array_equal(s.points,nc.points);np.testing.assert_array_equal(s.base_weights,nc.base_weights)
        self.assertTrue(np.all(nc.weights[self.bank.materialize()['history']['strict_free']]==0))
    def test_geometry_bank_reused_and_observation_changes_key(self):
        self.assertIs(m.build_particle_bank(self.mapper),self.bank)
        before=m.observed_input_signature(self.mapper);self.mapper.quality[(0,0,0)]['bits']^=1
        after=m.observed_input_signature(self.mapper);self.mapper.quality[(0,0,0)]['bits']^=1
        self.assertNotEqual(before,after)
    def test_mean_collapse_preserves_structure_mass_without_refitting(self):
        full=self.bank.distribution(.06,True)['tracks'][0]
        c=m.FullTemplateModel(self.mapper,True,sigma_m=.06,extent_mode='mean_collapse',bank=self.bank)
        for h,row in enumerate(c.objects[0]['mean_collapse']):
            self.assertAlmostEqual(row['retained_shape_mass'],full['shape_probability_mass'][h]);self.assertFalse(row['likelihood_recomputed'])
            expected=sum(p['depth_m']*full['joint_probability_mass'][i][h] for i,p in enumerate(full['particles']))/full['shape_probability_mass'][h]
            self.assertAlmostEqual(row['depth_m'],expected)
    def test_repeated_route_view_cannot_increase_union(self):
        model=m.FullTemplateModel(self.mapper,False,sigma_m=.06,bank=self.bank,history_rejection=False)
        pose=self.mapper.keyframes[0].world_from_camera
        with patch('env.virtual3d.camera_pose',return_value=pose):
            one,c1=b7tests.fixtures.adapter.route_union_area(model,[(0,0,0)],None)
            twice,c2=b7tests.fixtures.adapter.route_union_area(model,[(0,0,0),(0,0,0)],None)
        self.assertGreater(one,0);self.assertEqual(one,twice);self.assertEqual(c2,1)
    def test_actual_geometry_duplicate_particles_leave_visible_area_unchanged(self):
        obj,cloud,mapper,base=object_fixture();dup=m.build_particle_bank(mapper,base)
        v=dup.objects[0]['variants']['uncertainty'];particles=[]
        for p in v['particles']:
            for _ in range(2):
                q=dict(p);q['prior_mass']/=2;particles.append(q)
        v['particles']=particles;v['prior_mass']=np.repeat(v['prior_mass']/2,2)
        for key in ('energy_m2','surface_energy_m2','free_energy_m2'):v[key]=np.repeat(v[key],2,axis=0)
        a=m.FullTemplateModel(self.mapper,True,sigma_m=.06,bank=self.bank,history_rejection=False)
        b=m.FullTemplateModel(mapper,True,sigma_m=.06,bank=dup,history_rejection=False)
        with patch('env.virtual3d.camera_pose',return_value=mapper.keyframes[0].world_from_camera):
            ga=a.gain((0,0),0,None).sum();gb=b.gain((0,0),0,None).sum()
        self.assertGreater(ga,0);self.assertAlmostEqual(float(ga),float(gb),places=12)
    def test_active_only_gain_exactly_matches_full_basis_and_zero_scatter(self):
        a=m.FullTemplateModel(self.mapper,True,sigma_m=.06,bank=self.bank,history_rejection=False)
        a.weights[::3]=0
        with patch('env.virtual3d.camera_pose',return_value=self.mapper.keyframes[0].world_from_camera):
            expected=m.REPAIR.trace_gain(a,(0,0),0,None,repair_enabled=True)[0]
            actual=a.gain((0,0),0,None)
            np.testing.assert_array_equal(actual,expected)
            self.assertTrue(np.all(actual[::3]==0))
            self.assertEqual(a._occlusion_repair_summary['gain_calls'],1)
            self.assertEqual(a._occlusion_repair_summary['stage_sums_across_gain_calls']['all']['points'],np.count_nonzero(a.weights))
            a.weights[:]=0
            np.testing.assert_array_equal(a.gain((0,0),0,None),np.zeros(len(a.points)))
            self.assertEqual(a._occlusion_repair_summary['gain_calls'],2)
    def test_missing_frozen_scale_cannot_bind_policy(self):
        with self.assertRaisesRegex(ValueError,'calibrated'):m.FullTemplateModel(self.mapper,False,bank=self.bank)
    def test_invalid_depth_does_not_supply_free_evidence(self):
        obj,cloud,mapper,base=object_fixture();mapper.keyframes[0].depth_m[:]=0
        bank=m.build_particle_bank(mapper,base);row=bank.distribution()['tracks'][0]
        self.assertEqual(row['fixed_ray_audit']['ray_count'],0);self.assertTrue(np.all(np.asarray(row['free_energy_m2'])==0))

class SourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):b7tests.SourceParityTests.setUpClass();cls.old=b7tests.SourceParityTests();cls.obj,cls.cloud,cls.bits=cls.old.prepared_object()
    def test_parts_and_quadrature_match_B6_at_same_depth_and_position(self):
        obj=self.obj;R=np.asarray(obj['rotation']);center=np.asarray(obj['center'])@R
        parts=m.primitive_parts(obj,obj['dims'][1],center[1]-obj['dims'][1]/2,self.bits)
        source=self.old.model
        for h,part in enumerate(parts):
            xyz,a=m.quadrature(part,R);indices=source.hypothesis_ids==h
            np.testing.assert_allclose(xyz,source.points[indices],atol=2e-14,rtol=0)
            self.assertEqual(len(xyz),int(indices.sum()))
            _,scene=m.mesh_scene(part,R)
            dist=scene.compute_distance(m.o3d.core.Tensor(source.points[indices].astype(np.float32)),nthreads=1).numpy()
            self.assertLess(float(dist.max()),2e-6)

if __name__=='__main__':unittest.main(verbosity=2)
