import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,importlib.util,unittest,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
B=ROOT/'dual_template_exploration_20261119';B6=ROOT/'dual_template_parts_20261120'
SOURCE=ROOT/'reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source'
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(B))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
p=load('tested_B7_profile',HERE/'profile.py');b6=load('B7_frozen_B6',B6/'method/model.py')
fixtures=load('B7_independent_fixtures',B/'method/test_adapter.py')

def one_frame(depth):return SimpleNamespace(world_from_camera=np.eye(4),intrinsic=np.eye(3),depth_m=np.array([[depth]],float))

class SelectionTests(unittest.TestCase):
    def test_flat_curve_keeps_prior(self):
        d=[.3,.4,.65,.8,1.3];a=p.select_profile(d,[.2]*5,np.zeros((5,2)),.65)
        self.assertFalse(a['adopted']);self.assertEqual(a['point_estimate_m'],.65)

    def test_measured_lower_bound_optimum_can_have_internal_upper_limit(self):
        d=[.3,.4,.5,.65,1.3];f=np.array([[0,0],[.001,.001],[.005,.005],[.02,.02],[.02,.02]])
        a=p.select_profile(d,[0,.4,1.2,2.1,3],f,.65)
        self.assertTrue(a['adopted']);self.assertEqual(a['point_estimate_m'],.4)
        self.assertEqual(a['plausible_grid_interval_m'],[.3,.4]);self.assertTrue(a['lower_end_is_measured_span'])
        self.assertEqual(a['internal_upper_crossing_bracket_m'],[.4,.5])
        self.assertFalse(a['exact_hidden_depth_identified'])

    def test_artificial_upper_boundary_does_not_establish_depth(self):
        d=[.3,.4,.65,.8,1.3];a=p.select_profile(d,[3,3,2.5,1,0],np.ones((5,2))*.01,.65)
        self.assertFalse(a['adopted']);self.assertEqual(a['point_estimate_m'],.65)
        self.assertEqual(a['reason'],'artificial_upper_search_boundary_unresolved')

    def test_without_free_discrimination_surface_fit_does_not_force_update(self):
        a=p.select_profile([.3,.4,.65,1.3],[0,.4,2.5,3],np.zeros((4,2)),.65)
        self.assertFalse(a['adopted'])
        self.assertEqual(a['reason'],'no_shared_measured_free_contradiction_reduction')

class EvidenceTests(unittest.TestCase):
    def test_free_veto_unknown_and_closer_hit_behavior(self):
        scene=p._scene(p.o3d.geometry.TriangleMesh.create_box(1,1,.1).translate((-.5,-.5,1.)))
        cloud=np.array([[0,0,1.]]);rays=np.array([[0,0,0,0,0,1]],np.float32)
        _,free,a=p.measured_terms(scene,cloud,rays,np.array([2.]),[one_frame(2.)])
        self.assertGreater(free,0);self.assertEqual(a['strict_free_hits'],1)
        _,free,a=p.measured_terms(scene,cloud,rays,np.array([2.]),[one_frame(2.),one_frame(1.)])
        self.assertEqual(free,0);self.assertEqual(a['measured_surface_vetoes'],1)
        for depth in (0.,np.nan,np.inf,-1.):
            _,free,a=p.measured_terms(scene,cloud,rays,np.array([2.]),[one_frame(depth)])
            self.assertEqual(free,0)
        _,free,a=p.measured_terms(scene,cloud,rays,np.array([.5]),[one_frame(.5)])
        self.assertEqual(free,0);self.assertEqual(a['front_hit_free_candidates'],0)

    def test_nonunit_camera_ray_t_remains_camera_depth(self):
        scene=p._scene(p.o3d.geometry.TriangleMesh.create_box(4,4,.1).translate((-2,-2,1.)))
        rays=np.array([[0,0,0,.3,.4,1.]],np.float32)
        hit=scene.cast_rays(p.o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()[0]
        self.assertAlmostEqual(float(hit),1.,places=6)
        _,free,_=p.measured_terms(scene,np.array([[.3,.4,1.]]),rays,np.array([1.]),[one_frame(1.)])
        self.assertEqual(free,0)

    def test_front_only_observation_leaves_hidden_depth_curve_flat(self):
        obj=dict(center=[0,.325,.5],dims=[.9,.65,1.],rotation=np.eye(3).tolist(),
            geometry_fit_audit={'extent_uncertainty':[{},dict(low_face_coordinate_m=0.,high_face_coordinate_m=None)]},
            observed_layers_audit={'final_intervals':[dict(lower_m=a,upper_m=a+.08) for a in (.15,.5,.85)]})
        pose=np.eye(4);pose[:3,:3]=np.array([[1,0,0],[0,0,1],[0,-1,0]]);pose[:3,3]=[0,-1,.5]
        frame=SimpleNamespace(world_from_camera=pose,intrinsic=np.array([[25,0,10],[0,25,10],[0,0,1.]]),depth_m=np.ones((21,21)))
        rays,depths,_=p.fixed_observed_rays(obj,1.3,[frame])
        x,z=np.meshgrid(np.linspace(-.3,.3,5),np.linspace(.2,.8,5));cloud=np.column_stack([x.ravel(),np.zeros(x.size),z.ravel()])
        nll=[];free=[]
        for depth in (.2,.4,.65,1.,1.3):
            terms=[p.measured_terms(p._scene(mesh),cloud,rays,depths,[frame])[:2] for mesh in p.primitive_meshes(obj,depth,np.array([1<<2]))]
            energy=np.array([sum(t) for t in terms])/(2*.06**2)
            nll.append(float(-np.logaddexp(-energy[0],-energy[1])+np.log(2.)));free.append([t[1] for t in terms])
        np.testing.assert_allclose(nll,nll[0],atol=1e-10,rtol=0)
        self.assertFalse(p.select_profile([.2,.4,.65,1.,1.3],nll,free,.65)['adopted'])

class SourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Tests.setUpClass();cls.mapper=fixtures.Tests().mapper(4)
        assigned={k:(1 if r['label'] in (2,3) else 0) for k,r in cls.mapper.quality.items()}
        class Index:
            def memberships_for_keys(self,keys):return np.array([assigned[tuple(k)] for k in keys],np.int64)
        cls.mapper.instance_support_index=Index();cls.mapper.asset_grouping_profile='B7_test_track'
        cls.model=b6.configure(outer_orientation=True,observed_layers=True)(cls.mapper,False)

    def prepared_object(self):
        obj=copy.deepcopy(self.model.objects[0]);R=np.array(obj['rotation']);local=np.array(obj['center'])@R
        obj['geometry_fit_audit']['extent_uncertainty'][1].update(low_face_coordinate_m=float(local[1]-obj['dims'][1]/2),high_face_coordinate_m=None)
        rows=list(self.mapper.quality.items());keys=np.array([k for k,_ in rows]);tracks=self.mapper.instance_support_index.memberships_for_keys(keys)
        selected=[row for (_,row),track in zip(rows,tracks) if track==obj['support_track']]
        return obj,np.array([q['point'] for q in selected]),np.array([q['bits'] for q in selected])

    def test_rebuilt_primitives_match_original_B6_surfaces(self):
        obj,cloud,bits=self.prepared_object();meshes=p.primitive_meshes(obj,obj['dims'][1],bits)
        for index,mesh in enumerate(meshes):
            points=self.model.points[self.model.hypothesis_ids==index]
            rebuilt=p._scene(mesh)
            distances=rebuilt.compute_distance(p.o3d.core.Tensor(points.astype(np.float32)),nthreads=1).numpy()
            self.assertLess(float(distances.max()),2e-6)
            before=self.model.hypothesis_rays[index].compute_distance(p.o3d.core.Tensor(cloud.astype(np.float32)),nthreads=1).numpy()
            after=rebuilt.compute_distance(p.o3d.core.Tensor(cloud.astype(np.float32)),nthreads=1).numpy()
            np.testing.assert_allclose(before,after,atol=2e-6,rtol=0)

    def test_fixed_rays_deduplicate_and_cover_all_layer_heights(self):
        obj,cloud,bits=self.prepared_object();obj['observed_layers_audit']['final_intervals'].append(dict(lower_m=-.1,upper_m=2.2))
        a,da,aa=p.fixed_observed_rays(obj,1.3,self.mapper.keyframes)
        b,db,bb=p.fixed_observed_rays(obj,1.3,self.mapper.keyframes*2)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(da,db)
        self.assertGreater(len(a),0);self.assertGreater(bb['duplicate_rays_removed'],aa['duplicate_rays_removed'])
        self.assertEqual(aa['roi_z_bounds_m'],[-.1,2.2])

    def test_zero_span_and_all_invalid_depth_keep_prior(self):
        obj,cloud,bits=self.prepared_object()
        saved=obj['geometry_fit_audit']['extent_uncertainty'][1]['observed_span_lower_bound_m']
        obj['geometry_fit_audit']['extent_uncertainty'][1]['observed_span_lower_bound_m']=0.
        result=p.profile_object(obj,cloud,bits,self.mapper.keyframes)
        self.assertEqual(result['status'],'retained_inconsistent_or_degenerate_domain')
        obj['geometry_fit_audit']['extent_uncertainty'][1]['observed_span_lower_bound_m']=min(saved,.3)
        frames=[SimpleNamespace(world_from_camera=f.world_from_camera,intrinsic=f.intrinsic,depth_m=np.zeros_like(f.depth_m)) for f in self.mapper.keyframes]
        result=p.profile_object(obj,cloud,bits,frames)
        self.assertEqual(result['fixed_ray_audit']['ray_count'],0)
        self.assertFalse(result['selection']['adopted'])

    def test_no_anchor_is_reported_without_center_adjustment(self):
        obj,cloud,bits=self.prepared_object();center=copy.deepcopy(obj['center'])
        obj['geometry_fit_audit']['extent_uncertainty'][1].update(low_face_coordinate_m=None,high_face_coordinate_m=None)
        result=p.profile_object(obj,cloud,bits,self.mapper.keyframes)
        self.assertEqual(result['status'],'retained_without_single_observed_anchor')
        np.testing.assert_array_equal(obj['center'],center)

if __name__=='__main__':unittest.main(verbosity=2)
