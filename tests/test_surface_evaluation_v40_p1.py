"""Analytic triangle fixtures only; no World, planner, fusion or trajectory."""
import json
import unittest
import numpy as np

from nso.surface_evaluation_v40 import CandidateViewV40,freeze_reference_v40,evaluate_surface_v40


def box(low=(0,0,0),high=(1,1,1),owner=0):
    x,y,z=low;X,Y,Z=high
    v=np.array([[x,y,z],[X,y,z],[X,Y,z],[x,Y,z],[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]],float)
    t=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
        [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]],np.int64)
    return v,t,np.full(len(t),owner,np.int64)


def combine(*meshes):
    vertices=[];triangles=[];owners=[];offset=0
    for v,t,ids in meshes:
        vertices.append(v);triangles.append(t+offset);owners.append(ids);offset+=len(v)
    return np.concatenate(vertices),np.concatenate(triangles),np.concatenate(owners)


def camera(origin,target=(.5,.5,.5),name='view',far=10.):
    origin=np.asarray(origin,float);forward=np.asarray(target,float)-origin;forward/=np.linalg.norm(forward)
    up=np.array([0.,0.,1.])
    if abs(forward@up)>.99: up=np.array([0.,1.,0.])
    right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right)
    t=np.eye(4);t[:3,:3]=np.column_stack((right,down,forward));t[:3,3]=origin
    return CandidateViewV40(np.array([[64.,0,63.5],[0,64.,47.5],[0,0,1.]]),t,128,96,far_m=far,view_id=name)


def views():
    return tuple(camera(o,name=f'view{i}') for i,o in enumerate(((.5,-3,.5),(4,.5,.5),(.5,4,.5),(-3,.5,.5),(.5,.5,4))))


class StaticSurfaceTests(unittest.TestCase):
    def reference(self): return freeze_reference_v40(*box(),views(),sample_spacing_m=.2)

    def test_reference_contains_horizontal_and_vertical_but_not_unseen_bottom(self):
        ref=self.reference()
        self.assertTrue(np.any(np.isclose(ref.points[:,2],1)))
        self.assertTrue(np.any(np.isclose(ref.points[:,1],0)))
        self.assertFalse(np.any(np.isclose(ref.points[:,2],0)))
        self.assertAlmostEqual(ref.area_weights.sum(),5.)

    def test_exact_complete_mesh_is_one_and_coverage_is_separate(self):
        v,t,_=box();result=evaluate_surface_v40(self.reference(),v,t,C_map=.8)
        self.assertAlmostEqual(result['Q'],1.);self.assertAlmostEqual(result['J'],.8)
        self.assertAlmostEqual(result['full_mesh_global_precision'],1.)
        self.assertAlmostEqual(result['correct_target_out_of_reference_area_m2'],1.)
        self.assertAlmostEqual(result['extra_false_positive_area_m2'],0.)
        json.dumps(result,allow_nan=False)

    def test_background_occludes_target_without_xray(self):
        wall=(np.array([[-.5,-1,-.5],[1.5,-1,-.5],[1.5,-1,1.5],[-.5,-1,1.5]]),
            np.array([[0,1,2],[0,2,3]],np.int64),np.array([-1,-1],np.int64))
        ref=freeze_reference_v40(*combine(box(),wall),views()[:2],sample_spacing_m=.2)
        self.assertFalse(np.any(np.isclose(ref.points[:,1],0)))
        self.assertTrue(np.all(np.isclose(ref.points[:,0],1)))
        self.assertAlmostEqual(ref.area_weights.sum(),1.)

    def test_fully_hidden_instance_is_invalid_reference_not_dropped(self):
        meshes=combine(box((0,0,0),(2,2,2),0),box((.5,.5,.5),(1.5,1.5,1.5),1))
        candidates=(camera((1,-3,1),target=(1,1,1)),)
        with self.assertRaisesRegex(ValueError,'no observable'):
            freeze_reference_v40(*meshes,candidates,sample_spacing_m=.2)

    def test_closed_internal_shell_does_not_enter_reference(self):
        meshes=combine(box((0,0,0),(2,2,2),0),box((.5,.5,.5),(1.5,1.5,1.5),0))
        ref=freeze_reference_v40(*meshes,(camera((1,-3,1),target=(1,1,1)),),sample_spacing_m=.2)
        self.assertTrue(np.all(np.isclose(ref.points[:,1],0)))
        self.assertAlmostEqual(ref.area_weights.sum(),4.)

    def test_missing_whole_facility_remains_in_macro_denominator(self):
        mesh=combine(box(),box((3,0,0),(4,1,1),1))
        candidates=(camera((2,-4,2),target=(2,.5,.5),name='front'),camera((2,.5,5),target=(2,.5,.5),name='top'))
        ref=freeze_reference_v40(*mesh,candidates,sample_spacing_m=.2)
        v,t,_=box();result=evaluate_surface_v40(ref,v,t,C_map=1.)
        self.assertEqual(len(result['per_instance']),2)
        self.assertAlmostEqual(result['per_instance'][0]['f1'],1.)
        self.assertEqual(result['per_instance'][1]['f1'],0.)
        self.assertAlmostEqual(result['macro_f1'],.5)

    def test_extra_far_surface_reduces_macro_and_global_precision(self):
        v,t,_=box();extra=np.array([[20,20,20],[22,20,20],[20,22,20]],float)
        pv=np.concatenate((v,extra));pt=np.concatenate((t,np.array([[8,9,10]],np.int64)))
        result=evaluate_surface_v40(self.reference(),pv,pt,C_map=1.)
        self.assertAlmostEqual(result['extra_false_positive_area_m2'],2.)
        self.assertAlmostEqual(result['full_mesh_global_precision'],.75)
        self.assertLess(result['macro_f1'],1.)
        self.assertAlmostEqual(result['per_instance'][0]['allocated_false_positive_area_m2'],2.)

    def test_duplicate_correct_triangles_cannot_hide_false_positive_area(self):
        v,t,_=box();pv=np.concatenate((v,np.array([[20,20,20],[22,20,20],[20,22,20]])))
        pt=np.concatenate((t,np.array([[8,9,10]],np.int64)))
        duplicated=np.concatenate((pt,np.repeat(t[2:3],20,axis=0)))
        ref=self.reference();a=evaluate_surface_v40(ref,pv,pt,C_map=1.);b=evaluate_surface_v40(ref,pv,duplicated,C_map=1.)
        self.assertEqual(a['Q'],b['Q']);self.assertEqual(a['full_mesh_global_precision'],b['full_mesh_global_precision'])
        self.assertEqual(b['duplicate_prediction_triangles_removed'],20)
        self.assertAlmostEqual(b['duplicate_prediction_area_removed_m2'],10.)

    def test_correct_background_is_accounted_for_but_wrong_background_is_fp(self):
        mesh=combine(box(),box((-2,-2,-.1),(3,3,0),-1))
        ref=freeze_reference_v40(*mesh,views(),sample_spacing_m=.3)
        correct=evaluate_surface_v40(ref,mesh[0],mesh[1],C_map=1.,sample_spacing_m=.3)
        self.assertAlmostEqual(correct['Q'],1.);self.assertGreater(correct['correct_background_area_m2'],1.)
        extra=np.array([[-1,-1,-1],[0,-1,-1],[-1,0,-1]],float)
        v=np.concatenate((mesh[0],extra));n=len(mesh[0]);t=np.concatenate((mesh[1],np.array([[n,n+1,n+2]],np.int64)))
        wrong=evaluate_surface_v40(ref,v,t,C_map=1.,sample_spacing_m=.3)
        self.assertAlmostEqual(wrong['extra_false_positive_area_m2'],.5)
        self.assertLess(wrong['Q'],correct['Q'])
        self.assertLess(wrong['full_mesh_global_precision'],1.)

    def test_empty_prediction_is_measured_failure_not_empty_target(self):
        result=evaluate_surface_v40(self.reference(),np.empty((0,3)),np.empty((0,3),np.int64),C_map=.7)
        self.assertTrue(result['prediction_empty']);self.assertEqual(result['Q'],0.);self.assertEqual(result['J'],0.)

    def test_distance_threshold_changes_match_without_surface_resampling_bias(self):
        v,t,_=box();shift=v+np.array([.2,0,0])
        ref=self.reference()
        tight=evaluate_surface_v40(ref,shift,t,C_map=1.,threshold_m=.05)
        loose=evaluate_surface_v40(ref,shift,t,C_map=1.,threshold_m=.3)
        self.assertLess(tight['Q'],loose['Q']);self.assertAlmostEqual(loose['Q'],1.)

    def test_fov_and_range_reject_unobservable_target(self):
        with self.assertRaisesRegex(ValueError,'no observable'):
            freeze_reference_v40(*box(),(camera((.5,-3,.5),far=1.),),sample_spacing_m=.2)
        with self.assertRaisesRegex(ValueError,'no observable'):
            freeze_reference_v40(*box(),(camera((.5,-3,.5),target=(.5,-4,.5)),),sample_spacing_m=.2)

    def test_axial_depth_clipping_accepts_off_axis_surface_beyond_radial_far(self):
        view=CandidateViewV40(np.array([[64.,0,63.5],[0,64.,47.5],[0,0,1.]]),np.eye(4),128,96,far_m=4.)
        ref=freeze_reference_v40(*box((2.,-.2,3.5),(3.,.2,3.7)),(view,),sample_spacing_m=.1)
        self.assertTrue(np.all(np.linalg.norm(ref.points,axis=1)>4.))
        self.assertTrue(np.all(ref.points[:,2]<4.))
        self.assertIn('optical axial z',ref.manifest()['depth_convention'])
        with self.assertRaisesRegex(ValueError,'no observable'):
            freeze_reference_v40(*box((0.,-.2,4.1),(.5,.2,4.3)),(view,),sample_spacing_m=.1)

    def test_reference_deterministic_and_no_route_or_semantic_input(self):
        a,b=self.reference(),self.reference()
        self.assertEqual(a.fingerprint,b.fingerprint);np.testing.assert_array_equal(a.points,b.points)
        with self.assertRaises(TypeError): freeze_reference_v40(*box(),views(),planner_route=[])
        v,t,_=box()
        with self.assertRaises(TypeError): evaluate_surface_v40(a,v,t,C_map=1.,prediction_owner=[0])

    def test_invalid_or_meaningless_meshes_and_duplicate_triangles_rejected(self):
        v,t,ids=box()
        with self.assertRaises(ValueError): freeze_reference_v40(v,t,np.full(len(t),-1),views())
        with self.assertRaises(ValueError): freeze_reference_v40(np.empty((0,3)),np.empty((0,3),np.int64),np.empty(0,np.int64),views())
        bad=t.copy();bad[0]=[0,0,1]
        with self.assertRaises(ValueError): freeze_reference_v40(v,bad,ids,views())
        with self.assertRaises(ValueError): freeze_reference_v40(v,np.concatenate((t,t[:1])),np.concatenate((ids,ids[:1])),views())
        with self.assertRaises(ValueError): evaluate_surface_v40(self.reference(),v,bad,C_map=1.)

    def test_sampling_cap_and_coverage_validation(self):
        with self.assertRaisesRegex(ValueError,'sample limit'):
            freeze_reference_v40(*box(),views(),sample_spacing_m=.001,max_samples=100)
        with self.assertRaisesRegex(ValueError,'sample limit'):
            freeze_reference_v40(*box(),views(),sample_spacing_m=1e-300,max_samples=100)
        v,t,_=box();ref=self.reference()
        for coverage in (-.1,1.1,float('nan'),True):
            with self.assertRaises(ValueError): evaluate_surface_v40(ref,v,t,C_map=coverage)

    def test_reference_mutation_detected_before_scoring(self):
        ref=self.reference();ref.points.flags.writeable=True;ref.points[0,0]+=1.
        v,t,_=box()
        with self.assertRaisesRegex(ValueError,'reference content changed'):
            evaluate_surface_v40(ref,v,t,C_map=1.)


if __name__=='__main__': unittest.main()
