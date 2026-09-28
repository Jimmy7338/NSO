"""Analytic paid RGB-D contracts only; no World or trajectory experiment."""
import unittest
import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_v41 import ObservedInstancesV41,support_digest

PALETTE={'a':[40,100,220],'b':[220,60,40]}


def model(mode='S',**kwargs):
    return ObservedInstancesV41(palette=PALETTE,structure_names=['front','side','both'],
        class_structure_prior={'a':[.7,.2,.1],'b':[.1,.2,.7]},mode=mode,**kwargs)


def planar(step,patches=((8,20,'a'),),*,z=2.,origin_x=0.,frame_id=None,rgb_delta=0):
    rgb=np.full((32,64,3),127+rgb_delta,np.uint8);depth=np.zeros((32,64),float)
    for first,last,label in patches:
        depth[10:22,first:last]=z
        if label is not None: rgb[14:18,first+2:first+6]=PALETTE[label]
    k=np.array([[40.,0,31.5],[0,40.,15.5],[0,0,1.]])
    t=np.eye(4);t[0,3]=origin_x
    return PaidRGBDObservationV40(frame_id or f'paid-{step}',step,rgb,depth,k,t)


def box_observation(step,origin,*,marker=False):
    """Analytic ray/AABB depths of one closed box, only inside this test fixture.

    No owner, box bounds or ray-hit identity is passed to the ledger.
    """
    h,w=48,64;k=np.array([[50.,0,31.5],[0,50.,23.5],[0,0,1.]])
    origin=np.asarray(origin,float);target=np.array([0.,0.,2.45]);z=target-origin;z/=np.linalg.norm(z)
    x=np.cross(z,[0.,1.,0.]);x/=np.linalg.norm(x);y=np.cross(z,x)
    t=np.eye(4);t[:3,:3]=np.column_stack((x,y,z));t[:3,3]=origin
    yy,xx=np.indices((h,w));rays=np.column_stack((xx.ravel(),yy.ravel(),np.ones(h*w)))@np.linalg.inv(k).T
    directions=rays@t[:3,:3].T;lo=np.array([-.45,-.45,2.]);hi=np.array([.45,.45,2.9])
    with np.errstate(divide='ignore',invalid='ignore'):
        a=(lo-origin)/directions;b=(hi-origin)/directions
    near=np.minimum(a,b).max(axis=1);far=np.maximum(a,b).min(axis=1)
    hit=(far>=np.maximum(near,0))&(near>.1)&(near<=4.)
    depth=np.where(hit,near,0).reshape(h,w);rgb=np.full((h,w,3),127,np.uint8)
    points=origin+directions*near[:,None]
    visible_marker=hit&(np.abs(points[:,2]-2.)<1e-6)&(np.abs(points[:,0])<.18)&(np.abs(points[:,1])<.18)
    if marker: rgb.reshape(-1,3)[visible_marker]=PALETTE['a']
    return PaidRGBDObservationV40(f'box-{step}',step,rgb,depth,k,t)


class ObservedInstanceTests(unittest.TestCase):
    def test_separate_instances_and_semantic_independence(self):
        s,g=model('S',minimum_semantic_views=1),model('G',minimum_semantic_views=1)
        obs=planar(0,((4,16,'a'),(44,56,'b')))
        for ledger in (s,g): self.assertEqual(len(ledger.observe(obs)['accepted']),2)
        self.assertEqual(s.geometry_snapshot(),g.geometry_snapshot())
        self.assertTrue(all(x['semantic_conditioning_used'] for x in s.snapshot()['instances']))
        self.assertFalse(any(x['semantic_conditioning_used'] for x in g.snapshot()['instances']))
        self.assertFalse(any(x['class_conflict'] for x in s.snapshot()['instances']))

    def test_palette_names_do_not_change_geometry_association(self):
        a=model();b=ObservedInstancesV41(palette={'other_a':PALETTE['a'],'other_b':PALETTE['b']},
            structure_names=['front','side','both'],class_structure_prior={'other_a':[.1,.2,.7],'other_b':[.7,.2,.1]})
        for step,patch in enumerate((((8,20,'a'),),((16,28,None),))):
            obs=planar(step,patch);a.observe(obs);b.observe(obs)
            self.assertEqual(a.geometry_snapshot(),b.geometry_snapshot())

    def test_markerless_patch_bridges_to_new_measured_support(self):
        ledger=model();first=ledger.observe(planar(0));key=first['accepted'][0]['instance_id']
        middle=ledger.observe(planar(1,((16,28,None),)))
        final=ledger.observe(planar(2,((24,36,None),)))
        self.assertEqual([x['instance_id'] for x in middle['accepted']],[key])
        self.assertEqual([x['instance_id'] for x in final['accepted']],[key])
        self.assertEqual(final['accepted'][0]['marker_pixel_indices'],[])
        self.assertTrue(final['accepted'][0]['geometry_feedback_eligible'])
        unbridged=model();unbridged.observe(planar(0))
        self.assertFalse(unbridged.observe(planar(1,((24,36,None),)))['accepted'])

    def test_box_front_side_back_requires_paid_intermediate_views(self):
        ledger=model(max_growth_distance_m=.55)
        origins=[(0,0,0),(-1.5,0,1.5),(-1.5,0,2.5),(-1.5,0,3.5),(0,0,4.8)]
        receipts=[ledger.observe(box_observation(step,origin,marker=step==0)) for step,origin in enumerate(origins)]
        self.assertTrue(all(len(r['accepted'])==1 for r in receipts),[r['rejected'] for r in receipts])
        self.assertEqual({r['accepted'][0]['instance_id'] for r in receipts},{'instance_0000'})
        self.assertEqual(receipts[-1]['accepted'][0]['marker_pixel_indices'],[])
        self.assertTrue(receipts[-1]['accepted'][0]['geometry_feedback_eligible'])
        direct=model();direct.observe(box_observation(0,origins[0],marker=True))
        self.assertFalse(direct.observe(box_observation(1,origins[-1]))['accepted'])

    def test_many_to_one_occlusion_and_two_instance_overlap_rejected(self):
        ledger=model(minimum_semantic_views=1)
        ledger.observe(planar(0,((4,16,'a'),(28,40,'b'))))
        mixed=ledger.observe(planar(1,((4,40,None),)))
        self.assertFalse(mixed['accepted'])
        self.assertEqual(mixed['rejected'][0]['reason'],'ambiguous_geometric_association')
        self.assertTrue(all(i['association_uncertain'] for i in mixed['instances']))
        clear=ledger.observe(planar(2,((4,16,None),(28,40,None))))
        self.assertEqual(len(clear['accepted']),2)
        self.assertFalse(any(i['association_uncertain'] for i in clear['instances']))
        single=model();single.observe(planar(0,((8,28,'a'),)))
        split=single.observe(planar(1,((8,16,None),(20,28,None))))
        self.assertFalse(split['accepted'])
        self.assertTrue(all(r['reason']=='multiple_components_claim_instance' for r in split['rejected']))

    def test_same_support_does_not_reinforce_on_rgb_or_tiny_pose_change(self):
        ledger=model();receipt=ledger.observe(planar(0));a=receipt['accepted'][0]
        applied=ledger.apply_geometry_feedback(a['instance_id'],frame_id=receipt['frame_id'],
            observation_sha256=receipt['observation_sha256'],log_likelihoods=[0,-2,-2])
        self.assertTrue(applied['applied'])
        second=ledger.observe(planar(1,rgb_delta=1,origin_x=.001));b=second['accepted'][0]
        self.assertFalse(b['geometry_feedback_eligible'])
        repeated=ledger.apply_geometry_feedback(b['instance_id'],frame_id=second['frame_id'],
            observation_sha256=second['observation_sha256'],log_likelihoods=[0,-2,-2])
        self.assertFalse(repeated['applied'])
        self.assertEqual(applied['instance']['geometry_log_scores'],repeated['instance']['geometry_log_scores'])

    def test_rotation_with_new_field_of_view_is_eligible(self):
        # Camera stays at one origin; change its yaw and actual planar depth.
        h,w=32,64;k=np.array([[40.,0,31.5],[0,40.,15.5],[0,0,1.]])
        def view(step,yaw):
            t=np.eye(4);c,s=np.cos(yaw),np.sin(yaw);t[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
            yy,xx=np.indices((h,w));rays=np.stack(((xx-31.5)/40,(yy-15.5)/40,np.ones((h,w))),axis=-1)
            direction=rays@t[:3,:3].T;z=2/direction[:,:,2];points=direction*z[:,:,None]
            valid=(np.abs(points[:,:,0])<1)&(np.abs(points[:,:,1])<.4)
            rgb=np.full((h,w,3),127,np.uint8)
            if step==0: rgb[14:18,30:34]=PALETTE['a']
            return PaidRGBDObservationV40(f'rotate-{step}',step,rgb,np.where(valid,z,0),k,t)
        ledger=model(bootstrap_radius_m=.5)
        ledger.observe(view(0,0.));out=ledger.observe(view(1,.2))
        self.assertEqual(len(out['accepted']),1)
        self.assertTrue(out['accepted'][0]['geometry_feedback_eligible'])
        self.assertEqual(out['accepted'][0]['marker_pixel_indices'],[])

    def test_duplicate_frame_id_future_and_old_feedback_rejected(self):
        ledger=model();obs=planar(0);first=ledger.observe(obs);before=ledger.snapshot()
        for duplicate in (obs,PaidRGBDObservationV40(obs.frame_id,1,obs.rgb,obs.depth_m,obs.intrinsic,obs.world_from_camera),planar(2)):
            with self.assertRaises(ValueError): ledger.observe(duplicate)
        self.assertEqual(before,ledger.snapshot())
        ledger.observe(planar(1,((16,28,None),)))
        with self.assertRaises(ValueError): ledger.apply_geometry_feedback('instance_0000',frame_id='paid-0',
            observation_sha256=first['observation_sha256'],log_likelihoods=[0,-1,-1])

    def test_new_paid_identical_measurement_accepted_without_support_or_belief_growth(self):
        ledger=model();obs=planar(0,((8,40,'a'),));first=ledger.observe(obs)
        ledger.apply_geometry_feedback('instance_0000',frame_id=obs.frame_id,
            observation_sha256=obs.sha256(),log_likelihoods=[0,-2,-1])
        before=ledger.snapshot()['instances'][0]
        repeated=PaidRGBDObservationV40('new-paid-1',1,obs.rgb,obs.depth_m,obs.intrinsic,obs.world_from_camera)
        second=ledger.observe(repeated);association=second['accepted'][0]
        self.assertTrue(second['duplicate_measurement']);self.assertTrue(second['no_new_support'])
        self.assertTrue(association['duplicate_measurement']);self.assertTrue(association['no_new_support'])
        self.assertEqual(association['novel_support_voxels'],0)
        self.assertFalse(association['geometry_feedback_eligible'])
        self.assertEqual(association['observation_sha256'],repeated.sha256())
        self.assertNotEqual(association['observation_sha256'],first['observation_sha256'])
        after=ledger.snapshot()['instances'][0]
        for key in ('support_sha256','distinct_class_supports','structure_probabilities','geometry_log_scores','novel_support_frames'):
            self.assertEqual(before[key],after[key],key)
        self.assertFalse(ledger.apply_geometry_feedback('instance_0000',frame_id=repeated.frame_id,
            observation_sha256=repeated.sha256(),log_likelihoods=[0,-2,-1])['applied'])
        with self.assertRaises(ValueError): ledger.apply_geometry_feedback('instance_0000',frame_id=repeated.frame_id,
            observation_sha256=obs.sha256(),log_likelihoods=[0,-2,-1])

    def test_consecutive_paid_all_missing_frames_are_accepted_without_evidence(self):
        ledger=model()
        for step in range(3):
            result=ledger.observe(planar(step,()))
            self.assertFalse(result['accepted']);self.assertEqual(result['paid_step'],step)
            self.assertTrue(result['no_new_support'])
            self.assertEqual(result['duplicate_measurement'],step>0)

    def test_feedback_requires_exact_hash_unique_mask_and_only_once(self):
        ledger=model();receipt=ledger.observe(planar(0));a=receipt['accepted'][0]
        with self.assertRaises(ValueError): ledger.apply_geometry_feedback(a['instance_id'],frame_id='paid-0',observation_sha256='bad',log_likelihoods=[0,-2,-1])
        kwargs=dict(frame_id='paid-0',observation_sha256=receipt['observation_sha256'],log_likelihoods=[0,-2,-1])
        self.assertTrue(ledger.apply_geometry_feedback(a['instance_id'],**kwargs)['applied'])
        self.assertFalse(ledger.apply_geometry_feedback(a['instance_id'],**kwargs)['applied'])
        self.assertEqual(a['support_sha256'],support_digest(a['points_world_m']))
        obs=planar(0);pixels=np.asarray(a['pixel_indices']);y,x=np.divmod(pixels,64)
        points=np.column_stack((x,y,np.ones(len(x))))@np.linalg.inv(obs.intrinsic).T*obs.depth_m[y,x,None]
        np.testing.assert_array_equal(points,a['points_world_m'])

    def test_axial_clipping_off_axis_and_out_of_range(self):
        ledger=model(bootstrap_radius_m=1.2,depth_link_distance_m=.25)
        obs=planar(0,((54,64,'a'),),z=3.5)
        out=ledger.observe(obs)
        self.assertEqual(len(out['accepted']),1)
        self.assertTrue(np.any(np.linalg.norm(out['accepted'][0]['points_world_m'],axis=1)>4.))
        for step,z in ((1,4.1),(2,.05)):
            self.assertFalse(ledger.observe(planar(step,z=z))['accepted'])

    def test_unknown_labels_empty_depth_and_strict_input(self):
        ledger=model();unknown=planar(0,((8,20,None),));self.assertFalse(ledger.observe(unknown)['accepted'])
        blank=planar(1,());self.assertFalse(ledger.observe(blank)['accepted'])
        with self.assertRaises(TypeError): ledger.observe(dict(frame_id='gt',owner=0))
        with self.assertRaises(ValueError): PaidRGBDObservationV40.from_mapping({'owner':0})

    def test_class_conflict_is_local_and_conservative(self):
        ledger=model(minimum_semantic_views=1);ledger.observe(planar(0,((4,16,'a'),(44,56,'b'))))
        out=ledger.observe(planar(1,((4,16,'b'),(44,56,'b')),rgb_delta=1))
        self.assertTrue(out['instances'][0]['class_conflict'])
        self.assertFalse(out['instances'][0]['semantic_conditioning_used'])
        self.assertFalse(out['instances'][1]['class_conflict'])

    def test_close_bootstraps_reject_both_instead_of_ordered_winner(self):
        ledger=model()
        out=ledger.observe(planar(0,((8,20,'a'),(21,33,'b'))))
        self.assertFalse(out['accepted']);self.assertFalse(out['instances'])
        self.assertEqual(len(out['rejected']),2)
        self.assertTrue(all(r['reason']=='ambiguous_same_frame_bootstrap' for r in out['rejected']))

    def test_oversize_connected_background_is_not_absorbed_by_marker(self):
        out=model().observe(planar(0,((0,64,'a'),)))
        self.assertFalse(out['accepted']);self.assertFalse(out['instances'])
        self.assertEqual(out['rejected'][0]['reason'],'oversize_depth_component')

    def test_growth_capacity_and_geometric_score_caps(self):
        ledger=model(maximum_evidence_frames=2,minimum_semantic_views=1,total_geometry_cap=3.)
        for step,patch in enumerate((((8,20,'a'),),((16,28,None),),((24,36,None),))):
            receipt=ledger.observe(planar(step,patch));a=receipt['accepted'][0]
            result=ledger.apply_geometry_feedback(a['instance_id'],frame_id=receipt['frame_id'],
                observation_sha256=receipt['observation_sha256'],log_likelihoods=[0,-100,-100])
            self.assertGreaterEqual(min(result['instance']['geometry_log_scores']),-3.)
            if step==2: self.assertFalse(result['applied'])


if __name__=='__main__': unittest.main()
