"""Analytic marker/geometry contracts, without simulator or reconstruction."""
import json
import unittest

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40,InstanceBeliefV40

PALETTE={2:[40,100,220],3:[220,60,40]}


def observation(step,patches=((2,18),(3,42)),origin_x=0.,frame_id=None,valid_depth=True):
    rgb=np.full((48,64,3),127,np.uint8);depth=np.zeros((48,64),np.float32)
    for label,x in patches:
        x=int(round(x-origin_x*10))
        rgb[22:26,x:x+4]=PALETTE[label]
        if valid_depth: depth[22:26,x:x+4]=2.
    k=np.array([[20.,0,31.5],[0,20.,23.5],[0,0,1.]])
    t=np.eye(4);t[0,3]=origin_x
    return PaidRGBDObservationV40.from_mapping(dict(frame_id=frame_id or f'paid-{step}',paid_step=step,
        rgb=rgb,depth_m=depth,intrinsic=k,world_from_camera=t))


def ledger(mode='S',**kwargs):
    return InstanceBeliefV40(palette=PALETTE,structure_names=['shape_0','shape_1'],
        class_structure_prior={2:[.9,.1],3:[.1,.9]},mode=mode,**kwargs)


class InstanceContractTests(unittest.TestCase):
    def test_two_classes_are_two_instances_not_global_conflict(self):
        model=ledger();model.observe(observation(0));model.observe(observation(1,origin_x=.5))
        instances=model.snapshot()['instances']
        self.assertEqual(len(instances),2)
        self.assertTrue(all(not i['class_conflict'] and i['semantic_conditioning_used'] for i in instances))
        np.testing.assert_allclose(instances[0]['structure_probabilities'],[.9,.1])
        np.testing.assert_allclose(instances[1]['structure_probabilities'],[.1,.9])

    def test_backprojection_world_anchor_invariant_under_camera_translation(self):
        model=ledger();a=model.observe(observation(0));b=model.observe(observation(1,origin_x=.5))
        np.testing.assert_allclose(a['accepted'][0]['anchor_world_m'],[-1.2,0.,2.],atol=1e-12)
        np.testing.assert_allclose(a['accepted'][0]['anchor_world_m'],b['accepted'][0]['anchor_world_m'],atol=1e-12)
        self.assertEqual([x['instance_id'] for x in a['accepted']],[x['instance_id'] for x in b['accepted']])

    def test_geometry_equal_in_s_and_g_but_structure_conditioning_differs(self):
        s,g=ledger('S'),ledger('G')
        for step,origin in enumerate((0.,.5)):
            obs=observation(step,origin_x=origin);s.observe(obs);g.observe(obs)
        self.assertEqual(s.geometry_snapshot(),g.geometry_snapshot())
        self.assertTrue(s.snapshot()['instances'][0]['semantic_conditioning_used'])
        self.assertFalse(g.snapshot()['instances'][0]['semantic_conditioning_used'])
        np.testing.assert_allclose(g.snapshot()['instances'][0]['structure_probabilities'],[.5,.5])

    def test_world_anchor_respects_camera_rotation_and_translation(self):
        obs=observation(0,((2,18),));t=np.eye(4)
        t[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];t[:3,3]=[1,2,3]
        transformed=PaidRGBDObservationV40('rotated',0,obs.rgb,obs.depth_m,obs.intrinsic,t)
        receipt=ledger().observe(transformed)
        np.testing.assert_allclose(receipt['accepted'][0]['anchor_world_m'],[1,.8,5],atol=1e-12)

    def test_palette_permutation_and_class_renaming_do_not_change_association(self):
        a=ledger();b=InstanceBeliefV40(palette={'renamed_blue':PALETTE[3],'renamed_red':PALETTE[2]},
            structure_names=['shape_0','shape_1'],
            class_structure_prior={'renamed_blue':[.9,.1],'renamed_red':[.1,.9]})
        for step,origin in enumerate((0.,.5)):
            obs=observation(step,origin_x=origin);a.observe(obs);b.observe(obs)
            self.assertEqual(a.geometry_snapshot(),b.geometry_snapshot())

    def test_three_structures_and_multimodal_class_prior(self):
        model=InstanceBeliefV40(palette=PALETTE,structure_names=['front','side','both'],
            class_structure_prior={2:[.6,.3,.1],3:[.3,.5,.2]})
        model.observe(observation(0));model.observe(observation(1,origin_x=.5))
        np.testing.assert_allclose(model.snapshot()['instances'][0]['structure_probabilities'],[.6,.3,.1])
        out=model.apply_geometry_feedback('instance_0000',frame_id='paid-1',log_likelihoods=[-3,-3,0])
        self.assertGreater(out['instance']['structure_probabilities'][2],.6)

    def test_small_pose_change_across_grid_boundary_does_not_add_evidence(self):
        model=ledger();model.observe(observation(0,origin_x=.2499))
        first=model.apply_geometry_feedback('instance_0000',frame_id='paid-0',log_likelihoods=[0,-2])
        model.observe(observation(1,origin_x=.2501))
        repeat=model.apply_geometry_feedback('instance_0000',frame_id='paid-1',log_likelihoods=[0,-2])
        self.assertTrue(first['applied']);self.assertFalse(repeat['applied'])
        self.assertEqual(first['instance']['geometry_log_scores'],repeat['instance']['geometry_log_scores'])
        self.assertFalse(repeat['instance']['semantic_conditioning_used'])

    def test_repeated_frame_and_nonconsecutive_paid_step_are_rejected(self):
        model=ledger();obs=observation(0);model.observe(obs);before=model.snapshot()
        with self.assertRaises(ValueError): model.observe(obs)
        with self.assertRaises(ValueError): model.observe(observation(2))
        with self.assertRaises(ValueError): model.observe(observation(1,frame_id='paid-0'))
        self.assertEqual(before,model.snapshot())

    def test_ambiguous_association_suspends_prior_then_clear_observation_restores(self):
        model=ledger();patches=((2,25),(3,35))
        model.observe(observation(0,patches));model.observe(observation(1,patches,origin_x=.5))
        self.assertTrue(all(x['semantic_conditioning_used'] for x in model.snapshot()['instances']))
        bad=model.observe(observation(2,((2,30),)))
        self.assertEqual(bad['accepted'],[])
        self.assertEqual(bad['rejected'][0]['reason'],'ambiguous_geometric_association')
        self.assertTrue(all(x['association_uncertain'] and not x['semantic_conditioning_used'] for x in bad['instances']))
        good=model.observe(observation(3,patches))
        self.assertTrue(all(not x['association_uncertain'] and x['semantic_conditioning_used'] for x in good['instances']))

    def test_nearby_new_components_are_rejected_without_fabricating_owner(self):
        model=ledger();r=model.observe(observation(0,((2,25),(3,30))))
        self.assertEqual(len(r['rejected']),2);self.assertEqual(r['instances'],[])

    def test_label_conflict_is_instance_local_and_geometry_still_updates(self):
        model=ledger();model.observe(observation(0));model.observe(observation(1,origin_x=.5))
        r=model.observe(observation(2,((3,18),(3,42)),origin_x=1.))
        self.assertTrue(r['instances'][0]['class_conflict'])
        self.assertFalse(r['instances'][0]['semantic_conditioning_used'])
        self.assertFalse(r['instances'][1]['class_conflict'])
        self.assertTrue(r['instances'][1]['semantic_conditioning_used'])
        result=model.apply_geometry_feedback('instance_0000',frame_id='paid-2',log_likelihoods=[-4,0])
        self.assertGreater(result['instance']['structure_probabilities'][1],.95)

    def test_geometry_feedback_cannot_cross_instances_or_future_frames(self):
        model=ledger();model.observe(observation(0));model.observe(observation(1,((3,42),),origin_x=.5))
        with self.assertRaises(ValueError): model.apply_geometry_feedback('instance_0000',frame_id='paid-1',log_likelihoods=[0,-1])
        with self.assertRaises(ValueError): model.apply_geometry_feedback('instance_0000',frame_id='paid-9',log_likelihoods=[0,-1])
        before=model.snapshot()['instances'][1]['geometry_log_scores']
        model.apply_geometry_feedback('instance_0000',frame_id='paid-0',log_likelihoods=[0,-4])
        self.assertEqual(before,model.snapshot()['instances'][1]['geometry_log_scores'])

    def test_duplicate_geometry_feedback_is_idempotent(self):
        model=ledger();model.observe(observation(0))
        first=model.apply_geometry_feedback('instance_0000',frame_id='paid-0',log_likelihoods=[0,-100])
        second=model.apply_geometry_feedback('instance_0000',frame_id='paid-0',log_likelihoods=[0,-100])
        self.assertTrue(first['applied']);self.assertFalse(second['applied'])
        self.assertEqual(first['instance']['geometry_log_scores'],[0.,-6.])
        self.assertEqual(first['instance']['geometry_log_scores'],second['instance']['geometry_log_scores'])

    def test_empty_unknown_color_and_depthless_frames_are_legal_without_instances(self):
        model=ledger();empty=observation(0,())
        self.assertEqual(model.observe(empty)['instances'],[])
        rgb=empty.rgb.copy();rgb[20:28,20:28]=[1,2,3]
        unknown=PaidRGBDObservationV40('paid-1',1,rgb,np.ones_like(empty.depth_m),empty.intrinsic,empty.world_from_camera)
        self.assertEqual(model.observe(unknown)['instances'],[])
        depthless=model.observe(observation(2,valid_depth=False))
        self.assertEqual(depthless['instances'],[])
        self.assertEqual({r['reason'] for r in depthless['rejected']},{'insufficient_measured_depth'})

    def test_strict_input_whitelist_rejects_truth_and_future_fields(self):
        obs=observation(0);payload={name:getattr(obs,name) for name in ('frame_id','paid_step','rgb','depth_m','intrinsic','world_from_camera')}
        for forbidden in ('owner','hypothesis','semantic','future_depth','world','ground_truth'):
            with self.subTest(field=forbidden),self.assertRaises(ValueError):
                PaidRGBDObservationV40.from_mapping(dict(payload,**{forbidden:0}))
        with self.assertRaises(TypeError): ledger().observe(payload)
        model=ledger();model.observe(obs)
        with self.assertRaises(TypeError): model.apply_geometry_feedback('instance_0000',frame_id='paid-0',log_likelihoods=[0,-1],ground_truth=1)

    def test_input_arrays_owned_readonly_and_snapshots_do_not_mutate_state(self):
        obs=observation(0)
        with self.assertRaises(ValueError): obs.rgb[0,0]=0
        model=ledger();r=model.observe(obs);json.dumps(r,allow_nan=False)
        snap=model.geometry_snapshot();snap['instances'][0]['anchor_world_m'][0]=999.
        self.assertNotEqual(model.geometry_snapshot()['instances'][0]['anchor_world_m'][0],999.)
        json.dumps(model.snapshot(),allow_nan=False)

    def test_invalid_calibration_and_nonfinite_feedback_rejected(self):
        obs=observation(0);bad=obs.world_from_camera.copy();bad[0,0]=2
        with self.assertRaises(ValueError): PaidRGBDObservationV40('a',0,obs.rgb,obs.depth_m,obs.intrinsic,bad)
        model=ledger();model.observe(obs)
        for score in ([float('nan'),0],[float('inf'),0],[0,1,2]):
            with self.assertRaises(ValueError): model.apply_geometry_feedback('instance_0000',frame_id='paid-0',log_likelihoods=score)

    def test_view_capacity_disables_new_feedback_without_reusing_old_bin(self):
        model=ledger(maximum_views=1,minimum_semantic_views=1)
        model.observe(observation(0));r=model.observe(observation(1,origin_x=.5))
        self.assertIsNone(r['accepted'][0]['view_id'])
        self.assertFalse(model.apply_geometry_feedback('instance_0000',frame_id='paid-1',log_likelihoods=[0,-1])['applied'])


if __name__=='__main__': unittest.main()
