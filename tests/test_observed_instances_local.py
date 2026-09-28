"""Bounded local proposal tests on analytic paid RGB-D; zero World calls."""
import unittest

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_local import ObservedInstancesLocal
from nso.observed_instances_v41 import ObservedInstancesV41


PALETTE = {'a': (40,100,220), 'b': (220,60,40)}


def ledger(cls=ObservedInstancesLocal, mode='S', **kwargs):
    return cls(palette=PALETTE, structure_names=('front','side','both'),
        class_structure_prior={'a':(.7,.2,.1), 'b':(.1,.2,.7)}, mode=mode, **kwargs)


def packet(step, *, full=True, markers=((32,'a'),), patches=((26,46),)):
    rgb = np.full((32,96,3),127,np.uint8)
    depth = np.full((32,96),2.) if full else np.zeros((32,96))
    if not full:
        for start,end in patches:
            depth[8:24,start:end]=2.
    for start,label in markers:
        rgb[13:19,start:start+6]=PALETTE[label]
    intrinsic=np.array([[40.,0.,47.5],[0.,40.,15.5],[0.,0.,1.]])
    return PaidRGBDObservationV40(f'local-{step}',step,rgb,depth,intrinsic,np.eye(4))


class ObservedInstancesLocalTests(unittest.TestCase):
    def test_exact_acceleration_is_default_and_explicit_reference_agrees(self):
        fast=ledger()
        reference=ledger(exact_distance_acceleration=False)
        self.assertTrue(fast.exact_distance_acceleration)
        self.assertFalse(reference.exact_distance_acceleration)
        for step in range(3):
            self.assertEqual(fast.observe(packet(step)),reference.observe(packet(step)))
            self.assertEqual(fast.snapshot(),reference.snapshot())

    def test_large_connected_region_is_bounded_before_rejection(self):
        observation=packet(0)
        old=ledger(ObservedInstancesV41).observe(observation)
        self.assertFalse(old['accepted'])
        self.assertEqual(old['rejected'][0]['reason'],'oversize_depth_component')
        new=ledger().observe(observation)
        self.assertEqual(len(new['accepted']),1)
        points=np.asarray(new['accepted'][0]['points_world_m'])
        anchor=np.asarray(new['instances'][0]['marker_anchor_world_m'])
        self.assertLessEqual(np.linalg.norm(points-anchor,axis=1).max(),.6+1e-12)
        self.assertLess(len(points),observation.depth_m.size)
        self.assertGreater(new['local_proposal_audit'][0]['measured_raw_extent_m'][0],2.5)

    def test_two_overlapping_marker_regions_are_rejected(self):
        result=ledger().observe(packet(0,markers=((30,'a'),(40,'b'))))
        self.assertFalse(result['accepted'])
        self.assertFalse(result['instances'])
        self.assertTrue(any(r['reason'] in ('multiple_markers_within_local_support',
            'overlapping_local_support_proposals') for r in result['rejected']))

    def test_distant_disjoint_marker_regions_keep_separate_instances(self):
        result=ledger().observe(packet(0,markers=((12,'a'),(72,'b'))))
        self.assertEqual(len(result['accepted']),2)
        sets=[set(row['pixel_indices']) for row in result['accepted']]
        self.assertFalse(sets[0]&sets[1])

    def test_known_instance_growth_overlap_is_ambiguous(self):
        model=ledger()
        first=model.observe(packet(0,full=False,markers=((30,'a'),(48,'b')),
                                    patches=((28,40),(46,58))))
        self.assertEqual(len(first['accepted']),2)
        second=model.observe(packet(1,markers=()))
        self.assertFalse(second['accepted'])
        self.assertTrue(all(row['association_uncertain'] for row in second['instances']))
        self.assertTrue(any(row['reason']=='overlapping_local_support_proposals' for row in second['rejected']))

    def test_g_and_s_share_geometry_and_palette_names_do_not_select_regions(self):
        g,s=ledger(mode='G'),ledger(mode='S')
        renamed=ObservedInstancesLocal(palette={'x':PALETTE['a'],'y':PALETTE['b']},
            structure_names=('front','side','both'),
            class_structure_prior={'x':(.1,.2,.7),'y':(.7,.2,.1)})
        for model in (g,s,renamed):
            model.observe(packet(0,markers=((12,'a'),(72,'b'))))
        self.assertEqual(g.geometry_snapshot(),s.geometry_snapshot())
        self.assertEqual(g.geometry_snapshot(),renamed.geometry_snapshot())

    def test_duplicate_packet_does_not_create_support_or_feedback(self):
        model=ledger(); first=model.observe(packet(0));key=first['accepted'][0]['instance_id']
        model.apply_geometry_feedback(key,frame_id=first['frame_id'],
            observation_sha256=first['observation_sha256'],log_likelihoods=[0,-2,-1])
        prior=model.snapshot()['instances'][0]
        repeated=model.observe(packet(1))
        self.assertTrue(repeated['duplicate_measurement'])
        self.assertTrue(repeated['no_new_support'])
        self.assertTrue(repeated['accepted'])
        self.assertFalse(repeated['accepted'][0]['geometry_feedback_eligible'])
        after=model.snapshot()['instances'][0]
        for field in ('support_sha256','geometry_log_scores','distinct_class_supports','novel_support_frames'):
            self.assertEqual(prior[field],after[field])

    def test_normal_extent_behavior_matches_original(self):
        a,b=ledger(ObservedInstancesV41),ledger()
        for step in range(2):
            old=a.observe(packet(step,full=False));new=b.observe(packet(step,full=False))
            new.pop('local_proposal_audit');new.pop('proposal_construction')
            self.assertEqual(old,new)
        self.assertEqual(a.geometry_snapshot(),b.geometry_snapshot())

    def test_same_paid_feedback_hash_and_once_only_contract_remains(self):
        model=ledger(); receipt=model.observe(packet(0));key=receipt['accepted'][0]['instance_id']
        with self.assertRaises(ValueError):
            model.apply_geometry_feedback(key,frame_id=receipt['frame_id'],
                observation_sha256='wrong',log_likelihoods=[0,-1,-2])
        args=dict(frame_id=receipt['frame_id'],observation_sha256=receipt['observation_sha256'],
                  log_likelihoods=[0,-1,-2])
        self.assertTrue(model.apply_geometry_feedback(key,**args)['applied'])
        self.assertFalse(model.apply_geometry_feedback(key,**args)['applied'])


if __name__=='__main__':
    unittest.main()
