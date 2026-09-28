"""Meaningful input/safety boundaries of the independent observed-ground revision."""
import unittest
import numpy as np
from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_article_v1 import ObservedInstancesArticleV1
from nso.observed_instances_article_ground_v2 import (
    GROUND_POLICY, ObservedInstancesArticleGroundV2, ground_mask_from_observation)


def floor_packet(*, ground_z=0., noise_z=0., narrow=False):
    height,width=72,96
    yy,xx=np.indices((height,width));k=np.array([[48.,0,47.5],[0,48.,35.5],[0,0,1.]])
    t=np.eye(4);t[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];t[:3,3]=[.75,.75,.9]
    dz=-(yy-35.5)/48.;z=np.full((height,width),ground_z)+noise_z*((xx+yy)%3-1.)
    valid=dz < -.24
    if narrow:valid &= (xx>=46)&(xx<50)
    depth=np.zeros((height,width),np.float32)
    depth[valid]=((z[valid]-.9)/dz[valid]).astype(np.float32)
    depth[depth>4.]=0.
    rgb=np.zeros((height,width,3),np.uint8)
    return PaidRGBDObservationV40('synthetic_paid:0',0,rgb,depth,k,t)


def replace(obs, **changed):
    values={k:getattr(obs,k) for k in ('frame_id','paid_step','rgb','depth_m','intrinsic','world_from_camera')}
    values.update(changed)
    return PaidRGBDObservationV40(**values)


def frontend(cls):
    return cls(palette={'cabinet':[40,100,220]},structure_names=['planar','recessed','louvered','open_frame'],
               class_structure_prior={'cabinet':[.4,.3,.2,.1]}, mode='S', maximum_instances=8)


class GroundAssociationV2Tests(unittest.TestCase):
    def test_measured_broad_floor_is_accepted(self):
        obs=floor_packet(noise_z=.004)
        mask,r=ground_mask_from_observation(obs)
        self.assertTrue(r['accepted']);self.assertGreater(r['excluded_pixels'],64)
        self.assertTrue(all(v>=.5 for v in r['validated_xy_span_m']))
        self.assertFalse(mask.flags.writeable)
        self.assertFalse(r['class_values_read']);self.assertFalse(r['private_geometry_read'])

    def test_insufficient_and_absent_ground_fall_back(self):
        obs=floor_packet();depth=np.zeros_like(obs.depth_m);depth[-1,:10]=obs.depth_m[-1,:10]
        mask,r=ground_mask_from_observation(replace(obs,depth_m=depth))
        self.assertFalse(mask.any());self.assertEqual(r['reason'],'insufficient_near_ground_support')
        mask,r=ground_mask_from_observation(replace(obs,depth_m=np.zeros_like(depth)))
        self.assertFalse(mask.any());self.assertFalse(r['accepted'])

    def test_noise_inconsistent_with_declared_sensor_falls_back(self):
        mask,r=ground_mask_from_observation(floor_packet(noise_z=.06))
        self.assertFalse(mask.any());self.assertEqual(r['reason'],'measured_ground_residual_exceeds_noise_contract')

    def test_wrong_ground_or_mount_height_falls_back(self):
        mask,r=ground_mask_from_observation(floor_packet(ground_z=.06))
        self.assertFalse(mask.any());self.assertEqual(r['reason'],'ground_height_inconsistent_with_public_mount')
        obs=floor_packet();t=obs.world_from_camera.copy();t[2,3]+=.1
        # A world-frame translation changes both reconstructed floor and its
        # expected public-height hypothesis equally: translation is not error.
        moved_mask,moved=ground_mask_from_observation(replace(obs,world_from_camera=t))
        self.assertTrue(moved['accepted']);self.assertTrue(moved_mask.any())

    def test_narrow_strip_cannot_certify_a_floor(self):
        mask,r=ground_mask_from_observation(floor_packet(narrow=True))
        self.assertFalse(mask.any());self.assertEqual(r['reason'],'insufficient_ground_spatial_extent')

    def test_forbidden_truth_fields_and_packet_subclasses_are_rejected(self):
        obs=floor_packet();fields={k:getattr(obs,k) for k in ('frame_id','paid_step','rgb','depth_m','intrinsic','world_from_camera')}
        for key in ('ground_truth_owner','floor_height','triangle_instance_id','private_geometry'):
            with self.assertRaises(ValueError):PaidRGBDObservationV40.from_mapping(dict(fields,**{key:0}))
        with self.assertRaises(TypeError):ground_mask_from_observation(fields)
        with self.assertRaises(TypeError):ground_mask_from_observation(obs, ground_truth_floor=0.)
        class ExtendedPacket(PaidRGBDObservationV40):
            pass
        with self.assertRaises(TypeError):ground_mask_from_observation(ExtendedPacket(**fields))

    def test_raw_inputs_unchanged_and_color_does_not_select_ground(self):
        obs=floor_packet();before=obs.sha256();copies=[a.copy() for a in (obs.rgb,obs.depth_m,obs.intrinsic,obs.world_from_camera)]
        m1,r1=ground_mask_from_observation(obs)
        color=np.zeros_like(obs.rgb);color[:]=[40,100,220]
        m2,r2=ground_mask_from_observation(replace(obs,rgb=color))
        np.testing.assert_array_equal(m1,m2);self.assertEqual(r1['mask_sha256'],r2['mask_sha256'])
        self.assertEqual(before,obs.sha256())
        for value,old in zip((obs.rgb,obs.depth_m,obs.intrinsic,obs.world_from_camera),copies):
            np.testing.assert_array_equal(value,old);self.assertFalse(value.flags.writeable)

    def test_contact_boundary_is_only_an_association_exclusion(self):
        obs=floor_packet();depth=obs.depth_m.copy();row,col=60,48;dz=-(row-35.5)/48.
        depth[row,col]=(.03-.9)/dz;depth[row,col+1]=(.08-.9)/dz
        packet=replace(obs,depth_m=depth);mask,r=ground_mask_from_observation(packet)
        self.assertTrue(r['accepted']);self.assertTrue(mask[row,col]);self.assertFalse(mask[row,col+1])
        self.assertEqual(packet.depth_m[row,col],depth[row,col])
        self.assertTrue(r['full_frame_fusion_preserved'])

    def test_fallback_preserves_parent_association_exactly(self):
        obs=floor_packet(ground_z=.06)
        old=frontend(ObservedInstancesArticleV1).observe(obs)
        new=frontend(ObservedInstancesArticleGroundV2).observe(obs)
        receipt=new.pop('article_ground_association')
        self.assertFalse(receipt['accepted']);self.assertEqual(old,new)

    def test_association_floor_is_not_cached_as_an_object(self):
        obs=floor_packet();color=np.zeros_like(obs.rgb);color[55:60,46:51]=[40,100,220]
        obs=replace(obs,rgb=color)
        new=frontend(ObservedInstancesArticleGroundV2);receipt=new.observe(obs)
        self.assertTrue(receipt['article_ground_association']['accepted'])
        self.assertFalse(receipt['accepted']);self.assertFalse(new.snapshot()['instances'])
        for name in ('association_distance_m','maximum_support_points','maximum_evidence_frames','minimum_semantic_views'):
            self.assertEqual(getattr(new,name),getattr(frontend(ObservedInstancesArticleV1),name))
        with self.assertRaises(TypeError):GROUND_POLICY['camera_height_m']=1.


if __name__=='__main__':unittest.main()
