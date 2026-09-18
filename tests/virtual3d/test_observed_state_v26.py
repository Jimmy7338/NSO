"""Analytic boundary tests only: no simulated world, motion or TSDF fusion."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import json
import unittest
import numpy as np
from nso.cpu_sensor_contract_v10 import SensorPacket
from nso.observed_state_v26 import (GeometryPatchV26, GeometryStateV26, SemanticCueV26,
    VisibleSemanticMemoryV26, geometry_state_v26)
from utils.rgbd_contract import RGBDFrame, PlanarScan


def packet(action_id=0, *, colors=(), depth=2., episode='observed-episode', frame_id=None):
    size=16; focal=8.; position=(15,15)
    k=np.array([[focal,0.,7.5],[0.,focal,7.5],[0.,0.,1.]])
    t=np.array([[1.,0.,0.,3.1],[0.,0.,1.,3.1],[0.,-1.,0.,.8],[0.,0.,0.,1.]])
    rgb=np.zeros((size,size,3),np.uint8)
    for rows,cols,color in colors:rgb[rows,cols]=color
    frame=RGBDFrame(float(action_id),np.full((size,size),depth,np.float32),rgb,k,t,np.zeros((size,size),np.int32))
    laser=np.array([[0.,-1.,0.,3.1],[1.,0.,0.,3.1],[0.,0.,1.,.25],[0.,0.,0.,1.]])
    scan=PlanarScan(float(action_id),np.full(16,5.,np.float32),-np.pi,2*np.pi/16,5.,laser)
    return SensorPacket('analytic',episode,frame_id or f'frame-{action_id}',action_id,frame,scan,
        position,0,'analytic_measured_input','declared_discrete_pose',None if action_id==0 else 'right')


class NoLabelRow(dict):
    def __getitem__(self,key):
        if key=='label':raise AssertionError('G read a class label')
        return super().__getitem__(key)


def mapper(p, number=10):
    quality={}
    for i in range(number):
        quality[(i,0,0)]=NoLabelRow(point=np.array([1.+i*.15,3.,.7]),normal=np.array([0.,-1.,0.]),
            n=1,bits=1,best_range=2.,residual=.001,label=i%4,information=.25)
    return SimpleNamespace(belief=np.zeros((31,31),np.int8),
        config=SimpleNamespace(resolution_m=.2,robot_radius_m=.2,max_depth_m=5.,fov_deg=90.,
            width_px=16,height_px=16,camera_height_m=.8,private_class='must never be copied'),
        quality=quality,frames=p.action_id+1,current_footprint_conflict_details=dict(
            map_version=p.action_id+1,timestamp_s=p.frame.timestamp_s,current_cell=p.position))


def red():return (slice(5,9),slice(5,9),(220,60,40))
def blue():return (slice(5,9),slice(5,9),(40,100,220))


class GeometryStateTests(unittest.TestCase):
    def state(self,m,p,**kw):return geometry_state_v26(m,p,(15,15,0),50,**kw)

    def test_geometry_discovery_does_not_read_any_appearance_or_label(self):
        p=packet(colors=[red()]);m=mapper(p)
        first=self.state(m,p,max_patches=3)
        # Even inaccessible/non-array appearance is irrelevant at this boundary.
        changed=replace(p,episode_id='different-hidden-category-name',
            frame=replace(p.frame,color_rgb=object(),semantic=object()))
        second=self.state(m,changed,max_patches=3)
        self.assertEqual(first.geometry_sha256,second.geometry_sha256)
        self.assertEqual(first.patches,second.patches)
        self.assertEqual([x.key for x in first.patches],[(0,0,0),(4,0,0),(9,0,0)])
        self.assertEqual(len(first.patches),3)  # label 0/1 support also discovered.
        self.assertNotIn('config',vars(first));self.assertNotIn('mapper',vars(first))
        self.assertNotIn('episode_id',vars(first));self.assertNotIn('label',vars(first.patches[0]))

    def test_sampling_is_stable_across_mapping_insertion_order(self):
        p=packet();m=mapper(p);a=self.state(m,p,max_patches=4)
        m.quality=dict(reversed(list(m.quality.items())))
        b=self.state(m,p,max_patches=4)
        self.assertEqual(a.geometry_sha256,b.geometry_sha256)
        self.assertEqual(a.patches,b.patches)

    def test_snapshot_survives_mapper_mutation_and_cannot_be_made_writable(self):
        p=packet();m=mapper(p);state=self.state(m,p);point=state.patches[0].point
        m.belief[0,0]=1;m.quality[(0,0,0)]['point'][0]=5.;m.quality.clear()
        self.assertEqual(state.belief[0,0],0);self.assertEqual(state.patches[0].point,point)
        with self.assertRaises(ValueError):state.belief[0,0]=1
        with self.assertRaises(ValueError):state.belief.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):state.remaining_budget=0
        with self.assertRaises(FrozenInstanceError):state.patches[0].n=100

    def test_geometric_changes_and_budget_change_the_hash(self):
        p=packet();m=mapper(p);original=self.state(m,p).geometry_sha256
        m.quality[(0,0,0)]['residual']=.003
        self.assertNotEqual(original,self.state(m,p).geometry_sha256)
        m=mapper(p)
        self.assertNotEqual(original,geometry_state_v26(m,p,(15,15,0),49).geometry_sha256)
        m.belief[1,1]=-1
        self.assertNotEqual(original,self.state(m,p).geometry_sha256)

    def test_unknown_cells_are_preserved_and_no_quality_is_allowed(self):
        p=packet();m=mapper(p,0);m.belief[1,1]=-1;m.belief[2,2]=1
        state=self.state(m,p)
        self.assertEqual(state.belief[1,1],-1);self.assertEqual(state.belief[2,2],1)
        self.assertEqual(state.patches,())

    def test_bad_budget_pose_map_and_stale_mapper_rejected(self):
        p=packet();m=mapper(p)
        for amount in (-1,True,1.5,float('nan')):
            with self.subTest(amount=amount),self.assertRaises(ValueError):geometry_state_v26(m,p,(15,15,0),amount)
        with self.assertRaises(ValueError):self.state(m,p,max_patches=0)
        with self.assertRaises(ValueError):self.state(m,replace(p,heading=2))
        with self.assertRaises(ValueError):geometry_state_v26(m,p,(15,15,4),10)
        m.belief[1,1]=2
        with self.assertRaises(ValueError):self.state(m,p)
        m=mapper(p);m.current_footprint_conflict_details['timestamp_s']=1.
        with self.assertRaises(ValueError):self.state(m,p)

    def test_bad_quality_and_sensor_geometry_rejected(self):
        p=packet()
        mutations=[('point',[np.nan,1,1]),('normal',[0,0,0]),('n',0),('bits',256),('best_range',0),('residual',-.1)]
        for key,value in mutations:
            with self.subTest(key=key):
                m=mapper(p);m.quality[(0,0,0)][key]=value
                with self.assertRaises(ValueError):self.state(m,p)
        m=mapper(p);m.config.max_depth_m=1.
        with self.assertRaises(ValueError):self.state(m,p)
        malformed=p.frame.depth_m.copy();malformed[0,0]=np.nan
        with self.assertRaises(ValueError):self.state(mapper(p),replace(p,frame=replace(p.frame,depth_m=malformed)))


class VisibleSemanticMemoryTests(unittest.TestCase):
    def test_semantic_channel_cannot_create_or_change_cues(self):
        p=packet();p=replace(p,frame=replace(p.frame,semantic=np.full((16,16),3,np.int32)))
        memory=VisibleSemanticMemoryV26();self.assertEqual(memory.update(p),())
        actual=packet(1,colors=[red()]);actual=replace(actual,frame=replace(actual.frame,semantic=np.full((16,16),2,np.int32)))
        cues=memory.update(actual)
        self.assertEqual(len(cues),1);self.assertEqual(cues[0].class_id,3)
        retry=replace(actual,frame=replace(actual.frame,semantic=object()))
        self.assertEqual(memory.update(retry),cues)

    def test_known_color_requires_actual_same_pixel_depth_and_enough_support(self):
        for depth in (0.,.1,6.):
            with self.subTest(depth=depth):self.assertEqual(VisibleSemanticMemoryV26().update(packet(colors=[red()],depth=depth)),())
        p=packet(colors=[red()]);d=p.frame.depth_m.copy();d[5,5]=0.
        p=replace(p,frame=replace(p.frame,depth_m=d))
        self.assertEqual(VisibleSemanticMemoryV26().update(p),())
        p=packet();p.frame.color_rgb[0,0]=[220,60,40]
        self.assertEqual(VisibleSemanticMemoryV26().update(p),())

    def test_seed_is_actual_pixel_and_outward_is_first_observed_view(self):
        p=packet(colors=[red()]);cue=VisibleSemanticMemoryV26().update(p)[0]
        source=json.loads(cue.source);r,c=source['first_pixel'];d=p.frame.depth_m[r,c]
        xyz=(np.linalg.inv(p.frame.intrinsic)@np.array([c,r,1.])*d)@p.frame.world_from_camera[:3,:3].T+p.frame.world_from_camera[:3,3]
        np.testing.assert_allclose(cue.center,xyz)
        delta=p.frame.world_from_camera[:3,3]-xyz;delta[2]=0;delta/=np.linalg.norm(delta)
        np.testing.assert_allclose(cue.outward,delta)
        self.assertEqual(cue.class_id,3);self.assertEqual(cue.confidence,1.)
        self.assertIn('not_calibrated_network',source['kind'])
        self.assertEqual(cue.action_id,0)
        with self.assertRaises(FrozenInstanceError):cue.class_id=2

    def test_position_identity_is_independent_of_class_and_memory_is_cumulative(self):
        memory=VisibleSemanticMemoryV26();first=memory.update(packet(colors=[red()]))[0]
        second=memory.update(packet(1,colors=[blue()]))[0]
        self.assertEqual(first.cue_id,second.cue_id);self.assertEqual(first.center,second.center)
        self.assertEqual(second.confidence,.5)
        self.assertEqual(memory.update(packet(2)),(second,))
        self.assertEqual(first.class_id,3);self.assertEqual(first.confidence,1.)

    def test_two_visible_cues_keep_spatial_identity_after_color_swap(self):
        left=(slice(5,9),slice(1,5),(220,60,40));right=(slice(5,9),slice(11,15),(40,100,220))
        memory=VisibleSemanticMemoryV26();a=memory.update(packet(colors=[left,right]))
        swapped=[(left[0],left[1],right[2]),(right[0],right[1],left[2])]
        b=memory.update(packet(1,colors=swapped))
        self.assertEqual(len(a),2);self.assertEqual(len(set(x.cue_id for x in b)),2)
        self.assertEqual([(x.cue_id,x.center) for x in a],[(x.cue_id,x.center) for x in b])
        self.assertTrue(all(x.confidence==.5 for x in b))

    def test_retries_are_idempotent_and_bad_order_or_identity_is_rejected_atomically(self):
        memory=VisibleSemanticMemoryV26();p=packet(colors=[red()]);first=memory.update(p)
        self.assertEqual(memory.update(p),first)
        invalid=[packet(2),packet(1,episode='other'),packet(1,frame_id=p.frame_id),packet(colors=[blue()])]
        for bad in invalid:
            with self.assertRaises(ValueError):memory.update(bad)
            self.assertEqual(memory.update(p),first)
        self.assertEqual(memory.update(packet(1)),first)
        with self.assertRaises(ValueError):memory.update(p)

    def test_manually_constructed_cues_validate_the_public_boundary(self):
        cue=SemanticCueV26('visible-1',[1.,2.,1.],(0.,-1.),3,.8,1,'actual_color_observation')
        self.assertEqual(cue.center,(1.,2.,1.));self.assertEqual(cue.outward,(0.,-1.,0.))
        bad=[dict(class_id=1),dict(class_id=True),dict(confidence=float('nan')),dict(confidence=1.1),
            dict(confidence=-.1),dict(center=(1.,2.,float('inf'))),dict(outward=(0.,-2.,0.)),
            dict(outward=(0.,0.,1.)),dict(action_id=-1),dict(cue_id=''),dict(source='')]
        for fields in bad:
            with self.subTest(fields=fields),self.assertRaises(ValueError):replace(cue,**fields)

    def test_invalid_rgb_depth_calibration_and_time_rejected(self):
        p=packet(colors=[red()]);memory=VisibleSemanticMemoryV26()
        for field,value in [('color_rgb',np.zeros((16,16,3),float)),('depth_m',np.full((16,16),np.nan)),
                ('intrinsic',np.zeros((3,3))),('timestamp_s',-1.)]:
            with self.subTest(field=field),self.assertRaises(ValueError):memory.update(replace(p,frame=replace(p.frame,**{field:value})))
        first=memory.update(p);bad=packet(1)
        bad=replace(bad,frame=replace(bad.frame,timestamp_s=0.),scan=replace(bad.scan,timestamp_s=0.))
        with self.assertRaises(ValueError):memory.update(bad)
        self.assertEqual(memory.update(p),first)


if __name__=='__main__':unittest.main()
