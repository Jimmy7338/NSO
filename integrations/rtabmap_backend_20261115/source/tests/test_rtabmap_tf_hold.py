import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as N
import unittest

root=Path(__file__).resolve().parents[1]
def load(name):
    s=importlib.util.spec_from_file_location(name,root/'runtime'/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('rtabmap_tf_hold');b=load('rtabmap_mapping_backend')
class Stamp:
    def __init__(self,n):self.n=n
    def to_nsec(self):return self.n
def header(n,frame='odom'):return N(stamp=Stamp(n),frame_id=frame,seq=4)
def transform(x=2.):return N(translation=N(x=x,y=-3.,z=.1),rotation=N(x=0.,y=0.,z=.6,w=.8))
def correction(n=100,x=2.):
    return N(header=header(n),child_frame_id='nso_pair_map',transform=transform(x),
             _connection_header={'callerid':'/nso_rtabmap_mapping_backend'})
def odom(n=200):
    return N(header=header(n),child_frame_id='base_link',
             _connection_header={'callerid':'/nso_csm_odometry_bridge'})

class Tests(unittest.TestCase):
    def test_matches_backend_hold_exactly(self):
        a=b.EpochAdapter();h=m.CorrectionHold();c=correction()
        a.tf_epoch=(100,copy.deepcopy(c.transform));h.receive_correction(c)
        for ns in [99,100,101,10**16]:
            expected=a.odom(odom(ns));actual=h.odom(odom(ns))
            if expected is None:self.assertIsNone(actual)
            else:
                self.assertEqual(vars(expected[0].translation),vars(actual[0].translation))
                self.assertEqual(vars(expected[0].rotation),vars(actual[0].rotation))
                self.assertEqual(actual[1].stamp.to_nsec(),ns)
                self.assertEqual(actual[2],expected[2])
    def test_requires_correction_and_rejects_older_epoch(self):
        h=m.CorrectionHold();self.assertIsNone(h.odom(odom()))
        self.assertTrue(h.receive_correction(correction(200,5.)))
        self.assertFalse(h.receive_correction(correction(100,9.)))
        self.assertFalse(h.receive_correction(correction(200,9.)))
        self.assertEqual(h.odom(odom(201))[0].translation.x,5.)
    def test_loop_update_replaces_not_compounds(self):
        h=m.CorrectionHold();h.receive_correction(correction(100,2.))
        h.receive_correction(correction(200,10.))
        self.assertIsNone(h.odom(odom(199)))
        self.assertEqual(h.odom(odom(201))[0].translation.x,10.)
    def test_invalid_correction_cannot_replace_epoch(self):
        for change in ['caller','frame','child','zero_stamp','nonfinite','quaternion']:
            h=m.CorrectionHold();h.receive_correction(correction());c=correction(200)
            if change=='caller':c._connection_header['callerid']='/other'
            if change=='frame':c.header.frame_id='map'
            if change=='child':c.child_frame_id='base_link'
            if change=='zero_stamp':c.header.stamp=Stamp(0)
            if change=='nonfinite':c.transform.translation.x=float('nan')
            if change=='quaternion':c.transform.rotation.w=0.
            with self.assertRaises(ValueError):h.receive_correction(c)
            self.assertEqual(h.epoch[0],100)
    def test_invalid_odom_rejected(self):
        for change in ['caller','frame','child','zero_stamp']:
            h=m.CorrectionHold();h.receive_correction(correction());o=odom()
            if change=='caller':o._connection_header['callerid']='/other'
            if change=='frame':o.header.frame_id='map'
            if change=='child':o.child_frame_id='other'
            if change=='zero_stamp':o.header.stamp=Stamp(0)
            with self.assertRaises(ValueError):h.odom(o)
    def test_input_and_result_ownership(self):
        h=m.CorrectionHold();c=correction();o=odom();h.receive_correction(c)
        c.transform.translation.x=42.;r=h.odom(o);r[0].translation.x=99.
        r[1].stamp=Stamp(3)
        self.assertEqual(h.odom(o)[0].translation.x,2.)
        self.assertEqual(o.header.stamp.to_nsec(),200)

if __name__=='__main__':unittest.main()
