"""Candidate-only synthetic source/alignment contracts, no ROS or hardware."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xmlrpc.client
from test_csm_odometry_bridge import rospy
from test_csm_odometry_bridge import bridge, pose, fake_feedback, vec

TOPIC='/nso_csm_zero_keyframe/pose_stamped'
CALLER='/nso_csm_zero_keyframe/matcher'
FRAME='odom_zero_keyframe_20261107'

class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'alignment.json'
        self.transform=(10.,20.,math.pi/2)
        self.data=dict(schema='csm_fixed_se2_alignment/1',source_topic=TOPIC,expected_caller=CALLER,
            source_frame=FRAME,target_frame='odom',odom_from_source=list(self.transform),
            anchor=dict(source_stamp_ns=1000000000,odom_stamp_ns=1000000000,
                        source_pose=[1.,0.,3.1],odom_pose=list(bridge.transform_sample(self.transform,(1.,0.,3.1)))))
        self.write()
    def write(self):self.path.write_text(json.dumps(self.data))
    def create(self, source='csm_se2', feedback=None):
        return bridge.CsmOdometryBridge(source,feedback,pose_topic=TOPIC,expected_caller=CALLER,
                                         source_frame=FRAME,alignment_json=self.path)
    def msg(self, ns, **kw):
        p=pose(ns,**kw);p.header.frame_id=FRAME;p._connection_header['callerid']=CALLER;return p
    def test_anchor_and_sha_bound_to_real_source(self):
        transform,receipt=bridge.load_alignment(self.path,TOPIC,CALLER,FRAME)
        self.assertEqual(transform,self.transform)
        self.assertEqual(receipt['sha256'],hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertLess(receipt['anchor_position_error_m'],1e-12)
        self.assertFalse(receipt['physical_accuracy_verified'])
    def test_wrong_binding_mismatched_time_nonfinite_or_false_anchor_rejected(self):
        original=copy.deepcopy(self.data)
        for change in [lambda x:x.update(source_frame='wrong'),lambda x:x.update(expected_caller='/fake'),
                       lambda x:x.update(source_topic='/wrong'),lambda x:x.update(target_frame='map'),
                       lambda x:x['anchor'].update(odom_stamp_ns=1000000001),
                       lambda x:x['anchor'].update(source_stamp_ns=True),
                       lambda x:x.update(odom_from_source=[0,0,float('nan')]),
                       lambda x:x['anchor'].update(odom_pose=[0,0,0])]:
            self.data=copy.deepcopy(original);change(self.data);self.write()
            with self.assertRaises((ValueError,TypeError)):
                bridge.load_alignment(self.path,TOPIC,CALLER,FRAME)
    def test_nondefault_source_without_alignment_rejected(self):
        with self.assertRaises(ValueError):bridge.CsmOdometryBridge(pose_topic=TOPIC,expected_caller=CALLER,source_frame=FRAME)
    def test_default_pose_object_and_values_remain_original(self):
        b=bridge.CsmOdometryBridge();b.callback(pose(1000000000));m=pose(1100000000,x=.01,z=.12)
        b.callback(m);self.assertIs(b.pub.items[0].pose.pose,m.pose)
        self.assertIs(b.pub.items[0].header.stamp,m.header.stamp)
        self.assertIsNone(b.pose_source_pub)
    def test_rotation_wrap_pose_stamp_and_TF_agree_without_mutation(self):
        b=self.create();b.callback(self.msg(1000000000,x=1,y=0,yaw=3.1))
        m=self.msg(1100000000,x=1.01,y=.02,yaw=-3.1,z=.123);saved=copy.deepcopy(m)
        b.callback(m);o=b.pub.items[0];t=b.tf.items[0];expected=bridge.transform_sample(self.transform,bridge.planar_sample(m))
        self.assertAlmostEqual(o.pose.pose.position.x,expected[0]);self.assertAlmostEqual(o.pose.pose.position.y,expected[1])
        q=o.pose.pose.orientation;yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        self.assertAlmostEqual(yaw,expected[2]);self.assertEqual(o.pose.pose.position.z,.123)
        self.assertEqual(o.header.frame_id,'odom');self.assertEqual(t.header.frame_id,'odom')
        self.assertIs(o.header.stamp,m.header.stamp);self.assertIs(t.header.stamp,m.header.stamp)
        self.assertEqual(t.transform.translation.x,o.pose.pose.position.x);self.assertEqual(t.transform.rotation,o.pose.pose.orientation)
        self.assertEqual(m,saved)
        receipt=json.loads(b.pose_source_pub.items[0].data)
        self.assertEqual(receipt['source_caller'],CALLER);self.assertEqual(receipt['source_stamp_ns'],1100000000)
        self.assertEqual(receipt['output_stamp_ns'],1100000000)
    def test_body_twist_invariant_under_fixed_world_SE2(self):
        a=(1.,2.,3.1);b=(1.01,2.02,-3.1)
        expected=bridge.derived_twist(a,b,.1)
        actual=bridge.derived_twist(bridge.transform_sample(self.transform,a),bridge.transform_sample(self.transform,b),.1)
        for x,y in zip(actual,expected):self.assertAlmostEqual(x,y,places=10)
    def test_wrong_caller_or_frame_rejected_without_mutating_previous(self):
        b=self.create();wrong=self.msg(1000000000);wrong._connection_header['callerid']='/nso_csm/matcher';b.callback(wrong)
        self.assertIsNone(b.previous)
        wrong=self.msg(1000000000);wrong.header.frame_id='odom';b.callback(wrong);self.assertIsNone(b.previous)
        self.assertEqual(len(b.pub.items),0)
    def test_CAN_selected_at_original_stamp_without_interface_change(self):
        selected=[]
        feedback=vec(for_scan=lambda stamp:(selected.append(stamp) or fake_feedback(.02,.1)),
                     in_startup_grace=lambda:False,source_receipt=lambda:dict(source_ttl_ns=150000000))
        b=self.create('can_feedback',feedback);b.callback(self.msg(1000000000));b.callback(self.msg(1100000000,x=.01))
        self.assertEqual(selected,[1100000000]);self.assertEqual(b.control_pub.items[0].twist.twist.linear.x,.02)
        self.assertEqual(b.control_pub.items[0].twist.twist.angular.z,.1)
        p=json.loads(b.provenance_pub.items[0].data);self.assertFalse(p['raw_pose_stamp_tf_unchanged'])
        self.assertTrue(p['source_pose_stamp_unchanged']);self.assertTrue(p['fixed_coordinate_transform_applied']);self.assertTrue(p['raw_registration_preserved']);self.assertEqual(p['can_source']['source_ttl_ns'],150000000)
        self.assertEqual(p['pose_alignment']['sha256'],b.alignment_receipt['sha256'])
    def test_loaded_alignment_stays_constant_if_file_later_changes(self):
        b=self.create();old=copy.deepcopy(b.alignment_receipt);self.data['odom_from_source']=[0,0,0];self.write()
        b.callback(self.msg(1000000000));b.callback(self.msg(1100000000,x=.01))
        self.assertEqual(b.alignment,self.transform);self.assertEqual(b.alignment_receipt,old)
        self.assertAlmostEqual(b.pub.items[0].pose.pose.position.y,20.01)
    def test_CAN_module_bytes_unchanged(self):
        root=Path(__file__).parents[1]
        self.assertEqual((root/'runtime/can_control_feedback.py').read_bytes(),(root/'source_before/can_control_feedback.py').read_bytes())

    def test_real_XMLRPC_int64_receipt_is_safe_only_in_parameter_copy(self):
        ns=1794053131681172132
        self.data['anchor'].update(source_stamp_ns=ns,odom_stamp_ns=ns);self.write()
        b=self.create();params=bridge.pose_source_parameters(TOPIC,CALLER,FRAME,b.alignment_receipt)
        wire=xmlrpc.client.dumps((params,),allow_none=False)
        decoded=xmlrpc.client.loads(wire)[0][0]
        self.assertEqual(decoded['alignment']['anchor']['source_stamp_ns'],str(ns))
        self.assertIs(type(b.alignment_receipt['anchor']['source_stamp_ns']),int)
        self.assertIs(type(json.loads(self.path.read_text())['anchor']['source_stamp_ns']),int)
        with self.assertRaises(OverflowError):xmlrpc.client.dumps((b.alignment_receipt,),allow_none=False)
    def test_aligned_to_default_main_overwrites_same_private_param_store(self):
        store={}
        args=['bridge.py','--pose-topic',TOPIC,'--expected-caller',CALLER,'--source-frame',FRAME,
              '--alignment-json',str(self.path)]
        for argv in [args,['bridge.py']]:
            with patch.object(rospy,'myargv',return_value=argv,create=True), \
                 patch.object(rospy,'init_node',create=True),patch.object(rospy,'spin',create=True), \
                 patch.object(rospy,'set_param',side_effect=lambda k,v:store.__setitem__(k,v)):
                self.assertEqual(bridge.main(),0)
            if argv==args:self.assertTrue(store['~pose_source']['fixed_coordinate_transform_applied'])
        self.assertEqual(store['~pose_source']['profile'],'original_csm_pose_source_v1')
        self.assertFalse(store['~pose_source']['fixed_coordinate_transform_applied'])
        self.assertEqual(store['~pose_source']['alignment'],{})
        xmlrpc.client.dumps((store['~pose_source'],),allow_none=False)
