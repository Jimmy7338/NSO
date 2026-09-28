import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import numpy as np
from scripts.ros1_recording_v3_contract import convert_ranges,convert_depth,rectified_projection,transform_response
from scripts.record_ros1_rgbd_v3 import main
from scripts.audit_ros1_recording_v3 import check_metadata_sequence
from utils.rgbd_contract import RGBDFrame,PlanarScan


def fixture_messages():
    def header(frame):
        stamp=S(secs=100,nsecs=10_000_000,to_sec=lambda:100.01)
        return S(stamp=stamp,seq=1,frame_id=frame)
    depth=S(header=header('left_optical'),encoding='16UC1',array=np.array([[1000,0],[5000,2000]],np.uint16))
    rgb=S(header=header('left_optical'),encoding='rgb8',array=np.zeros((2,2,3),np.uint8))
    scan=S(header=header('laser'),ranges=[1.,5.,float('inf'),float('nan'),.1],intensities=[],
        angle_min=-.2,angle_max=.2,angle_increment=.1,range_min=.2,range_max=4.,time_increment=.001,scan_time=.1)
    info=S(header=header('left_optical'),width=8,height=8,binning_x=2,binning_y=2,
        roi=S(x_offset=2,y_offset=2,width=4,height=4,do_rectify=False),
        K=[8.,0.,4.,0.,8.,4.,0.,0.,1.],P=[8.,0.,4.,0.,0.,8.,4.,0.,0.,0.,1.,0.],
        R=[0.,-1.,0.,1.,0.,0.,0.,0.,1.],D=[0.]*5,distortion_model='plumb_bob')
    return depth,rgb,scan,info


def mock_recording(output):
    depth,rgb,scan,info=fixture_messages();state={};queries=[]
    def info_sub(topic,cls,callback,**kwargs):state['info_callback']=callback
    class Sync:
        def __init__(self,*args):pass
        def registerCallback(self,callback):state['callback']=callback
    class Buffer:
        def __init__(self,**kwargs):pass
        def lookup_transform(self,target,source,stamp,timeout):
            queries.append((target,source,stamp.secs,stamp.nsecs))
            return S(header=S(frame_id=target,stamp=stamp),child_frame_id=source,
                transform=S(translation=S(x=1.,y=2.,z=3.),rotation=S(x=0.,y=0.,z=0.,w=1.)))
    class Bridge:
        def imgmsg_to_cv2(self,message,desired_encoding):return message.array.copy()
    def spin():
        state['info_callback'](info);state['callback'](depth,rgb,scan)
        state['shutdown']()
    def param(name,default=None):return str(output) if name=='~output' else default
    rospy=S(init_node=lambda *a,**k:None,get_param=param,Duration=lambda v:v,Subscriber=info_sub,
        on_shutdown=lambda cb:state.__setitem__('shutdown',cb),loginfo=lambda *a:None,
        logwarn_throttle=lambda *a:state.setdefault('warnings',[]).append(a),spin=spin)
    modules=dict(rospy=rospy,message_filters=S(Subscriber=lambda *a:S(),ApproximateTimeSynchronizer=Sync),
        tf2_ros=S(Buffer=Buffer,TransformListener=lambda b:S()),cv_bridge=S(CvBridge=Bridge))
    modules['sensor_msgs']=S();modules['sensor_msgs.msg']=S(Image=object,CameraInfo=object,LaserScan=object)
    with patch.dict(sys.modules,modules):main()
    return dict(queries=queries,warnings=state.get('warnings',[]))


class RecordingV3Tests(unittest.TestCase):
    def test_finite_above_range_is_invalid_not_a_false_maximum_hit(self):
        raw=np.array([.1,.2,1.,4.,5.,np.inf,-np.inf,np.nan],np.float32)
        result,audit=convert_ranges(raw,.2,4.)
        np.testing.assert_array_equal(result,np.array([0,.2,1,4,0,0,0,0],np.float32))
        self.assertEqual(audit['counts']['finite_above_maximum'],1)
        self.assertFalse(audit['finite_above_maximum_clipped'])

    def test_infinity_clear_policy_is_explicit_and_does_not_clip_finite_invalids(self):
        result,_=convert_ranges(np.array([np.inf,5.]),.2,4.,'clear_to_max')
        np.testing.assert_array_equal(result,[4,0])
        with self.assertRaises(ValueError):convert_ranges([1.],.2,4.,'guess')

    def test_depth_encoding_and_invalid_values(self):
        np.testing.assert_array_equal(convert_depth(np.array([[1000,5000]],np.uint16),'16UC1',4.),[[1.,0.]])
        with self.assertRaises(ValueError):convert_depth(np.ones((2,2),np.float32),'16UC1',4.)

    def test_cropped_binned_projection_and_rectification_basis(self):
        info=fixture_messages()[3]
        K,T,audit=rectified_projection(info,(2,2))
        np.testing.assert_array_equal(K,[[4,0,1],[0,4,1],[0,0,1]])
        # A rectified +x ray maps to -y in the original optical coordinates.
        np.testing.assert_array_equal(T[:3,:3]@np.array([1,0,0]),[0,-1,0])
        self.assertEqual(audit['roi']['x_offset'],2)

    def test_unrectified_roi_and_discarded_stereo_baseline_are_rejected(self):
        info=fixture_messages()[3];info.roi.do_rectify=True
        with self.assertRaisesRegex(ValueError,'ROI requires'):rectified_projection(info,(2,2))
        info.roi.do_rectify=False;info.P[3]=-.4
        with self.assertRaisesRegex(ValueError,'stereo translation'):rectified_projection(info,(2,2))

    def test_tf_response_must_match_requested_frames(self):
        message=S(header=S(frame_id='odom'),child_frame_id='laser')
        with self.assertRaisesRegex(ValueError,'frames differ'):transform_response(message,'map','laser')

    def test_actual_recorder_callback_saves_raw_and_passes_offline_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'recording';result=mock_recording(out)
            self.assertFalse(result['warnings'],result)
            manifest=json.loads((out/'sequence.json').read_text())
            self.assertEqual(manifest['schema_version'],'ros1_rgbd_recording/3')
            self.assertEqual(manifest['frames'],1);self.assertEqual(manifest['rejected'],0)
            with np.load(out/'raw/00000.npz',allow_pickle=False) as raw:
                self.assertEqual(raw['depth_source'].dtype,np.dtype('uint16'))
                self.assertTrue(np.isposinf(raw['ranges_source'][2]))
                self.assertTrue(np.isnan(raw['ranges_source'][3]))
            np.testing.assert_array_equal(PlanarScan.load(out/'scans/00000.npz').ranges_m,[1,0,0,0,0])
            audit=check_metadata_sequence(out)
            self.assertEqual(audit['status'],'passed',audit['errors'])
            self.assertFalse(audit['robot_interface_ready'])

    def test_forged_saved_ray_is_detected_even_with_updated_payload_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'recording';mock_recording(out)
            p=out/'scans/00000.npz';scan=PlanarScan.load(p);scan.ranges_m[1]=4.;scan.save(p)
            meta=out/'metadata/00000.json';m=json.loads(meta.read_text())
            m['payload_sha256']['scans/00000.npz']=hashlib.sha256(p.read_bytes()).hexdigest();meta.write_text(json.dumps(m))
            audit=check_metadata_sequence(out)
            self.assertEqual(audit['status'],'failed');self.assertIn('raw conversion',json.dumps(audit['errors']))

    def test_omitting_rectification_from_stored_pose_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'recording';mock_recording(out)
            p=out/'frames/00000.npz';f=RGBDFrame.load(p);f.world_from_camera[:3,:3]=np.eye(3);f.save(p)
            meta=out/'metadata/00000.json';m=json.loads(meta.read_text())
            m['payload_sha256']['frames/00000.npz']=hashlib.sha256(p.read_bytes()).hexdigest();meta.write_text(json.dumps(m))
            audit=check_metadata_sequence(out)
            self.assertEqual(audit['status'],'failed');self.assertIn('rectification composition',json.dumps(audit['errors']))


if __name__=='__main__':unittest.main()
