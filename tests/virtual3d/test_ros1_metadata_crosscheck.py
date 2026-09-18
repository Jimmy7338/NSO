"""Source/payload contradictions and exact integer sync boundaries."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest
import numpy as np
from scripts.audit_ros1_source_metadata import check_metadata_sequence
from scripts.record_ros1_rgbd import sensor_metadata
from utils.rgbd_contract import RGBDFrame,PlanarScan


def create_recording_fixture(path,count=3):
    path=Path(path)
    for name in ('frames','scans','metadata'):(path/name).mkdir()
    config=dict(schema_version='ros1_rgbd_recording/2',frames=count,slop_s=.05,
        max_depth_m=4.,world_frame='map',optical_frame='left_optical',semantic_topic='')
    (path/'sequence.json').write_text(json.dumps(config))
    for i in range(count):
        start=1_789_000_000_000_000_001+i*250_000_000
        def message(ns,frame,encoding):
            sec,nano=divmod(ns,1_000_000_000)
            return S(header=S(stamp=S(secs=sec,nsecs=nano),seq=i,frame_id=frame),encoding=encoding)
        depth=message(start,'left_optical','32FC1')
        rgb=message(start+1,'left_optical','rgb8')
        laser=message(start+50_000_000,'laser','unused')
        info=message(0,'left_optical','unused')
        intrinsic=np.array([[5.,0.,4.],[0.,5.,4.],[0.,0.,1.]])
        info.height=info.width=8
        info.K=[7.,0.,4.,0.,7.,4.,0.,0.,1.]
        info.P=np.column_stack([intrinsic,np.zeros(3)]).ravel().tolist()
        info.R=np.eye(3).ravel().tolist();info.D=[0.]*5;info.distortion_model='plumb_bob'
        laser.ranges=[1.,2.,3.];laser.time_increment=.0001;laser.scan_time=.1
        laser.angle_min=-.1;laser.angle_increment=.1;laser.angle_max=.1
        laser.range_min=.01;laser.range_max=4.
        sec=lambda m:float(m.header.stamp.secs)+float(m.header.stamp.nsecs)/1e9
        frame=RGBDFrame(sec(depth),np.ones((8,8),np.float32),np.zeros((8,8,3),np.uint8),
            intrinsic,np.eye(4),np.zeros((8,8),np.uint8))
        scan=PlanarScan(sec(laser),np.asarray(laser.ranges,np.float32),-.1,.1,4.,np.eye(4))
        stem=f'{i:05d}';frame.save(path/'frames'/f'{stem}.npz');scan.save(path/'scans'/f'{stem}.npz')
        meta=sensor_metadata(depth,rgb,laser,info,None,config)
        meta['payload_sha256']={f'{d}/{stem}.npz':hashlib.sha256((path/d/f'{stem}.npz').read_bytes()).hexdigest()
                                for d in ('frames','scans')}
        (path/'metadata'/f'{stem}.json').write_text(json.dumps(meta))


class MetadataCrosscheckTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        create_recording_fixture(self.path)

    def tearDown(self):self.temp.cleanup()

    def alter(self,fn,stem='00000'):
        path=self.path/'metadata'/f'{stem}.json';m=json.loads(path.read_text());fn(m)
        path.write_text(json.dumps(m))

    def failed(self,substring):
        r=check_metadata_sequence(self.path)
        self.assertEqual(r['status'],'failed');self.assertIn(substring,json.dumps(r['errors']))
        json.dumps(r,allow_nan=False)

    def test_large_timestamps_exact_boundary_and_static_calibration(self):
        r=check_metadata_sequence(self.path)
        self.assertEqual(r['status'],'passed',r['errors'])
        self.assertTrue(all(v['integer_span_ns']==50_000_000 for v in r['records']))
        self.assertFalse(r['robot_interface_ready']);self.assertFalse(r['extrinsic_correctness_verified'])
        self.assertGreater(r['float_seconds_representation_allowance_s'],0)

    def test_one_nanosecond_past_boundary_is_rejected_even_if_float_rounds_equal(self):
        def corrupt(m):
            m['headers']['scan']['nsecs']+=1;m['headers']['scan']['timestamp_ns']+=1
            m['synchronized_message_span_ns']+=1;m['depth_relative_offsets_ns']['scan']+=1
            m['tf_queries']['laser_stamp_ns']+=1
        self.alter(corrupt);self.failed('integer source timestamp span')

    def test_integer_stamp_and_offset_tampering(self):
        self.alter(lambda m:m['headers']['rgb'].__setitem__('timestamp_ns',1))
        self.failed('integer timestamp fields disagree')

    def test_payload_tampering_is_detected(self):
        p=self.path/'frames/00000.npz';f=RGBDFrame.load(p)
        replace(f,depth_m=f.depth_m*2).save(p)
        self.failed('payload hash mismatch')

    def test_wrong_camera_intrinsic_fails_crosscheck(self):
        self.alter(lambda m:m['camera_info']['P'].__setitem__(0,9.))
        self.failed('stored intrinsic differs')

    def test_uncalibrated_camera_is_not_certified_by_valid_payload(self):
        self.alter(lambda m:m['camera_info']['K'].__setitem__(0,0.))
        self.failed('uncalibrated CameraInfo')

    def test_tf_frame_or_query_stamp_mismatch(self):
        self.alter(lambda m:m['tf_queries'].__setitem__('laser_source_frame','wrong_laser'))
        self.failed('TF query does not match')

    def test_calibration_from_another_camera_is_not_accepted_by_same_dimensions(self):
        self.alter(lambda m:m['headers']['camera_info'].__setitem__('frame_id','right_optical'))
        self.failed('CameraInfo optical frame differs')

    def test_laser_endpoint_mismatch(self):
        self.alter(lambda m:m['laser'].__setitem__('angle_max_rad',.5))
        self.failed('laser angular endpoint')

    def test_nonfinite_laser_timing_returns_serializable_failure(self):
        self.alter(lambda m:m['laser'].__setitem__('time_increment_s',float('nan')))
        self.failed('invalid numeric time_increment')

    def test_missing_metadata_is_not_legacy_success(self):
        (self.path/'metadata/00001.json').unlink()
        self.failed('noncontiguous_missing_or_extra_files')

    def test_payload_time_must_match_source_not_just_sync_window(self):
        p=self.path/'scans/00000.npz';s=PlanarScan.load(p)
        replace(s,timestamp_s=s.timestamp_s-.01).save(p)
        self.alter(lambda m:m['payload_sha256'].__setitem__('scans/00000.npz',hashlib.sha256(p.read_bytes()).hexdigest()))
        self.failed('payload/source timestamp differs')

    def test_false_deskew_claim_rejected(self):
        self.alter(lambda m:m['laser'].__setitem__('motion_compensation_applied',True))
        self.failed('cannot claim deskew')


if __name__=='__main__':unittest.main()
