"""Malformed and asynchronous records must fail offline robot-data checks."""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from scripts.audit_ros1_rgbd_sequence import check_sequence
from utils.rgbd_contract import RGBDFrame,PlanarScan


class SequenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.path=Path(self.temp.name)
        (self.path/"frames").mkdir();(self.path/"scans").mkdir()
        self.frame=RGBDFrame(1.,np.ones((8,8),np.float32),np.zeros((8,8,3),np.uint8),
            np.array([[5.,0,4],[0,5.,4],[0,0,1.]]),np.eye(4),np.zeros((8,8),np.uint8))
        self.scan=PlanarScan(1.01,np.ones(8,np.float32),0.,.1,4.,np.eye(4))
        self.frame.save(self.path/"frames/00000.npz");self.scan.save(self.path/"scans/00000.npz")

    def tearDown(self): self.temp.cleanup()

    def test_valid_storage_does_not_claim_robot_readiness(self):
        result=check_sequence(self.path)
        self.assertEqual(result["status"],"passed")
        self.assertFalse(result["robot_interface_ready"])

    def test_time_skew_and_missing_scan_are_rejected(self):
        replace(self.scan,timestamp_s=1.2).save(self.path/"scans/00000.npz")
        self.assertEqual(check_sequence(self.path)["status"],"failed")
        (self.path/"scans/00000.npz").unlink()
        self.assertEqual(check_sequence(self.path)["status"],"failed")

    def test_nonrigid_laser_transform_is_rejected(self):
        pose=np.eye(4);pose[0,0]=2.
        replace(self.scan,world_from_laser=pose).save(self.path/"scans/00000.npz")
        result=check_sequence(self.path)
        self.assertIn("invalid_world_from_laser",result["records"][0]["errors"])

    def test_duplicate_timestamps_are_rejected(self):
        self.frame.save(self.path/"frames/00001.npz");self.scan.save(self.path/"scans/00001.npz")
        self.assertEqual(check_sequence(self.path)["status"],"failed")

    def test_nonfinite_timestamp_produces_a_serializable_failure_record(self):
        replace(self.scan,timestamp_s=float("nan")).save(self.path/"scans/00000.npz")
        result=check_sequence(self.path)
        self.assertEqual(result["status"],"failed")
        json.dumps(result,allow_nan=False)


if __name__=="__main__":unittest.main()
