"""Pure offline overlay checks: no ROS, sockets, camera, or worker processes."""
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from types import SimpleNamespace

PATH = Path(__file__).parents[1]/'runtime/dashboard_overlay_server.py'
spec = importlib.util.spec_from_file_location('overlay', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'semantic_state.json'
        self.now = 100.
        self.overlay = module.Overlay(self.path, clock=lambda: self.now)

    def pose(self, x=1., y=0., stamp=100.):
        return dict(x=x, y=y, yaw=0., stamp=stamp)

    def test_planar_forward_inverse(self):
        transform = [10, 20, math.pi/2]
        moved = module.point(transform, [2, 3])
        self.assertAlmostEqual(moved[0], 7)
        self.assertAlmostEqual(moved[1], 22)
        back = module.point(module.inverse(transform), moved)
        self.assertAlmostEqual(back[0], 2)
        self.assertAlmostEqual(back[1], 3)

    def test_trajectory_accumulates_without_snapshots_and_is_bounded(self):
        for index in range(5010):
            self.now = 100+index*.2
            self.overlay.sample(self.pose(x=index*.02, stamp=self.now), [0, 0, 0])
        self.assertEqual(len(self.overlay.trajectory), 5000)
        result = self.overlay.snapshot({})
        self.assertEqual(len(result['trajectory']), 5000)
        self.assertTrue(result['freshness']['pose']['fresh'])

    def test_snapshot_reprojects_entire_history_with_current_map_correction(self):
        self.overlay.sample(self.pose(), [0, 0, 0])
        self.now += .2
        self.overlay.sample(self.pose(x=2, stamp=self.now), [0, 0, 0])
        result = self.overlay.snapshot({}, current_transform=[10, 20, math.pi/2])
        self.assertAlmostEqual(result['trajectory'][0][0], 10)
        self.assertAlmostEqual(result['trajectory'][0][1], 21)
        self.assertAlmostEqual(result['pose']['y'], 22)
        self.assertEqual(result['pose']['frame'], 'map')
        self.assertEqual(result['pose']['stamp'], 100.2)

    def test_semantics_transforms_polygon_preserves_source_age(self):
        self.path.write_text(json.dumps(dict(frame='odom', stamp=95., status='ok', objects=[dict(
            id=1, label='chair', label_zh='椅子', confidence=.8, x=1, y=2,
            polygon=[[0, 0], [1, 0]], last_seen_stamp=94.)])))
        self.overlay.sample(self.pose(), [10, 20, math.pi/2])
        result = self.overlay.snapshot({})['semantics']
        self.assertEqual(result['stamp'], 95.)
        self.assertEqual(result['freshness']['age_s'], 5.)
        self.assertEqual(result['objects'][0]['last_seen_stamp'], 94.)
        self.assertAlmostEqual(result['objects'][0]['x'], 8)
        self.assertAlmostEqual(result['objects'][0]['y'], 21)
        self.assertAlmostEqual(result['objects'][0]['polygon'][1][1], 21)
        self.assertFalse(result['camera_extrinsics_verified'])
        self.now = 107
        aged = self.overlay.snapshot({})['semantics']
        self.assertEqual(aged['freshness']['age_s'], 12)
        self.assertFalse(aged['freshness']['fresh'])

    def test_timeout_future_and_backend_rejection_remain_stale(self):
        self.overlay.sample(self.pose(), [0, 0, 0])
        self.now = 104
        result = self.overlay.snapshot(dict(camera=dict(stamp=104), freshness=dict(camera=dict(fresh=False))))
        self.assertFalse(result['freshness']['pose']['fresh'])
        self.assertFalse(result['freshness']['camera']['fresh'])
        self.assertEqual(result['pose']['stamp'], 100.)
        for stamp in [105, -1, None, float('nan')]:
            self.assertFalse(module.freshness(stamp, self.now, 3)['fresh'])
        with self.assertRaises(ValueError):
            self.overlay.sample(self.pose(stamp=105), [0, 0, 0])

    def test_seed_roundtrips_old_map_history_only_when_corroborated(self):
        self.overlay.sample(self.pose(), [10, 0, 0])
        upstream = dict(stamp=100, pose=dict(x=11, y=0, yaw=0, stamp=100, frame='map'),
                        trajectory=[[10, 0], [11, 0]], freshness=dict(pose=dict(fresh=True)))
        self.overlay.seed(upstream)
        self.assertEqual(self.overlay.seed_receipt['status'], 'accepted')
        self.assertEqual(self.overlay.snapshot({})['trajectory'][:2], upstream['trajectory'])

    def test_different_epoch_seed_is_not_mixed(self):
        self.overlay.sample(self.pose(), [0, 0, 0])
        self.overlay.seed(dict(pose=dict(x=30, y=0, yaw=0, stamp=100, frame='map'),
                               trajectory=[[30, 0]], freshness=dict(pose=dict(fresh=True))))
        self.assertEqual(self.overlay.seed_receipt['status'], 'rejected')
        self.assertEqual(list(self.overlay.trajectory), [[1., 0.]])

    def test_bad_or_partial_semantic_file_never_claims_fresh_data(self):
        self.overlay.sample(self.pose(), [0, 0, 0])
        for contents in ['{', '{"frame":"map","objects":[]}',
                         '{"frame":"odom","stamp":100,"objects":[{"x":null,"y":0}]}']:
            self.path.write_text(contents)
            result = self.overlay.snapshot({})['semantics']
            self.assertEqual(result['objects'], [])
            self.assertFalse(result['freshness']['fresh'])

    def test_calibration_requires_source_hash_and_actual_matching_camera_TF(self):
        source = Path(self.directory.name)/'left.yaml'
        source.write_text('actual calibration source')
        receipt = Path(self.directory.name)/'calibration.json'
        expected = np.eye(4)
        expected[:3, 3] = [.2, .06, .3]
        row = dict(schema='nso.camera_mount_tf_verification.v1', parent_frame='base_link',
                   camera_frame='zed2i_left_camera_optical_frame',
                   source_yaml=str(source), source_yaml_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   expected_base_from_camera_optical=expected.tolist())
        receipt.write_text(json.dumps(row))
        loaded = module.camera_calibration_metadata(receipt, expected)
        self.assertTrue(loaded['camera_extrinsics_loaded'])
        self.assertFalse(loaded['camera_extrinsics_verified'])
        wrong = expected.copy(); wrong[0, 3] += .01
        self.assertFalse(module.camera_calibration_metadata(receipt, wrong)['camera_extrinsics_loaded'])
        self.assertFalse(module.camera_calibration_metadata(receipt)['camera_extrinsics_loaded'])
        source.write_text('changed calibration')
        rejected = module.camera_calibration_metadata(receipt, expected)
        self.assertFalse(rejected['camera_extrinsics_loaded'])
        self.assertEqual(rejected['camera_calibration_status'], 'calibration_source_hash_mismatch')

    def test_camera_rigid_matrix_preserves_full_optical_rotation(self):
        transform = SimpleNamespace(translation=SimpleNamespace(x=1., y=2., z=3.),
                                    rotation=SimpleNamespace(x=0., y=0., z=math.sqrt(.5), w=math.sqrt(.5)))
        matrix = module.rigid_matrix(transform)
        self.assertTrue(np.allclose(matrix @ [1, 0, 0, 1], [1, 3, 3, 1]))
        transform.rotation.w = 2.
        with self.assertRaises(ValueError):
            module.rigid_matrix(transform)


if __name__ == '__main__':
    unittest.main()
