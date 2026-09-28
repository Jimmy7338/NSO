import unittest
from types import SimpleNamespace as S

from scripts.record_ros1_rgbd import sensor_metadata


class SourceMetadataTests(unittest.TestCase):
    def setUp(self):
        def image(nanoseconds=1):
            return S(header=S(stamp=S(secs=1_789_000_000, nsecs=nanoseconds), seq=7,
                              frame_id='left_optical'), encoding='32FC1')
        self.depth, self.rgb, self.scan, self.info = [image() for _ in range(4)]
        self.scan.time_increment = .0001
        self.scan.scan_time = .1
        self.scan.ranges = [1., 2., 3.]
        self.scan.angle_min = -.1
        self.scan.angle_max = .1
        self.scan.angle_increment = .1
        self.scan.range_min = .01
        self.scan.range_max = 20.
        self.info.height = self.info.width = 2
        self.info.K = self.info.R = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        self.info.P = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]
        self.info.D = []
        self.info.distortion_model = 'plumb_bob'
        self.config = dict(slop_s=.05, world_frame='map', optical_frame='left_optical')

    def call(self, semantic=None):
        return sensor_metadata(self.depth, self.rgb, self.scan, self.info, semantic, self.config)

    def test_nanosecond_precision_and_latched_calibration_are_preserved(self):
        self.rgb.header.stamp.nsecs = 2
        self.info.header.stamp.secs = self.info.header.stamp.nsecs = 0
        result = self.call()
        self.assertEqual(result['depth_relative_offsets_ns']['rgb'], 1)
        self.assertEqual(result['headers']['depth']['timestamp_ns'], 1_789_000_000_000_000_001)
        self.assertEqual(result['headers']['camera_info']['timestamp_ns'], 0)
        self.assertEqual(result['laser']['time_increment_s'], .0001)
        self.assertFalse(result['laser']['motion_compensation_applied'])
        self.assertFalse(result['calibration_correctness_verified'])

    def test_pairwise_span_rejects_images_and_laser_on_opposite_sides(self):
        self.depth.header.stamp.nsecs = 50_000_000
        self.rgb.header.stamp.nsecs = 10_000_000
        self.scan.header.stamp.nsecs = 90_000_000
        with self.assertRaisesRegex(ValueError, 'timestamp span'):
            self.call()

    def test_stale_semantic_image_is_not_silently_accepted(self):
        semantic = S(header=S(stamp=S(secs=1_789_000_000, nsecs=90_000_000),
                              seq=3, frame_id='left_optical'), encoding='mono8')
        with self.assertRaisesRegex(ValueError, 'timestamp span'):
            self.call(semantic)

    def test_nonfinite_laser_timing_is_rejected(self):
        self.scan.time_increment = float('nan')
        with self.assertRaisesRegex(ValueError, 'laser acquisition timing'):
            self.call()


if __name__ == '__main__':
    unittest.main()
