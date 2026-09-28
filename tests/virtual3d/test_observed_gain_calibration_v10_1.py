"""V10.1 candidate and feedback repair tests; no held-out efficacy claim."""
from types import SimpleNamespace
import unittest
import numpy as np

from nso.hierarchical_options_v10 import generate_options
from nso.observed_gain_calibration_v10_1 import ObservedGainCalibration
from tests.virtual3d.test_hierarchical_options_v10 import MeasuredMapperFixture


class GainMapper:
    def __init__(self):
        self.shape = (15, 15)
        self.config = SimpleNamespace(resolution_m=.2, robot_radius_m=0.,
            max_depth_m=1.2, fov_deg=90.)
        self.belief = np.full(self.shape, -1, np.int8)
        self.belief[5:10, 5:10] = 0
        self.camera_seen = np.zeros(self.shape, bool)

    def grid_cell(self, point):
        return self.shape[0] - 1 - int(np.floor(point[1] / .2)), int(np.floor(point[0] / .2))


class ObservedGainCalibrationTests(unittest.TestCase):
    def test_actual_only_consumption_keeps_unrealized_predicted_cells_eligible(self):
        mapper = GainMapper()
        gain = ObservedGainCalibration(mapper.shape, consume_predicted_attempts=False)
        history = [np.eye(4)]
        attempted = gain.attempted_camera_mask(mapper, history)
        self.assertFalse(attempted.any())
        _, camera = gain.route_masks(mapper,
            [(7, 7, 0)], attempted)
        self.assertFalse(np.any(camera & mapper.camera_seen))

    def test_total_diverse_pool_retains_translation_and_consumes_attempted_mask(self):
        mapper = MeasuredMapperFixture((15, 21))
        current = (7, 7, 0)
        mapper.belief[:, 15:] = -1
        mapper.camera_seen[:, 15:] = False
        routes, audit = generate_options(mapper, current[:2], current[2], 18, current,
            coverage_strategy="total_diverse", coverage_slots=4,
            attempted_camera_mask=np.zeros(mapper.shape, bool))
        self.assertEqual(audit["coverage_strategy"], "total_diverse")
        self.assertGreaterEqual(len({tuple(r["pose"][:2]) for r in routes}), 2)
        self.assertTrue(any(tuple(r["pose"][:2]) != current[:2] for r in routes))
        masked, _ = generate_options(mapper, current[:2], current[2], 18, current,
            coverage_strategy="total_diverse", coverage_slots=4,
            attempted_camera_mask=np.ones(mapper.shape, bool))
        self.assertTrue(all(r["group"].startswith("coverage_total") for r in masked))
        with self.assertRaises(ValueError):
            generate_options(mapper, current[:2], current[2], 18, current,
                coverage_strategy="total_diverse", attempted_camera_mask=np.zeros(mapper.shape, np.uint8))

    def test_paid_feedback_updates_beta_counts_and_no_feedback_freezes_posterior(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                mapper = GainMapper(); gain = ObservedGainCalibration(mapper.shape, feedback_enabled=enabled)
                prediction = gain.prepare_action(mapper, position=(7, 7), heading=0,
                    action="right", action_id=1, planning_camera_poses=[])
                self.assertGreater(prediction["predicted_camera_cells"], 0)
                pending = gain._pending[1]
                predicted = pending["camera"].copy()
                ids = np.argwhere(predicted)
                for cell in ids[:max(1, len(ids)//3)]:
                    mapper.belief[tuple(cell)] = 0
                    mapper.camera_seen[tuple(cell)] = True
                before = gain.snapshot()["posterior_mean"]
                event = gain.observe(mapper, action_id=1)
                self.assertEqual(event["camera"]["predicted_cells"], int(predicted.sum()))
                self.assertEqual(event["camera"]["realized_predicted_cells"], max(1, len(ids)//3))
                after = gain.snapshot()["posterior_mean"]
                self.assertNotEqual(before, after) if enabled else self.assertEqual(before, after)
                with self.assertRaises(ValueError):
                    gain.observe(mapper, action_id=1)

    def test_actual_pose_history_suppresses_repeated_expected_camera_support(self):
        mapper = GainMapper(); gain = ObservedGainCalibration(mapper.shape)
        pose = np.eye(4); pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        pose[:3, 3] = [(7+.5)*.2, (mapper.shape[0]-7-.5)*.2, .8]
        attempted = gain.attempted_camera_mask(mapper, [pose])
        _, unmasked = gain.route_masks(mapper, [(7, 7, 0)], np.zeros(mapper.shape, bool))
        _, masked = gain.route_masks(mapper, [(7, 7, 0)], attempted)
        self.assertGreater(unmasked.sum(), 0)
        self.assertEqual(masked.sum(), 0)


if __name__ == "__main__":
    unittest.main()
