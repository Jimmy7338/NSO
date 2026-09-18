"""Small real sensor contract tests, explicitly counted; no mapper or fusion."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from env.information_pixel_v34 import (InformationPixelWorldV34, cue_from_rgb,
    nonsemantic_rgb, marker_codes_v34, sensor_counts_v34, swept_clear_v34)
from nso.cpu_sensor_contract_v10 import SensorPacket
from nso.sensor_contract_v34 import SensorPacketV34, validate_sensor_packet_v34
from nso.decision_replay_v13 import save_packet, load_packet


TEST_RECEIPT_V34 = {}


class InformationPixelV34Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = dict(sensor_counts_v34())
        # Three small contract fixtures, never the formal four-world/416 table.
        cls.a = InformationPixelWorldV34('P00', 0, 'v34-unit-a')
        cls.b = InformationPixelWorldV34('P00', 1, 'v34-unit-b')
        cls.rotated = InformationPixelWorldV34('P01', 1, 'v34-unit-rotated')
        if any(w.counts['packets'] for w in (cls.a, cls.b, cls.rotated)):
            raise AssertionError('construction performed implicit sensing')
        cls.pa = cls.a.packet_at(cls.a.pose)
        cls.pb = cls.b.packet_at(cls.b.pose)
        cls.pr = cls.rotated.packet_at(cls.rotated.pose)

    @classmethod
    def tearDownClass(cls):
        global TEST_RECEIPT_V34
        after = dict(sensor_counts_v34())
        TEST_RECEIPT_V34 = dict(scope='contract_tests_not_formal_preflight_or_main_tasks',
            calls={k: after[k]-cls.before[k] for k in after},
            per_world={w.episode_id: dict(w.counts) for w in (cls.a, cls.b, cls.rotated)},
            world_geometry_modified=False, mapper_updates=0, TSDF_integrations=0,
            main_tasks=0, all416poses_queried=False)
        print('V34_TEST_CALL_RECEIPT '+json.dumps(TEST_RECEIPT_V34, sort_keys=True))

    def test_class_is_only_actual_color_and_pair_depth_scan_are_equal(self):
        for frame, expected in ((self.pa.frame, 2), (self.pb.frame, 3), (self.pr.frame, 3)):
            cue = cue_from_rgb(frame)
            self.assertEqual(cue['class_id'], expected)
            self.assertGreaterEqual(cue['pixel_count'], 16)
            self.assertEqual(cue['valid_depth_pixel_counts'][expected], cue['pixel_count'])
            self.assertEqual(cue_from_rgb(frame.color_rgb)['type'], cue['type'])
        np.testing.assert_array_equal(self.pa.frame.depth_m, self.pb.frame.depth_m)
        np.testing.assert_array_equal(self.pa.scan.ranges_m, self.pb.scan.ranges_m)
        np.testing.assert_array_equal(nonsemantic_rgb(self.pa.frame.color_rgb),
                                      nonsemantic_rgb(self.pb.frame.color_rgb))
        self.assertFalse(np.array_equal(self.pa.frame.color_rgb, self.pb.frame.color_rgb))

    def test_marker_exists_only_on_first_hit_front_decal_and_rotation_is_real(self):
        for world, packet in ((self.a, self.pa), (self.rotated, self.pr)):
            code = marker_codes_v34(packet.frame.color_rgb)
            v, u = np.nonzero(code)
            depth = packet.frame.depth_m[v, u]
            optical = np.column_stack(((u-47.5)/48*depth, (v-35.5)/48*depth, depth))
            points = optical @ packet.frame.world_from_camera[:3, :3].T+packet.frame.world_from_camera[:3, 3]
            points -= world.shift
            for _ in range((-world.parent['device_frame']['quarter_turns_ccw']) % 4):
                x, y = points[:, 0].copy(), points[:, 1].copy()
                points[:, 0], points[:, 1] = -y, x
            np.testing.assert_allclose(points[:, 1], 1.1, rtol=0, atol=1e-6)
            self.assertTrue(np.all(np.abs(points[:, 0]) <= .35+1e-6))
            self.assertTrue(np.all((points[:, 2] >= .55-1e-6) & (points[:, 2] <= 1.25+1e-6)))
        rear = self.a.packet_at((0, 4, 2))
        self.assertEqual(cue_from_rgb(rear.frame)['pixel_count'], 0)
        self.assertIsNone(cue_from_rgb(rear.frame)['type'])

    def test_rgb_removal_keeps_position_and_does_not_use_fake_semantics(self):
        frame = self.pa.frame
        original = frame.color_rgb.copy(); codes = marker_codes_v34(original)
        clean = nonsemantic_rgb(original)
        np.testing.assert_array_equal(original, frame.color_rgb)
        np.testing.assert_array_equal(clean[codes == 0], original[codes == 0])
        self.assertTrue(np.all(clean[codes > 0] == [127, 127, 127]))
        fake = replace(frame, color_rgb=np.zeros_like(original), semantic=np.full_like(frame.semantic, 3))
        self.assertIsNone(cue_from_rgb(fake)['type'])
        # An arbitrary nearby natural colour must not be silently erased.
        example = np.array([[[41, 100, 220], [40, 100, 220], [220, 60, 40]]], np.uint8)
        self.assertEqual(nonsemantic_rgb(example)[0, 0].tolist(), [41, 100, 220])

    def test_camera4_and_laser8_are_validated_without_range_spoofing(self):
        packet = self.pa
        self.assertIsInstance(packet, SensorPacket)
        self.assertIsInstance(packet, SensorPacketV34)
        self.assertEqual(self.a.config.max_depth_m, 4.)
        self.assertEqual(self.a.config.laser_range_m, 8.)
        self.assertEqual(packet.scan.range_max_m, 8.)
        self.assertGreater(float(packet.scan.ranges_m.max()), 4.)
        self.assertTrue(np.all(packet.frame.depth_m*self.a.ray_norm <= 4.+1e-6))
        before = dict(self.a.counts)
        self.assertIs(validate_sensor_packet_v34(packet, self.a.transform, self.a.config), packet)
        self.assertEqual(dict(self.a.counts), before)
        # V10 stays frozen and still rejects this intentionally new calibration.
        with self.assertRaises(ValueError): SensorPacket.validate(packet, self.a.transform, self.a.config)
        with self.assertRaises(ValueError):
            validate_sensor_packet_v34(replace(packet, scan=replace(packet.scan, range_max_m=4.)),
                                       self.a.transform, self.a.config)

    def test_queries_are_stateless_single_render_and_noise_is_paired(self):
        before = dict(self.a.counts); state = (self.a.pose, self.a.position, self.a.heading, self.a.step_count)
        first = self.a.packet_at(tuple(self.a.parent['anchor']), step=7, noise_model='iid_025px')
        cached = self.a.last_clean_depth.copy()
        second = self.a.packet_at(tuple(self.a.parent['anchor']), step=7, noise_model='iid_025px')
        paired = self.b.packet_at(tuple(self.b.parent['anchor']), step=7, noise_model='iid_025px')
        self.assertEqual(first.sha256(), second.sha256())
        np.testing.assert_array_equal(first.frame.depth_m, paired.frame.depth_m)
        np.testing.assert_array_equal(cached, self.pa.frame.depth_m)
        self.assertFalse(np.array_equal(first.frame.depth_m, cached))
        self.assertEqual((self.a.pose, self.a.position, self.a.heading, self.a.step_count), state)
        for key in ('packets', 'clean_depth_queries', 'scan_queries'):
            self.assertEqual(self.a.counts[key]-before[key], 2)
        with self.assertRaises(TypeError): self.a.counts['packets'] = 0
        with self.assertRaises(TypeError): sensor_counts_v34()['worlds'] = 0
        self.assertFalse(self.a.last_clean_depth.flags.writeable)

    def test_public_map_shift_centres_and_full_raster_are_separate(self):
        for world, expected_grid in ((self.a, 24), (self.rotated, 28)):
            for point in world.parent['nav_cells']:
                for h in range(4):
                    pose = (*point, h); cell = world.pose_to_cell(pose)
                    self.assertEqual(world.cell_to_pose(cell, h), pose)
                    np.testing.assert_allclose(world.transform.cell_to_world(cell), world.v33_to_world(point), atol=1e-10)
            before = dict(world.counts)
            floor = world.evaluation_floor()
            self.assertEqual(floor['grid_denominator'], expected_grid)
            self.assertGreater(floor['raster_denominator'], expected_grid)
            self.assertEqual(floor['raster_denominator'], int(floor['reachable'].sum()))
            self.assertTrue(np.all(floor['reachable'][tuple(floor['declared_grid_cells'].T)]))
            self.assertEqual(dict(world.counts), before)
            self.assertEqual(world.shape, (25, 35) if world.parent_id == 'P00' else (35, 30))
            self.assertFalse(floor['reachable'].flags.writeable)

    def test_step_pays_collision_and_rotation_without_extra_sensing(self):
        # This test uses the rotated fixture's real original state. Its forward
        # path immediately meets the common guard; no geometry is modified.
        world = self.rotated; origin = world.pose; before = dict(world.counts)
        hit = world.step('forward')
        self.assertTrue(hit.collision); self.assertEqual(world.pose, origin)
        self.assertEqual(hit.action_id, 1); self.assertEqual(world.collisions, 1)
        turn = world.step('left')
        self.assertFalse(turn.collision); self.assertEqual(turn.action_id, 2)
        self.assertEqual(world.pose[:2], origin[:2]); self.assertEqual(world.pose[2], (origin[2]-1) % 4)
        for key in ('packets', 'clean_depth_queries', 'scan_queries', 'step_calls'):
            self.assertEqual(world.counts[key]-before[key], 2)

    def test_whole_raised_footprint_and_tangency_match_v33(self):
        raised = [(0., 1., 0., 1., 2., 3.)]
        self.assertTrue(swept_clear_v34((-1., -.2), (2., -.2), raised))
        self.assertFalse(swept_clear_v34((-1., -.199), (2., -.199), raised))
        with self.assertRaises(ValueError): swept_clear_v34((0., 0.), (1., 1.), raised)

    def test_old_packet_io_preserves_new_sensor_arrays_and_new_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'packet.npz'
            save_packet(path, self.pa)
            loaded = load_packet(path)
            self.assertEqual(type(loaded), SensorPacket)
            self.assertEqual(loaded.sha256(), self.pa.sha256())
            self.assertIs(validate_sensor_packet_v34(loaded, self.a.transform, self.a.config), loaded)

    def test_bad_input_depth_pose_identity_and_range_are_rejected(self):
        for packet in (replace(self.pa, action_id=-1), replace(self.pa, heading=4),
                       replace(self.pa, episode_id=''), replace(self.pa, collision=1)):
            with self.assertRaises(ValueError): validate_sensor_packet_v34(packet, self.a.transform, self.a.config)
        depth = self.pa.frame.depth_m.copy(); depth[0, 0] = 3.9
        invalid = replace(self.pa, frame=replace(self.pa.frame, depth_m=depth))
        with self.assertRaisesRegex(ValueError, 'Euclidean'):
            validate_sensor_packet_v34(invalid, self.a.transform, self.a.config)
        scan = replace(self.pa.scan, timestamp_s=1.)
        with self.assertRaises(ValueError):
            validate_sensor_packet_v34(replace(self.pa, scan=scan), self.a.transform, self.a.config)
        before = dict(self.a.counts)
        for pose in ((0, 0, 4), (0., 0, 0), (0, 1, 0)):
            with self.assertRaises(ValueError): self.a.packet_at(pose)
        self.assertEqual(dict(self.a.counts), before)


if __name__ == '__main__':
    unittest.main()
