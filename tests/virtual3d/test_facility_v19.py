import unittest
import numpy as np
from env.facility_documentation_v19 import FacilityWorldV19, stereo_depth_v19, PARENTS_V19, ASSIGNMENTS_V19


class FacilityV19Tests(unittest.TestCase):
    def test_paired_initial_geometry_and_safe_floor(self):
        for parent in PARENTS_V19:
            worlds = [FacilityWorldV19(parent, assignment) for assignment in ASSIGNMENTS_V19]
            a, b = worlds; fa, fb = a.sense(), b.sense()
            self.assertEqual(len(a.objects), 6)
            self.assertTrue(np.array_equal(a.reachable, ~a._blocked))
            self.assertTrue(a.reachable[a.start])
            np.testing.assert_array_equal(fa.depth_m, fb.depth_m)
            np.testing.assert_array_equal(a.scan().ranges_m, b.scan().ranges_m)
            np.testing.assert_array_equal(fa.semantic > 0, fb.semantic > 0)
            self.assertEqual(set(np.unique(fa.semantic)) - {0}, {2, 3})
            marked = fa.semantic > 0
            np.testing.assert_array_equal(fa.color_rgb[~marked], fb.color_rgb[~marked])
            # At the initial pose, only the two paired front labels are visible.
            np.testing.assert_array_equal(fa.semantic[marked] + fb.semantic[marked], np.full(marked.sum(), 5))
            np.testing.assert_array_equal(a.sense().depth_m, fa.depth_m)
            for world in worlds:
                for item in world.objects:
                    self.assertGreater(world.instance_mesh(item['id']).get_surface_area(), 0)
                bodies = np.asarray(world._solid_primitives)
                self.assertGreaterEqual(float(bodies[bodies[:, 6] >= 100, 3:6].min()), .12)

    def test_semantic_intervention_does_not_change_sensor_geometry(self):
        aligned = FacilityWorldV19()
        frame = aligned.sense()
        for condition in ('shuffled', 'absent'):
            changed = FacilityWorldV19(semantic_condition=condition).sense()
            np.testing.assert_array_equal(frame.depth_m, changed.depth_m)
            np.testing.assert_array_equal(frame.color_rgb, changed.color_rgb)
            np.testing.assert_array_equal(frame.world_from_camera, changed.world_from_camera)
            if condition == 'absent':
                self.assertFalse(changed.semantic.any())
        changed_seed = FacilityWorldV19(noise_seed=1902).sense()
        self.assertFalse(np.array_equal(frame.depth_m, changed_seed.depth_m))

    def test_runtime_observes_two_separate_paired_marked_assets(self):
        from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
        from nso.observed_asset_axes_v16 import measured_assets_v16
        from nso.axis_history_view_v16 import AxisHistoryViewV16
        from nso.cpu_sensor_contract_v10 import digest
        for parent in PARENTS_V19:
            geometries = []
            for assignment in ASSIGNMENTS_V19:
                world = FacilityWorldV19(parent, assignment)
                frame = world.sense()
                mapper = ObservedRuntimeMapperV10(world.shape, world.config)
                mapper.update(frame, world.scan())
                assets = measured_assets_v16(AxisHistoryViewV16(mapper, [frame]))
                marked = [row for row in assets if row['marked_points'] > 0]
                self.assertEqual(len(marked), 2, [(a['aabb_center'], a['marked_points']) for a in assets])
                self.assertEqual({row['class_vote'] for row in marked}, {-1., 1.})
                geometries.append(digest([{k: v for k, v in a.items()
                    if k not in ('class_vote', 'marked_points')} for a in assets]))
            self.assertEqual(geometries[0], geometries[1])

    def test_disparity_noise_distance_response_and_determinism(self):
        clean = np.full((72, 96), 4., np.float32)
        far = stereo_depth_v19(clean)
        near = stereo_depth_v19(clean/4)
        np.testing.assert_array_equal(far, stereo_depth_v19(clean))
        self.assertFalse(np.array_equal(far, stereo_depth_v19(clean, step=1)))
        self.assertGreater(float(np.std(far-4)), 12*float(np.std(near-1)))
        np.testing.assert_array_equal(stereo_depth_v19(clean, model='ideal'), clean)
        clean[0] = 0
        self.assertFalse(stereo_depth_v19(clean)[0].any())
        with self.assertRaises(ValueError):
            stereo_depth_v19(clean, model='label_dependent')


if __name__ == '__main__':
    unittest.main()
