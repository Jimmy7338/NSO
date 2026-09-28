"""All four fixed worlds: real prefix sensors, never future candidate rewards."""
from dataclasses import fields
import unittest

import numpy as np

from env.virtual3d_competition_v8 import (
    COMPETITION_ARRANGEMENTS, CompetitionConfigV8, collect_competition_prefix,
    create_competition_world, get_competition_context, load_competition_contexts,
)
from env.virtual3d_inspection_v4 import read_inspection_markers_rgb
from utils.rgbd_contract import RGBDFrame, PlanarScan


class CompetitionWorldV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worlds, cls.prefixes = {}, {}
        for parent in ("P0", "P1"):
            for arrangement in COMPETITION_ARRANGEMENTS:
                key = (parent, arrangement)
                cls.worlds[key] = create_competition_world(*key)
                cls.prefixes[key] = collect_competition_prefix(cls.worlds[key])

    def test_finite_parent_list_cannot_open_unseen_seeds(self):
        manifest = load_competition_contexts()
        self.assertEqual([r["context_id"] for r in manifest["contexts"]], ["P0", "P1"])
        self.assertEqual([r["outer_seed"] for r in manifest["contexts"]], [751, 752])
        self.assertEqual(len({r["sensor_seed"] for r in manifest["contexts"]}), 2)
        for key in ("753", "754", "301", "401", "P2"):
            with self.assertRaises(ValueError):
                get_competition_context(key)

    def test_every_paid_prefix_frame_has_exact_paired_nonlabel_sensors(self):
        for parent in ("P0", "P1"):
            a_rows, b_rows = [self.prefixes[parent, arrangement] for arrangement in COMPETITION_ARRANGEMENTS]
            for a, b in zip(a_rows, b_rows):
                with self.subTest(parent=parent, step=a["step"]):
                    self.assertEqual(a["step"], b["step"])
                    self.assertEqual(a["action"], b["action"])
                    for key in ("depth_m", "world_from_camera", "intrinsic", "timestamp_s"):
                        np.testing.assert_array_equal(getattr(a["frame"], key), getattr(b["frame"], key))
                    for field in fields(PlanarScan):
                        np.testing.assert_array_equal(getattr(a["scan"], field.name), getattr(b["scan"], field.name))
                    masks = [read_inspection_markers_rgb(row["frame"].color_rgb) > 0 for row in (a, b)]
                    np.testing.assert_array_equal(*masks)
                    np.testing.assert_array_equal(a["frame"].color_rgb[~masks[0]], b["frame"].color_rgb[~masks[1]])

    def test_physical_asset_markers_are_actually_observed_and_only_rgb_is_read(self):
        for key, rows in self.prefixes.items():
            seen = set()
            for row in rows:
                frame = row["frame"]
                np.testing.assert_array_equal(frame.semantic, read_inspection_markers_rgb(frame.color_rgb))
                seen.update(np.unique(frame.semantic))
                self.assertEqual(set(vars(frame)), {f.name for f in fields(RGBDFrame)})
                self.assertEqual(set(vars(row["scan"])), {f.name for f in fields(PlanarScan)})
            self.assertEqual(seen, {0, 2, 3}, key)
        for parent in ("P0", "P1"):
            a, b = [self.prefixes[parent, arrangement] for arrangement in COMPETITION_ARRANGEMENTS]
            for ra, rb in zip(a, b):
                labels = ra["frame"].semantic
                np.testing.assert_array_equal(rb["frame"].semantic,
                    np.where(labels == 2, 3, np.where(labels == 3, 2, 0)))

    def test_hidden_union_and_total_area_match_physical_swap_without_duplicate_surfaces(self):
        for parent in ("P0", "P1"):
            a, b = [self.worlds[parent, arrangement] for arrangement in COMPETITION_ARRANGEMENTS]
            for key in ("occupancy", "_blocked", "reachable"):
                np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
            np.testing.assert_array_equal(np.asarray(a._solid_primitives[:a.common_primitive_count])[:, :6],
                                          np.asarray(b._solid_primitives[:b.common_primitive_count])[:, :6])
            self.assertEqual(a.common_primitive_count, 15)
            self.assertFalse(np.array_equal(np.asarray(a._solid_primitives)[:, :6],
                                             np.asarray(b._solid_primitives)[:, :6]))
            self.assertAlmostEqual(a.mesh.get_surface_area(), b.mesh.get_surface_area(), delta=1e-10)
            self.assertEqual(a.seed, b.seed)
            for world in (a, b):
                self.assertEqual(sorted(obj["category"] for obj in world.objects), [2, 3])
                self.assertEqual(set(world._geometry_labels.values()), {1})
                union = world.competition_truth["union_surface"]
                self.assertFalse(union["internal_faces_emitted"])
                self.assertFalse(union["coplanar_overlap_faces_emitted"])

    def test_actual_prefix_cost_position_and_occluding_origin_contract(self):
        for key, world in self.worlds.items():
            rows = self.prefixes[key]
            self.assertEqual(len(rows), 151)
            self.assertEqual(world.step_count, 150)
            self.assertEqual(world.moves, 120)
            self.assertEqual(world.collisions, 0)
            self.assertEqual(world.heading, world.context.final_prefix_heading)
            self.assertEqual(sum(r["action"] in ("left", "right") for r in rows), 30)
            self.assertTrue(all(not r["collision"] and not r["done"] for r in rows))
            np.testing.assert_allclose(rows[0]["frame"].world_from_camera[:2, 3], world.context.initial_xy_m, atol=1e-12)
            np.testing.assert_allclose(rows[-1]["frame"].world_from_camera[:2, 3], world.context.final_prefix_xy_m, atol=1e-12)
            self.assertLessEqual(max(r["frame"].world_from_camera[1, 3] for r in rows), 4.7 + 1e-12)
            for previous, current in zip(rows, rows[1:]):
                displacement = np.linalg.norm(current["frame"].world_from_camera[:3, 3]
                                              - previous["frame"].world_from_camera[:3, 3])
                self.assertAlmostEqual(displacement, .2 if current["action"] == "forward" else 0., delta=1e-12)
            with self.assertRaises(ValueError):
                collect_competition_prefix(world)

    def test_parent_mirror_changes_side_without_bypassing_paid_actions(self):
        for arrangement in COMPETITION_ARRANGEMENTS:
            p0, p1 = [self.prefixes[parent, arrangement] for parent in ("P0", "P1")]
            for a, b in zip(p0, p1):
                pose_a, pose_b = a["frame"].world_from_camera, b["frame"].world_from_camera
                self.assertAlmostEqual(pose_a[0, 3] + pose_b[0, 3], 8.4, delta=1e-12)
                np.testing.assert_allclose(pose_a[1:3, 3], pose_b[1:3, 3], atol=1e-12)
                self.assertEqual(b["action"], {None: None, "forward": "forward", "left": "right", "right": "left"}[a["action"]])

    def test_interpretation_only_interventions_cannot_change_physical_sensors(self):
        worlds = [create_competition_world("P0", "shelf_west", semantic_condition=c)
                  for c in ("aligned", "shuffled", "absent")]
        # Initial frames have markers in front; no unregistered future pose.
        frames = [w.sense() for w in worlds]
        self.assertTrue(frames[0].semantic.any())
        for frame in frames[1:]:
            for key in ("depth_m", "color_rgb", "intrinsic", "world_from_camera", "timestamp_s"):
                np.testing.assert_array_equal(getattr(frames[0], key), getattr(frame, key))
        labels = frames[0].semantic
        np.testing.assert_array_equal(frames[1].semantic, np.where(labels == 2, 3, np.where(labels == 3, 2, 0)))
        self.assertFalse(frames[2].semantic.any())

    def test_sensor_contract_cannot_silently_change_occlusion_or_cost(self):
        for change in (dict(laser_height_m=.4), dict(pose_noise_m=.01),
                       dict(depth_sigma_m=.03), dict(max_steps=197), dict(action_duration_s=0.)):
            with self.assertRaises(ValueError):
                CompetitionConfigV8(**change)


if __name__ == "__main__":
    unittest.main()
