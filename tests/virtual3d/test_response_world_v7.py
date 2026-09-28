"""First predeclared parent only: sensor isolation, not candidate effectiveness."""
from dataclasses import fields
import unittest

import numpy as np

from env.virtual3d_response_v7 import (
    RESPONSE_FAMILIES, ResponseConfigV7, collect_response_prefix,
    create_response_world, get_response_context, load_response_contexts,
)
from env.virtual3d_inspection_v4 import read_inspection_markers_rgb
from utils.rgbd_contract import RGBDFrame, PlanarScan


class ResponseWorldV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worlds = [create_response_world("T0", family) for family in RESPONSE_FAMILIES]
        cls.prefixes = [collect_response_prefix(world) for world in cls.worlds]

    def test_parent_manifest_is_finite_and_grouped_without_opening_worlds(self):
        manifest = load_response_contexts()
        rows = manifest["contexts"]
        self.assertEqual([r["context_id"] for r in rows], [f"T{i}" for i in range(8)] + [f"C{i}" for i in range(4)])
        self.assertEqual(len({r["sensor_seed"] for r in rows}), 12)
        for role, expected in (("train", 4), ("calibration", 2)):
            for seed in (751, 752):
                self.assertEqual(sum(r["role"] == role and r["outer_seed"] == seed for r in rows), expected)
        self.assertFalse({r["outer_seed"] for r in rows} & set(manifest["excluded_outer_seeds"]))
        with self.assertRaises(ValueError):
            get_response_context("753")

    def test_all_twenty_paid_prefix_steps_have_identical_geometry_and_scan(self):
        for a, b in zip(*self.prefixes):
            self.assertEqual(a["step"], b["step"])
            self.assertEqual(a["action"], b["action"])
            for key in ("depth_m", "world_from_camera", "intrinsic", "timestamp_s"):
                np.testing.assert_array_equal(getattr(a["frame"], key), getattr(b["frame"], key))
            for field in fields(PlanarScan):
                np.testing.assert_array_equal(getattr(a["scan"], field.name), getattr(b["scan"], field.name))
            ma, mb = [read_inspection_markers_rgb(r["frame"].color_rgb) > 0 for r in (a, b)]
            np.testing.assert_array_equal(ma, mb)
            np.testing.assert_array_equal(a["frame"].color_rgb[~ma], b["frame"].color_rgb[~ma])

    def test_rgb_identifies_physical_asset_family_only_on_visible_marker(self):
        for category, rows in zip((2, 3), self.prefixes):
            self.assertGreater(np.count_nonzero(rows[0]["frame"].semantic), 0)
            self.assertGreater(np.count_nonzero(rows[-1]["frame"].semantic), 0)
            for row in rows:
                frame = row["frame"]
                np.testing.assert_array_equal(frame.semantic, read_inspection_markers_rgb(frame.color_rgb))
                self.assertLessEqual(set(np.unique(frame.semantic)), {0, category})
                self.assertEqual(set(vars(frame)), {f.name for f in fields(RGBDFrame)})
                self.assertEqual(set(vars(row["scan"])), {f.name for f in fields(PlanarScan)})

    def test_hidden_solids_differ_but_common_primitives_and_footprint_match(self):
        a, b = self.worlds
        for key in ("occupancy", "_blocked", "reachable"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        np.testing.assert_array_equal(np.asarray(a._solid_primitives[:a.common_primitive_count])[:, :6],
                                      np.asarray(b._solid_primitives[:b.common_primitive_count])[:, :6])
        self.assertFalse(np.array_equal(np.asarray(a.mesh.vertices), np.asarray(b.mesh.vertices)))
        self.assertEqual(a.seed, b.seed)
        self.assertEqual(a.start, b.start)
        self.assertFalse(a.inspection_truth["union_surface"]["internal_faces_emitted"])
        self.assertFalse(b.inspection_truth["union_surface"]["coplanar_overlap_faces_emitted"])

    def test_prefix_does_not_teleport_or_grant_free_observations(self):
        for world, rows in zip(self.worlds, self.prefixes):
            self.assertEqual(len(rows), 21)
            self.assertEqual(world.step_count, 20)
            self.assertEqual(world.moves, 10)
            self.assertEqual(world.collisions, 0)
            self.assertEqual(world.heading, 0)  # only T0 is constructed by this test suite
            self.assertEqual(sum(row["action"] == "left" for row in rows), 5)
            self.assertEqual(sum(row["action"] == "right" for row in rows), 5)
            self.assertAlmostEqual(np.linalg.norm(rows[-1]["frame"].world_from_camera[:3, 3]
                                                  - rows[0]["frame"].world_from_camera[:3, 3]), 2.)
            with self.assertRaises(ValueError):
                collect_response_prefix(world)

    def test_interpretation_interventions_leave_actual_sensor_color_unchanged(self):
        frames = [create_response_world("T0", RESPONSE_FAMILIES[0], semantic_condition=condition).sense()
                  for condition in ("aligned", "shuffled", "absent")]
        for frame in frames[1:]:
            for key in ("depth_m", "color_rgb", "intrinsic", "world_from_camera"):
                np.testing.assert_array_equal(getattr(frames[0], key), getattr(frame, key))
        np.testing.assert_array_equal(frames[1].semantic, np.where(frames[0].semantic == 2, 3, 0))
        self.assertFalse(frames[2].semantic.any())

    def test_invalid_sensor_contract_cannot_silently_break_occlusion(self):
        with self.assertRaises(ValueError):
            ResponseConfigV7(laser_height_m=.4)
        with self.assertRaises(ValueError):
            ResponseConfigV7(pose_noise_m=.01)


if __name__ == "__main__":
    unittest.main()
