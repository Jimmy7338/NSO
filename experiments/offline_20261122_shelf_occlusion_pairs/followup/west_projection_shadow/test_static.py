"""Cheap endpoint cases and recorded sensor support, without any actor or renderer."""

from math import floor
from unittest import TestCase, main

import numpy as np

from inputs import INITIAL, OPPOSITE_SCAN, WEST_DEPTH, WEST_SCAN, read
from projection import cell_tolerance, directed_endpoint
from validate import edge_admissions, return_edges


class EndpointProjection(TestCase):
    shape = (45, 51)
    resolution = .2

    @staticmethod
    def logged_pose_to_xy(info):
        row, col, _ = info["pose"]
        return ((col + .5) * .2, (45 - row - .5) * .2)

    def hit(self, origin, point):
        raw = (44 - floor(point[1] / .2), floor(point[0] / .2))
        return directed_endpoint(origin, point, self.resolution, self.shape, raw)

    def test_opposite_observed_faces(self):
        initial = read(INITIAL)
        self.assertEqual(tuple(initial["shape"]), self.shape)
        self.assertEqual(initial["grid_resolution_m"], self.resolution)
        depth = read(WEST_DEPTH)["first_local_rejection_and_anchor_disconnection"]["61"]
        lidar = read(WEST_SCAN)
        reverse = read(OPPOSITE_SCAN)["first_direct_witness"]
        self.assertEqual(len(depth["occupied_update_samples"][0]["RGB_D"]), 44)
        self.assertEqual(sum(len(cell["laser"]) for cell in lidar["nearby_blocking_obstacle_endpoints"]), 3)
        src_depth = self.logged_pose_to_xy(depth)
        src_lidar = self.logged_pose_to_xy(lidar)
        src_reverse = self.logged_pose_to_xy(reverse)
        for group, samples, origin, col in (("camera", depth["occupied_update_samples"], src_depth, 17),
                                           ("laser", lidar["nearby_blocking_obstacle_endpoints"], src_lidar, 17),
                                           ("laser", reverse["first_lost_occupied_sensor_endpoints"], src_reverse, 9)):
            for cell in samples:
                for item in cell["RGB_D" if group == "camera" else "laser"]:
                    with self.subTest(group=group, point=item["xyz_m"]):
                        self.assertEqual(self.hit(origin, item["xyz_m"])[1], col)

    def test_same_plane_both_directions(self):
        high, low = float(np.nextafter(3.4, np.inf)), float(np.nextafter(3.4, -np.inf))
        self.assertEqual(self.hit((2.9, 4.3), (low, 4.3)), (23, 17))
        self.assertEqual(self.hit((3.9, 4.3), (high, 4.3)), (23, 16))

    def test_nominal_and_ambiguity_fall_back(self):
        # Nonboundary hit, two-axis intersection, same-origin cell, and an
        # almost tangent plane whose ULP slab would cross a second boundary.
        for origin, near, raw in (
            ((2.9, 4.3), (3.3999, 4.3), (23,16)),
            ((2.9, 4.3), (3.399999999, 4.400000001), (22,16)),
            ((3.35, 4.3), (3.399999999, 4.3), (23,16)),
            ((3.4 - 4*cell_tolerance(17)*.2, 4.25), (3.399999999, 4.3999), (23,16)),
            ((0.1, 4.3), (-1e-10, 4.3), (23,-1)),
        ):
            with self.subTest(near=near):
                self.assertEqual(self.hit(origin, near), raw)


class GuardSweeps(TestCase):
    def test_directed_edges_and_turn_checks(self):
        directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
        safe = np.array([[0, 0, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0]], bool)
        edges = edge_admissions(safe, directions)
        self.assertTrue(edges[1, 0, 1])
        self.assertTrue(edges[1, 1, 3])
        self.assertFalse(edges[1, 0, 3])
        self.assertFalse(edges[1, 1, 0])
        moves = ("forward", "right", "forward")
        self.assertTrue(return_edges([1, 0], 1, moves, [2, 1, 2], edges, safe, directions))
        edges[1, 1, 2] = False
        self.assertFalse(return_edges([1, 0], 1, moves, [2, 1, 2], edges, safe, directions))


if __name__ == "__main__":
    main()
