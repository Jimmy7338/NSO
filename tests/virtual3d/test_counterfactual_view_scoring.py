"""Fixed-path semantic interventions must not alter geometry or cost accounting."""
from dataclasses import replace
import json
import unittest

import numpy as np
import open3d as o3d

from env.grid_exploration import GridConfig
from env.virtual3d import VirtualConfig
from nso.counterfactual_view_scoring import CounterfactualScoreConfig, score_routes, fixed_horizon_gain_auc
from nso.semantic_completion_v3 import SemanticHistoryMapperV3
from utils.inspection_benchmark import InspectionWorld


class MeasuredMapper:
    """Small synthetic *measured* ledger, independent of any simulator GT."""
    def __init__(self, mode):
        self.config = VirtualConfig(depth_sigma_m=0., dropout=0., voxel_m=.03)
        self.shape = (30, 40)
        self.belief = np.zeros(self.shape, np.int8)
        self.belief[:, 24:] = -1
        self.visible = self.belief == 0
        self.camera_seen = np.zeros(self.shape, bool)
        self.frames = 1
        self.keyframes = []
        self.quality = {}
        # A measured vertical face with enough extent for the common model.
        for y in np.linspace(2.5, 3.15, 6):
            for z in np.linspace(.3, 1.2, 6):
                point = np.array([4., y, z])
                key = tuple(np.floor(point / .15).astype(int))
                category = {"aligned": 2, "shuffled": 3, "absent": 0}[mode]
                self.quality[key] = dict(point=point, normal=np.array([1., 0., 0.]),
                    n=1, bits=1, label=category, information=.1, best_range=2.,
                    residual=.001, normal_dispersion=0.)

    def mesh(self):
        return o3d.geometry.TriangleMesh()

    def quality_evidence(self, max_points=1600):
        rows = list(self.quality.values())[:max_points]
        if not rows:
            return None
        return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}

    def evidence(self):
        q = self.quality_evidence()
        if q is None:
            return np.empty((0, 3)), np.empty(0, int), np.empty(0, int)
        return q["point"], q["bits"], q["label"]


class CounterfactualScoringTests(unittest.TestCase):
    def setUp(self):
        self.mappers = {mode: MeasuredMapper(mode) for mode in ("aligned", "shuffled", "absent")}
        virtual = self.mappers["aligned"].config
        self.config = CounterfactualScoreConfig(GridConfig(resolution_m=.2,
            robot_radius_m=.2, sensor_range_m=4., sensor_fov_deg=360., max_steps=280), virtual)
        self.states = [(15, 10, 1), (15, 10, 2), (15, 10, 3), (15, 10, 0), (15, 10, 1)]

    def test_same_pool_semantics_only_changes_hidden_weights(self):
        result = score_routes(self.mappers, [{"candidate_id": 0, "states": self.states}], self.config)
        scores = result["candidates"][0]["scores"]
        self.assertEqual(scores["G"], scores["M"])
        self.assertEqual(scores["G"], scores["O"])
        self.assertTrue(result["diagnostics"]["objectness_degenerate_with_geometry"])
        self.assertEqual(len(set(result["invariants"]["shape_geometry_sha256"].values())), 1)
        weights = result["invariants"]["shape_weights_sha256"]
        self.assertNotEqual(weights["S"], weights["X"])
        self.assertGreater(len(result["objects"]["S"]), 0)
        self.assertEqual(scores["N"]["hidden_area_m2"], 0.)
        for key in ("observed_quality_gain", "camera_coverage_cells", "radar_coverage_cells", "cost"):
            self.assertEqual(len({row[key] for row in scores.values()}), 1)
        json.dumps(result, allow_nan=False)

    def test_full_path_max_union_deduplicates_repeated_round_trip(self):
        repeated = self.states + self.states[1:]
        result = score_routes(self.mappers, [
            {"candidate_id": 0, "states": self.states},
            {"candidate_id": 1, "states": repeated}], self.config)
        a, b = result["candidates"]
        self.assertEqual(a["union_hashes"], b["union_hashes"])
        for name in result["scorers"]:
            first, second = a["scores"][name], b["scores"][name]
            for key in ("hidden_area_m2", "observed_quality_gain", "camera_coverage_cells", "radar_coverage_cells"):
                self.assertEqual(first[key], second[key])
            self.assertAlmostEqual(first["score"], 2 * second["score"])

    def test_common_budget_does_not_reward_repeating_identical_evidence(self):
        result = score_routes(self.mappers, [
            {"candidate_id": 0, "states": self.states},
            {"candidate_id": 1, "states": self.states + self.states[1:]}],
            replace(self.config, score_objective="horizon_auc"))
        a, b = result["candidates"]
        for name in result["scorers"]:
            self.assertAlmostEqual(a["scores"][name]["score"], b["scores"][name]["score"])
            self.assertAlmostEqual(a["scores"][name]["rate_score"], 2*b["scores"][name]["rate_score"])
        self.assertEqual(result["selected"]["G"], result["selected"]["M"])

    def test_common_budget_values_earlier_gain_without_cost_ratio_amplification(self):
        self.assertAlmostEqual(fixed_horizon_gain_auc([0., 1.], 4), .875)
        self.assertAlmostEqual(fixed_horizon_gain_auc([0., 0., 0., 0., 1.], 4), .125)
        self.assertGreater(fixed_horizon_gain_auc([0., 0., 0., 1.], 4),
                           fixed_horizon_gain_auc([0., .1], 4))
        with self.assertRaises(ValueError):
            fixed_horizon_gain_auc([0., 1., 1.], 1)

    def test_geometry_mutation_cannot_be_hidden_by_label_intervention(self):
        first = next(iter(self.mappers["shuffled"].quality.values()))
        first["information"] += .01
        with self.assertRaisesRegex(ValueError, "quality_nonlabel"):
            score_routes(self.mappers, [], self.config)

    def test_missing_labels_are_checked_instead_of_aliasing_geometry(self):
        next(iter(self.mappers["absent"].quality.values()))["label"] = 3
        with self.assertRaisesRegex(ValueError, "absent quality"):
            score_routes(self.mappers, [], self.config)

    def test_swapped_labels_must_be_an_actual_two_three_swap(self):
        next(iter(self.mappers["shuffled"].quality.values()))["label"] = 2
        with self.assertRaisesRegex(ValueError, "not a 2/3 swap"):
            score_routes(self.mappers, [], self.config)

    def test_motion_accounting_rejects_free_frames_and_teleports(self):
        bad = [self.states[0], self.states[0]]
        with self.assertRaisesRegex(ValueError, "teleport, free observation"):
            score_routes(self.mappers, [{"candidate_id": 0, "states": bad}], self.config)
        with self.assertRaisesRegex(ValueError, "cost differs"):
            score_routes(self.mappers, [{"candidate_id": 0, "states": self.states, "cost": 3}], self.config)

    def test_empty_pool_remains_an_explicit_no_selection(self):
        result = score_routes(self.mappers, [], self.config)
        self.assertEqual(result["candidates"], [])
        self.assertTrue(all(value is None for value in result["selected"].values()))

    def test_wrong_metric_resolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "0.2 m"):
            CounterfactualScoreConfig(GridConfig(), self.mappers["aligned"].config)

    def test_real_rgbd_replay_has_identical_tsdf_and_missing_pathway(self):
        # Existing unit fixture, not a protocol development/check seed.
        world = InspectionWorld(self.config.virtual_config, 223, 'box')
        pose, _ = world.pose(1.4, -90)
        frame = world.capture(pose, 0, 'none')
        mappers = {}
        labels = frame.semantic
        for mode in ("aligned", "shuffled", "absent"):
            semantic = labels if mode == "aligned" else (
                np.where(labels == 2, 3, np.where(labels == 3, 2, labels))
                if mode == "shuffled" else np.zeros_like(labels))
            mapper = SemanticHistoryMapperV3((30, 40), world.config)
            mapper.update(replace(frame, semantic=semantic.astype(np.uint8)))
            mappers[mode] = mapper
        row, col = mappers["aligned"].grid_cell(pose[:3, 3])
        states = [(row, col, h) for h in (0, 1, 2, 3, 0)]
        result = score_routes(mappers, [{"candidate_id": 0, "states": states}], self.config)
        self.assertEqual(result["candidates"][0]["scores"]["G"],
                         result["candidates"][0]["scores"]["M"])
        hashes = result["invariants"]["mapper_hashes"]
        self.assertEqual(len({row["tsdf_mesh_sha256"] for row in hashes.values()}), 1)

    def test_changed_tsdf_cannot_pass_shared_geometry_invariant(self):
        self.mappers["shuffled"].mesh = lambda: o3d.geometry.TriangleMesh.create_box()
        with self.assertRaisesRegex(ValueError, "tsdf_mesh"):
            score_routes(self.mappers, [], self.config)


if __name__ == "__main__":
    unittest.main()
