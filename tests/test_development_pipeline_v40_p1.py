"""Check that evaluator-private reachability cannot jump across thin obstacles."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('development_surface_pipeline', ROOT/'scripts/verify_development_surface_pipeline_v40.py')
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


class ReachableCandidateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT/'configs/virtual3d/v40_scene_protocol_20260920.json').read_text())
        self.metadata = {'public_workspace': {'room_inner_bounds_xy_m': [[0, 0], [4.5, 4.5]],
                                              'start_position_world_m': [.75, .75, .9]},
                         'private_instances': [], 'background_boxes': []}

    def test_thin_wall_cannot_be_skipped_between_free_endpoints(self):
        self.metadata['background_boxes'] = [[1.45, 1.55, 0, 4.5, 0, 2]]
        views, receipt = PIPELINE.reachable_candidates(self.metadata, self.protocol)
        self.assertEqual(receipt['collision_free_cells'], 9)
        self.assertEqual(receipt['reachable_cells'], 3)
        self.assertEqual(len(views), 12)
        self.assertTrue(all(view.world_from_camera[0, 3] < 1.45 for view in views))

    def test_connected_gap_restores_reachability(self):
        self.metadata['background_boxes'] = [[1.45, 1.55, 0, 2.99, 0, 2]]
        views, receipt = PIPELINE.reachable_candidates(self.metadata, self.protocol)
        self.assertEqual(receipt['reachable_cells'], 9)
        self.assertEqual(len(views), 36)

    def test_blocked_start_is_not_silently_relocated(self):
        self.metadata['background_boxes'] = [[.5, 1., .5, 1., 0, 2]]
        with self.assertRaisesRegex(ValueError, 'predeclared start is blocked'):
            PIPELINE.reachable_candidates(self.metadata, self.protocol)

    def test_floor_does_not_block_all_candidate_poses(self):
        self.metadata['background_boxes'] = [[0, 4.5, 0, 4.5, -.1, 0]]
        views, receipt = PIPELINE.reachable_candidates(self.metadata, self.protocol)
        self.assertEqual(receipt['reachable_cells'], 9)
        self.assertTrue(all(view.world_from_camera[2, 3] == .9 for view in views))
        self.assertTrue(all(view.far_m == 4 for view in views))


if __name__ == '__main__':
    unittest.main()
