"""Finite arithmetic/geometry fixtures only: no scene, sensor or trajectory."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nso.offline_evaluation_v44 import (ROOT, GEOMETRY_ROOT, _array_sha, _frozen_inputs, sha256,
    conservative_floor_domain_v44, measure_navigation_coverage_v44,
    load_reference_v44, evaluate_saved_episode_v44)
from nso.surface_evaluation_v40 import CandidateViewV40, freeze_reference_v40


def metadata():
    return dict(public_workspace=dict(bounds_xy_m=[[0., 0.], [2., 2.]],
        room_inner_bounds_xy_m=[[0., 0.], [2., 2.]], start_position_world_m=[.5, .5, .9]),
        private_instances=[], background_boxes=[])


class CoverageContractTests(unittest.TestCase):
    def test_fixed_floor_denominator_and_unknown_zero(self):
        domain, desc = conservative_floor_domain_v44(metadata())
        self.assertEqual(desc['denominator_cells'], 256)
        measured = measure_navigation_coverage_v44(np.full((20, 20), -1, np.int8), domain, desc)
        self.assertEqual(measured['C_nav'], 0.)
        self.assertFalse(measured['last_frame_observed_mask_used'])

    def test_free_area_not_touched_or_last_frame(self):
        domain, desc = conservative_floor_domain_v44(metadata())
        belief = np.full(domain.shape, -1, np.int8)
        indexes = np.argwhere(domain)
        for row, col in indexes[:64]:
            belief[row, col] = 0
        for row, col in indexes[64:128]:
            belief[row, col] = 1
        result = measure_navigation_coverage_v44(belief, domain, desc)
        self.assertEqual(result['C_nav'], .25)
        self.assertEqual(result['touched_domain_fraction'], .5)
        self.assertEqual(result['domain_false_occupied_cells'], 64)

    def test_disconnected_floor_is_fixed_before_prediction(self):
        data = metadata()
        data['background_boxes'] = [[.999, 1.001, 0., 2., 0., 1.]]
        domain, desc = conservative_floor_domain_v44(data)
        self.assertGreater(desc['disconnected_free_cells'], 0)
        self.assertFalse(domain[:, 12:].any())
        self.assertTrue(domain[5, 5])
        self.assertFalse(desc['uses_planner_trajectory'])

    def test_all_facility_aabbs_count_even_without_detection(self):
        base, _ = conservative_floor_domain_v44(metadata())
        data = metadata()
        data['private_instances'] = [dict(world_aabb_m=[[1., 1., 0.], [1.3, 1.3, 1.]])]
        domain, desc = conservative_floor_domain_v44(data)
        self.assertLess(domain.sum(), base.sum())
        self.assertFalse(desc['uses_detected_instances'])

    def test_start_cannot_be_replaced(self):
        data = metadata()
        data['private_instances'] = [dict(world_aabb_m=[[.3, .3, 0.], [.7, .7, 1.]])]
        with self.assertRaisesRegex(ValueError, 'start cell'):
            conservative_floor_domain_v44(data)

    def test_tampered_mask_or_noninteger_occupancy_rejected(self):
        domain, desc = conservative_floor_domain_v44(metadata())
        altered = domain.copy(); altered[5, 5] = False
        with self.assertRaises(ValueError):
            measure_navigation_coverage_v44(np.zeros(domain.shape, np.int8), altered, desc)
        with self.assertRaises(ValueError):
            measure_navigation_coverage_v44(np.zeros(domain.shape, float), domain, desc)

    def test_reference_sha_pin_required(self):
        with self.assertRaisesRegex(ValueError, 'presealed'):
            load_reference_v44('/path/not/read', manifest_sha256='', asset_id='DEV_A_00')

    def test_frozen_input_sources_still_match_original_seals(self):
        _, hashes = _frozen_inputs('DEV_A_00', ROOT)
        self.assertEqual(len(hashes), 6)
        with self.assertRaisesRegex(ValueError, 'development asset'):
            _frozen_inputs('TEST_A_00', ROOT)


class SavedEndpointWiringTests(unittest.TestCase):
    """Loader/reference are explicit analytic doubles, never alleged DEV data."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.episode_root = self.root/'episode'
        (self.episode_root/'prediction').mkdir(parents=True)
        workspace = metadata()['public_workspace']
        asset = self.root/GEOMETRY_ROOT/'DEV_A_00'
        asset.mkdir(parents=True)
        (asset/'public_workspace.json').write_text(json.dumps(workspace))
        domain, desc = conservative_floor_domain_v44(metadata())
        belief = np.full(domain.shape, -1, np.int8)
        for row, col in np.argwhere(domain)[:128]:
            belief[row, col] = 0
        self.belief = belief
        self.vertices = np.array([[0, 0, 2], [0, 1, 2], [1, 1, 2], [1, 0, 2],
                                  [1.5, 0, 2], [1.5, 1, 2], [2.5, 1, 2], [2.5, 0, 2]], float)
        self.triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        view = CandidateViewV40(np.array([[10, 0, 30], [0, 10, 30], [0, 0, 1.]]), np.eye(4), 100, 100)
        surface = freeze_reference_v40(self.vertices, self.triangles, np.array([0, 0, 1, 1]),
                                      [view], sample_spacing_m=.3, seed=4001)
        self.reference = (surface, domain, dict(coverage=desc))
        self.episode = SimpleNamespace(root=self.episode_root,
            started=dict(slot=dict(asset_id='DEV_A_00', mode='S'), run_id='analytic_fixture',
                         task_kind='autonomous_controller'),
            runtime=dict(world_created=False), public_workspace=workspace, frames=(object(),),
            task_success=True, status='controller_stop')
        mapper = dict(backend_poisoned=False, shape=desc['shape'], resolution_m=.1,
            origin_xy_m=desc['origin_xy_m'], grid_convention=desc['grid_convention'], frames=1,
            occupancy_sha256=_array_sha(belief))
        (self.episode_root/'prediction/mapper.json').write_text(json.dumps(mapper))
        (self.episode_root/'artifact_manifest.json').write_text('{}')
        self.save_occupancy()
        self.save_mesh(self.triangles)

    def save_occupancy(self):
        np.savez(self.episode_root/'prediction/occupancy.npz', belief=self.belief,
                 observed=np.zeros(self.belief.shape, bool))

    def save_mesh(self, triangles):
        np.savez(self.episode_root/'prediction/mesh.npz', vertices=self.vertices,
                 triangles=triangles, vertex_colors=np.zeros_like(self.vertices))

    def evaluate(self):
        with patch('nso.saved_replay_v44.load_saved_episode_v44', return_value=self.episode), \
                patch('nso.offline_evaluation_v44.load_reference_v44', return_value=self.reference):
            return evaluate_saved_episode_v44(self.episode_root, 'explicit_analytic_reference_double',
                reference_manifest_sha256='0'*64, source_root=self.root,
                expected_episode_manifest_sha256=sha256(self.episode_root/'artifact_manifest.json'))

    def test_whole_prediction_and_measured_coverage_product(self):
        result = self.evaluate()
        self.assertAlmostEqual(result['coverage']['C_nav'], .5)
        self.assertAlmostEqual(result['metrics']['Q'], 1.)
        self.assertAlmostEqual(result['metrics']['J_nav'], .5)
        self.assertNotIn('C_map', result['metrics'])
        self.assertNotIn('J', result['metrics'])
        self.assertFalse(result['eligible_successful_development_endpoint'])
        self.assertEqual(result['new_worlds'], 0)

    def test_undiscovered_missing_facility_remains_zero(self):
        self.save_mesh(self.triangles[:2])
        result = self.evaluate()
        self.assertEqual(len(result['metrics']['per_instance']), 2)
        self.assertEqual(result['metrics']['per_instance'][1]['f1'], 0.)
        self.assertAlmostEqual(result['metrics']['Q'], .5)
        self.assertAlmostEqual(result['metrics']['J_nav'], .25)

    def test_nonreturn_does_not_become_successful_endpoint(self):
        self.episode.task_success = False
        self.episode.status = 'stopped_without_confirmed_return'
        result = self.evaluate()
        self.assertEqual(result['status'], 'diagnostic_non_success_scored')
        self.assertFalse(result['eligible_successful_development_endpoint'])

    def test_occupancy_modified_without_mapper_binding_is_rejected(self):
        self.belief[5, 5] = 1
        self.save_occupancy()
        with self.assertRaisesRegex(ValueError, 'occupancy content'):
            self.evaluate()

    def test_empty_mesh_keeps_all_facilities_and_zero_quality(self):
        np.savez(self.episode_root/'prediction/mesh.npz', vertices=np.empty((0, 3)),
                 triangles=np.empty((0, 3), dtype=np.int64), vertex_colors=np.empty((0, 3)))
        result = self.evaluate()
        self.assertEqual(len(result['metrics']['per_instance']), 2)
        self.assertEqual(result['metrics']['Q'], 0.)
        self.assertEqual(result['metrics']['J_nav'], 0.)

    def test_extra_outside_surface_is_not_cropped(self):
        vertices = np.vstack([self.vertices, np.array([[10, 0, 2], [10, 1, 2], [11, 1, 2]])])
        triangles = np.vstack([self.triangles, [8, 9, 10]])
        np.savez(self.episode_root/'prediction/mesh.npz', vertices=vertices,
                 triangles=triangles, vertex_colors=np.zeros_like(vertices))
        result = self.evaluate()
        self.assertGreater(result['metrics']['extra_false_positive_area_m2'], 0.)
        self.assertLess(result['metrics']['Q'], 1.)
        self.assertAlmostEqual(result['coverage']['C_nav'], .5)

    def test_unexpected_roi_or_owner_array_rejected(self):
        np.savez(self.episode_root/'prediction/mesh.npz', vertices=self.vertices,
                 triangles=self.triangles, vertex_colors=np.zeros_like(self.vertices), owner=np.zeros(4))
        with self.assertRaisesRegex(ValueError, 'complete original mapper mesh'):
            self.evaluate()

    def test_external_episode_digest_is_required(self):
        with self.assertRaisesRegex(ValueError, 'externally pinned'):
            evaluate_saved_episode_v44(self.episode_root, 'not_opened',
                reference_manifest_sha256='0'*64, expected_episode_manifest_sha256='1'*64)

    def test_finite_fixture_cannot_be_paired_with_development_gt(self):
        self.episode.started['finite_fixture'] = True
        with self.assertRaisesRegex(ValueError, '^finite fixture cannot be scored as a development episode$'):
            self.evaluate()


if __name__ == '__main__':
    unittest.main()
