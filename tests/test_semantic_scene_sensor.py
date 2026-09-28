"""New-asset adapter boundary tests; no study World is constructed."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from env import development_sensor_v41 as physical
from env.semantic_scene_sensor import SemanticSceneSensor, validate_sensor_contract, marker_patches_from_asset


class SemanticSceneSensorBoundaryTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.spec = json.loads((root/'configs/virtual3d/v40_scene_protocol_20260920.json').read_text())['public_defaults']
        self.kwargs = dict(expected_manifest_sha256='a'*64, episode_id='boundary_fixture',
                           noise_seed=0, persistent_output_root=root, expected_batch_peak_bytes=1)

    def test_resource_failure_precedes_spec_or_private_asset_read(self):
        before = physical.runtime_counts_v41()
        report = dict(passed=False, status='blocked_before_world_creation',
                      reason='insufficient_persistent_space', free_bytes=0, required_free_bytes=1)
        with patch.object(physical, 'storage_report_v41', return_value=report):
            with self.assertRaises(physical.ResourceGateBlocked):
                SemanticSceneSensor('/nonexistent/private', None, **self.kwargs)
        after = physical.runtime_counts_v41()
        self.assertEqual(after['worlds_created'], before['worlds_created'])
        self.assertEqual(after['blocked_before_world_creation'], before['blocked_before_world_creation']+1)

    def test_manifest_pin_required_before_asset_loader(self):
        self.kwargs['expected_manifest_sha256'] = ''
        with patch.object(physical, 'storage_report_v41', return_value=dict(passed=True)):
            with self.assertRaisesRegex(ValueError, 'external asset manifest'):
                SemanticSceneSensor('/nonexistent/private', self.spec, **self.kwargs)

    def test_runtime_cannot_silently_change_observation_noise(self):
        validate_sensor_contract(self.spec)
        for noise in (0., .02):
            changed = deepcopy(self.spec)
            changed['sensor']['depth_noise_relative_std'] = noise
            with self.assertRaisesRegex(ValueError, 'fixed shared'):
                validate_sensor_contract(changed)

    def test_runtime_cannot_change_scan_or_footprint(self):
        for key, value in (('scan_rays', 180), ('robot_radius_m', .1), ('laser_height_m', .9)):
            with patch.object(physical, 'storage_report_v41', return_value=dict(passed=True)):
                with self.assertRaisesRegex(ValueError, 'shared 0.2m radius'):
                    SemanticSceneSensor('/nonexistent/private', self.spec, **self.kwargs, **{key:value})

    def test_registered_private_asset_payload_matches_sensor_contract(self):
        from nso.semantic_scene_assets import load_semantic_scene_asset
        root = Path(__file__).resolve().parents[1]
        path = root/'audit_results/semantic_scene_assets_20260923/SEM_P00__nominal_relationship'
        if not path.exists():
            self.skipTest('registered development assets are not present')
        before = physical.runtime_counts_v41()
        asset = load_semantic_scene_asset(path,
            expected_manifest_sha256='1566282bc88ef557daa7498c2816e4968c08ae7088bcc6e3d1413b6a96dadfd1')
        patches = marker_patches_from_asset(asset)
        self.assertEqual(len(patches),4)
        self.assertEqual(patches,asset['markers'])
        self.assertIsNot(patches,asset['markers'])
        self.assertEqual(before,physical.runtime_counts_v41())

    def test_wrapped_marker_json_cannot_masquerade_as_loader_payload(self):
        with self.assertRaisesRegex(ValueError,'marker-patch list'):
            marker_patches_from_asset(dict(markers=dict(marker_patches=[])))


if __name__ == '__main__':
    unittest.main()
