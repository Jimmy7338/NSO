"""Saved finite-packet verification; no development World or physical replay."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
import numpy as np

from nso.episode_driver_v43 import canonical_bytes
from nso.saved_replay_v44 import (load_saved_episode_v44, inspect_saved_inventory_v44,
    verify_saved_observations_v44, _verify_loaded_v44, _canonical_mesh,
    SavedEpisodeIntegrityErrorV44, FailedSavedEpisodeV44)
from v44_episode_fixture import create_finite_episode_v44, FiniteMapperV44, FiniteControllerV44


def reseal(path):
    path = Path(path)
    manifest = json.loads((path/'artifact_manifest.json').read_text())
    manifest['files'] = {str(file.relative_to(path)): dict(bytes=file.stat().st_size,
        sha256=hashlib.sha256(file.read_bytes()).hexdigest()) for file in path.rglob('*')
        if file.is_file() and file.name != 'artifact_manifest.json'}
    (path/'artifact_manifest.json').write_bytes(canonical_bytes(manifest))


def alter_json(path, mutate):
    data = json.loads(path.read_text()); mutate(data); path.write_bytes(canonical_bytes(data))


class SavedReplayV44Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)/'episode'
        create_finite_episode_v44(self.root)
    def tearDown(self):
        self.temp.cleanup()
    def test_complete_finite_observations_verify_without_sensor_calls(self):
        episode = load_saved_episode_v44(self.root)
        self.assertFalse(episode.task_success); self.assertFalse(episode.eligible_study_episode)
        self.assertEqual(len(episode.frames), 2)
        with patch('env.development_sensor_v41.create_development_sensor', side_effect=AssertionError('no World')):
            result = _verify_loaded_v44(episode, FiniteMapperV44(), FiniteControllerV44())
        self.assertEqual(result['status'], 'verified'); self.assertEqual(result['frames_verified'], 2)
        self.assertTrue(result['source_is_finite_fixture']); self.assertFalse(result['counterfactual_trajectory'])
    def test_byte_corruption_rejected_before_backend(self):
        with (self.root/'packets/000_rgbd.npz').open('ab') as stream: stream.write(b'x')
        with self.assertRaisesRegex(SavedEpisodeIntegrityErrorV44, 'SHA256'):
            load_saved_episode_v44(self.root)
    def test_missing_frame_rejected_even_if_manifest_rewritten(self):
        (self.root/'packets/001_scan.npz').unlink(); reseal(self.root)
        with self.assertRaisesRegex(SavedEpisodeIntegrityErrorV44, 'sequence'):
            load_saved_episode_v44(self.root)
    def test_extra_frame_rejected_even_if_manifest_rewritten(self):
        (self.root/'packets/002_receipt.json').write_text('{}'); reseal(self.root)
        with self.assertRaisesRegex(SavedEpisodeIntegrityErrorV44, 'sequence'):
            load_saved_episode_v44(self.root)
    def test_source_archive_mismatch_rejected(self):
        (self.root/'source/nso/controller_v43.py').write_text('# wrong source\n'); reseal(self.root)
        with self.assertRaisesRegex(SavedEpisodeIntegrityErrorV44, 'source mismatch'):
            load_saved_episode_v44(self.root)
    def test_current_source_root_mismatch_rejected(self):
        with self.assertRaisesRegex(SavedEpisodeIntegrityErrorV44, 'source mismatch'):
            load_saved_episode_v44(self.root, source_root=self.root)
    def test_failed_episode_preserved_but_not_replayed(self):
        alter_json(self.root/'result.json', lambda data: data.update(status='episode_error', error={'stage': 'mapping'}))
        reseal(self.root)
        self.assertEqual(inspect_saved_inventory_v44(self.root)['result']['status'], 'episode_error')
        with self.assertRaises(FailedSavedEpisodeV44): load_saved_episode_v44(self.root)
    def test_packet_action_disagreement_rejected(self):
        alter_json(self.root/'packets/001_receipt.json', lambda data: data['execution'].update(action='turn_left'))
        reseal(self.root)
        with self.assertRaises(ValueError): load_saved_episode_v44(self.root)
    def test_duplicate_packet_identity_rejected(self):
        with np.load(self.root/'packets/001_rgbd.npz') as old:
            data = {name: old[name].copy() for name in old.files}
        with np.load(self.root/'packets/000_rgbd.npz') as first:
            data['frame_id'] = first['frame_id'].copy()
        np.savez_compressed(self.root/'packets/001_rgbd.npz', **data)
        reseal(self.root)
        with self.assertRaises(SavedEpisodeIntegrityErrorV44): load_saved_episode_v44(self.root)
    def test_future_packets_not_processed_after_action_divergence(self):
        episode = load_saved_episode_v44(self.root)
        class Divergent(FiniteControllerV44):
            def choose(self): return dict(action='left')
        mapper = FiniteMapperV44()
        result = _verify_loaded_v44(episode, mapper, Divergent())
        self.assertEqual(result['status'], 'diverged'); self.assertEqual(result['divergent_step'], 0)
        self.assertEqual(len(mapper.receipts), 1); self.assertFalse(result['future_saved_frames_processed'])
    def test_manifest_pin_and_duplicate_json_fields_rejected(self):
        with self.assertRaisesRegex(SavedEpisodeIntegrityErrorV44, 'external'):
            load_saved_episode_v44(self.root, expected_manifest_sha256='0'*64)
        (self.root/'result.json').write_text('{"status": "controller_stop", "status": "episode_error"}')
        reseal(self.root)
        with self.assertRaisesRegex(SavedEpisodeIntegrityErrorV44, 'duplicate JSON'):
            load_saved_episode_v44(self.root)
    def test_saved_map_must_bind_final_observed_prediction(self):
        alter_json(self.root/'prediction/mapper.json', lambda data: data.update(occupancy_sha256='0'*64))
        reseal(self.root)
        with self.assertRaisesRegex(SavedEpisodeIntegrityErrorV44, 'final mapper'):
            load_saved_episode_v44(self.root)
    def test_fixture_cannot_be_relabelled_as_live_without_ledger(self):
        alter_json(self.root/'started.json', lambda data: data.pop('finite_fixture'))
        reseal(self.root)
        with self.assertRaisesRegex(SavedEpisodeIntegrityErrorV44, 'actual World'):
            load_saved_episode_v44(self.root)
    def test_gzip_steps_are_lossless_and_double_alias_rejected(self):
        compressed = Path(self.temp.name)/'compressed'
        create_finite_episode_v44(compressed, compressed=True)
        result = _verify_loaded_v44(load_saved_episode_v44(compressed), FiniteMapperV44(), FiniteControllerV44())
        self.assertEqual(result['status'], 'verified')
        (compressed/'steps/000.json').write_bytes(gzip.decompress((compressed/'steps/000.json.gz').read_bytes()))
        reseal(compressed)
        with self.assertRaisesRegex(SavedEpisodeIntegrityErrorV44, 'exactly one'):
            load_saved_episode_v44(compressed)
    def test_real_cpu_controller_initial_grant_recomputes(self):
        actual = Path(self.temp.name)/'one_initial_grant'
        fixture = create_finite_episode_v44(actual, real_controller=True)
        self.assertEqual(fixture['result']['status'], 'controller_stop')
        self.assertEqual(fixture['result']['executed_paid_actions'], 0)
        result = verify_saved_observations_v44(actual)
        self.assertEqual(result['status'], 'verified', result)
        self.assertEqual(result['frames_verified'], 1)
    def test_mesh_vertex_and_triangle_order_does_not_change_comparison(self):
        vertices = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]])
        first = dict(vertices=vertices, triangles=np.array([[0,1,2]]), vertex_colors=np.zeros((3,3)))
        second = dict(vertices=vertices[[2,0,1]], triangles=np.array([[2,1,0]]), vertex_colors=np.zeros((3,3)))
        for left, right in zip(_canonical_mesh(first), _canonical_mesh(second)):
            np.testing.assert_array_equal(left, right)


if __name__ == '__main__': unittest.main()
