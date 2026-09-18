"""Pure collector boundary tests; no world, runtime, mapping, or measurement run.

Only tiny temporary files and mocked disk/verification prerequisites are used.
The independently reviewed replay control flow is not a new physical episode.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import scripts.run_observed_autonomous_v26 as collector


def blank_manifest():
    return dict(status='prepared', quota_start=4, quota_limit=36,
        main_attempts_started=0, cases=[dict(index=i, physical_status='unstarted',
            replay_status='unstarted') for i in range(4)])


def put(path, value):
    path.write_text(json.dumps(value))


class AutonomousCollectorV26Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='v26-collector-unit-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base/'batch'; self.root.mkdir()
        self.initial_progress = deepcopy(collector.PROGRESS)
        self.addCleanup(self.restore_progress)
        collector.PROGRESS.update(case=None, replay=False, owned=False, prepare_owned=False,
            phase='unstarted', last_attempted_action=None, last_saved_packet=None)
        # Any accidental acquisition/construction makes the pure test fail.
        for name in ('FacilityChoiceWorldV25', 'ObservedANSRuntimeV26', 'FacilityMeasurementV26'):
            stub = patch.object(collector, name, side_effect=AssertionError('unit test must not construct '+name))
            stub.start(); self.addCleanup(stub.stop)
        disk = patch.object(collector.shutil, 'disk_usage', return_value=SimpleNamespace(free=1024**3))
        disk.start(); self.addCleanup(disk.stop)

    def restore_progress(self):
        collector.PROGRESS.clear(); collector.PROGRESS.update(self.initial_progress)

    def claims(self):
        put(self.root/'manifest.json', blank_manifest())
        return patch.object(collector, 'check_frozen', side_effect=lambda root: collector.read(root/'manifest.json'))

    def test_public_configuration_excludes_world_and_task_identity(self):
        fields = {'resolution_m', 'robot_radius_m', 'camera_height_m', 'width_px', 'height_px',
            'fov_deg', 'max_depth_m', 'depth_sigma_m', 'dropout', 'action_duration_s', 'voxel_m',
            'laser_height_m', 'laser_rays', 'width_m', 'height_m', 'truncation_m', 'pose_noise_m',
            'stereo_model', 'stereo_reference_fx_px', 'stereo_baseline_m'}
        accesses = []
        class PrivateWorldConfig:
            __slots__ = ()
            def __getattr__(self, name):
                accesses.append(name)
                if name not in fields:
                    raise AssertionError('private field read: '+name)
                return 'iid_025px' if name == 'stereo_model' else .12 if name == 'truncation_m' else 1
        public = collector.public_config(PrivateWorldConfig())
        self.assertEqual(set(vars(public)), fields)
        self.assertEqual(set(accesses), fields)
        self.assertEqual(public.truncation_m, .12)
        self.assertEqual(public.stereo_model, 'iid_025px')
        for name in ('assignment', 'parent', 'objects', 'reachable', 'service_regions',
                     'prefix_proposal', 'evaluation_bounds', 'owner', 'max_steps'):
            self.assertFalse(hasattr(public, name))

    def test_claim_counts_fifth_attempt_and_cannot_be_reopened(self):
        with self.claims():
            manifest, folder = collector.claim_case(self.root, 0)
            self.assertEqual(folder, self.root/'case_00')
            self.assertEqual({p.name for p in folder.iterdir()}, {'packets', 'audit', 'meshes'})
            self.assertEqual(manifest['main_attempts_started'], 1)
            self.assertEqual(manifest['cases'][0]['main_attempt_number'], 5)
            self.assertEqual(manifest['cases'][0]['physical_status'], 'running')
            self.assertTrue(collector.PROGRESS['owned'])
            with self.assertRaises(ValueError):
                collector.claim_case(self.root, 0)
            self.assertEqual(collector.read(self.root/'manifest.json')['main_attempts_started'], 1)

    def test_capacity_refusal_before_ownership_does_not_spend_attempt(self):
        with self.claims(), patch.object(collector.shutil, 'disk_usage', return_value=SimpleNamespace(
                free=collector.CONFIG['task_cap_bytes']+collector.CONFIG['free_reserve_bytes']-1)):
            with self.assertRaises(OSError):
                collector.claim_case(self.root, 0)
        self.assertFalse((self.root/'case_00').exists())
        self.assertFalse(collector.PROGRESS['owned'])
        self.assertEqual(collector.read(self.root/'manifest.json')['main_attempts_started'], 0)

    def test_fixed_order_requires_prior_replay_and_ownership_matches_ledger(self):
        with self.claims():
            manifest = collector.read(self.root/'manifest.json')
            manifest['main_attempts_started'] = 1
            manifest['cases'][0]['physical_status'] = 'complete'
            (self.root/'case_00').mkdir(); put(self.root/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'earlier independent replays'):
                collector.claim_case(self.root, 1)
            self.assertFalse((self.root/'case_01').exists())
            manifest['cases'][0]['replay_status'] = 'complete'
            manifest['main_attempts_started'] = 0
            put(self.root/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'ownership'):
                collector.claim_case(self.root, 1)

    def test_failure_between_directory_claim_and_manifest_commit_keeps_quota(self):
        with self.claims():
            collector.PROGRESS.update(case=0, replay=False)
            with patch.object(collector, 'write', side_effect=OSError('injected manifest write failure')):
                with self.assertRaises(OSError):
                    collector.claim_case(self.root, 0)
            self.assertTrue((self.root/'case_00').exists())
            self.assertTrue(collector.PROGRESS['owned'])
            self.assertEqual(collector.read(self.root/'manifest.json')['main_attempts_started'], 0)
            collector.failure(self.root, OSError('injected manifest write failure'))
            saved = collector.read(self.root/'manifest.json')
            self.assertEqual(saved['main_attempts_started']+saved['quota_start'], 5)
            self.assertEqual(saved['cases'][0]['physical_status'], 'failed')
            self.assertTrue(saved['status'].startswith('stopped'))
            receipt = collector.read(self.root/'case_00/failure.json')
            self.assertFalse(receipt['quota_refunded'])
            collector.verify_inventory(self.root/'case_00')
            with self.assertRaises(ValueError):
                collector.claim_case(self.root, 0)

    def test_prepare_io_failure_has_receipt_but_no_main_attempt(self):
        interface, old = self.base/'interface', self.base/'old'
        interface.mkdir(); old.mkdir()
        put(interface/'result.json', {'status': 'complete_offline_interface_check'})
        put(interface/'manifest.json', {'source_sha256': {}})
        put(old/'manifest.json', {'status': 'complete', 'main_attempts_started': 4, 'source_sha256': {}})
        output = self.base/'new-preparation'
        with patch.object(collector, 'INTERFACE', interface), patch.object(collector, 'OLD', old), \
             patch.object(collector, 'verify_inventory'), patch.object(collector, 'source_files', return_value=set()), \
             patch.object(collector, 'save_bytes', side_effect=OSError('injected archive write failure')):
            with self.assertRaises(OSError):
                collector.prepare(output)
        self.assertTrue(output.exists())
        self.assertTrue(collector.PROGRESS['prepare_owned'])
        self.assertFalse(collector.PROGRESS['owned'])
        collector.failure(output, OSError('injected archive write failure'))
        receipt = collector.read(output/'failure_prepare.json')
        self.assertEqual(receipt['new_main_attempts'], 0)
        self.assertEqual(receipt['prior_main_attempts'], 4)
        self.assertEqual(list(output.glob('case_*')), [])
        collector.FacilityChoiceWorldV25.assert_not_called()

    def test_task_cap_includes_nested_logs_and_leaves_receipt_room(self):
        folder = self.root/'case_00'; (folder/'audit').mkdir(parents=True)
        (folder/'audit/record').write_bytes(b'abc')
        (self.root/'root-metadata').write_bytes(b'0123456789')
        self.assertEqual(collector.bytes_used(folder), 3)
        self.assertEqual(collector.bytes_used(self.root), 10)
        with patch.object(collector, 'bytes_used', return_value=
                collector.CONFIG['task_cap_bytes']-collector.CONFIG['receipt_reserve_bytes']-10):
            collector.reserve(folder, 10)
            with self.assertRaises(OSError):
                collector.reserve(folder, 11)
            collector.reserve(folder, 11, receipt=True)
        with patch.object(collector, 'bytes_used', return_value=collector.CONFIG['task_cap_bytes']):
            with self.assertRaises(OSError):
                collector.reserve(folder, 1, receipt=True)

    def test_real_free_reserve_applies_to_receipts_and_refusal_keeps_old_file(self):
        path = self.root/'immutable-existing'; path.write_bytes(b'original')
        with patch.object(collector.shutil, 'disk_usage', return_value=SimpleNamespace(
                free=collector.CONFIG['free_reserve_bytes'])):
            with self.assertRaises(OSError):
                collector.save_bytes(self.root, path, b'replacement')
            with self.assertRaises(OSError):
                collector.reserve(self.root, 1, receipt=True)
        self.assertEqual(path.read_bytes(), b'original')
        self.assertFalse(path.with_name(path.name+'.tmp').exists())

    def test_replay_ignores_only_planning_wallclock_not_actions_cues_or_geometry(self):
        expected = dict(next_action='forward', pose=[2, 3, 0], geometry_sha256='geometry-a',
            nested=[dict(planning_seconds=1., confidence=.5, observed=np.array([1, 2]))], elapsed_s=3.)
        actual = deepcopy(expected); actual['nested'][0]['planning_seconds'] = 999.
        collector.require_replay_equal(actual, expected, 'same autonomous decision')
        changes = [dict(next_action='right'), dict(pose=[2, 4, 0]),
            dict(geometry_sha256='geometry-b'), dict(elapsed_s=4.)]
        for change in changes:
            with self.subTest(change=change):
                bad = deepcopy(actual); bad.update(change)
                with self.assertRaisesRegex(ValueError, 'replay mismatch'):
                    collector.require_replay_equal(bad, expected, 'changed observation or issued action')
        bad = deepcopy(actual); bad['nested'][0]['confidence'] = .6
        with self.assertRaises(ValueError):
            collector.require_replay_equal(bad, expected, 'changed cue')
        self.assertEqual(expected['nested'][0]['planning_seconds'], 1.)

    def test_frozen_source_input_archive_and_config_changes_are_rejected(self):
        source, observed_input = self.base/'source.py', self.base/'input.json'
        source.write_text('# frozen unit source\n'); observed_input.write_text('{}')
        archive = self.root/'sources.zip'; archive.write_bytes(b'unit archive bytes')
        manifest = dict(config=deepcopy(collector.CONFIG), quota_start=4,
            source_sha256={'source.py': collector.sha(source)},
            input_sha256={'input.json': collector.sha(observed_input)},
            source_archive_sha256=collector.sha(archive), versions={'unit': 1})
        put(self.root/'manifest.json', manifest)
        with patch.object(collector, 'ROOT', self.base), patch.object(collector, 'versions', return_value={'unit': 1}):
            collector.check_frozen(self.root)
            for path in (source, observed_input, archive):
                with self.subTest(path=path.name):
                    old = path.read_bytes(); path.write_bytes(old+b'changed')
                    with self.assertRaises(ValueError):
                        collector.check_frozen(self.root)
                    path.write_bytes(old)
            manifest['config']['budget'] = 401; put(self.root/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                collector.check_frozen(self.root)


if __name__ == '__main__':
    unittest.main()
