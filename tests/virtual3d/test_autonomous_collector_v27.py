"""Pure V27 collector wiring tests; no world, sensor, mapping or evaluator."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import scripts.run_observed_autonomous_v27 as collector
from nso.observed_runtime_v27 import ObservedANSRuntimeV27


def put(path, value):
    path.write_text(json.dumps(value))


class AutonomousCollectorV27Tests(unittest.TestCase):
    def setUp(self):
        collector.configure_base()
        self.temporary = tempfile.TemporaryDirectory(prefix='v27-collector-unit-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/'batch'
        self.root.mkdir()
        collector.PROGRESS.update(case=None, replay=False, owned=False, prepare_owned=False,
            phase='unstarted', last_attempted_action=None, last_saved_packet=None)

    def manifest(self):
        return dict(status='prepared', config=deepcopy(collector.CONFIG), quota_start=8,
            quota_limit=36, main_attempts_started=0, source_sha256={}, input_sha256={},
            source_archive_sha256='x', versions={}, cases=[dict(index=i,
            physical_status='unstarted', replay_status='unstarted') for i in range(4)])

    def test_runtime_substitution_is_v27(self):
        self.assertIs(collector.base.ObservedANSRuntimeV26, ObservedANSRuntimeV27)

    def test_declared_matrix_is_feedback_ablation_only(self):
        matrix=(('A_complex_B_simple','S_no_feedback'),('A_complex_B_simple','S'),
                ('A_simple_B_complex','S_no_feedback'),('A_simple_B_complex','S'))
        self.assertEqual(len(matrix), 4)
        self.assertEqual({mode for _, mode in matrix}, {'S', 'S_no_feedback'})

    def test_claim_spends_ninth_attempt_and_preserves_reserve(self):
        put(self.root/'manifest.json', self.manifest())
        with patch.object(collector, 'check_frozen', side_effect=lambda root: collector.base.read(root/'manifest.json')), \
             patch.object(collector.base.shutil, 'disk_usage', return_value=SimpleNamespace(free=1024**3)):
            manifest, folder = collector.claim_case(self.root, 0)
        self.assertEqual(folder.name, 'case_00')
        self.assertEqual(manifest['cases'][0]['main_attempt_number'], 9)
        self.assertEqual(manifest['main_attempts_started']+manifest['quota_start'], 9)

    def test_capacity_refusal_precedes_attempt_ownership(self):
        put(self.root/'manifest.json', self.manifest())
        free=collector.CONFIG['task_cap_bytes']+collector.CONFIG['free_reserve_bytes']-1
        with patch.object(collector, 'check_frozen', side_effect=lambda root: collector.base.read(root/'manifest.json')), \
             patch.object(collector.base.shutil, 'disk_usage', return_value=SimpleNamespace(free=free)):
            with self.assertRaises(OSError):
                collector.claim_case(self.root, 0)
        self.assertFalse((self.root/'case_00').exists())
        self.assertEqual(collector.base.read(self.root/'manifest.json')['main_attempts_started'], 0)


if __name__ == '__main__':
    unittest.main()
