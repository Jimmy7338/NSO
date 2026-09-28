"""New-phase isolation and failure retention; finite mocks never construct World."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from nso import mechanism_development as m


class MechanismDevelopmentTests(unittest.TestCase):
    def test_blocked_gate_does_not_load_assets_reserve_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(m, 'storage_report_v41', return_value={'passed': False}) as gate, \
             patch.object(m, '_bundle') as bundle, \
             patch.object(m, 'create_development_sensor') as factory, \
             patch.object(m, 'DevelopmentStartLedgerV43') as ledger:
            output = Path(directory)/'episodes'
            result = m.run_pilot('pilot_A_G', output_root=output)
            self.assertEqual(result['status'], 'blocked_before_world_creation')
            self.assertFalse(output.exists())
            bundle.assert_not_called(); factory.assert_not_called(); ledger.assert_not_called()
            self.assertEqual(gate.call_count, 1)

    def test_failed_factory_reservation_is_new_phase_and_cannot_restart_elsewhere(self):
        config = m._json(m.ROOT/m.CONFIG)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            path = root/m.CONFIG; path.parent.mkdir(parents=True); path.write_text(json.dumps(config))
            old = root/'audit_results/v43_development_batch_20260921/start_ledger.json'
            old.parent.mkdir(parents=True); old.write_text('historical sentinel\n')
            bundle = dict(public_spec={'task': {'max_actions': 160}}, workspace={}, graph_spec={},
                          graph=SimpleNamespace(input_sha256='a'*64))
            stack.enter_context(patch.object(m, 'ROOT', root))
            stack.enter_context(patch.object(m, 'source_names', return_value=[m.CONFIG]))
            stack.enter_context(patch.object(m, 'storage_report_v41', return_value={'passed': True}))
            stack.enter_context(patch.object(m, '_bundle', return_value=bundle))
            stack.enter_context(patch.object(m, '_controller_and_mapper', return_value=(object(), object())))
            stack.enter_context(patch('nso.offline_evaluation_v44.load_reference_v44'))
            factory = stack.enter_context(patch.object(m, 'create_development_sensor', side_effect=RuntimeError('finite injected factory failure')))
            result = m.run_pilot('pilot_A_G', output_root=root/'episodes')
            self.assertEqual(result['status'], 'pilot_attempt_failed')
            self.assertFalse(result['world_created'])
            rows = m._json(root/config['ledger_relative_path'])['entries']
            self.assertEqual(len(rows), 1); self.assertEqual(rows[0]['status'], 'pilot_attempt_failed')
            self.assertEqual(rows[0]['metadata']['protocol_path'], m.CONFIG)
            with self.assertRaisesRegex(ValueError, 'already reserved'):
                m.run_pilot('pilot_A_G', output_root=root/'different_output')
            self.assertEqual(factory.call_count, 1)
            self.assertFalse((root/'different_output').exists())
            self.assertEqual(old.read_text(), 'historical sentinel\n')

    def test_review_rejects_mutated_artifact_before_reintegration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'packet.bin').write_bytes(b'original')
            manifest = dict(schema='mechanism.pilot.artifacts.v1', protocol_path=m.CONFIG,
                files={'packet.bin': dict(bytes=8, sha256=m.file_sha256(root/'packet.bin'))})
            (root/'artifact_manifest.json').write_text(json.dumps(manifest))
            sha = m.file_sha256(root/'artifact_manifest.json')
            (root/'packet.bin').write_bytes(b'modified')
            with patch.object(m, '_replay') as replay, self.assertRaisesRegex(ValueError, 'artifact changed'):
                m.review_pilot(root, expected_manifest_sha256=sha, output=root.parent/'unused_review.json')
            replay.assert_not_called()


if __name__ == '__main__':
    unittest.main()
