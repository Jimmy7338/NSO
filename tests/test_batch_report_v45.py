"""Batch reporting failures and pairing, using explicitly synthetic receipts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nso.batch_report_v45 import (PROTOCOL, LAUNCH_ROOT, REFERENCE_SHA,
    _paired_rows, _ratio, _verified_endpoint, report_batch_v45)

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class BatchReportV45Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = json.loads((ROOT/PROTOCOL).read_text())
        write(self.root/PROTOCOL, self.protocol)
        self.summary = self.root/'summary.json'
        write(self.summary, dict(schema='v45.development_batch.v1', status='blocked_before_world_creation', runs=[]))
        self.ledger_path = self.root/self.protocol['ledger_relative_path']

    def ledger(self, entries):
        write(self.ledger_path, dict(schema='v43.development_start_ledger.v1', maximum_slots=5, entries=entries))

    def terminal_failure(self, run_id='R3_A_G'):
        episode = self.root/'episodes'/run_id
        path = episode/'attempt_failure.json'
        write(path, dict(status='development_attempt_failed', message='finite injected fixture error'))
        return dict(run_id=run_id, status='development_attempt_failed', world_created=False,
            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            metadata=dict(output=str(episode), slot=self.protocol['slots'][run_id]))

    def test_zero_start_retains_all_slots_without_zero_quality_imputation(self):
        result = report_batch_v45(self.summary, source_root=self.root)
        self.assertEqual(result['planned_slots'], 5)
        self.assertEqual(result['reserved_slots'], 0)
        self.assertEqual(result['counts']['not_started'], 5)
        self.assertTrue(all(row['endpoint'] is None for row in result['rows']))
        self.assertTrue(all(row['delta_S_minus_G'] is None for row in result['paired_deltas']))
        self.assertFalse(result['missing_scores_imputed'])

    def test_unfinished_reservation_is_neither_unstarted_nor_retryable(self):
        self.ledger([dict(run_id='R3_A_G', status='reserved_before_factory', world_created=False)])
        result = report_batch_v45(self.summary, source_root=self.root)
        self.assertEqual(result['reserved_slots'], 1)
        self.assertEqual(result['counts']['unfinished_reservation_no_retry'], 1)
        self.assertEqual(result['counts']['not_started'], 4)
        self.assertIsNone(next(row for row in result['rows'] if row['run_id'] == 'R3_A_G')['world_created'])

    def test_failed_start_is_kept_without_fabricated_score(self):
        self.ledger([self.terminal_failure()])
        result = report_batch_v45(self.summary, source_root=self.root)
        self.assertEqual(result['counts']['terminal_without_verified_endpoint'], 1)
        row = next(row for row in result['rows'] if row['run_id'] == 'R3_A_G')
        self.assertEqual(row['episode_status'], 'development_attempt_failed')
        self.assertFalse(row['world_created'])
        self.assertIsNone(row['endpoint'])
        self.assertFalse(result['failures_excluded_from_rows'])

    def test_interrupted_launch_before_factory_reservation_is_retained(self):
        write(self.root/LAUNCH_ROOT/'R3_A_G.json', dict(run_id='R3_A_G', automatic_retry=False))
        result = report_batch_v45(self.summary, source_root=self.root)
        self.assertEqual(result['reserved_slots'], 0)
        self.assertEqual(result['recorded_launches'], 1)
        self.assertEqual(result['counts']['launch_without_reservation_no_retry'], 1)
        self.assertEqual(result['counts']['not_started'], 4)
        self.assertIsNone(next(row for row in result['rows'] if row['run_id'] == 'R3_A_G')['world_created'])

    def test_wrong_batch_source_closure_is_not_accepted(self):
        write(self.summary, dict(schema='v45.development_batch.v1', runs=[], source_sha256={}))
        result = report_batch_v45(self.summary, source_root=self.root)
        self.assertFalse(result['batch_sources_verified'])
        self.assertEqual(result['status'], 'report_with_integrity_errors')

    def test_tampered_failure_becomes_integrity_error_not_missing_case(self):
        entry = self.terminal_failure(); self.ledger([entry])
        write(Path(entry['metadata']['output'])/'attempt_failure.json', dict(status='controller_stop'))
        result = report_batch_v45(self.summary, source_root=self.root)
        self.assertEqual(len(result['rows']), 5)
        self.assertEqual(result['counts']['integrity_error'], 1)
        self.assertEqual(result['status'], 'report_with_integrity_errors')

    def test_orphan_derived_outcome_is_not_an_experiment(self):
        write(self.summary, dict(schema='v45.development_batch.v1', runs=[dict(
            run_id='R3_A_G', evaluation_result_path='/not/a/real/evaluation.json')]))
        result = report_batch_v45(self.summary, source_root=self.root)
        self.assertEqual(result['reserved_slots'], 0)
        self.assertEqual(result['counts']['integrity_error'], 1)

    def test_duplicate_slot_or_undeclared_reservation_rejected(self):
        entry = self.terminal_failure()
        for entries in ([entry, entry], [dict(entry, run_id='unregistered')]):
            self.ledger(entries)
            with self.assertRaises(ValueError):
                report_batch_v45(self.summary, source_root=self.root)

    def test_json_duplicates_and_nonfinite_ratio_rejected(self):
        self.summary.write_text('{"schema":"v45.development_batch.v1","runs":[],"runs":[]}')
        with self.assertRaises(ValueError):
            report_batch_v45(self.summary, source_root=self.root)
        for value in (float('nan'), float('inf'), True, -.01, 1.01):
            with self.assertRaises(ValueError): _ratio(value, 'Q')

    def pair_rows(self):
        rows = []
        for asset in ('DEV_A_00', 'DEV_C_00'):
            for mode in ('G', 'S'):
                endpoint = dict(metrics=dict(C_nav=.6, Q=.5, J_nav=.3),
                    task_success=mode == 'S', reference_manifest_sha256='same-'+asset,
                    public_contract_sha256={'graph': asset}, source_sha256={'policy': 'same-source'})
                if mode == 'S': endpoint['metrics'] = dict(C_nav=.5, Q=.8, J_nav=.4)
                rows.append(dict(asset_id=asset, mode=mode, run_id=asset+'-'+mode, noise_seed=0,
                    state='verified_endpoint', endpoint=endpoint))
        return rows

    def test_pair_reports_tradeoff_and_keeps_nonreturn_identity(self):
        pairs = _paired_rows(self.pair_rows())
        for pair in pairs:
            self.assertEqual(pair['comparison_status'], 'descriptive_development_pair')
            self.assertAlmostEqual(pair['delta_S_minus_G']['C_nav'], -.1)
            self.assertAlmostEqual(pair['delta_S_minus_G']['Q'], .3)
            self.assertAlmostEqual(pair['delta_S_minus_G']['J_nav'], .1)
            self.assertFalse(pair['both_task_successful'])

    def test_pair_needs_common_reference_source_navigation_and_noise(self):
        for field, changed in (('reference_manifest_sha256', 'wrong'),
                               ('source_sha256', {'policy': 'different'}),
                               ('public_contract_sha256', {'budget': 'different'})):
            rows = self.pair_rows(); rows[1]['endpoint'][field] = changed
            self.assertIsNone(_paired_rows(rows)[0]['delta_S_minus_G'])
        rows = self.pair_rows(); rows[1]['noise_seed'] = 1
        self.assertIsNone(_paired_rows(rows)[0]['delta_S_minus_G'])

    def test_missing_pair_side_does_not_become_zero_or_win(self):
        rows = self.pair_rows(); rows[0]['state'] = 'unfinished_reservation_no_retry'; rows[0]['endpoint'] = None
        pair = _paired_rows(rows)[0]
        self.assertIsNone(pair['delta_S_minus_G'])
        self.assertEqual(pair['comparison_status'], 'not_available_without_two_verified_endpoints')

    def synthetic_endpoint(self):
        """Metadata-only parser fixture: loader is replaced, never actual evidence."""
        slot = self.protocol['slots']['R3_A_G']
        episode_root = self.root/'episode'; episode_root.mkdir()
        write(episode_root/'artifact_manifest.json', {'synthetic_parser_fixture': True})
        for name in ('public_graph.json', 'public_spec.json', 'public_workspace.json'):
            write(episode_root/name, {'fixture': name})
        prediction = episode_root/'prediction'; prediction.mkdir()
        for name in ('mesh.npz', 'occupancy.npz', 'mapper.json'):
            (prediction/name).write_bytes(b'finite parser double, not numeric experiment data')
        evaluator_source = self.root/'nso/offline_evaluation_v44.py'
        evaluator_source.parent.mkdir(); evaluator_source.write_text('# finite source-binding fixture\n')
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        digest = sha(episode_root/'artifact_manifest.json')
        episode = SimpleNamespace(eligible_study_episode=True,
            started=dict(run_id='R3_A_G', slot=slot), status='controller_stop', task_success=True,
            frames=(None,), result=dict(executed_paid_actions=0, collisions=0),
            manifest={'source_sha256': {'fixture-policy': 'synthetic'}})
        entry = dict(run_id='R3_A_G', status='controller_stop', metadata=dict(output=str(episode_root)))
        verification = dict(schema='v44.saved_observation_verification.v1', status='verified',
            prediction_verified=True, eligible_study_episode=True, externally_pinned_manifest=True,
            source_unchanged_after_verification=True, source_manifest_sha256=digest,
            source_episode_status='controller_stop', source_task_success=True,
            frames_verified=1, physical_actions=0, new_worlds=0)
        evaluation = dict(schema='v44.saved_prediction_offline_evaluation.v1', run_id='R3_A_G',
            asset_id='DEV_A_00', mode='G', original_episode_status='controller_stop',
            episode_manifest_sha256=digest, reference_manifest_sha256=REFERENCE_SHA['DEV_A_00'],
            externally_pinned_episode_manifest=True, evaluation_source_sha256=sha(evaluator_source),
            all_task_instances_in_macro_denominator=True, prediction_roi_cropped=False,
            task_success=True, eligible_successful_development_endpoint=True,
            input_prediction_sha256={name: sha(prediction/name) for name in ('mesh.npz', 'occupancy.npz', 'mapper.json')},
            metrics=dict(C_nav=.5, Q=.6, J_nav=.3), coverage=dict(C_nav=.5))
        run = dict(episode_manifest_sha256=digest)
        for kind, value, script in (('verification', verification, 'replay_episode_v44.py'),
                                    ('evaluation', evaluation, 'evaluate_episode_v44.py')):
            directory = self.root/kind; directory.mkdir()
            path = directory/'result.json'; write(path, value)
            (directory/'stdout.txt').write_text(json.dumps(value)+'\n')
            (directory/'stderr.txt').write_bytes(b'')
            write(directory/'command.json', dict(status='exited', returncode=0,
                command=['python', '-B', str(self.root/'scripts'/script)],
                files={name: dict(bytes=(directory/name).stat().st_size, sha256=sha(directory/name))
                       for name in ('stdout.txt', 'stderr.txt')}))
            run[kind+'_result_path'] = str(path); run[kind+'_result_sha256'] = sha(path)
        return entry, run, slot, episode

    def test_endpoint_receipts_are_bound_to_batch_hashes(self):
        entry, run, slot, episode = self.synthetic_endpoint()
        with patch('nso.saved_replay_v44.load_saved_episode_v44', return_value=episode):
            self.assertEqual(_verified_endpoint(entry, run, slot, self.root)['metrics']['J_nav'], .3)
            for key in ('episode_manifest_sha256', 'verification_result_sha256', 'evaluation_result_sha256'):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'batch-pinned SHA256'):
                    _verified_endpoint(entry, dict(run, **{key: '0'*64}), slot, self.root)

    def test_score_change_cannot_pass_with_unchanged_batch_identity(self):
        entry, run, slot, episode = self.synthetic_endpoint()
        path = Path(run['evaluation_result_path']); evaluation = json.loads(path.read_text())
        evaluation['metrics'].update(Q=.8, J_nav=.4); write(path, evaluation)
        with patch('nso.saved_replay_v44.load_saved_episode_v44', return_value=episode):
            with self.assertRaisesRegex(ValueError, 'batch-pinned SHA256'):
                _verified_endpoint(entry, run, slot, self.root)
            # Even repinning only the result cannot change the saved child stdout.
            run['evaluation_result_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'independent child stdout'):
                _verified_endpoint(entry, run, slot, self.root)

    def test_mismatched_episode_identity_rejected_after_stream_checks(self):
        entry, run, slot, episode = self.synthetic_endpoint()
        episode.started['run_id'] = 'R3_A_S'
        with patch('nso.saved_replay_v44.load_saved_episode_v44', return_value=episode):
            with self.assertRaisesRegex(ValueError, 'reserved development episode'):
                _verified_endpoint(entry, run, slot, self.root)


if __name__ == '__main__':
    unittest.main()
