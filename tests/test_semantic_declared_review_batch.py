"""Offline batch control fixtures; all replay/evaluation/reuse operations mocked."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import review_semantic_declared_batch as m


class DeclaredReviewControlTests(unittest.TestCase):
    def items(self):
        return [dict(entry=dict(run_id=k, mode='full')) for k in ('A', 'B', 'C')]

    def test_declared_order_does_not_stop_for_low_or_zero_quality(self):
        calls = []
        def invoke(item):
            calls.append(item['entry']['run_id'])
            return dict(run_id=calls[-1], status='experiment_reviewed', metrics=dict(C_nav=0., Q=0., J_nav=0.))
        result = m.execute_sequence(self.items(), invoke, lambda: None, lambda event: None)
        self.assertEqual(calls, ['A', 'B', 'C'])
        self.assertEqual(result['status'], 'declared_review_batch_complete')

    def test_first_failure_stops_without_retry_or_later_invocation(self):
        invoke = Mock(side_effect=[dict(status='experiment_reviewed'), ValueError('synthetic failure')])
        result = m.execute_sequence(self.items(), invoke, lambda: None, lambda event: None)
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(result['unstarted_run_ids'], ['C'])
        self.assertEqual(result['status'], 'declared_review_batch_stopped')
        self.assertFalse(result['automatic_retry'])

    def test_binding_failure_does_not_invoke_current_item(self):
        invoke = Mock()
        result = m.execute_sequence(self.items(), invoke, Mock(side_effect=ValueError('binding changed')), lambda event: None)
        invoke.assert_not_called()
        self.assertEqual(result['unstarted_run_ids'], ['A', 'B', 'C'])


class DeclaredReviewFixtureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.addCleanup(patch.stopall)
        patch.object(m, 'ROOT', self.root).start()
        self.protocol = dict(status='frozen', phase_id='semantic_fixture',
            output_relative_path='audit_results/semantic_fixture/episodes',
            ledger_relative_path='audit_results/semantic_fixture/start_ledger.json',
            slots={key: dict(method=key, asset_id='synthetic', budget=120) for key in ('A', 'B', 'C')})
        self.protocol_path = self.root/'protocol.json'
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.declaration = dict(schema=m.SCHEMA, protocol_path='protocol.json',
            protocol_sha256=m.proof.file_sha256(self.protocol_path),
            entries=[dict(run_id='A', mode='full', manifest_sha256='a'*64)])
        patch.object(m, 'validate_protocol', side_effect=lambda value:value).start()
        self.checked = patch.object(m.proof, '_checked_episode', side_effect=self.episode).start()
        self.checked_review = patch.object(m.proof, '_checked_review', return_value={}).start()
        self.review = patch.object(m, 'review_experiment', side_effect=self.fake_review).start()
        self.reuse = patch.object(m.proof, 'reuse', side_effect=self.fake_reuse).start()

    def episode(self, path, pin):
        m.proof._pin(pin)
        return dict(root=Path(path), started=dict(run_id=Path(path).name), protocol=deepcopy(self.protocol),
                    slot=self.protocol['slots'][Path(path).name])

    def fake_review(self, root, *, expected_manifest_sha256, output, replay_only):
        result = dict(status='experiment_replay_verified' if replay_only else 'experiment_reviewed')
        if not replay_only: result['evaluation'] = dict(metrics=dict(C_nav=.5, Q=.4, J_nav=.2))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream: json.dump(result, stream)
        return result

    def fake_reuse(self, **kwargs):
        result = dict(status='experiment_endpoint_evaluation_reused',
                      evaluation=dict(metrics=dict(C_nav=.5, Q=.4, J_nav=.2)))
        kwargs['output'].parent.mkdir(parents=True, exist_ok=True)
        with kwargs['output'].open('x') as stream: json.dump(result, stream)
        return result

    def source(self):
        path = self.root/'source_review.json'; path.write_text('{}')
        return dict(run_id='B', manifest_sha256='b'*64, review_path='source_review.json',
                    review_sha256=m.proof.file_sha256(path))

    def prepared(self):
        return m.prepare(self.declaration)[2][0]

    def test_full_dispatch_calls_only_full_existing_reviewer(self):
        result = m.invoke_review(self.prepared())
        self.review.assert_called_once()
        self.assertFalse(self.review.call_args.kwargs['replay_only'])
        self.reuse.assert_not_called()
        self.assertEqual(result['evaluation_execution'], 'recomputed')
        self.assertEqual(result['metrics'], dict(C_nav=.5, Q=.4, J_nav=.2))

    def test_replay_then_reuse_uses_new_target_replay_hash_and_pinned_source(self):
        entry = self.declaration['entries'][0]
        entry.update(mode='replay_and_reuse', source=self.source())
        item = self.prepared(); result = m.invoke_review(item)
        self.assertTrue(self.review.call_args.kwargs['replay_only'])
        self.assertEqual(self.reuse.call_args.kwargs['source_review_sha256'], entry['source']['review_sha256'])
        self.assertEqual(self.reuse.call_args.kwargs['target_review_sha256'], m.proof.file_sha256(item['review_path']))
        self.assertEqual(result['evaluation_execution'], 'reused')

    def test_reuse_only_never_replays_and_requires_explicit_existing_target_pin(self):
        target = self.root/'existing_target_review.json'; target.write_text('{}')
        self.declaration['entries'][0].update(mode='reuse_only', source=self.source(),
            target_review=dict(path='existing_target_review.json', sha256=m.proof.file_sha256(target)))
        item = self.prepared(); m.invoke_review(item)
        self.review.assert_not_called(); self.reuse.assert_called_once()
        self.assertEqual(target.read_text(), '{}')
        changed = deepcopy(self.declaration)
        del changed['entries'][0]['target_review']['sha256']
        with self.assertRaises(ValueError): m.prepare(changed)

    def test_failed_replay_retains_artifact_without_trying_reuse(self):
        self.declaration['entries'][0].update(mode='replay_and_reuse', source=self.source())
        self.review.side_effect = None
        self.review.return_value = dict(status='experiment_review_failed', error={'synthetic': True})
        with self.assertRaisesRegex(ValueError, 'offline review failed'):
            m.invoke_review(self.prepared())
        self.reuse.assert_not_called()

    def test_preflight_rejects_existing_review_before_any_dispatch(self):
        self.declaration['entries'].append(dict(run_id='C', mode='full', manifest_sha256='c'*64))
        existing = self.root/'audit_results/semantic_fixture/reviews/C.json'
        existing.parent.mkdir(parents=True); existing.write_text('retained')
        with self.assertRaisesRegex(ValueError, 'existing output'):
            m.prepare(self.declaration)
        self.review.assert_not_called(); self.reuse.assert_not_called()
        self.assertEqual(existing.read_text(), 'retained')

    def test_duplicate_unknown_and_unsealed_runs_or_bad_protocol_pin_rejected(self):
        variants = []
        variant = deepcopy(self.declaration); variant['entries'] *= 2; variants.append(variant)
        variant = deepcopy(self.declaration); variant['entries'][0]['run_id'] = 'undeclared'; variants.append(variant)
        variant = deepcopy(self.declaration); variant['protocol_sha256'] = '0'*64; variants.append(variant)
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError): m.prepare(variant)
        self.checked.side_effect = ValueError('not a sealed complete episode')
        with self.assertRaisesRegex(ValueError, 'sealed complete'): m.prepare(self.declaration)
        self.review.assert_not_called(); self.reuse.assert_not_called()

    def test_changed_existing_target_pin_and_newly_occupied_output_stop_before_replay(self):
        item = self.prepared()
        item['reuse_path'].parent.mkdir(parents=True); item['reuse_path'].write_text('racing retained output')
        with self.assertRaisesRegex(ValueError, 'existing output'): m.invoke_review(item)
        self.review.assert_not_called(); self.reuse.assert_not_called()
        target = self.root/'target_review.json'; target.write_text('{}')
        item['entry'].update(mode='reuse_only', target_review=dict(sha256='0'*64))
        item['review_path'] = target
        item['reuse_path'] = self.root/'new_reuse.json'
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'): m.invoke_review(item)
        self.review.assert_not_called(); self.reuse.assert_not_called()

    def test_batch_archives_exact_declaration_and_wrapper_with_compact_result(self):
        self.protocol.update(asset_root='audit_results/synthetic_assets',
            navigation_root='audit_results/synthetic_navigation',
            reference_index_path='audit_results/synthetic_references/index.json')
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.declaration['protocol_sha256'] = m.proof.file_sha256(self.protocol_path)
        declaration_path = self.root/'explicit.json'
        declaration_path.write_text(json.dumps(self.declaration))
        scripts = self.root/'scripts'; scripts.mkdir()
        wrapper = scripts/'wrapper.py'; wrapper.write_text('# synthetic wrapper snapshot\n')
        helper = scripts/'reuse.py'; helper.write_text('# synthetic reuse snapshot\n')
        audit = self.root/'audit_results/synthetic_review_batch'
        with patch.object(m, '__file__', str(wrapper)), patch.object(m.proof, '__file__', str(helper)), \
                patch.object(m.proof, 'runtime_counts_v41', return_value={'worlds_created': 0, 'rgbd_frames': 0}), \
                patch('builtins.print'):
            result = m.run_batch(declaration_path, audit)
        self.assertEqual(result['status'], 'declared_review_batch_complete')
        self.assertEqual(result['new_worlds'], 0)
        self.assertEqual(result['new_sensor_packets'], 0)
        self.assertEqual((audit/'declaration.json').read_bytes(), declaration_path.read_bytes())
        self.assertEqual((audit/'wrapper.source.py').read_bytes(), wrapper.read_bytes())
        self.assertEqual((audit/'reuse.source.py').read_bytes(), helper.read_bytes())
        self.assertEqual(json.loads((audit/'result.json').read_text())['finished'][0]['metrics'],
                         dict(C_nav=.5, Q=.4, J_nav=.2))
        self.assertEqual(len((audit/'events.jsonl').read_text().splitlines()), 4)
        self.reuse.assert_not_called()


if __name__ == '__main__':
    unittest.main()
