"""Finite subprocess fakes exercise ordering/resumption; no World is constructed."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from nso.development_batch_v45 import (ROOT, PROTOCOL, REFERENCE_ROOT, STATE_ROOT, ORDER,
    _run_batch_v45, source_names_v45, bounded_subprocess_v45)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeCommands:
    def __init__(self, root, *, statuses=None, timeout=False, divergence=False):
        self.root = root; self.calls = []; self.statuses = statuses or {}
        self.timeout = timeout; self.divergence = divergence

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        script = Path(command[2]).name
        args = command[3:]
        def argument(name):
            return args[args.index(name)+1]
        if script == 'run_development_v44.py':
            if self.timeout:
                return dict(returncode=-9, status='hard_timeout', stdout=b'partial', stderr=b'', elapsed_s=.1)
            run_id = argument('--run-id')
            episode = Path(argument('--output-root'))/run_id
            status = self.statuses.get(run_id, 'controller_stop')
            slot = json.loads((self.root/PROTOCOL).read_text())['slots'][run_id]
            save(episode/'result.json', dict(status=status))
            save(episode/'artifact_manifest.json', dict(fake_finite_subprocess=True, run_id=run_id))
            protocol = json.loads((self.root/PROTOCOL).read_text())
            ledger = self.root/protocol['ledger_relative_path']
            data = json.loads(ledger.read_text()) if ledger.exists() else dict(
                schema='v43.development_start_ledger.v1', maximum_slots=5, entries=[])
            data['entries'].append(dict(run_id=run_id, status=status, world_created=True,
                result_sha256=sha(episode/'result.json'), metadata=dict(output=str(episode), slot=slot)))
            save(ledger, data)
            result = dict(status=status)
        elif script == 'replay_episode_v44.py':
            episode = Path(args[0])
            status = json.loads((episode/'result.json').read_text())['status']
            result = dict(status='diverged' if self.divergence else 'verified',
                source_manifest_sha256=argument('--expected-manifest-sha256'), prediction_verified=True,
                eligible_study_episode=True, source_unchanged_after_verification=True,
                source_task_success=status=='controller_stop')
        elif script == 'evaluate_episode_v44.py':
            episode = Path(argument('--episode'))
            result = dict(run_id=episode.name, original_episode_status=json.loads((episode/'result.json').read_text())['status'],
                episode_manifest_sha256=argument('--episode-manifest-sha256'),
                reference_manifest_sha256=argument('--reference-manifest-sha256'))
            save(Path(argument('--output')), result)
        else:
            raise AssertionError(script)
        return dict(returncode=0, status='exited', stdout=json.dumps(result).encode()+b'\n', stderr=b'', elapsed_s=.001)


class DevelopmentBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'repo'; self.root.mkdir()
        self.output = Path(self.tmp.name)/'output'
        for name in source_names_v45():
            dst = self.root/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, dst)
        shutil.copytree(ROOT/REFERENCE_ROOT, self.root/REFERENCE_ROOT)
        self.fake = FakeCommands(self.root)
        self.ready = lambda *args: dict(passed=True, non_ram_filesystem=True, free_bytes=11*1024**3, required_free_bytes=10*1024**3)

    def run_batch(self, **kw):
        return _run_batch_v45(self.output, root=self.root, resource_probe=self.ready, runner=self.fake, **kw)

    def test_resource_rejection_precedes_reference_ledger_and_all_side_effects(self):
        shutil.rmtree(self.root/REFERENCE_ROOT)
        result = _run_batch_v45(self.output, root=self.root, resource_probe=lambda *args: dict(passed=False), runner=self.fake)
        self.assertEqual(result['status'], 'blocked_before_world_creation')
        self.assertFalse(self.output.exists()); self.assertFalse((self.root/STATE_ROOT).exists())
        self.assertEqual(self.fake.calls, [])

    def test_external_episode_disk_does_not_require_second_10GiB_on_metadata_disk(self):
        def split_disks(path, peak):
            return dict(passed=Path(path)==self.output, non_ram_filesystem=True,
                        free_bytes=11*1024**3 if Path(path)==self.output else 1024**3)
        result = _run_batch_v45(self.output, root=self.root, preflight_only=True,
                               resource_probe=split_disks, runner=self.fake)
        self.assertEqual(result['status'], 'ready_without_world_creation')
        self.assertEqual(result['launch_journal_resource']['required_free_bytes'], 64*1024**2)
        self.assertFalse(self.output.exists())

    def test_metadata_journal_must_be_persistent_and_have_64MiB(self):
        def split_disks(path, peak):
            return dict(passed=True, non_ram_filesystem=Path(path)==self.output,
                        free_bytes=11*1024**3)
        result = _run_batch_v45(self.output, root=self.root, resource_probe=split_disks, runner=self.fake)
        self.assertEqual(result['status'], 'blocked_before_launch_journal_creation')
        self.assertFalse(self.output.exists()); self.assertEqual(self.fake.calls, [])

    def test_ready_preflight_has_no_outputs_no_world_commands(self):
        result = self.run_batch(preflight_only=True)
        self.assertEqual(result['status'], 'ready_without_world_creation')
        self.assertFalse(self.output.exists()); self.assertEqual(self.fake.calls, [])

    def test_sealed_reference_tamper_fails_before_outputs(self):
        (self.root/REFERENCE_ROOT/'DEV_A_00'/'surface.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'reference content changed'):
            self.run_batch()
        self.assertFalse(self.output.exists()); self.assertEqual(self.fake.calls, [])

    def test_complete_finite_subprocess_batch_in_declared_order(self):
        result = self.run_batch()
        self.assertEqual(result['status'], 'batch_completed')
        self.assertEqual([r['run_id'] for r in result['runs']], list(ORDER))
        self.assertEqual(len(self.fake.calls), 15)
        for index, run_id in enumerate(ORDER):
            self.assertIn(run_id, self.fake.calls[index*3])
            row = result['runs'][index]
            self.assertTrue(Path(row['verification_result_path']).is_file())
            self.assertTrue(Path(row['evaluation_result_path']).is_file())
        self.assertEqual(result['unattempted_runs'], [])

    def test_resume_completed_batch_reuses_sealed_commands_without_process(self):
        self.run_batch()
        result = self.run_batch()
        self.assertEqual(result['status'], 'batch_completed')
        self.assertEqual(len(self.fake.calls), 15)
        self.assertIn('attempt_001', result['summary_path'])

    def test_non_success_G_is_retained_and_S_still_runs(self):
        self.fake.statuses['R3_A_G'] = 'controller_blocked'
        result = self.run_batch()
        self.assertEqual(result['status'], 'batch_completed')
        self.assertEqual(result['runs'][1]['episode_status'], 'controller_blocked')
        self.assertEqual(result['runs'][1]['orchestration_status'], 'non_success_endpoint_recorded')
        self.assertEqual(result['runs'][2]['run_id'], 'R3_A_S')
        self.assertEqual(len(self.fake.calls), 15)

    def test_R2_task_failure_stops_before_any_autonomous_run(self):
        self.fake.statuses[ORDER[0]] = 'stopped_without_confirmed_return'
        result = self.run_batch()
        self.assertEqual(result['status'], 'batch_stopped_for_audit')
        self.assertEqual(len(self.fake.calls), 2)
        self.assertEqual(result['unattempted_runs'], list(ORDER[1:]))

    def test_R2_recomputation_divergence_stops_batch(self):
        self.fake.divergence = True
        result = self.run_batch()
        self.assertEqual(result['status'], 'batch_stopped_for_audit')
        self.assertEqual(len(self.fake.calls), 2)
        self.assertIn('verification did not pass', result['error']['message'])

    def test_timeout_retained_claim_blocks_restart_even_another_output_root(self):
        self.fake.timeout = True
        first = self.run_batch()
        self.assertEqual(first['runs'][0]['orchestration_status'], 'hard_timeout')
        second = _run_batch_v45(self.output.parent/'different', root=self.root,
                               resource_probe=self.ready, runner=self.fake)
        self.assertEqual(second['status'], 'batch_stopped_for_audit')
        self.assertEqual(len(self.fake.calls), 1)
        self.assertIn('retained launch', second['error']['message'])

    def test_reserved_slot_cannot_restart(self):
        protocol = json.loads((self.root/PROTOCOL).read_text())
        save(self.root/protocol['ledger_relative_path'], dict(schema='v43.development_start_ledger.v1',
            maximum_slots=5, entries=[dict(run_id=ORDER[0], status='reserved_before_factory', metadata={})]))
        result = self.run_batch()
        self.assertEqual(result['status'], 'batch_stopped_for_audit')
        self.assertEqual(self.fake.calls, [])

    def test_orphan_episode_does_not_start(self):
        (self.output/'episodes'/ORDER[0]).mkdir(parents=True)
        result = self.run_batch()
        self.assertEqual(result['status'], 'batch_stopped_for_audit'); self.assertEqual(self.fake.calls, [])

    def test_postprocessing_output_tamper_is_rejected_without_rerun(self):
        result = self.run_batch()
        path = Path(result['runs'][0]['verification_result_path']).parent/'stdout.txt'
        path.write_bytes(b'{"status":"verified"}\n')
        resumed = self.run_batch()
        self.assertEqual(resumed['status'], 'batch_stopped_for_audit')
        self.assertEqual(len(self.fake.calls), 15)

    def test_resource_depletion_between_slots_retains_summary(self):
        count = [0]
        def gate(*args):
            count[0] += 1
            return dict(passed=count[0] < 4, non_ram_filesystem=True, free_bytes=11*1024**3)
        result = _run_batch_v45(self.output, root=self.root, resource_probe=gate, runner=self.fake)
        self.assertEqual(result['status'], 'blocked_before_next_slot')
        self.assertEqual(len(self.fake.calls), 3)
        self.assertTrue(Path(result['summary_path']).is_file())

    def test_technical_episode_failure_is_not_restarted_or_relabelled(self):
        self.fake.statuses['R3_A_G'] = 'episode_error'
        result = self.run_batch()
        self.assertEqual(result['status'], 'batch_stopped_for_audit')
        self.assertEqual(result['runs'][-1]['episode_status'], 'episode_error')
        count = len(self.fake.calls)
        resumed = self.run_batch()
        self.assertEqual(resumed['status'], 'batch_stopped_for_audit')
        self.assertEqual(len(self.fake.calls), count)

    def test_interrupted_verification_is_not_retried(self):
        result = self.run_batch()
        path = Path(result['runs'][0]['verification_result_path']).parent/'command.json'
        path.unlink()
        resumed = self.run_batch()
        self.assertEqual(resumed['status'], 'batch_stopped_for_audit')
        self.assertEqual(len(self.fake.calls), 15)

    def test_source_change_prevents_next_world_start(self):
        fake = self.fake
        def mutate(command, **kw):
            result = fake(command, **kw)
            if len(fake.calls) == 3:
                path = self.root/'nso/development_batch_v45.py'
                path.write_text(path.read_text()+'\n# changed during finite fake\n')
            return result
        result = _run_batch_v45(self.output, root=self.root, resource_probe=self.ready, runner=mutate)
        self.assertEqual(result['status'], 'batch_stopped_for_audit')
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(result['unattempted_runs'], list(ORDER[1:]))

    def test_launch_receipt_is_bound_even_when_timeout_precedes_reservation(self):
        self.fake.timeout = True
        result = self.run_batch()
        row = result['runs'][0]
        self.assertEqual(sha(Path(row['launch_receipt_path'])), row['launch_receipt_sha256'])
        self.assertFalse(row['world_created'])
        self.assertIsNone(row['episode_manifest_sha256'])

    def test_corrupt_ledger_still_produces_failure_summary(self):
        fake = self.fake
        def corrupt(command, **kw):
            result = fake(command, **kw)
            protocol = json.loads((self.root/PROTOCOL).read_text())
            (self.root/protocol['ledger_relative_path']).write_text('{invalid')
            return result
        result = _run_batch_v45(self.output, root=self.root, resource_probe=self.ready, runner=corrupt)
        self.assertEqual(result['status'], 'batch_stopped_for_audit')
        self.assertIn('final_ledger_error', result)
        self.assertIsNone(result['unattempted_runs'])
        self.assertTrue(Path(result['summary_path']).is_file())

    def test_real_bounded_child_capture_is_not_a_simulator(self):
        result = bounded_subprocess_v45([sys.executable, '-c', 'print("finite stdout")'], cwd=self.root, timeout_s=5)
        self.assertEqual(result['returncode'], 0); self.assertEqual(result['stdout'], b'finite stdout\n')

    def test_real_child_stream_overflow_is_killed_and_bounded(self):
        result = bounded_subprocess_v45([sys.executable, '-c', 'import os; os.write(1,b"x"*10000)'],
                                       cwd=self.root, timeout_s=5, maximum_stream_bytes=1024)
        self.assertEqual(result['status'], 'output_limit'); self.assertEqual(len(result['stdout']), 1024)

    def test_closed_pipes_do_not_replace_process_deadline(self):
        result = bounded_subprocess_v45([sys.executable, '-c',
            'import os,time; os.write(1,b"retained"); os.close(1); os.close(2); time.sleep(.2)'],
            cwd=self.root, timeout_s=2)
        self.assertEqual(result['status'], 'exited'); self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['stdout'], b'retained')

    def test_closed_pipes_timeout_still_retains_prefix(self):
        result = bounded_subprocess_v45([sys.executable, '-c',
            'import os,time; os.write(1,b"retained"); os.close(1); os.close(2); time.sleep(5)'],
            cwd=self.root, timeout_s=.1)
        self.assertEqual(result['status'], 'hard_timeout'); self.assertEqual(result['stdout'], b'retained')

    def test_failed_process_spawn_is_a_record_not_an_unhandled_exception(self):
        result = bounded_subprocess_v45(['/nonexistent/v45/fake-command'], cwd=self.root, timeout_s=1)
        self.assertEqual(result['status'], 'spawn_failed'); self.assertEqual(result['stdout'], b'')

    def test_real_child_hard_timeout_is_killed(self):
        result = bounded_subprocess_v45([sys.executable, '-c', 'import time; time.sleep(5)'],
                                       cwd=self.root, timeout_s=.05)
        self.assertEqual(result['status'], 'hard_timeout'); self.assertNotEqual(result['returncode'], 0)


if __name__ == '__main__':
    unittest.main()
