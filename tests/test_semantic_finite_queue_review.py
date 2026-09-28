"""Finite streaming review logic and exact-input fixtures; no real replay/score."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import review_semantic_finite_queue as m
from tests import test_reuse_semantic_scene_endpoint_evaluation as fixtures


class FiniteReviewWaitTests(unittest.TestCase):
    def test_declared_batches_interleave_without_changing_within_batch_order(self):
        items=[dict(batches=[dict(run_ids=['A1a','A1b']),dict(run_ids=['A2']),dict(run_ids=['A3'])]),
            dict(batches=[dict(run_ids=['B1']),dict(run_ids=['B2a','B2b'])])]
        self.assertEqual(m.interleaved_slots(items),['A1a','A1b','B1','A2','B2a','B2b','A3'])
        self.assertEqual(m.interleaved_slots(items[::-1]),['B1','A1a','A1b','B2a','B2b','A2','A3'])

    def test_sealed_next_target_does_not_wait_for_queue_completion(self):
        sleep = Mock(); check = Mock(); result = {'manifest_sha256':'a'*64}
        self.assertEqual(m.wait_for_seal('A',sealed=lambda _:result,check_queues=check,
            deadline=0,poll_s=10,emit=lambda _:None,clock=lambda:100,sleep=sleep),result)
        sleep.assert_not_called(); check.assert_called_once()

    def test_unsealed_target_has_finite_wait_and_no_implicit_other_target(self):
        now = [0.]; seen = []; events = []
        def sleep(dt): now[0] += dt
        def sealed(key): seen.append(key); return None
        with self.assertRaises(TimeoutError):
            m.wait_for_seal('A',sealed=sealed,check_queues=lambda:None,deadline=3.,poll_s=2.,
                emit=events.append,clock=lambda:now[0],sleep=sleep)
        self.assertEqual(now[0],3.); self.assertEqual(seen,['A','A','A'])
        self.assertEqual(len(events),1)

    def test_dead_queue_stops_even_when_target_already_sealed(self):
        sealed = Mock(return_value={}); sleep = Mock()
        with self.assertRaisesRegex(ValueError,'dead PID'):
            m.wait_for_seal('A',sealed=sealed,check_queues=Mock(side_effect=ValueError('dead PID')),
                deadline=10,poll_s=1,emit=lambda _:None,sleep=sleep)
        sealed.assert_not_called(); sleep.assert_not_called()

    def test_explicit_order_zero_quality_continues_and_failure_stops(self):
        called=[]; events=[]
        def invoke(row):
            called.append(row['run_id'])
            if row['run_id']=='C': raise ValueError('technical failure')
            return dict(run_id=row['run_id'],metrics=dict(Q=0,J_nav=0))
        result=m.execute_reviews(['A','B','C','D'],lambda key:dict(run_id=key,manifest_sha256='a'*64),
            invoke,lambda:None,events.append)
        self.assertEqual(called,['A','B','C']);self.assertEqual(result['unstarted_run_ids'],['D'])
        self.assertEqual(result['failed_run_id'],'C');self.assertFalse(result['automatic_retry'])

    def test_queue_failure_after_review_retains_completed_receipt_accounting(self):
        check=Mock(side_effect=[None,ValueError('external queue failed during review')])
        invoke=Mock(return_value=dict(run_id='A',metrics=dict(Q=0)))
        result=m.execute_reviews(['A','B'],lambda key:dict(run_id=key,manifest_sha256='a'*64),
            invoke,check,lambda _:None)
        self.assertEqual([row['run_id'] for row in result['finished']],['A'])
        self.assertEqual(result['unstarted_run_ids'],['B']);self.assertIsNone(result['failed_run_id'])
        self.assertTrue(result['stopped_at_boundary_after_completed_review']);invoke.assert_called_once()

    def test_proc_identity_binds_queue_command_start_ticks_and_boot(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc=Path(temporary);process=proc/'123';process.mkdir()
            boot=proc/'sys/kernel/random/boot_id';boot.parent.mkdir(parents=True);boot.write_text('analytic-boot\n')
            (process/'cwd').symlink_to(m.ROOT,target_is_directory=True)
            fields=['S']+['0']*18+['987654']+['0']*3
            (process/'stat').write_text('123 (python worker) '+' '.join(fields))
            cmd=b'python\0scripts/run_semantic_finite_queue.py\0--queue\0audit_results/example/queue.json\0'
            (process/'cmdline').write_bytes(cmd)
            queue=m.ROOT/'audit_results/example/queue.json'
            identity=m.process_identity(123,queue,proc_root=proc)
            self.assertEqual(identity,dict(pid=123,start_ticks=987654,
                cmdline_sha256=hashlib.sha256(cmd).hexdigest(),boot_id='analytic-boot'))
            fields[19]='987655';(process/'stat').write_text('123 (python worker) '+' '.join(fields))
            self.assertNotEqual(m.process_identity(123,queue,proc_root=proc),identity)
            with self.assertRaisesRegex(ValueError,'explicit World queue'):
                m.process_identity(123,m.ROOT/'different.json',proc_root=proc)
            fields[0]='Z';(process/'stat').write_text('123 (python worker) '+' '.join(fields))
            with self.assertRaisesRegex(ValueError,'not live'):m.process_identity(123,queue,proc_root=proc)


class FiniteReviewInputTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.SemanticSceneEndpointReuseTests()
        fixture.setUpClass();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.f=fixture;self.source=fixture.episodes['source'];self.target=fixture.episodes['target']

    def test_exact_arrays_and_mapper_allow_same_inputs_before_any_replay(self):
        self.assertTrue(m.same_inputs(self.source,self.target,dict(coverage=self.f.coverage)))
        self.f.forbid_score.assert_not_called();self.f.forbid_replay.assert_not_called()

    def test_one_ulp_mesh_or_mapper_history_selects_full_normal_path(self):
        changed=self.f.vertices.copy();changed[1,0]=np.nextafter(1.,2.)
        self.f.save_mesh(self.target['root'],vertices=changed)
        self.assertFalse(m.same_inputs(self.source,self.target,dict(coverage=self.f.coverage)))
        self.f.save_mesh(self.target['root'])
        path=self.target['root']/'prediction/mapper.json';mapper=m.proof.read_json(path)
        mapper['tsdf_integration_count']+=1;path.write_bytes(m.proof.canonical_bytes(mapper))
        self.assertFalse(m.same_inputs(self.source,self.target,dict(coverage=self.f.coverage)))

    def test_nonmethod_slot_or_reference_mismatch_is_normal_nonreuse(self):
        target=deepcopy(self.target);target['slot']['budget']+=1
        self.assertFalse(m.same_inputs(self.source,target,dict(coverage=self.f.coverage)))
        target=deepcopy(self.target);target['reference']['manifest_sha256']='e'*64
        self.assertFalse(m.same_inputs(self.source,target,dict(coverage=self.f.coverage)))


class FiniteReviewTerminalTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        for name in ('queue.json','protocol.json','queue.source.py','batch.source.py'):
            (self.root/name).write_text(name)
        self.pin=m.proof.file_sha256(self.root/'protocol.json')
        self.item=dict(audit=self.root,sha256=m.proof.file_sha256(self.root/'queue.json'),
            run_ids=['A'],batches=[dict(run_ids=['A'])])
        started=dict(queue_sha256=self.item['sha256'],protocol_sha256=self.pin,run_ids=['A'])
        (self.root/'started.json').write_bytes(m.proof.canonical_bytes(started))
        self.result=dict(schema=m.world_queue.RESULT_SCHEMA,status=m.world_queue.COMPLETE_STATUS,
            queue_sha256=self.item['sha256'],protocol_sha256=self.pin,run_ids=['A'],
            unstarted_run_ids=[],unresolved_current_batch_run_ids=[],
            queue_script_sha256=m.proof.file_sha256(self.root/'queue.source.py'),
            batch_script_sha256=m.proof.file_sha256(self.root/'batch.source.py'),
            finished=[dict(run_ids=['A'],batch_invoked=True,result=dict(status='declared_batch_complete',
                protocol_sha256=self.pin,run_ids=['A'],finished=[dict(run_id='A',runner_invoked=True,
                    result=dict(status='controller_blocked'))]))])
        self.save()

    def save(self): (self.root/'result.json').write_bytes(m.proof.canonical_bytes(self.result))

    def test_complete_normal_unsuccessful_task_is_accepted_but_error_marker_stops(self):
        receipt=m.queue_terminal(self.item,self.pin);self.assertEqual(receipt['run_ids'],['A'])
        (self.root/'error.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'retained error marker'):m.queue_terminal(self.item,self.pin)

    def test_wrong_terminal_pin_or_technical_child_failure_cannot_mark_complete(self):
        self.result['protocol_sha256']='f'*64;self.save()
        with self.assertRaisesRegex(ValueError,'terminal binding'):m.queue_terminal(self.item,self.pin)
        self.result['protocol_sha256']=self.pin
        self.result['finished'][0]['result']['finished'][0]['result']['status']='blocked_before_world_creation';self.save()
        with self.assertRaisesRegex(ValueError,'every explicitly declared'):m.queue_terminal(self.item,self.pin)

    def test_partial_slot_finished_line_waits_then_binds_the_reported_manifest(self):
        batch=dict(audit_dir=str(self.root),run_ids=['A'])
        header=dict(schema='semantic.declared_batch.v1',protocol_sha256=self.pin,run_ids=['A'],
            batch_script_sha256=m.proof.file_sha256(m.ROOT/'scripts/run_semantic_declared_batch.py'))
        (self.root/'batch.json').write_bytes(m.proof.canonical_bytes(header))
        row=dict(event='slot_finished',run_id='A',runner_invoked=True,
            result=dict(run_id='A',status='controller_stop',artifact_manifest_sha256='a'*64))
        line=json.dumps(row,sort_keys=True).encode()+b'\n'
        path=self.root/'events.jsonl';path.write_bytes(line[:-1])
        # Even a complete JSON object without its publication newline is not
        # accepted; no manifest file access or independent review is triggered.
        self.assertIsNone(m.completed_slot_event(batch,'A',self.pin))
        path.write_bytes(line)
        receipt=m.completed_slot_event(batch,'A',self.pin)
        self.assertEqual(receipt['manifest_sha256'],'a'*64)
        self.assertEqual(receipt['complete_line_sha256'],hashlib.sha256(line).hexdigest())
        self.assertEqual(receipt['complete_line_bytes'],len(line))
        path.write_bytes(line+line)
        with self.assertRaisesRegex(ValueError,'duplicate slot_finished'):
            m.completed_slot_event(batch,'A',self.pin)

    def test_technical_slot_event_is_rejected_even_with_a_manifest_pin(self):
        row=dict(event='slot_finished',run_id='A',runner_invoked=True,
            result=dict(run_id='A',status='experiment_attempt_failed',artifact_manifest_sha256='a'*64))
        (self.root/'events.jsonl').write_bytes(m.proof.canonical_bytes(row))
        with self.assertRaisesRegex(ValueError,'technical failure'):
            m.completed_slot_event(dict(audit_dir=str(self.root),run_ids=['A']),'A',self.pin)


class FiniteReviewDispatchTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.target=dict(root=self.root/'episodes/A',started=dict(run_id='A'),manifest_sha256='a'*64)
        self.source=dict(run_id='B',manifest_sha256='b'*64,review_path=self.root/'old.json',review_sha256='c'*64)
        self.calls=[];self.events=[];self.sources=[]
        self.addCleanup(patch.stopall)
        self.choice=patch.object(m,'choose_source',side_effect=self.choose).start()
        self.review=patch.object(m,'review_experiment',side_effect=self.run_review).start()
        self.reuse=patch.object(m.proof,'reuse',side_effect=self.run_reuse).start()
        patch.object(m,'checked_source',return_value=({}, {}, {})).start()
        self.checked_review=patch.object(m.proof,'_checked_review',return_value=dict(status='experiment_replay_verified')).start()
        self.selected=self.source

    def choose(self,*_): self.calls.append('choose');return self.selected

    def run_review(self,root,*,expected_manifest_sha256,output,replay_only):
        self.calls.append('replay_only' if replay_only else 'full')
        value=dict(status='experiment_replay_verified' if replay_only else 'experiment_reviewed',
            evaluation=dict(metrics=dict(C_nav=0.,Q=0.,J_nav=0.)))
        output.parent.mkdir(parents=True);output.write_bytes(m.proof.canonical_bytes(value));return value

    def run_reuse(self,**kwargs):
        self.calls.append('reuse');output=kwargs['output'];output.parent.mkdir(parents=True)
        value=dict(evaluation=dict(metrics=dict(C_nav=0.,Q=0.,J_nav=0.)))
        output.write_bytes(m.proof.canonical_bytes(value));return value

    def invoke(self,existing=None):
        return m.review_one(self.target,sources=self.sources,existing=existing or {},protocol={},
            episode_root=self.root/'episodes',phase_root=self.root,emit=self.events.append)

    def test_preproved_equal_input_replays_once_then_reuses(self):
        row=self.invoke();self.assertEqual(self.calls,['choose','replay_only','reuse'])
        self.assertEqual(row['mode'],'replay_and_reuse');self.assertEqual(len(self.events),1)

    def test_different_input_calls_full_only_once(self):
        self.selected=None;row=self.invoke()
        self.assertEqual(self.calls,['choose','full']);self.reuse.assert_not_called()
        self.assertEqual(row['mode'],'full');self.assertEqual(len(self.sources),1)

    def test_post_replay_proof_failure_never_falls_back_to_full(self):
        self.reuse.side_effect=ValueError('source pin changed')
        with self.assertRaisesRegex(ValueError,'source pin changed'):self.invoke()
        self.assertEqual(self.calls,['choose','replay_only']);self.review.assert_called_once()
        self.assertTrue((self.root/'reviews/A.json').exists())

    def previous(self):
        path=self.root/'existing.json';path.write_text('{}')
        return dict(run_id='A',manifest_sha256='a'*64,review_path=path,review_sha256=m.proof.file_sha256(path))

    def test_existing_replay_can_only_fill_reuse_not_repeat_replay(self):
        row=self.invoke({'A':self.previous()})
        self.assertEqual(self.calls,['choose','reuse']);self.review.assert_not_called()
        self.assertEqual(row['mode'],'reuse_existing_replay')

    def test_existing_replay_without_numeric_source_is_not_completed(self):
        self.selected=None
        with self.assertRaisesRegex(ValueError,'no numeric result'):self.invoke({'A':self.previous()})
        self.review.assert_not_called();self.reuse.assert_not_called()

    def test_existing_full_is_validated_without_replaying(self):
        self.checked_review.return_value=dict(status='experiment_reviewed',evaluation=dict(metrics=dict(Q=0)))
        row=self.invoke({'A':self.previous()})
        self.assertEqual(row['mode'],'existing_full_validated');self.assertEqual(self.calls,[])
        self.review.assert_not_called();self.reuse.assert_not_called()

    def test_undeclared_existing_output_is_never_overwritten(self):
        path=self.root/'reviews/A.json';path.parent.mkdir();path.write_text('retained')
        with self.assertRaisesRegex(ValueError,'explicit descriptor'):self.invoke()
        self.assertEqual(path.read_text(),'retained');self.review.assert_not_called()


if __name__ == '__main__':unittest.main()
