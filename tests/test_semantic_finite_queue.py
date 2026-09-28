"""Finite queue control tests using in-memory invocation only; no World."""
import unittest
from unittest.mock import patch

from scripts import run_semantic_finite_queue as module
from scripts.run_semantic_declared_batch import execute_sequence


class SemanticFiniteQueueTests(unittest.TestCase):
    def setUp(self):
        self.protocol=dict(slots={key:{} for key in ('a','b','c','d','e','f')},
            storage_plan=dict(maximum_active_batch_slots=6))
        self.batches=[dict(run_ids=['a','b'],audit_dir='audit_A'),dict(run_ids=['c'],audit_dir='audit_B')]

    def test_whole_queue_duplicate_unknown_attempted_and_oversized_rejected(self):
        bad=[([dict(run_ids=['a','a'],audit_dir='A')],{},'duplicate'),
            ([dict(run_ids=['a'],audit_dir='A'),dict(run_ids=['a'],audit_dir='B')],{},'duplicate'),
            ([dict(run_ids=['unknown'],audit_dir='A')],{},'known slots'),
            ([dict(run_ids=['a','b','c','d'],audit_dir='A')],{},'1..3'),
            (self.batches,dict(attempted_run_ids=['c']),'already reserved'),
            (self.batches,dict(existing_run_ids=['a']),'already reserved'),
            ([dict(run_ids=['a'],audit_dir='A'),dict(run_ids=['b'],audit_dir='A')],{},'unique')]
        for batches,changes,message in bad:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError,message):module.validate_schedule(self.protocol,batches,**changes)
        self.assertEqual(module.validate_schedule(self.protocol,self.batches),['a','b','c'])

    def test_finite_order_and_normal_unsuccessful_task_statuses_continue(self):
        called=[];events=[];checks=[]
        statuses={'a':'stopped_without_confirmed_return','b':'controller_blocked','c':'budget_exhausted'}
        def invoke(batch):
            called.append(tuple(batch['run_ids']))
            return execute_sequence(batch['run_ids'],lambda key:dict(status=statuses[key]),lambda:None,lambda row:None)
        result=module.execute_queue(self.batches,invoke,lambda:checks.append('boundary'),lambda:None,events.append)
        self.assertEqual(called,[('a','b'),('c',)])
        self.assertEqual(result['status'],module.COMPLETE_STATUS)
        self.assertEqual(checks,['boundary','boundary'])
        self.assertEqual(result['unstarted_run_ids'],[])

    def test_technical_or_resource_failure_stops_without_retry(self):
        called=[]
        def invoke(batch):
            called.append(batch['run_ids'])
            return dict(status='batch_stopped_on_technical_or_resource_failure',unstarted_run_ids=['b'])
        result=module.execute_queue(self.batches,invoke,lambda:None,lambda:None,lambda row:None)
        self.assertEqual(called,[['a','b']]);self.assertEqual(result['unstarted_run_ids'],['b','c'])
        self.assertFalse(result['automatic_retry'])
        self.assertEqual(result['status'],'queue_stopped_on_technical_or_resource_failure')

    def test_peer_failure_is_observed_between_batches_without_killing_current(self):
        called=[];peer=[None]
        def invoke(batch):
            called.append(batch['run_ids']);peer[0]=dict(status='peer_technical_failure')
            return dict(status='declared_batch_complete')
        result=module.execute_queue(self.batches,invoke,lambda:None,lambda:peer[0],lambda row:None)
        self.assertEqual(called,[['a','b']]);self.assertEqual(len(result['finished']),1)
        self.assertEqual(result['status'],'queue_stopped_on_peer_signal')
        self.assertEqual(result['unstarted_run_ids'],['c'])

    def test_peer_success_does_not_stop_and_failure_or_invalid_terminal_does(self):
        class MemoryPath:
            name='result.json'
            def exists(self):return True
            def is_symlink(self):return False
            def is_file(self):return True
            def __str__(self):return 'explicit_peer/result.json'
        path=MemoryPath()
        with patch.object(module,'file_sha256',return_value='a'*64):
            with patch.object(module,'read_json',return_value=dict(schema=module.RESULT_SCHEMA,status=module.COMPLETE_STATUS)):
                self.assertIsNone(module.peer_stop_signal([path]))
            with patch.object(module,'read_json',return_value=dict(schema=module.RESULT_SCHEMA,status='queue_stopped_on_technical_or_resource_failure')):
                self.assertEqual(module.peer_stop_signal([path])['reason'],'peer_error_or_noncomplete_result')
            with patch.object(module,'read_json',side_effect=ValueError('invalid terminal')):
                self.assertEqual(module.peer_stop_signal([path])['reason'],'invalid_peer_terminal')

    def test_preinvocation_failure_and_partial_invocation_are_not_confused(self):
        called=[]
        def failed():raise ValueError('changed binding')
        result=module.execute_queue(self.batches,lambda row:called.append(row),failed,lambda:None,lambda row:None)
        self.assertFalse(called);self.assertEqual(result['unstarted_run_ids'],['a','b','c'])
        def invoke(batch):raise RuntimeError('may have left retained partial work')
        result=module.execute_queue(self.batches,invoke,lambda:None,lambda:None,lambda row:None)
        self.assertEqual(result['unstarted_run_ids'],['c'])
        self.assertEqual(result['unresolved_current_batch_run_ids'],['a','b'])
        self.assertFalse(result['automatic_retry'])


if __name__=='__main__':unittest.main()
