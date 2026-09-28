"""Ground administrative wrapper tests: only tiny child fixtures, never Worlds."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts import run_article_batch_20260928 as old
from scripts import run_article_ground_batch_20260928 as batch
from tests.test_article_batch_20260928 import fixture as original_fixture, factory, write


def fixture(root,count=3):
    path,p,phase=original_fixture(root,count)
    runner=root/batch.RUNNER;runner.write_text('# mock runner; not executed\n')
    p.update(schema=batch.SCHEMA,phase='ablation',source_sha256={batch.RUNNER:old.sha(runner)})
    write(path,p)
    return path,p,phase


def run(root,path,name='batch',**kwargs):
    return batch.run_batch(path,root/name,root=root,preflight=False,
        command_factory=kwargs.pop('command_factory',factory(root)),
        free_bytes=kwargs.pop('free_bytes',lambda:10*old.GIB),poll_seconds=.01,**kwargs)


class GroundBatchTests(unittest.TestCase):
    def test_wrapper_does_not_modify_old_module_and_preserves_no_retry(self):
        self.assertEqual(old.RUNNER,'scripts/run_article_experiment_20260928.py')
        self.assertIsNot(old.load_protocol,batch.load_protocol)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,p,phase=fixture(root)
            write(phase/'start_ledger.json',dict(schema='article.start_ledger.v1',
                protocol_sha256=old.sha(path),phase='ablation',slots=p['slots'],
                entries=[dict(run_id='r0',status='reserved')]))
            first=run(root,path,command_factory=factory(root,{'r1':'fail'}))
            second=run(root,path,'second')
            calls=[json.loads(x)['run_id'] for x in (root/'calls.jsonl').read_text().splitlines()]
            self.assertEqual(sorted(calls),['r1','r2'])
            self.assertEqual(first['status'],'complete_with_failures')
            self.assertTrue(all(x['status']=='skipped' for x in second['runs']))
            self.assertEqual(first['ground_runner'],batch.RUNNER)
            self.assertEqual(first['scheduler_sha256'],old.sha(Path(batch.__file__)))
            self.assertEqual(first['reused_scheduler_sha256'],old.sha(Path(old.__file__)))
        self.assertEqual(old.RUNNER,'scripts/run_article_experiment_20260928.py')

    def test_global_ablation_cap_still_24_and_resources_still_reserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,p,phase=fixture(root)
            rows=[dict(ledger='/other/ledger',run_id=f'old{i}',phase='ablation',protocol_sha256='old')
                  for i in range(24)]
            write(phase.parent/'execution_registry.json',dict(schema='article.execution_registry.v1',
                phase_caps=old.CAPS,entries=rows))
            result=run(root,path)
            self.assertEqual(result['stop_reason'],'cumulative_phase_cap')
            self.assertFalse((root/'calls.jsonl').exists())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,p,phase=fixture(root)
            result=run(root,path,free_bytes=lambda:old.GIB)
            self.assertEqual(result['stop_reason'],'resource_blocked')
            self.assertFalse((root/'calls.jsonl').exists())

    def test_protocol_mutation_stops_new_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,p,phase=fixture(root)
            result=run(root,path,workers=1,command_factory=factory(root,{'r0':'change'}))
            self.assertEqual(result['stop_reason'],'validation_failed')
            self.assertEqual(result['never_launched'],['r1','r2'])

    def test_unbounded_wrong_schema_and_changed_source_rejected(self):
        for problem in ('schema','slots','source'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path,p,phase=fixture(root,13 if problem=='slots' else 1)
                if problem=='schema':p['schema']='article.experiment_protocol.v1';write(path,p)
                if problem=='source':(root/batch.RUNNER).write_text('# changed\n')
                with self.assertRaises(ValueError):run(root,path)
                self.assertFalse((root/'calls.jsonl').exists())


if __name__=='__main__':unittest.main()
