"""Administrative scheduler fixtures: tiny subprocesses, no experimental World."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from scripts import run_article_batch_20260928 as batch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, count=3):
    runner = root/batch.RUNNER; runner.parent.mkdir(parents=True)
    runner.write_text('# placeholder scientific runner; never executed\n')
    archive = root/'archive.zip'; archive.write_bytes(b'fixture-not-scientific-archive')
    protocol = dict(schema='article.experiment_protocol.v1', status='frozen', phase='development',
        output_relative_path='audit_results/article_stage_20260928/fixture',
        reserve_bytes=batch.GIB, maximum_episode_bytes=batch.EPISODE_ALLOWANCE,
        source_sha256={batch.RUNNER:batch.sha(runner)}, source_archive='archive.zip',
        source_archive_sha256=batch.sha(archive),
        slots={f'r{i}':dict(method='G', scene_id='ART1_CELL_DEV',budget=1,noise_seed=1) for i in range(count)})
    path = root/'protocol.json'; write(path, protocol)
    return path, protocol, root/protocol['output_relative_path']


CHILD = '''import json,os,sys,time
from pathlib import Path
root,run_id,mode=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
fd=os.open(root/'calls.jsonl',os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
os.write(fd,(json.dumps(dict(run_id=run_id,blas=os.environ['OPENBLAS_NUM_THREADS']))+'\\n').encode());os.close(fd)
if mode=='change':
 p=root/'protocol.json';d=json.loads(p.read_text());d['changed']=True;p.write_text(json.dumps(d))
time.sleep(.08)
if mode in ('fail','blocked'):raise SystemExit(3)
'''


def factory(root, modes=None):
    return lambda run_id:[sys.executable,'-B','-c',CHILD,str(root),run_id,(modes or {}).get(run_id,'ok')]


def run(root, path, name='batch', **kwargs):
    return batch.run_batch(path, root/name, root=root, preflight=False,
        command_factory=kwargs.pop('command_factory', factory(root)),
        free_bytes=kwargs.pop('free_bytes', lambda:10*batch.GIB), poll_seconds=.01, **kwargs)


class ArticleBatchTests(unittest.TestCase):
    def test_only_never_started_and_no_retry_across_fresh_batch_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,p,phase=fixture(root,5);pin=batch.sha(path)
            write(phase/'start_ledger.json',dict(schema='article.start_ledger.v1',
                protocol_sha256=pin,phase='development',slots=p['slots'],entries=[
                    dict(run_id='r0',status='controller_stop'),dict(run_id='r1',status='reserved')]))
            write(phase.parent/'execution_registry.json',dict(schema='article.execution_registry.v1',
                phase_caps=batch.CAPS,entries=[dict(ledger=str(phase/'start_ledger.json'),
                    run_id='r2',phase='development',protocol_sha256=pin)]))
            first=run(root,path,command_factory=factory(root,{'r3':'fail','r4':'blocked'}))
            second=run(root,path,'second')
            calls=[json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()]
            self.assertEqual(sorted(row['run_id'] for row in calls),['r3','r4'])
            self.assertTrue(all(row['blas']=='1' for row in calls))
            self.assertEqual(len([r for r in first['runs'] if r.get('returncode')==3]),2)
            self.assertEqual(len([r for r in second['runs'] if r['status']=='skipped']),5)

    def test_reserve_includes_active_children_and_stops_new_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_,_=fixture(root)
            available=batch.GIB+batch.SCRATCH_ALLOWANCE+batch.EPISODE_ALLOWANCE
            result=run(root,path,free_bytes=lambda:available)
            self.assertEqual(result['stop_reason'],'resource_blocked')
            self.assertEqual(result['resource_gate']['required_bytes'],available+batch.EPISODE_ALLOWANCE)
            self.assertEqual(result['never_launched'],['r1','r2'])
            self.assertEqual(len((root/'calls.jsonl').read_text().splitlines()),1)
            self.assertEqual(result['runs'][0]['status'],'process_finished')

    def test_protocol_changed_by_child_stops_remaining_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_,_=fixture(root)
            result=run(root,path,workers=1,command_factory=factory(root,{'r0':'change'}))
            self.assertEqual(result['stop_reason'],'validation_failed')
            self.assertIn('protocol changed',result['validation_error'])
            self.assertEqual(result['never_launched'],['r1','r2'])

    def test_frozen_source_change_rejected_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_,_=fixture(root)
            (root/batch.RUNNER).write_text('# changed\n')
            with self.assertRaisesRegex(ValueError,'source changed'):run(root,path)
            self.assertFalse((root/'calls.jsonl').exists())

    def test_deadline_finishes_children_without_launching_remainder(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_,_=fixture(root)
            result=run(root,path,workers=1,max_wall_seconds=.03)
            self.assertEqual(result['stop_reason'],'launch_deadline')
            self.assertEqual(result['never_launched'],['r1','r2'])
            self.assertEqual(result['runs'][0]['returncode'],0)

    def test_two_worker_bound_and_fresh_output_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_,_=fixture(root,6)
            result=run(root,path)
            rows=[r for r in result['runs'] if 'started_unix_s' in r]
            peak=max(sum(r['started_unix_s']<=t<r['ended_unix_s'] for r in rows)
                     for t in [r['started_unix_s'] for r in rows])
            self.assertEqual(peak,2)
            with self.assertRaises(FileExistsError):run(root,path)
            with self.assertRaisesRegex(ValueError,'workers'):run(root,path,'invalid',workers=3)

    def test_exclusive_protocol_lock_blocks_second_scheduler(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_,phase=fixture(root)
            lock=phase.parent/'batch_control'/(batch.sha(path)+'.lock')
            with batch.exclusive(lock):
                with self.assertRaisesRegex(RuntimeError,'another batch'):run(root,path)
            self.assertFalse((root/'batch').exists())

    def test_malformed_global_registry_and_full_cap_never_spawn(self):
        for malformed in (False,True):
            with self.subTest(malformed=malformed),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path,_,phase=fixture(root)
                rows=[{}] if malformed else [dict(ledger='/other/ledger',run_id=f'old{i}',
                    phase='development',protocol_sha256='old') for i in range(12)]
                write(phase.parent/'execution_registry.json',dict(schema='article.execution_registry.v1',
                    phase_caps=batch.CAPS,entries=rows))
                result=run(root,path)
                self.assertEqual(result['stop_reason'],'validation_failed' if malformed else 'cumulative_phase_cap')
                self.assertFalse((root/'calls.jsonl').exists())


if __name__=='__main__':
    unittest.main()
