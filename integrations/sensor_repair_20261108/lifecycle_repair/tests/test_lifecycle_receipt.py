"""Real OS signal / cleanup receipt checks; no ROS or actuator imports."""
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

RUNTIME=Path(__file__).resolve().parents[1]/'runtime'
sys.path.insert(0,str(RUNTIME))
from lifecycle_receipt import LifecycleReceipt
from stop_feedback_wait import stop_feedback_cleanup_sequence

CHILD='''
import pathlib,signal,sys,threading,time
sys.path.insert(0,sys.argv[1])
from lifecycle_receipt import LifecycleReceipt
from stop_feedback_wait import stop_feedback_cleanup_sequence
r=LifecycleReceipt(sys.argv[2]);done=threading.Event()
for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,r.signal_handler(done))
r.record('ready')
while not done.wait(.01):pass
r.record('stop_requested');r.record('worker_close_begin');r.record('worker_close_end')
def close():r.record('actuator_close_begin');r.record('actuator_close_end')
stop_feedback_cleanup_sequence(lambda _:dict(confirmed=True,reason=None,snapshot={}),close,
 milestone=lambda phase,value:r.record(phase,verification=value))
r.record('root_saved')
'''

class TestLifecycleReceipt(unittest.TestCase):
    def test_real_signals_keep_cleanup_order(self):
        for sig in (signal.SIGTERM,signal.SIGINT):
            with self.subTest(sig=sig),tempfile.TemporaryDirectory() as td:
                p=subprocess.Popen([sys.executable,'-c',CHILD,str(RUNTIME),td])
                try:
                    deadline=time.monotonic()+5
                    while not list(Path(td).glob('*_ready.json')):
                        if p.poll() is not None or time.monotonic()>deadline:self.fail('child not ready')
                        time.sleep(.01)
                    p.send_signal(sig);self.assertEqual(p.wait(timeout=5),0)
                    rows=[json.loads(f.read_text()) for f in sorted(Path(td).glob('*.json'))]
                    self.assertEqual([r['event'] for r in rows],['ready','signal_received','stop_requested','worker_close_begin','worker_close_end','zero_before','actuator_close_begin','actuator_close_end','zero_after','root_saved'])
                    self.assertEqual(rows[1]['signal_number'],sig)
                    self.assertTrue(rows[5]['verification']['confirmed'])
                finally:
                    if p.poll() is None:p.kill();p.wait()

    def test_diagnostic_failure_cannot_skip_cleanup(self):
        calls=[]
        def bad(*_):raise OSError('diagnostic unavailable')
        result=stop_feedback_cleanup_sequence(lambda _:dict(confirmed=True,snapshot={}),lambda:calls.append('cleanup'),milestone=bad)
        self.assertEqual(calls,['cleanup']);self.assertTrue(result['after_close']['confirmed'])
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'file';file.write_text('existing')
            receipt=LifecycleReceipt(file);receipt.record('stop_requested')
            self.assertTrue(receipt.errors);self.assertEqual(file.read_text(),'existing')

if __name__=='__main__':unittest.main()
