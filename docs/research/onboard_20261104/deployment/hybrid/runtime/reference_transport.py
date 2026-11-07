"""Persistent science-Python transport; ROS supervisor retains execution control."""
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

from runtime_environment import bounded_environment

SCIENCE_PYTHON = Path('/home/ouc/nso_robot/envs/science312/bin/python')


class ReferenceWorker:
    def __init__(self, work, output, module='nso_bridge.reference_worker'):
        self.sequence = 0
        self.responses = queue.Queue()
        Path(output).mkdir(parents=True, exist_ok=True)
        self.log = (Path(output) / 'worker.log').open('xb')
        self.process = subprocess.Popen(
            [str(SCIENCE_PYTHON), '-s', '-B', '-m', module, '--output', str(output)],
            cwd=str(Path(work) / 'nso_bridge'), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1, env=bounded_environment())
        def read():
            for line in self.process.stdout:
                try:
                    self.responses.put(json.loads(line))
                except ValueError:
                    self.responses.put(dict(ok=False, error=dict(message='invalid reference worker JSON')))
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def call(self, method, params, heartbeat):
        self.sequence += 1
        self.process.stdin.write(json.dumps(dict(id=self.sequence, method=method, params=params), allow_nan=False)+'\n')
        self.process.stdin.flush()
        # Transport timeout exceeds the unchanged planner's original 20 s
        # compute limit; it never enters action selection or its unit budget.
        deadline = time.monotonic() + 45.
        while time.monotonic() < deadline:
            heartbeat()
            try:
                response = self.responses.get(timeout=.05)
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError('reference worker exited')
                continue
            if response.get('id') != self.sequence:
                raise RuntimeError('reference response sequence differs')
            if not response.get('ok'):
                raise RuntimeError('reference worker: '+str(response.get('error')))
            return response['result']
        raise RuntimeError('reference worker transport timeout')

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps(dict(id=-1, method='shutdown', params={}))+'\n')
                self.process.stdin.flush()
                self.process.wait(timeout=3.)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.log.close()
