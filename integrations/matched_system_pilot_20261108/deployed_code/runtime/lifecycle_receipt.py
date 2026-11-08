"""Best-effort immutable diagnostics; no execution or stop authority."""
import json
import os
from pathlib import Path
import signal
import time


class LifecycleReceipt:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.errors = []

    def record(self, event, **fields):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            row = dict(event=event, unix_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                       pid=os.getpid(), ppid=os.getppid(), pgid=os.getpgrp(),
                       sid=os.getsid(0), **fields)
            path = self.directory / (str(row['unix_ns'])+'_'+event+'.json')
            with path.open('x') as stream:
                json.dump(row, stream, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:
            self.errors.append(type(exc).__name__+': '+str(exc))

    def signal_handler(self, interrupted):
        def receive(signum, _frame):
            interrupted.set()
            self.record('signal_received', signal_number=signum,
                        signal_name=signal.Signals(signum).name)
        return receive
