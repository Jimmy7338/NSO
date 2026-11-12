"""Passive bounded CAN evidence for a specific scan-matcher fault probe.

Uses the existing receive-only decoder and original kernel timestamps. This
observer never supplies velocity to a controller and has no transmit method.
"""
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

RUNTIME = Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/runtime')


class BoundedCanCapture:
    def __init__(self, maximum_frames=800):
        sys.path.insert(0, str(RUNTIME))
        from can_control_feedback import CanVelocityFeedback
        self.reader = CanVelocityFeedback('can0')
        # Set before starting the receive thread; retain roughly 16 s at 50 Hz.
        self.reader.samples = deque(maxlen=maximum_frames)
        self.maximum_frames = maximum_frames
        self.started_wall_ns = time.time_ns()
        self.reader.start()

    def save_and_close(self, path):
        self.reader.close()
        with self.reader.lock:
            samples = [asdict(s) for s in self.reader.samples]
        receipt = self.reader.source_receipt()
        stamps = [s['kernel_stamp_ns'] for s in samples]
        intervals = [b-a for a, b in zip(stamps, stamps[1:])]
        result = dict(
            schema='nso.bounded_passive_can/1', started_wall_ns=self.started_wall_ns,
            finished_wall_ns=time.time_ns(), maximum_frames=self.maximum_frames,
            retained_frames=len(samples), source=receipt, samples=samples,
            oldest_kernel_stamp_ns=stamps[0] if stamps else None,
            newest_kernel_stamp_ns=stamps[-1] if stamps else None,
            maximum_interval_ns=max(intervals) if intervals else None,
            nonincreasing_intervals=sum(v <= 0 for v in intervals),
            scope='Receive-only independent diagnostic ring; original kernel '
                  'timestamps, raw frame bytes and decoded velocity; no '
                  'interpolation or controller input. Completeness must be '
                  'checked from timestamps and reader counters, not assumed.')
        Path(path).write_text(json.dumps(result, separators=(',', ':')))
        return result
