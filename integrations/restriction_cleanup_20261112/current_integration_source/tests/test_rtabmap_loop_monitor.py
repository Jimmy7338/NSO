import math
import sys
import threading
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from rtabmap_loop_monitor import Monitor, info_record, transform_record


def transform(x=0):
    return NS(translation=NS(x=x, y=0, z=0), rotation=NS(x=0, y=0, z=0, w=1))


def info(loop=0, proximity=0, ref=3):
    return NS(header=NS(stamp=NS(to_nsec=lambda: 123), frame_id='nso_loop_map'),
              refId=ref, loopClosureId=loop, proximityDetectionId=proximity,
              loopClosureTransform=transform(), wmState=[1,2,3])


class LoopMonitorTests(unittest.TestCase):
    def setUp(self):
        self.monitor = Monitor.__new__(Monitor)
        self.monitor.lock = threading.RLock()
        self.monitor.latest = dict(info_count=0, graph_count=0, loop_event_count=0,
                                   proximity_event_count=0, actual_loop_reported=False)
        self.monitor.seen = set()
        self.events = []
        self.monitor.event = lambda kind, data: self.events.append((kind, data))

    def test_no_loop_readiness_does_not_invent_loop(self):
        self.monitor.info(info())
        self.assertFalse(self.monitor.latest['actual_loop_reported'])
        self.assertEqual(self.events, [])

    def test_positive_loop_counted_once_per_pair(self):
        self.monitor.info(info(loop=1))
        self.monitor.info(info(loop=1))
        self.assertEqual(self.monitor.latest['loop_event_count'], 1)
        self.assertEqual(self.monitor.latest['info_count'], 2)

    def test_new_reference_is_separate_event(self):
        self.monitor.info(info(loop=1, ref=3))
        self.monitor.info(info(loop=1, ref=4))
        self.assertEqual(self.monitor.latest['loop_event_count'], 2)

    def test_proximity_separate_from_descriptor_loop(self):
        self.monitor.info(info(proximity=1))
        self.assertEqual(self.monitor.latest['proximity_event_count'], 1)
        self.assertEqual(self.monitor.latest['loop_event_count'], 0)
        self.assertFalse(self.monitor.latest['actual_loop_reported'])
        self.assertTrue(self.monitor.latest['actual_proximity_reported'])

    def test_nonfinite_correction_rejected(self):
        with self.assertRaises(ValueError):
            transform_record(transform(math.nan))

    def test_graph_is_bounded_summary_with_correction(self):
        msg = NS(header=info().header, posesId=[1,2,3],
                 links=[NS(type=0), NS(type=1)], mapToOdom=transform(.2))
        self.monitor.graph(msg)
        self.assertEqual(self.monitor.latest['graph']['nodes'], 3)
        self.assertEqual(self.monitor.latest['graph']['map_to_odom']['translation'], [.2,0,0])
        self.assertNotIn('poses', self.monitor.latest['graph'])


if __name__ == '__main__':
    unittest.main()
