"""Log sampling tests use synthetic feedback only; no ROS or device access."""
import json
from pathlib import Path
import tempfile
import unittest

from test_visual_velocity_adapter import adapter,hardware,ready_gate


class AdapterLogSamplingTests(unittest.TestCase):
    def test_motion_edges_fault_changes_and_final_zero_survive_sampling(self):
        sampler=adapter.EventSampler();kept=[]
        rows=[]
        for i in range(100):
            rows.append(dict(event='command',at_monotonic=i*.02,
                v=.1 if 10<=i<30 else 0.,w=0.,reason='current_proposal'))
        rows.append(dict(event='command',at_monotonic=2.01,v=0.,w=0.,reason='laser_source_stale'))
        rows.append(dict(event='command',at_monotonic=2.03,v=0.,w=0.,reason='laser_source_stale'))
        for row in rows:
            if sampler.keep(row):kept.append(row)
        kept.extend(sampler.finish())
        self.assertLess(len(kept),10)
        self.assertIn(rows[10],kept)  # First actual nonzero command.
        self.assertIn(rows[30],kept)  # First actual zero after motion.
        self.assertIn(rows[-2],kept) # Fault/source transition immediately.
        self.assertIn(rows[-1],kept) # Actual final zero, never synthesized.

    def test_sparse_CAN_keeps_full_rate_distance_and_motion_boundary_times(self):
        gate=ready_gate();self.assertTrue(gate.arm(2.,1002.)[0])
        sampler=adapter.EventSampler();kept=[]
        for i in range(1,101):
            now=2.+i*.02;v=100 if 20<=i<70 else 0
            hardware(gate,now,v=v)
            row=dict(event='CAN_feedback',id='0x221',receipt_monotonic=now,
                source_monotonic=now,v=v*.001,w=0.,moving=v!=0,ever_armed=True,fault=gate.fault,
                can_distance_m=gate.can_distance,can_absolute_yaw_rad=gate.can_absolute_yaw)
            if sampler.keep(row):kept.append(row)
        kept.extend(sampler.finish());kept.sort(key=lambda r:r['source_monotonic'])
        self.assertLess(len(kept),10)
        self.assertAlmostEqual(kept[-1]['can_distance_m'],gate.can_distance)
        self.assertAlmostEqual(gate.can_distance,.1)
        transitions=[b for a,b in zip(kept,kept[1:]) if a['moving']!=b['moving']]
        self.assertEqual(len(transitions),2)
        self.assertAlmostEqual(transitions[1]['source_monotonic']-transitions[0]['source_monotonic'],1.)

    def test_state_change_and_method_events_are_not_periodically_discarded(self):
        sampler=adapter.EventSampler()
        status=dict(event='CAN_feedback',id='0x211',receipt_monotonic=1.,state=0,mode=1,error=0)
        self.assertTrue(sampler.keep(status))
        self.assertFalse(sampler.keep(dict(status,receipt_monotonic=1.01)))
        self.assertTrue(sampler.keep(dict(status,receipt_monotonic=1.02,error=5)))
        for event in ('goal','goal_result','arrival_evidence','NSO_observation','NSO_decision_from_observation','finished'):
            self.assertTrue(sampler.keep(dict(event=event,receipt_monotonic=1.03)))
            self.assertTrue(sampler.keep(dict(event=event,receipt_monotonic=1.04)))

    def test_journal_close_retains_actual_last_odom_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'events.jsonl';journal=adapter.Journal(path)
            for i in range(101):
                journal.add(dict(event='odom',receipt_monotonic=i*.02,x=i*.001,
                    y=0.,yaw=0.,caller='bridge',frame='odom',child_frame='base_link'))
            journal.close()
            self.assertIsNone(journal.error)
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertLessEqual(len(rows),4)
            self.assertEqual(rows[-1]['receipt_monotonic'],2.)
            self.assertEqual(rows[-1]['x'],.1)


if __name__=='__main__':unittest.main()
