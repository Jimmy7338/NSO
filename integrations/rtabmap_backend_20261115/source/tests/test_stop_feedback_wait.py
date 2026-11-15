"""No ROS, sockets, hardware or motion: real Gate predicate, simulated CAN time."""
import pathlib
import struct
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'runtime'))
from stop_feedback_wait import wait_for_stop_feedback, stop_feedback_cleanup_sequence
from visual_velocity_adapter import Gate


class Clock:
    def __init__(self):self.now=10.
    def __call__(self):return self.now
    def sleep(self, seconds):self.now += seconds


def sampler(clock, velocity=lambda t: (0., 0.), stale=False, bad_status=False):
    gate=Gate(60.)
    def sample(now):
        source=now-1. if stale else now
        status=bytes([0, 1, 0, 0, 0, int(bad_status), 0, 0])
        v,w=velocity(now)
        gate.hardware(0x211,status,source,now)
        gate.hardware(0x221,struct.pack('>hh',round(v*1000),round(w*1000))+b'\0'*4,source,now)
        reason=gate.hardware_reason(now)
        confirmed=(reason is None and gate.can_zero_since is not None and now-gate.can_zero_since>=1.)
        return dict(confirmed=confirmed,reason=reason if reason else None if confirmed else 'zero_interval_less_than_1s',
                    snapshot=dict(zero_since=gate.can_zero_since,
                                  status_source=gate.can_status['source'],velocity_source=gate.can_velocity['source'],
                                  status_source_age_s=now-gate.can_status['source'],velocity_source_age_s=now-gate.can_velocity['source'],
                                  v=v,w=w,state=gate.can_status['state'],mode=gate.can_status['mode'],error=gate.can_status['error'],
                                  hardware_reason=reason))
    return sample


class StopFeedbackWaitTests(unittest.TestCase):
    def run_wait(self, clock, sample, timeout=5.):
        return wait_for_stop_feedback(sample,timeout,clock=clock,sleep=clock.sleep,poll_interval_s=.02)

    def test_delayed_settling_requires_full_original_one_second(self):
        c=Clock();r=self.run_wait(c,sampler(c,lambda t:(0.,.01 if t<12. else 0.)))
        self.assertTrue(r['confirmed']);self.assertGreaterEqual(r['elapsed_s'],3.)
        self.assertLess(r['elapsed_s'],3.06)
        self.assertGreaterEqual(r['ended_monotonic']-r['last_sample']['snapshot']['zero_since'],1.)

    def test_nonzero_restarts_interval(self):
        c=Clock();r=self.run_wait(c,sampler(c,lambda t:(0.,.01 if 10.6<=t<10.8 else 0.)))
        self.assertTrue(r['confirmed']);self.assertGreaterEqual(r['elapsed_s'],1.8)
        self.assertLess(r['elapsed_s'],1.86)

    def test_zero_but_stale_never_passes(self):
        c=Clock();r=self.run_wait(c,sampler(c,stale=True),3.)
        self.assertFalse(r['confirmed']);self.assertTrue(r['timeout'])
        self.assertEqual(r['last_sample']['reason'],'chassis_status_stale')
        self.assertEqual(r['last_sample']['snapshot']['velocity_source_age_s'],1.)
        self.assertAlmostEqual(r['elapsed_s'],3.)

    def test_zero_bad_hardware_status_never_passes(self):
        c=Clock();r=self.run_wait(c,sampler(c,bad_status=True),3.)
        self.assertFalse(r['confirmed']);self.assertTrue(r['timeout'])
        self.assertEqual(r['last_sample']['reason'],'chassis_state_mode_or_fault')

    def test_continued_motion_is_bounded_timeout(self):
        c=Clock();r=self.run_wait(c,sampler(c,lambda t:(0.,.01)),5.)
        self.assertTrue(r['timeout']);self.assertFalse(r['confirmed'])
        self.assertAlmostEqual(c.now,15.)
        self.assertIsNone(r['last_sample']['snapshot']['zero_since'])

    def test_callback_exception_returns_failure_for_cleanup(self):
        c=Clock()
        def bad(now):raise RuntimeError('receive-only sample failure')
        r=self.run_wait(c,bad)
        self.assertEqual(r['status'],'error');self.assertFalse(r['confirmed'])
        self.assertIn('receive-only sample failure',r['error'])

    def test_zero_timeout_still_samples_current_evidence_once(self):
        c=Clock();r=self.run_wait(c,sampler(c),0.)
        self.assertTrue(r['timeout']);self.assertEqual(r['sample_count'],1)

    def test_scheduler_oversleep_cannot_accept_late_success(self):
        c=Clock();calls=[]
        def sample(now):calls.append(now);return dict(confirmed=False,reason='pending',snapshot={})
        def oversleep(seconds):c.now+=2.
        r=wait_for_stop_feedback(sample,1.,clock=c,sleep=oversleep)
        self.assertTrue(r['timeout']);self.assertEqual(len(calls),1)

    def test_before_wait_then_native_cleanup_then_after_wait(self):
        c=Clock();events=[];phase=['before']
        def sample(now):
            events.append(phase[0])
            return dict(confirmed=True,reason=None,snapshot={'sampled_monotonic':now})
        def cleanup():events.append('native_close');phase[0]='after'
        r=stop_feedback_cleanup_sequence(sample,cleanup,clock=c,sleep=c.sleep)
        self.assertEqual(events,['before','native_close','after'])
        self.assertEqual(r['before_close']['timeout_s'],5.)
        self.assertEqual(r['after_close']['timeout_s'],3.)
        self.assertTrue(r['after_close']['confirmed'])

    def test_before_exception_cannot_skip_cleanup_or_after_wait(self):
        c=Clock();events=[];phase=['before']
        def sample(now):
            events.append(phase[0])
            if phase[0]=='before':raise RuntimeError('sample failed')
            return dict(confirmed=True,reason=None,snapshot={})
        def cleanup():events.append('native_close');phase[0]='after'
        r=stop_feedback_cleanup_sequence(sample,cleanup,clock=c,sleep=c.sleep)
        self.assertEqual(events,['before','native_close','after'])
        self.assertEqual(r['before_close']['status'],'error')
        self.assertTrue(r['after_close']['confirmed'])

    def test_timeouts_cannot_skip_cleanup_and_are_independently_bounded(self):
        c=Clock();closed=[]
        def sample(now):return dict(confirmed=False,reason='pending',snapshot={})
        r=stop_feedback_cleanup_sequence(sample,lambda:closed.append(c.now),clock=c,sleep=c.sleep)
        self.assertEqual(closed,[15.]);self.assertAlmostEqual(c.now,18.)
        self.assertTrue(r['before_close']['timeout']);self.assertTrue(r['after_close']['timeout'])

    def test_cleanup_exception_still_collects_after_evidence(self):
        c=Clock();calls=[]
        def sample(now):calls.append(now);return dict(confirmed=True,reason=None,snapshot={})
        def cleanup():raise RuntimeError('native cleanup failed')
        r=stop_feedback_cleanup_sequence(sample,cleanup,clock=c,sleep=c.sleep)
        self.assertEqual(len(calls),2);self.assertTrue(r['after_close']['confirmed'])
        self.assertIn('native cleanup failed',r['cleanup_error'])

    def test_callback_taking_deadline_cannot_report_late_confirmation(self):
        c=Clock()
        def sample(now):c.now+=2.;return dict(confirmed=True,reason=None,snapshot={})
        r=wait_for_stop_feedback(sample,1.,clock=c,sleep=c.sleep)
        self.assertFalse(r['confirmed']);self.assertTrue(r['timeout'])


if __name__=='__main__':unittest.main()
