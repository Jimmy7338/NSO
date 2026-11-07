"""Bounded, receive-only stop verification orchestration.

The sample callback owns the existing hardware/zero predicate and an atomic
snapshot. This helper neither interprets CAN nor grants motion permission.
Callbacks must be nonblocking; the deadline bounds polling, not callback I/O.
"""
import math
import time


def wait_for_stop_feedback(sample, timeout_s, *, clock=time.monotonic,
                           sleep=time.sleep, poll_interval_s=.02):
    """Poll sample(now), preserving its predicate and diagnostic evidence.

    sample returns {'confirmed': bool, 'reason': str or None, 'snapshot': dict}.
    Snapshot should retain zero_since, CAN source times/ages and status. The
    caller must obtain predicate and snapshot together under its existing lock.
    Ordinary sampling/clock/sleep errors become failed results, allowing the
    caller's cleanup to continue. A timeout remains a failed stop confirmation.
    """
    if (not math.isfinite(timeout_s) or timeout_s < 0 or
            not math.isfinite(poll_interval_s) or poll_interval_s <= 0):
        raise ValueError('finite nonnegative timeout and positive poll interval required')
    started = None
    last_now = None
    last_sample = None
    count = 0

    def result(status, reason, error=None):
        return dict(confirmed=status == 'confirmed', status=status,
                    timeout=status == 'timeout', reason=reason, error=error,
                    timeout_s=float(timeout_s), started_monotonic=started,
                    ended_monotonic=last_now,
                    elapsed_s=None if started is None or last_now is None else last_now-started,
                    sample_count=count, last_sample=last_sample)

    try:
        started = clock()
        if not math.isfinite(started):
            raise ValueError('nonfinite stop-verification clock')
        deadline = started + timeout_s
        while True:
            now = clock()
            if not math.isfinite(now) or now < (started if last_now is None else last_now):
                raise ValueError('invalid or regressed stop-verification clock')
            last_now = now
            # Do not admit a success sampled after the allotted deadline.
            if now > deadline:
                return result('timeout', 'stop_feedback_wait_timeout')
            current = sample(now)
            if (not isinstance(current, dict) or type(current.get('confirmed')) is not bool
                    or not isinstance(current.get('snapshot'), dict)):
                raise ValueError('invalid stop-verification sample')
            last_sample = current
            count += 1
            sampled_end = clock()
            if not math.isfinite(sampled_end) or sampled_end < now:
                raise ValueError('invalid or regressed stop-verification clock')
            last_now = sampled_end
            if sampled_end > deadline:
                return result('timeout', 'stop_feedback_wait_timeout')
            if current['confirmed']:
                return result('confirmed', current.get('reason'))
            remaining = deadline-last_now
            if remaining <= 0:
                return result('timeout', 'stop_feedback_wait_timeout')
            sleep(min(poll_interval_s, remaining))
    except Exception as exc:
        return result('error', 'stop_feedback_wait_error',
                      type(exc).__name__+': '+str(exc))


def stop_feedback_cleanup_sequence(sample, cleanup, *, clock=time.monotonic,
                                   sleep=time.sleep):
    """Keep native/adapter cleanup between independent bounded observations.

    A failed/exceptional observation never prevents cleanup. Cleanup exceptions
    are recorded, and post-close verification still runs; the caller retains
    cleanup failures separately from the CAN stop predicate.
    """
    before = wait_for_stop_feedback(sample, 5., clock=clock, sleep=sleep)
    cleanup_error = None
    try:
        cleanup()
    except Exception as exc:
        cleanup_error = type(exc).__name__+': '+str(exc)
    after = wait_for_stop_feedback(sample, 3., clock=clock, sleep=sleep)
    return dict(before_close=before, after_close=after, cleanup_error=cleanup_error)
