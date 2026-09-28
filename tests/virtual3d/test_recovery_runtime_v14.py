from types import SimpleNamespace as S
import unittest

from nso.recovery_runtime_v14 import RecoveryRuntimeV14


class Harness(RecoveryRuntimeV14):
    def __init__(self, allowed, *, phase='outbound', return_available=True, remaining=20, return_actions=None):
        self.plans = 0
        self.audit = []
        self.full_shape = (8, 8)
        self.remaining = remaining
        ledger = S(remaining_budget=remaining)
        self.states = [dict(closed=False, pending=None, packet=S(position=(3, 3), heading=0,
                            action_id=53, frame_id='frame-53'), active_actions=['forward'],
                            option={'option_id':'old','selection_call_id':1}, phase=phase)]
        self.checks = []
        values = iter(allowed)
        def assess(i, action):
            self.checks.append(action)
            ok = next(values)
            return dict(allowed=ok, reason='ok' if ok else 'next_footprint_not_known_safe', action=action)
        self.components = S(_cpu_backend=S(scenes=[dict(ledger=ledger)]),
            bind_execution_option=lambda i, option: None, assess_local_action=assess,
            plan_observed_return=lambda i: S(available=return_available,
                actions=['left'] if return_actions is None else return_actions, reason='unavailable'))

    def choose_goal(self, i, proposed, bounds):
        self.plans += 1
        self.states[i].update(phase='outbound', active_actions=['right'],
                              option={'option_id':'new','selection_call_id':2})

    def close_sensor_episode(self, i, *, reason):
        self.states[i].update(closed=True, reason=reason)


class RecoveryRuntimeTests(unittest.TestCase):
    def test_recovery_action_is_guarded_and_requires_observation(self):
        r = Harness([False, True])
        self.assertEqual(r.next_local_action(0), 'right')
        self.assertEqual(r.checks, ['forward','right'])
        self.assertEqual(r.plans, 1)
        self.assertEqual(r.states[0]['phase'], 'outbound')
        self.assertTrue(r.states[0]['pending']['allowed'])
        with self.assertRaisesRegex(RuntimeError, 'pending'):
            r.next_local_action(0)

    def test_second_denial_falls_back_after_exactly_one_retry(self):
        r = Harness([False, False, True])
        self.assertEqual(r.next_local_action(0), 'left')
        self.assertEqual(r.plans, 1)
        self.assertEqual(r.checks, ['forward','right','left'])
        self.assertEqual(r.states[0]['phase'], 'return')

    def test_return_phase_never_restarts_exploration(self):
        r = Harness([False, True], phase='return')
        self.assertEqual(r.next_local_action(0), 'left')
        self.assertEqual(r.plans, 0)

    def test_no_safe_return_never_replans_or_authorizes(self):
        r = Harness([False], return_available=False)
        self.assertIsNone(r.next_local_action(0))
        self.assertEqual(r.plans, 0)
        self.assertTrue(r.states[0]['closed'])
        self.assertIsNone(r.states[0]['pending'])

    def test_only_return_budget_left_never_replans(self):
        r = Harness([False, True], remaining=1)
        self.assertEqual(r.next_local_action(0), 'left')
        self.assertEqual(r.plans, 0)


if __name__ == '__main__':
    unittest.main()
