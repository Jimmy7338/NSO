"""Check outbound commitment without disabling observation or return safety."""
import unittest

from nso.continuation_v13_2 import force_outbound_candidate
from nso.cpu_sensor_contract_v10 import digest
from scripts.collect_semantic_gain_v13_history import packet
from tests.virtual3d.test_continuation_v13 import ContinuationInterventionTests


class OutboundInterventionTests(unittest.TestCase):
    setUp = ContinuationInterventionTests.setUp

    def test_outbound_completes_with_return_reserve_and_live_feedback(self):
        route = self.routes[0]
        option = force_outbound_candidate(self.runtime, route["candidate_id"], digest(self.routes))
        state = self.runtime.states[0]
        self.assertEqual(state["active_actions"], route["outbound_actions"])
        self.assertEqual(option["actions"], route["actions"])
        self.assertEqual(option["return_cost"], route["return_cost"])
        self.assertGreater(option["cost"], len(state["active_actions"]))
        for expected in route["outbound_actions"]:
            action = self.runtime.next_local_action(0)
            self.assertEqual(action, expected)
            self.assertTrue(state["pending"]["allowed"])
            frame, collision, done = self.world.step(action)
            self.assertFalse(collision)
            observed = packet(self.world, self.config, action, frame, collision, done)
            self.runtime.observe(0, self.world.step_count, None, None, None, sensor_packet=observed)
        self.assertEqual(state["active_actions"], [])
        self.assertEqual([*self.world.position, self.world.heading], route["states"][len(route["outbound_actions"])] )
        backend_state = self.backend.scenes[0]
        self.assertEqual(backend_state["gain"].snapshot()["observed_actions"], route["outbound_cost"])
        self.assertGreaterEqual(backend_state["ledger"].remaining_budget, route["return_cost"])

    def test_changed_pool_cannot_start_motion(self):
        with self.assertRaisesRegex(ValueError, "pool changed"):
            force_outbound_candidate(self.runtime, self.routes[0]["candidate_id"], "incorrect")
        self.assertEqual(self.world.step_count, 0)
        self.assertIsNone(self.runtime.states[0]["pending"])

    def test_pending_observation_cannot_be_overridden(self):
        force_outbound_candidate(self.runtime, self.routes[0]["candidate_id"], digest(self.routes))
        self.runtime.next_local_action(0)
        with self.assertRaisesRegex(ValueError, "idle decision"):
            force_outbound_candidate(self.runtime, self.routes[0]["candidate_id"], digest(self.routes))


if __name__ == "__main__":
    unittest.main()
