"""Real observed-map checks at the forced-candidate experiment boundary."""
import json
from pathlib import Path
import unittest

from env.virtual3d_competition_v9 import create_competition_world
from nso.continuation_v13 import describe_candidates, force_full_candidate
from nso.cpu_sensor_contract_v10 import GridTransform, digest
from scripts.collect_semantic_gain_v13_history import packet, start


class ContinuationInterventionTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.config = json.loads((root / "configs/virtual3d/semantic_gain_v13_history_smoke.json").read_text())
        self.world = create_competition_world("Q0", "shelf_east")
        self.transform = GridTransform(tuple(self.world.shape), self.world.config.resolution_m)
        self.runtime = start(self.config, self.world, self.transform, packet(self.world, self.config, None))
        self.backend = self.runtime.components._cpu_backend
        self.backend.select_target(0)
        self.routes = self.backend.scenes[0]["last_selection"]["candidates"]

    def test_complete_roundtrip_is_bound_and_live_guard_authorizes_observation(self):
        route = self.routes[0]
        force_full_candidate(self.runtime, route["candidate_id"], digest(self.routes))
        state = self.runtime.states[0]
        self.assertGreater(len(state["active_actions"]), len(route["outbound_actions"]))
        self.assertEqual(len(state["active_actions"]), route["cost"])
        self.assertEqual(route["states"][-1], route["return_anchor"])
        action = self.runtime.next_local_action(0)
        self.assertTrue(state["pending"]["allowed"])
        frame, collision, done = self.world.step(action)
        observed = packet(self.world, self.config, action, frame, collision, done)
        self.runtime.observe(0, 1, None, None, None, sensor_packet=observed)
        feedback = self.backend.scenes[0]["ledger"].snapshot()
        self.assertEqual(feedback["paid_actions"], 1)
        self.assertEqual(self.backend.scenes[0]["gain"].snapshot()["observed_actions"], 1)

    def test_changed_pool_rejected_before_any_paid_action(self):
        with self.assertRaisesRegex(ValueError, "pool changed"):
            force_full_candidate(self.runtime, self.routes[0]["candidate_id"], "wrong-hash")
        self.assertEqual(self.backend.scenes[0]["ledger"].snapshot()["paid_actions"], 0)
        self.assertIsNone(self.runtime.states[0]["pending"])

    def test_pending_sensor_prevents_second_intervention(self):
        force_full_candidate(self.runtime, self.routes[0]["candidate_id"], digest(self.routes))
        self.runtime.next_local_action(0)
        with self.assertRaisesRegex(ValueError, "idle decision"):
            force_full_candidate(self.runtime, self.routes[0]["candidate_id"], digest(self.routes))

    def test_features_keep_unobserved_direction_and_confidence_separate(self):
        rows = describe_candidates(self.runtime, self.routes)
        self.assertEqual(len(rows), len(self.routes))
        for row in rows:
            self.assertEqual(row["remaining_budget"], 96)
            self.assertEqual(row["roundtrip_cost"], row["outbound_cost"] + row["return_cost"])
            self.assertGreaterEqual(row["unknown_allocated_grid_fraction"], 0.)
            self.assertLessEqual(row["unknown_allocated_grid_fraction"], 1.)
            for asset in row["observed_assets"]:
                self.assertEqual(len(asset["direction_unobserved_fractions"]), 8)
                self.assertGreaterEqual(asset["semantic_confidence"], 0.)
                self.assertLessEqual(asset["semantic_confidence"], 1.)


if __name__ == "__main__":
    unittest.main()
