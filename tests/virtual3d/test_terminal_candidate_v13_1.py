"""Regression: budget closure must not relabel a preceding candidate pool."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.collect_semantic_gain_v13_history_v13_1 import main


class TerminalCandidateTests(unittest.TestCase):
    def test_short_real_episode_never_records_a_zero_budget_candidate(self):
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / "configs/virtual3d/semantic_gain_v13_history_smoke.json").read_text())
        config.update(total_budget=4, checkpoint_decision_ordinals=list(range(1, 7)))
        with tempfile.TemporaryDirectory(prefix=".v13-terminal-test-", dir=root) as directory:
            directory = Path(directory)
            protocol = directory / "protocol.json"
            protocol.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                main(directory / "run", protocol)
            result = json.loads((directory / "run/result.json").read_text())
            checkpoints = json.loads((directory / "run/checkpoints.json").read_text())
            self.assertEqual(result["paid_actions"], 4)
            self.assertEqual(result["termination"]["reason"], "budget_exhausted")
            self.assertTrue(checkpoints)
            for row in checkpoints:
                remaining = 4 - row["action_id"]
                self.assertGreater(remaining, 0)
                self.assertTrue(all(0 < c["cost"] <= remaining for c in row["candidates"]))


if __name__ == "__main__":
    unittest.main()
