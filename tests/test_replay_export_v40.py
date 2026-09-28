"""Meaningful replay integrity checks: saved actions, data boundary and time gating.

Run after the bounded exporter. These tests never create a simulated world,
sensor observation, plan, TSDF reconstruction, or quality measurement.
"""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v40_replay_export", ROOT / "scripts/build_replay_viewer_v40.py")
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)
OUT = ROOT / "docs/thesis/demos/v40_replay"


def load_artifact():
    html = (OUT / "index.html").read_text()
    match = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', html, re.S)
    if not match:
        raise AssertionError("missing standalone data payload")
    return html, json.loads(match.group(1))


class ReplayUnitTests(unittest.TestCase):
    def test_absent_depth_has_distinct_grey_not_zero_distance_color(self):
        values = np.array([[0, np.nan, np.inf, -1, 4.1, .01, 2, 4]], dtype=np.float32)
        rgb = export.depth_rgb(values)
        for col in range(5):
            self.assertEqual(rgb[0, col].tolist(), [215, 220, 226])
        self.assertNotEqual(rgb[0, 5].tolist(), [215, 220, 226])
        self.assertNotEqual(rgb[0, 6].tolist(), rgb[0, 7].tolist())

    def test_packet_export_ignores_hidden_metadata_and_semantic_array(self):
        def fake_packet(secret):
            buf = io.BytesIO()
            np.savez(buf, frame__timestamp_s=4., frame__depth_m=np.ones((72, 96), dtype=np.float32),
                     frame__color_rgb=np.zeros((72, 96, 3), dtype=np.uint8),
                     frame__intrinsic=np.array([[48., 0, 47.5], [0, 48., 35.5], [0, 0, 1.]]),
                     frame__world_from_camera=np.eye(4), metadata=secret,
                     frame__semantic=np.full((72, 96), secret, dtype=object))
            return buf.getvalue()
        a, _ = export.export_packet(fake_packet("HIDDEN_A"), np.zeros(3), export.Assets())
        b, _ = export.export_packet(fake_packet("HIDDEN_B"), np.zeros(3), export.Assets())
        self.assertEqual(a, b)
        self.assertNotIn("HIDDEN", json.dumps(a))

    def test_infeasible_candidate_sentinel_is_not_displayed_as_gain(self):
        raw = {"planning": {"phase": "online", "secret_actual_Q": .9,
               "candidates": [{"action": "left", "return_feasible": False,
                               "expected_proxy": -1e30, "hidden_structure": "LEAK"},
                              {"action": "right", "return_feasible": True,
                               "expected_proxy": .7, "forecast_information": True}]}}
        sanitized = export.sanitize_planning(raw)
        self.assertIsNone(sanitized["candidates"][0]["expected_proxy"])
        self.assertEqual(sanitized["candidates"][1]["expected_proxy"], .7)
        self.assertNotIn("LEAK", json.dumps(sanitized))
        self.assertNotIn("secret_actual_Q", sanitized)

    def test_only_reached_endpoint_can_be_visible(self):
        self.assertIsNone(export.checkpoint_at(0))
        self.assertIsNone(export.checkpoint_at(17))
        self.assertEqual(export.checkpoint_at(18), "prefix")
        self.assertEqual(export.checkpoint_at(41), "prefix")
        self.assertEqual(export.checkpoint_at(42), "final")
        for step in (-1, 43):
            with self.assertRaises(ValueError):
                export.checkpoint_at(step)

    def test_source_registry_rejects_changes_and_outside_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.json").write_text('{"value":1}')
            registry = export.Sources(root)
            registry.read("source.json")
            (root / "source.json").write_text('{"value":2}')
            with self.assertRaises(ValueError):
                registry.verify()
            with self.assertRaises(ValueError):
                registry.read("../forbidden")

    def test_no_simulation_or_mapping_imports(self):
        tree = ast.parse((ROOT / "scripts/build_replay_viewer_v40.py").read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        self.assertFalse(any(name.split(".")[0] in {"env", "nso", "open3d", "rospy"} for name in names))


@unittest.skipUnless((OUT / "index.html").exists(), "build the saved replay artifact first")
class SavedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html, cls.data = load_artifact()
        cls.manifest = json.loads((OUT / "manifest.json").read_text())

    def test_fixed_cases_and_real_saved_temporal_alignment(self):
        self.assertEqual(set(self.data["runs"]), {"case00", "case01", "case04", "case05"})
        for case, run in self.data["runs"].items():
            folder = ROOT / export.COHORT / case
            trace = json.loads((folder / "trace.json").read_text())
            controller = json.loads((folder / "controller.json").read_text())
            export.validate_alignment(trace, controller["history"], controller["plans"])
            corrupted = copy.deepcopy(trace)
            corrupted[19]["action"] = "not_the_executed_action"
            with self.assertRaises(ValueError):
                export.validate_alignment(corrupted, controller["history"], controller["plans"])
            self.assertEqual(len(run["frames"]), 43)
            for step, frame in enumerate(run["frames"]):
                self.assertEqual(set(frame), set(export.FRAME_KEYS))
                self.assertEqual(frame["step"], step)
                self.assertEqual(frame["pose"], trace[step]["pose_v33"])
                self.assertEqual(frame["next_action"], trace[step]["next_action"])
                if step < 42:
                    self.assertEqual(frame["next_action"], trace[step + 1]["action"])
                self.assertEqual(frame["posterior"]["weights"], controller["history"][step]["posterior"]["probabilities"])

    def test_h0_null_and_h1_decision_event_are_reported_without_selecting_new_case(self):
        conditions = {row["id"]: row for row in self.data["conditions"]}
        self.assertIsNone(conditions["h0"]["first_divergence_step"])
        self.assertEqual(conditions["h1"]["first_divergence_step"], 18)
        for case in ("case00", "case01", "case04", "case05"):
            run = self.data["runs"][case]
            self.assertEqual(run["checkpoints"]["prefix"]["step"], 18)
            self.assertEqual(run["checkpoints"]["final"]["step"], 42)
            self.assertEqual(set(run["checkpoints"]), {"prefix", "final"})
        self.assertEqual(self.data["evaluation"]["case00"]["final"]["J5"],
                         self.data["evaluation"]["case01"]["final"]["J5"])

    def test_source_and_output_hashes_and_artifact_cap(self):
        for path, record in self.manifest["sources"].items():
            raw = (ROOT / path).read_bytes()
            self.assertEqual(len(raw), record["bytes"], path)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"], path)
        raw = (OUT / "index.html").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), self.manifest["outputs"]["index.html"]["sha256"])
        self.assertLessEqual(sum(p.stat().st_size for p in OUT.iterdir()), 20_000_000)
        self.assertEqual(self.manifest["saved_sensor_frames"], 172)
        self.assertEqual(self.manifest["displayed_mesh_checkpoints"], 8)
        self.assertFalse(self.manifest["scope"]["GT_geometry_loaded"])

    def test_offline_and_online_layers_do_not_mix(self):
        self.assertIn("evaluation", self.data)
        for run in self.data["runs"].values():
            self.assertNotIn("evaluation", run)
            for frame in run["frames"]:
                encoded = json.dumps(frame)
                for key in ('"metadata"', '"hypothesis"', '"F1_5cm"', '"J5"',
                            '"reference"', '"masks_hex"', '"potential"', '"frame__semantic"'):
                    self.assertNotIn(key, encoded)
        self.assertNotRegex(self.html, r'<(?:script|link)[^>]+(?:src|href)=["\']https?://')
        self.assertNotIn("fetch(", self.html)
        self.assertNotIn("new WebSocket", self.html)
        self.assertIn('type="checkbox" id="evaluation">', self.html)  # opt-in, not prechecked

    @unittest.skipUnless(shutil.which("node"), "Node not available for JS guard check")
    def test_actual_browser_checkpoint_function_has_no_future_mesh(self):
        body = re.search(r"function checkpointAt\(t\)\{[^}]+\}", self.html).group()
        program = body + "\nconst assert=require('assert');" + "\n".join(
            f"assert.strictEqual(checkpointAt({step}), {json.dumps(export.checkpoint_at(step))});"
            for step in range(43)) + "\nassert.throws(()=>checkpointAt(43));assert.throws(()=>checkpointAt(1.5));"
        subprocess.run([shutil.which("node"), "-e", program], check=True, capture_output=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
