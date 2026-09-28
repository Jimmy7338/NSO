"""V37 configuration gate tests; no ROS imports, messages, or recordings."""
import copy
import json
from pathlib import Path
import shlex
import tempfile
import unittest

from scripts.preflight_ros1_field_capture_v37 import (
    ROOT, VERIFICATION_KEYS, capture_commands, recorder_parameter_names, validate_configuration,
)


class CaptureChecklistTests(unittest.TestCase):
    def setUp(self):
        self.example = json.loads((ROOT / "configs/virtual3d/ros1_field_capture_v37.example.json").read_text())
        self.temporary = tempfile.TemporaryDirectory(prefix="nso-v37-config-")
        self.addCleanup(self.temporary.cleanup)
        self.filled = copy.deepcopy(self.example)
        self.filled["recorder_private_parameters"].update(
            output=str(Path(self.temporary.name) / "new recording"),
            world_frame="map", camera_optical_frame="zed_left_camera_optical_frame")
        self.filled["raw_bag"]["output"] = str(Path(self.temporary.name) / "new raw bag")
        self.filled["raw_bag"]["auxiliary_topics"].update(pose="/zed/pose", imu="/imu/data", gps="/fix")
        self.filled["verification"] = {key: True for key in VERIFICATION_KEYS}
        self.filled["verification_evidence"] = {key: "synthetic configuration-test declaration, not robot evidence"
                                                  for key in VERIFICATION_KEYS}

    def assertRejected(self, configuration):
        self.assertFalse(validate_configuration(configuration)["ready_to_print_capture_commands"])
        with self.assertRaises(ValueError):
            capture_commands(configuration)

    def test_example_and_each_unverified_flag_cannot_export(self):
        self.assertRejected(self.example)
        for key in VERIFICATION_KEYS:
            with self.subTest(key=key):
                config = copy.deepcopy(self.filled)
                config["verification"][key] = False
                self.assertRejected(config)
        config = copy.deepcopy(self.filled)
        config["verification_evidence"].clear()
        self.assertRejected(config)

    def test_private_parameter_schema_is_actual_v3_not_output_aliases(self):
        self.assertEqual(set(self.example["recorder_private_parameters"]), recorder_parameter_names())
        for wrong, right in (("optical_frame", "camera_optical_frame"), ("info_topic", "camera_info_topic"),
                             ("slop_s", "sync_slop_s"), ("pose_topic", None)):
            with self.subTest(wrong=wrong):
                config = copy.deepcopy(self.filled)
                config["recorder_private_parameters"][wrong] = config["recorder_private_parameters"].pop(right) if right else "/pose"
                self.assertRejected(config)

    def test_missing_tf_invalid_units_and_motion_are_rejected(self):
        for mutation in (lambda c: c["raw_bag"]["auxiliary_topics"].pop("tf_static"),
                         lambda c: c["capture_policy"]["depth_encodings_to_verify"].update({"32FC1": "millimetres"}),
                         lambda c: c["capture_policy"].update(publishes_motion=True),
                         lambda c: c["recorder_private_parameters"].update(world_frame=""),
                         lambda c: c["raw_bag"]["auxiliary_topics"].update(pose="/VERIFY_POSE_TOPIC")):
            config = copy.deepcopy(self.filled)
            mutation(config)
            self.assertRejected(config)

    def test_filled_config_prints_quoted_complete_commands_without_execution(self):
        result = validate_configuration(self.filled)
        self.assertTrue(result["ready_to_print_capture_commands"], result["errors"])
        self.assertFalse(result["live_ros_verified"])
        commands = capture_commands(self.filled)
        bag = shlex.split(commands["terminal_a_raw_bag"])
        recorder = shlex.split(commands["terminal_b_v3_recorder"])
        self.assertEqual(bag[bag.index("-O") + 1], self.filled["raw_bag"]["output"])
        self.assertIn("/tf_static", bag)
        self.assertIn("/zed/pose", bag)
        self.assertIn("_output:=" + self.filled["recorder_private_parameters"]["output"], recorder)
        self.assertFalse(any(a.startswith("_semantic_topic:=") for a in recorder))
        self.assertEqual(len([a for a in recorder if a.startswith("_")]), len(recorder_parameter_names()) - 1)
        self.assertFalse(Path(self.filled["recorder_private_parameters"]["output"]).exists())
        self.assertNotIn("cmd_vel", " ".join(commands.values()))


if __name__ == "__main__":
    unittest.main()
