#!/usr/bin/env python3
"""Validate a filled V37 capture checklist and print commands; never execute ROS.

This is a configuration gate, not evidence that topic/TF/physical checks passed.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
RECORDER = ROOT / "scripts/record_ros1_rgbd_v3.py"
VERIFICATION_KEYS = {
    "actual_topics_and_message_types", "world_and_optical_frames",
    "raw_bag_pose_imu_gps_topics", "registered_rectified_rgb_depth",
    "depth_encoding_units_and_axial_convention",
    "camera_info_projection_roi_binning_and_R",
    "tf_at_sensor_timestamps_and_extrinsics",
    "laser_range_limits_and_invalid_policy", "source_clock_and_sync_tolerance",
    "fresh_absolute_output_paths_and_free_space",
}
DEPTH_CONTRACT = {"16UC1": "uint16_axial_millimetres", "32FC1": "float32_axial_metres"}
PLACEHOLDER = re.compile(r"(?:^|[/_])(?:VERIFY|VERIFIED|ABSOLUTE|TODO|REPLACE|CHANGE_ME)(?:$|[/_])", re.I)
TOPIC = re.compile(r"/[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*")
FRAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*")


def recorder_parameter_names(source=RECORDER):
    """Read literal rospy private parameters from the actual V3 source AST."""
    names = set()
    for node in ast.walk(ast.parse(Path(source).read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "rospy" and node.func.attr == "get_param"
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str) and node.args[0].value.startswith("~")):
            names.add(node.args[0].value[1:])
    if not names:
        raise ValueError("no literal private parameters found in V3 source")
    return names


def validate_configuration(config, recorder_source=RECORDER):
    errors = []

    def need(condition, message):
        if not condition:
            errors.append(message)

    def object_at(parent, key):
        value = parent.get(key)
        if not isinstance(value, dict):
            errors.append(key + " must be an object")
            return {}
        return value

    def concrete_name(value, pattern):
        return isinstance(value, str) and bool(pattern.fullmatch(value)) and not PLACEHOLDER.search(value)

    def fresh_path(value, key, bag_prefix=False):
        if not isinstance(value, str) or not value or PLACEHOLDER.search(value):
            errors.append(key + " must be a concrete new absolute path")
            return None
        path = Path(value)
        if not path.is_absolute() or path == Path("/") or ".." in path.parts:
            errors.append(key + " must be a concrete new absolute path without parent traversal")
            return None
        need(not path.exists(), key + " already exists")
        need(path.parent.is_dir(), key + " parent directory does not exist on this host")
        if bag_prefix and path.parent.is_dir():
            need(not any(p.name.startswith(path.name) for p in path.parent.iterdir()),
                 key + " collides with an existing bag prefix")
        return path

    if not isinstance(config, dict):
        return {"ready_to_print_capture_commands": False, "errors": ["configuration must be an object"]}
    need(config.get("schema_version") == "ros1_field_capture/v37", "unsupported schema_version")
    params = object_at(config, "recorder_private_parameters")
    expected = recorder_parameter_names(recorder_source)
    need(set(params) == expected, "recorder parameter set differs from actual V3 source: missing="
         + repr(sorted(expected - set(params))) + "; unsupported=" + repr(sorted(set(params) - expected)))
    for key in ("rgb_topic", "depth_topic", "camera_info_topic", "scan_topic"):
        need(concrete_name(params.get(key), TOPIC), key + " must be a concrete absolute topic")
    for key in ("world_frame", "camera_optical_frame"):
        need(concrete_name(params.get(key), FRAME), key + " must be a concrete TF frame without leading slash")
    need(params.get("semantic_topic") == "", "V37 stationary capture keeps semantic_topic empty")
    for key, strictly_positive in (("max_depth_m", True), ("sync_slop_s", False), ("min_interval_s", False)):
        value = params.get(key)
        need(type(value) in (int, float) and math.isfinite(value)
             and (value > 0 if strictly_positive else value >= 0), key + " must be a valid finite numeric value")
    need(params.get("positive_inf_policy") in ("invalid", "clear_to_max"), "unsupported positive_inf_policy")
    recording_path = fresh_path(params.get("output"), "output")
    raw_bag = object_at(config, "raw_bag")
    bag_path = fresh_path(raw_bag.get("output"), "raw_bag.output", bag_prefix=True)
    if recording_path is not None and bag_path is not None:
        need(recording_path != bag_path, "recorder and bag output paths must differ")
    auxiliary = object_at(raw_bag, "auxiliary_topics")
    need(set(auxiliary) == {"pose", "imu", "gps", "tf", "tf_static"},
         "raw bag needs exactly pose, imu, gps, tf and tf_static auxiliary topic roles")
    for key in ("pose", "imu", "gps", "tf", "tf_static"):
        need(concrete_name(auxiliary.get(key), TOPIC), "raw bag " + key + " must be a concrete absolute topic")
    # V3's TransformListener uses these standard names; this exporter adds no remappings.
    need(auxiliary.get("tf") == "/tf" and auxiliary.get("tf_static") == "/tf_static",
         "raw bag must preserve /tf and /tf_static used by V3 TransformListener")
    need(raw_bag.get("include_recorder_image_and_scan_topics") is True
         and raw_bag.get("include_camera_info") is True, "raw bag must include recorder sensors and CameraInfo")
    split = raw_bag.get("split_size_mib")
    need(type(split) is int and split > 0, "split_size_mib must be a positive integer")
    topics = [params.get(k) for k in ("rgb_topic", "depth_topic", "camera_info_topic", "scan_topic")]
    topics += [auxiliary.get(k) for k in ("pose", "imu", "gps", "tf", "tf_static")]
    if all(isinstance(v, str) for v in topics):
        need(len(set(topics)) == len(topics), "distinct sensor/auxiliary roles must not alias the same topic")
    policy = object_at(config, "capture_policy")
    need(policy.get("publishes_motion") is False, "publishes_motion must be false")
    need(policy.get("phase") == "stationary_capture_only", "only stationary capture is in this V37 gate")
    need(policy.get("raw_bag_required") is True, "raw_bag_required must be true")
    need(policy.get("gps_used_for_localization") is False, "GPS localization remains disabled until separately validated")
    need(policy.get("depth_encodings_to_verify") == DEPTH_CONTRACT, "depth encoding/unit contract differs from V3")
    need(policy.get("camera_info_association") == "latest_received_not_timestamp_synchronized",
         "CameraInfo association must reflect V3's unsynchronized latest value")
    need(policy.get("lidar_motion_compensation") == "not_implemented", "V3 has no per-beam lidar motion compensation")
    verification = object_at(config, "verification")
    evidence = object_at(config, "verification_evidence")
    need(set(verification) == VERIFICATION_KEYS, "verification key set must match the V37 checklist")
    for key in sorted(VERIFICATION_KEYS):
        need(verification.get(key) is True, "unverified: " + key)
        need(isinstance(evidence.get(key), str) and bool(evidence.get(key).strip()), "missing evidence reference: " + key)
    return {
        "schema_version": "ros1_field_capture_preflight/v37",
        "ready_to_print_capture_commands": not errors,
        "errors": errors,
        "recorder_source": str(Path(recorder_source).resolve()),
        "recorder_source_sha256": hashlib.sha256(Path(recorder_source).read_bytes()).hexdigest(),
        "recorder_parameter_names": sorted(expected),
        "scope": "offline_configuration_only; manual verification declarations are not physical certification",
        "ros_imported": False, "ros_master_started": False, "motion_published": False,
        "live_ros_verified": False, "robot_interface_ready": False,
    }


def capture_commands(config, recorder_source=RECORDER):
    result = validate_configuration(config, recorder_source)
    if not result["ready_to_print_capture_commands"]:
        raise ValueError("configuration not ready; no commands produced: " + "; ".join(result["errors"]))
    params = config["recorder_private_parameters"]
    recorder = ["python3", str(Path(recorder_source).resolve())]
    # Leave the empty optional semantic topic at V3's default. ROS command-line
    # YAML parsing of an explicit empty value can produce None on some stacks.
    recorder += ["_" + key + ":=" + str(value) for key, value in params.items() if key != "semantic_topic"]
    bag = config["raw_bag"]
    topics = [params[k] for k in ("rgb_topic", "depth_topic", "camera_info_topic", "scan_topic")]
    topics += [bag["auxiliary_topics"][k] for k in ("pose", "imu", "gps", "tf", "tf_static")]
    return {"terminal_a_raw_bag": shlex.join(["rosbag", "record", "--split", "--size=" + str(bag["split_size_mib"]),
                                              "-O", bag["output"]] + topics),
            "terminal_b_v3_recorder": shlex.join(recorder)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--print-commands", action="store_true", help="Print shell-quoted commands only if all checks pass; execute nothing")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        result = validate_configuration(config)
        result["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        if args.print_commands and result["ready_to_print_capture_commands"]:
            result["commands"] = capture_commands(config)
    except (ValueError, OSError, TypeError, SyntaxError) as exc:
        result = {"ready_to_print_capture_commands": False, "errors": [str(exc)]}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["ready_to_print_capture_commands"] else 2


if __name__ == "__main__":
    sys.exit(main())
