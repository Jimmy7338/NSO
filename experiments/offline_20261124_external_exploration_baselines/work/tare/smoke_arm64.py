"""Exercise the unmodified upstream TARE node with five recorded sensor packets."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
PINNED_SOURCE_ARCHIVE_SHA256 = "a3adfcab09a669a2a1b9bf0aaec734171cc46ce0fa5cc16a3b54fc51fdb36024"
RUNTIME = ROOT / "runtime"
PACKETS = ROOT / "smoke_packets"
SOURCE = ROOT / "catkin/src/tare_planner"
RECORDINGS = PROJECT / "analyses/reproduction_20261104/current_source_pair/prepared/Q0/shelf_west/prefix"
OLD_BRIDGE = PROJECT / "workspaces/nso_research_20261029/NSO/scripts/tare_ros_bridge_v39.py"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_samples():
    PACKETS.mkdir(exist_ok=True)
    frame_fields = ("depth_m", "intrinsic", "world_from_camera")
    scan_fields = ("ranges_m", "angle_min_rad", "angle_increment_rad", "range_max_m", "world_from_laser")
    metadata = []
    for index in range(5):
        frame_path = RECORDINGS / f"frames/{index:04d}.npz"
        scan_path = RECORDINGS / f"scans/{index:04d}.npz"
        with np.load(frame_path, allow_pickle=False) as frame, np.load(scan_path, allow_pickle=False) as scan:
            packet = {"frame__" + field: frame[field] for field in frame_fields}
            packet.update({"scan__" + field: scan[field] for field in scan_fields})
        path = PACKETS / f"{index:04d}.npz"
        np.savez_compressed(path, **packet)
        metadata.append({"path": str(path), "sha256": sha256(path),
                         "original_frame_sha256": sha256(frame_path), "original_scan_sha256": sha256(scan_path)})
    return metadata


def main():
    assert sha256(ROOT / "tare_4450059.tar.gz") == PINNED_SOURCE_ARCHIVE_SHA256
    for directory in (RUNTIME, RUNTIME / "logs", RUNTIME / "home"):
        directory.mkdir(exist_ok=True)
    executable = ROOT / "catkin/devel/lib/tare_planner/tare_planner_node"
    node = RUNTIME / "tare_planner_node"
    if not node.is_symlink():
        node.symlink_to(executable)
    assert node.resolve() == executable and node.is_file()
    os.environ["ROS_HOME"] = str(RUNTIME / "home")
    os.environ["ROS_LOG_DIR"] = str(RUNTIME / "logs")
    os.environ["ROS_IP"] = "127.0.0.1"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    spec = importlib.util.spec_from_file_location("historical_tare_bridge", OLD_BRIDGE)
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    historical.PINNED_NODE_SHA256 = sha256(executable)
    bridge = historical.Bridge.__new__(historical.Bridge)
    bridge.native, bridge.master, bridge.log = None, None, None
    args = SimpleNamespace(runtime=RUNTIME, parameters=SOURCE / "config/indoor.yaml", overrides_json=None,
                           port=port, public_ground_z=0.0, planning_wait_s=8.0)
    samples = prepare_samples()
    try:
        bridge.__init__(args)
        status = bridge.status()
        print(json.dumps({"status": status["status"], "node": status["node_name"],
                          "node_sha256": status["node_sha256"], "master": status["master_uri"]}), flush=True)
        results = [bridge.observe({"op": "observe", "packet_path": sample["path"], "action_id": action_id})
                   for action_id, sample in enumerate(samples)]
        assert [row["unique_paid_observations"] for row in results] == list(range(1, 6))
        assert all(row["rgbd_points"] > 0 and row["planar_hit_points"] > 0 for row in results)
        assert results[-1]["expected_native_planning_due"]
        final = {"scope": "Native ROS transport and planner input smoke using a recorded prior simulation prefix",
                 "new_simulator_actions": 0, "formal_experiment": False, "upstream_commit":
                 "44500592b86138257273e0cab264e6a847ccefc7", "upstream_package": str(SOURCE),
                 "bridge_sha256": sha256(OLD_BRIDGE), "startup": status, "samples": samples, "results": results,
                 "fresh_waypoint_received": bool(results[-1]["fresh_waypoint"])}
        (ROOT / "smoke_receipt.json").write_text(json.dumps(final, indent=2) + "\n")
        print(json.dumps({key: final[key] for key in ("scope", "new_simulator_actions",
                                                     "formal_experiment", "fresh_waypoint_received")}), flush=True)
        if not final["fresh_waypoint_received"]:
            raise RuntimeError("native registration and sensor transport passed but no fresh waypoint was returned")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
