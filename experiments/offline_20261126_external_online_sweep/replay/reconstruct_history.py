"""Recreate only the observed packets at registered archived poses."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
ARCHIVE = ANALYSES / "external_support_continuity_20261126"
SOURCE_SCENE = ANALYSES / "ras_20261126/heterogeneous/scenes"
FROZEN = (ANALYSES / "reproduction_20261104/simulation_publication_stage/experiments"
          / "unknown_exploration_20261104/source")
LAYOUT = "b26_hm3d_00770_s1"


def read(path):
    return json.loads(path.read_bytes())


def hash_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class RecordedPoseSensor:
    """Use original first-hit sensor routines without World construction or stepping."""

    def __init__(self, cold, layout):
        sys.path[:0] = [str(FROZEN), str(SOURCE_SCENE)]
        from scene import ReplicaCADConfig, ReplicaCADWorld, SPECS, TexturedRayRenderer, source_scene
        self.config = ReplicaCADConfig(**cold["actual_config"])
        self.shape = tuple(cold["shape"])
        self.seed = int(cold["seed"])
        spec = SPECS[layout]
        self.source_from_world = np.linalg.inv(np.asarray(spec["world_from_source"]))
        self._renderer = TexturedRayRenderer(source_scene(spec["source_floor"]))
        self._ray = self._renderer.scene
        self._frame_cache = {}
        self._open_gids, self._closed_regions = [], []
        c = self.config
        focal = c.width_px / (2 * np.tan(np.deg2rad(c.fov_deg) / 2))
        self.intrinsic = np.array([[focal, 0, (c.width_px - 1) / 2], [0, focal, (c.height_px - 1) / 2], [0, 0, 1.]])
        self._render = ReplicaCADWorld._render.__get__(self)
        self.sense = ReplicaCADWorld.sense.__get__(self)
        self.scan = ReplicaCADWorld.scan.__get__(self)

    def place(self, row):
        self.position = tuple(row["position"])
        self.heading, self.step_count = int(row["heading"]), int(row["step"])


def load_history(max_paid=38, arm="region_S"):
    run = ARCHIVE / "execution/formal" / LAYOUT / arm / "attempt_001/run"
    cold = read(ARCHIVE / "scenes/prepared" / LAYOUT / "initial_state.json")
    steps = read(run / "steps.json")
    if not 0 <= max_paid <= len(steps):
        raise ValueError("Requested state is outside the archived trajectory")
    first = {
        **read(run / "initial_raw_rgbd_scan_sha256.json"),
        "position": cold["anchor"][:2], "heading": cold["anchor"][2],
        "action": None, "collision": False, "done": False,
    }
    sensor = RecordedPoseSensor(cold, LAYOUT)
    from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10

    transform = GridTransform(sensor.shape, sensor.config.resolution_m)
    mapper = ObservedRuntimeMapperV10(sensor.shape, sensor.config, truncation_m=sensor.config.truncation_m)
    packets = []
    with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as saved:
        beliefs = saved["belief"][: max_paid + 1]
        np.testing.assert_array_equal(saved["paid"][: max_paid + 1], np.arange(max_paid + 1))

    for index, row in enumerate([first, *steps[:max_paid]]):
        assert row["step"] == index
        sensor.place(row)
        frame, scan = sensor.sense(), sensor.scan()
        actual = {
            "depth_values_sha256": hash_array(frame.depth_m),
            "rgb_values_sha256": hash_array(frame.color_rgb),
            "scan_ranges_sha256": hash_array(scan.ranges_m),
        }
        if any(row[key] != value for key, value in actual.items()):
            raise ValueError(f"{arm} paid {index}: reproduced raw sensor values disagree with archive")
        packet = SensorPacket(
            cold["scene_id"], "single_observation_cold_start", f"frame-{index}", index, frame, scan,
            sensor.position, sensor.heading, "current_first_hit_RGBD_scan_visible_ideal_structure_labels",
            "exact_discrete_simulator_odometry", row["action"], bool(row["collision"]), bool(row["done"]),
        ).validate(transform, sensor.config)
        if packet.sha256() != row["packet_sha256"]:
            raise ValueError(f"{arm} paid {index}: full packet provenance disagrees")
        packets.append(packet)
        mapper.update(frame, scan)
        np.testing.assert_array_equal(mapper.belief, beliefs[index], err_msg=f"{arm} paid {index}")
    return {
        "packets": packets, "beliefs": beliefs, "mapper": mapper, "transform": transform,
        "run": run, "cold": cold,
    }


if __name__ == "__main__":
    result = load_history()
    print(f"Verified {len(result['packets'])} full packets and complete observed grids through paid action 38")
