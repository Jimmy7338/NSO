#!/usr/bin/env python3
"""habitat-sim 0.2.x 冒烟：加载 NSO 测试场景并读取 RGB。"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

import habitat_sim
import numpy as np


def main():
    scene = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    if not os.path.isfile(scene):
        print("场景不存在:", scene, file=sys.stderr)
        sys.exit(1)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene
    sim_cfg.gpu_device_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

    rgb_cfg = habitat_sim.CameraSensorSpec()
    rgb_cfg.uuid = "rgb"
    rgb_cfg.sensor_type = habitat_sim.SensorType.COLOR
    rgb_cfg.resolution = [256, 256]
    rgb_cfg.position = [0.0, 1.25, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_cfg]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    sim.initialize_agent(0)
    obs = sim.get_sensor_observations()["rgb"]
    rgb = obs[:, :, :3] if obs.shape[-1] >= 3 else obs
    print("habitat_sim", habitat_sim.__version__)
    print("scene OK:", scene)
    print("rgb shape", rgb.shape, "max", int(rgb.max()))
    sim.close()
    print("PASS")


if __name__ == "__main__":
    main()
