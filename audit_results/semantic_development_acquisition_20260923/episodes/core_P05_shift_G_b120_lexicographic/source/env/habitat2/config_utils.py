"""Habitat 2 配置加载与 NSO 参数覆盖。"""
from __future__ import annotations

import os
from typing import List, Optional

from ._lab import setup_habitat2_lab

setup_habitat2_lab()

from habitat.config import read_write
from habitat.config.default import get_config as _get_config
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

_HABITAT2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

_TASK_TO_BENCHMARK = {
    "pointnav_habitat_test.yaml": "benchmark/nav/pointnav/pointnav_habitat_test.yaml",
    "pointnav_gibson.yaml": "benchmark/nav/pointnav/pointnav_gibson.yaml",
    "pointnav_mp3d.yaml": "benchmark/nav/pointnav/pointnav_mp3d.yaml",
}


def resolve_benchmark_path(task_config: str) -> str:
    if task_config.startswith("benchmark/"):
        return task_config
    name = os.path.basename(task_config)
    if name in _TASK_TO_BENCHMARK:
        return _TASK_TO_BENCHMARK[name]
    legacy = os.path.join(_HABITAT2_ROOT, "configs", "habitat2", name)
    if os.path.isfile(legacy):
        return os.path.abspath(legacy)
    return task_config


def get_nso_config(
    task_config: str,
    overrides: Optional[List[str]] = None,
):
    path = resolve_benchmark_path(task_config)
    if os.path.isabs(path) or path.startswith("benchmark/"):
        return _get_config(path, overrides=overrides)
    return _get_config(path, overrides=overrides)


def apply_nso_env_overrides(config, args, rank: int, scenes: List[str], gpu_id: int):
    """按 NSO arguments 覆盖 habitat 2 配置。"""
    with read_write(config):
        h = config.habitat
        h.dataset.split = args.split
        if scenes:
            i = rank
            scene_split_size = max(1, int(len(scenes) // max(args.num_processes, 1)))
            start = i * scene_split_size
            end = (i + 1) * scene_split_size if i < args.num_processes - 1 else len(scenes)
            h.dataset.content_scenes = scenes[start:end]

        h.environment.max_episode_steps = args.max_episode_length
        if hasattr(h.environment, "iterator_options"):
            h.environment.iterator_options.shuffle = False

        sim = h.simulator
        sim.turn_angle = 10
        sim.habitat_sim_v0.gpu_device_id = gpu_id

        agent = sim.agents.main_agent
        for key in ("rgb_sensor", "depth_sensor"):
            if key in agent.sim_sensors:
                agent.sim_sensors[key].width = args.env_frame_width
                agent.sim_sensors[key].height = args.env_frame_height
                if key == "rgb_sensor" and hasattr(agent.sim_sensors[key], "hfov"):
                    agent.sim_sensors[key].hfov = int(args.hfov)
                if hasattr(agent.sim_sensors[key], "position"):
                    agent.sim_sensors[key].position = [
                        0, float(args.camera_height), 0
                    ]

        sim.action_space_config = "CustomActionSpaceConfiguration"
    return config


def get_scenes_for_config(config) -> List[str]:
    return PointNavDatasetV1.get_scenes_to_load(config.habitat.dataset)
