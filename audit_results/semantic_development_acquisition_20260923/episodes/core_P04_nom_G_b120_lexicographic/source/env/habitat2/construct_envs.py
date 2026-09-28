# Habitat 2 环境构造实现
from __future__ import annotations

import os
import sys

from ._lab import setup_habitat2_lab

setup_habitat2_lab()

import numpy as np
import torch
from habitat.config import read_write
from habitat.core.vector_env import VectorEnv
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

from env.habitat.sync_vector_env import SyncVectorEnv

from .config_utils import apply_nso_env_overrides, get_nso_config, get_scenes_for_config
from .exploration_env import Exploration_Env
from .vector_env_compat import VectorEnvCompat

_HABITAT2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _check_habitat2():
    try:
        import habitat  # noqa: F401
        return True
    except ImportError:
        return False


def construct_envs(args):
    if not _check_habitat2():
        print(
            "[Habitat2] habitat-lab 未安装。\n"
            "  1) 将 habitat-lab-0.2.4.tar.gz 放到项目根目录\n"
            "  2) bash scripts/unpack_local_archives.sh\n"
            "  3) bash scripts/install_habitat2.sh",
            file=sys.stderr,
        )
        raise RuntimeError("habitat-lab not installed for Habitat 2")

    os.chdir(_HABITAT2_ROOT)

    env_configs = []
    args_list = []

    basic_config = get_nso_config(args.task_config)
    with read_write(basic_config):
        basic_config.habitat.dataset.split = args.split
    scenes = get_scenes_for_config(basic_config)

    if getattr(args, "priority_scene", None) and args.priority_scene in scenes:
        scenes = [args.priority_scene] + [s for s in scenes if s != args.priority_scene]

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce num_processes: not enough scenes")
        scene_split_size = int(np.floor(len(scenes) / args.num_processes))

    def make_env_fn(args, config_env, rank):
        dataset = PointNavDatasetV1(config_env.habitat.dataset)
        with read_write(config_env):
            if dataset.episodes:
                config_env.habitat.simulator.scene = dataset.episodes[0].scene_id
        env = Exploration_Env(
            args=args,
            rank=rank,
            config_env=config_env,
            config_baseline=None,
            dataset=dataset,
        )
        env.seed(rank)
        return env

    for i in range(args.num_processes):
        config_env = get_nso_config(args.task_config)
        with read_write(config_env):
            config_env.habitat.dataset.split = args.split

        if len(scenes) > 0:
            with read_write(config_env):
                config_env.habitat.dataset.content_scenes = scenes[
                    i * scene_split_size: (i + 1) * scene_split_size
                ]

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        else:
            gpu_id = 0

        config_env = apply_nso_env_overrides(
            config_env, args, i, scenes, gpu_id)
        env_configs.append(config_env)
        args_list.append(args)

    env_fn_args = tuple(
        zip(args_list, env_configs, range(args.num_processes))
    )
    if args.num_processes == 1:
        envs = SyncVectorEnv(make_env_fn=make_env_fn, env_fn_args=env_fn_args)
    else:
        envs = VectorEnvCompat(
            VectorEnv(
                make_env_fn=make_env_fn,
                env_fn_args=env_fn_args,
                multiprocessing_start_method="forkserver",
            )
        )
    return envs
