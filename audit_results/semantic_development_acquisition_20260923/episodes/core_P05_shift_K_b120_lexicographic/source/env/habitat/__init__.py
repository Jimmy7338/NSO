# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api

import os

# Habitat 2 仅复用 exploration_env / sync_vector_env，勿加载 vendored habitat_api
if os.environ.get("NSO_HABITAT_VERSION") == "2":
    from .sync_vector_env import SyncVectorEnv  # noqa: F401
else:
    import numpy as np
    import torch
    from habitat.config.default import get_config as cfg_env
    from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

    from .exploration_env import Exploration_Env
    from .habitat_api.habitat.core.vector_env import ThreadedVectorEnv, VectorEnv
    from .sync_vector_env import SyncVectorEnv
    from .habitat_api.habitat_baselines.config.default import get_config as cfg_baseline


def make_env_fn(args, config_env, config_baseline, rank):  # pragma: no cover - H1 only
    print(f"[环境初始化] 进程 {rank}: 开始加载数据集...")
    dataset = PointNavDatasetV1(config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    print(f"[环境初始化] 进程 {rank}: 正在加载场景 {config_env.SIMULATOR.SCENE}")
    config_env.freeze()

    print(f"[环境初始化] 进程 {rank}: 正在创建 Exploration_Env...")
    env = Exploration_Env(args=args, rank=rank,
                          config_env=config_env, config_baseline=config_baseline, dataset=dataset
                          )

    env.seed(rank)
    print(f"[环境初始化] 进程 {rank}: 环境创建完成")
    return env


def construct_envs(args):
    env_configs = []
    baseline_configs = []
    args_list = []

    basic_config = cfg_env(config_paths=
                           ["env/habitat/habitat_api/configs/" + args.task_config])
    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    basic_config.freeze()

    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

    # 如果指定了优先场景，将其移到列表最前面
    if hasattr(args, 'priority_scene') and args.priority_scene:
        priority = args.priority_scene
        if priority in scenes:
            scenes.remove(priority)
            scenes.insert(0, priority)
            print(f"[Scene] 优先使用场景: {priority}")
        else:
            print(f"[Scene] 警告: 场景 '{priority}' 不在可用场景列表中")
            print(f"[Scene] 可用场景: {', '.join(scenes[:10])}..." if len(scenes) > 10 else f"[Scene] 可用场景: {', '.join(scenes)}")

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(np.floor(len(scenes) / args.num_processes))

    for i in range(args.num_processes):
        config_env = cfg_env(config_paths=
                             ["env/habitat/habitat_api/configs/" + args.task_config])
        config_env.defrost()

        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scenes[
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
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

        agent_sensors = []
        agent_sensors.append("RGB_SENSOR")
        agent_sensors.append("DEPTH_SENSOR")

        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

        config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.TURN_ANGLE = 10
        config_env.DATASET.SPLIT = args.split

        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)

        args_list.append(args)

    print(f"[环境初始化] 正在创建 {args.num_processes} 个并行环境...")
    print(f"[环境初始化] 这可能需要一些时间，特别是首次加载场景数据时...")
    # 单进程时用线程版 VectorEnv，避免子进程 GLX/EGL 与 xvfb 不兼容
    env_fn_args = tuple(
        tuple(
            zip(args_list, env_configs, baseline_configs,
                range(args.num_processes))
        )
    )
    use_main_thread_vis = (
        args.num_processes == 1
        and (getattr(args, "visualize", 0) or getattr(args, "print_images", 0))
    )
    if use_main_thread_vis:
        print("[环境初始化] 可视化模式：主线程单环境（SyncVectorEnv）")
        envs = SyncVectorEnv(
            make_env_fn=make_env_fn,
            env_fn_args=env_fn_args,
        )
    elif args.num_processes == 1:
        envs = ThreadedVectorEnv(
            make_env_fn=make_env_fn,
            env_fn_args=env_fn_args,
        )
    else:
        envs = VectorEnv(
            make_env_fn=make_env_fn,
            env_fn_args=env_fn_args,
            multiprocessing_start_method="forkserver",
        )
    print(f"[环境初始化] 环境创建完成")

    return envs
