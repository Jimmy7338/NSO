import argparse
import math
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Active-Neural-SLAM')

    ## General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--auto_gpu_config', type=int, default=1)
    parser.add_argument('--total_num_scenes', type=str, default="auto")
    parser.add_argument('-n', '--num_processes', type=int, default=4,
                        help="""how many training processes to use (default:4)
                                Overridden when auto_gpu_config=1
                                and training on gpus """)
    parser.add_argument('--num_processes_per_gpu', type=int, default=11)
    parser.add_argument('--num_processes_on_first_gpu', type=int, default=1,
                        help='第一个GPU上的进程数（默认1，只显示一个窗口）')
    parser.add_argument('--num_episodes', type=int, default=1000000,
                        help='number of training episodes (default: 1000000)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--eval', type=int, default=0,
                        help='1: evaluate models (default: 0)')
    parser.add_argument('--train_global', type=int, default=1,
                        help="""0: Do not train the Global Policy
                                1: Train the Global Policy (default: 1)""")
    parser.add_argument('--train_local', type=int, default=1,
                        help="""0: Do not train the Local Policy
                                1: Train the Local Policy (default: 1)""")
    parser.add_argument('--train_slam', type=int, default=1,
                        help="""0: Do not train the Neural SLAM Module
                                1: Train the Neural SLAM Module (default: 1)""")


    parser.add_argument('--use_semantic', default=False, action='store_true',
                        help='是否使用语义信息')
    parser.add_argument('--semantic_conf_thresh', type=float, default=0.2,
                        help='语义检测置信度阈值（默认0.2，平衡检测数量和准确率）')
    parser.add_argument('--semantic_use_all_classes', action='store_true', default=False,
                        help='使用所有YOLO类别（不进行类别映射过滤），可检测更多对象')
    parser.add_argument('--semantic_indoor_only', action='store_true', default=False,
                        help='仅使用室内场景相关类别（过滤掉飞机、火车等室外物体）')
    parser.add_argument('--semantic_no_indoor_filter', action='store_true', default=False,
                        help='禁用室内场景过滤（允许检测所有类别，包括不合理的室外物体）')
    parser.add_argument('--train_semantic', default=False, action='store_true',
                        help='是否训练语义模型')
    parser.add_argument('--load_semantic', type=str, default="0",
                        help='预训练语义模型的路径')
    parser.add_argument('--semantic_interval', type=int, default=1,
                        help='处理语义信息的间隔步数')
    parser.add_argument('--semantic_reward_coeff', type=float, default=0.1,
                        help='语义奖励系数')
    parser.add_argument('--structural_reward_coeff', type=float, default=0.1,
                        help='结构内容奖励系数（门框/狭窄通道/开阔区）')
    parser.add_argument('--frontier_reward_coeff', type=float, default=0.15,
                        help='前沿区域奖励系数（可见但未访问的区域）')
    parser.add_argument('--w_struct_door', type=float, default=2.0,
                        help='门框区域权重（提高以优先探索门框）')
    parser.add_argument('--w_struct_narrow', type=float, default=1.0,
                        help='狭窄通道区域权重')
    parser.add_argument('--w_struct_open', type=float, default=0.5,
                        help='开阔区域权重')
    parser.add_argument('--door_boost_distance', type=float, default=5.0,
                        help='门框增强距离（格子数，门框附近区域额外奖励）')
    parser.add_argument('--room_exploration_boost', type=float, default=1.5,
                        help='房间探索增强系数（通过门框进入未探测房间的额外奖励）')
    parser.add_argument('--narrow_width_cells', type=int, default=4,
                        help='狭窄通道宽度阈值（格子）')
    parser.add_argument('--open_kernel', type=int, default=9,
                        help='开阔区域均值核大小（奇数）')

    parser.add_argument('--use_loop_detection', action='store_true', default=False,
                        help='启用语义增强 NetVLAD 回环检测')
    parser.add_argument('--loop_interval', type=int, default=100,
                        help='执行回环检测的步间隔（默认100，较大值可提升性能）')
    parser.add_argument('--loop_min_gap', type=int, default=200,
                        help='同一轨迹两关键帧之间的最小间隔，用于过滤短周期回环')
    parser.add_argument('--loop_top_k', type=int, default=5,
                        help='检索候选的数量（FAISS top-k）')
    parser.add_argument('--loop_sim_thresh', type=float, default=0.75,
                        help='NetVLAD 相似度阈值（内积）')
    parser.add_argument('--loop_sem_thresh', type=float, default=0.6,
                        help='语义向量余弦相似度阈值')
    parser.add_argument('--loop_use_lightweight', action='store_true', default=False,
                        help='使用轻量级特征提取（仅语义直方图，不使用NetVLAD，速度更快但精度较低）')

    # Logging, loading models, visualization
    parser.add_argument('--log_interval', type=int, default=10,
                        help="""log interval, one log per n updates
                                (default: 10) """)
    parser.add_argument('--save_interval', type=int, default=1,
                        help="""save interval""")
    parser.add_argument('-d', '--dump_location', type=str, default="./tmp/",
                        help='path to dump models and log (default: ./tmp/)')
    parser.add_argument('--exp_name', type=str, default="exp1",
                        help='experiment name (default: exp1)')
    parser.add_argument('--save_periodic', type=int, default=500000,
                        help='Model save frequency in number of updates')
    parser.add_argument('--load_slam', type=str, default="0",
                        help="""model path to load,
                                0 to not reload (default: 0)""")
    parser.add_argument('--load_global', type=str, default="0",
                        help="""model path to load,
                                0 to not reload (default: 0)""")
    parser.add_argument('--load_local', type=str, default="0",
                        help="""model path to load,
                                0 to not reload (default: 0)""")
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help='1:Render the frame (default: 0)')
    parser.add_argument('--vis_type', type=int, default=1,
                        help='1: Show predicted map, 2: Show GT map')
    parser.add_argument('--print_images', type=int, default=0,
                        help='1: save visualization as images')
    parser.add_argument('--save_trajectory_data', type=str, default="0")

    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=256,
                        help='Frame width (default:84)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=256,
                        help='Frame height (default:84)')
    parser.add_argument('-fw', '--frame_width', type=int, default=128,
                        help='Frame width (default:84)')
    parser.add_argument('-fh', '--frame_height', type=int, default=128,
                        help='Frame height (default:84)')
    parser.add_argument('-el', '--max_episode_length', type=int, default=1000,
                        help="""Maximum episode length in seconds for
                                Doom (default: 180)""")
    parser.add_argument("--sim_gpu_id", type=int, default=0,
                        help="gpu id on which scenes are loaded")
    parser.add_argument("--task_config", type=str,
                        default="tasks/pointnav_gibson.yaml",
                        help="path to config yaml containing task information")
    parser.add_argument("--split", type=str, default="train",
                        help="dataset split (train | val | val_mini) ")
    parser.add_argument("--priority_scene", type=str, default=None,
                        help="优先使用的场景名称（如：Cantwell, Denmark等），该场景会被优先加载")
    parser.add_argument('-na', '--noisy_actions', type=int, default=1)
    parser.add_argument('-no', '--noisy_odometry', type=int, default=1)
    parser.add_argument('--camera_height', type=float, default=1.25,
                        help="agent camera height in metres")
    parser.add_argument('--hfov', type=float, default=90.0,
                        help="horizontal field of view in degrees")
    parser.add_argument('--randomize_env_every', type=int, default=1000,
                        help="randomize scene in a thread every k episodes")

    ## Global Policy RL PPO Hyperparameters
    parser.add_argument('--global_lr', type=float, default=2.5e-5,
                        help='global learning rate (default: 2.5e-5)')
    parser.add_argument('--global_hidden_size', type=int, default=256,
                        help='local_hidden_size')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RL Optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RL Optimizer alpha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use_gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy_coef', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--num_global_steps', type=int, default=40,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo_epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num_mini_batch', type=str, default="auto",
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--use_recurrent_global', type=int, default=0,
                        help='use a recurrent global policy')

    # Local Policy
    parser.add_argument('--local_optimizer', type=str,
                        default='adam,lr=0.0001')
    parser.add_argument('--num_local_steps', type=int, default=25,
                        help="""Number of steps the local can
                            perform between each global instruction""")
    parser.add_argument('--local_hidden_size', type=int, default=512,
                        help='local_hidden_size')
    parser.add_argument('--short_goal_dist', type=int, default=1,
                        help="""Maximum distance between the agent
                                and the short term goal""")
    parser.add_argument('--local_policy_update_freq', type=int, default=5)
    parser.add_argument('--use_recurrent_local', type=int, default=1,
                        help='use a recurrent local policy')
    parser.add_argument('--use_deterministic_local', type=int, default=0,
                        help="use classical deterministic local policy")

    # Neural SLAM Module
    parser.add_argument('-pe', '--use_pose_estimation', type=int, default=2)
    parser.add_argument('--goals_size', type=int, default=2)
    parser.add_argument('-pt', '--pretrained_resnet', type=int, default=1)

    parser.add_argument('--slam_optimizer', type=str, default='adam,lr=0.0001')
    parser.add_argument('-sbs', '--slam_batch_size', type=int, default=72)
    parser.add_argument('-sit', '--slam_iterations', type=int, default=10)
    parser.add_argument('-sms', '--slam_memory_size', type=int, default=500000)
    parser.add_argument('--proj_loss_coeff', type=float, default=1.0)
    parser.add_argument('--pose_loss_coeff', type=float, default=10000.0)
    parser.add_argument('--exp_loss_coeff', type=float, default=1.0)
    parser.add_argument('--global_downscaling', type=int, default=2)
    parser.add_argument('--map_pred_threshold', type=float, default=0.5)

    parser.add_argument('--vision_range', type=int, default=64)
    parser.add_argument('--obstacle_boundary', type=int, default=5)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=2)
    parser.add_argument('--map_size_cm', type=int, default=2400)
    parser.add_argument('-ot', '--obs_threshold', type=float, default=1)
    parser.add_argument('-ct', '--collision_threshold', type=float, default=0.20)
    parser.add_argument('-nl', '--noise_level', type=float, default=1.0)

    # parse arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        if args.auto_gpu_config:
            num_gpus = torch.cuda.device_count()
            if args.total_num_scenes != "auto":
                args.total_num_scenes = int(args.total_num_scenes)
            elif "gibson" in args.task_config and \
                    "train" in args.split:
                args.total_num_scenes = 72
            elif "gibson" in args.task_config and \
                    "val_mt" in args.split:
                args.total_num_scenes = 14
            elif "gibson" in args.task_config and \
                    "val" in args.split:
                args.total_num_scenes = 1
            else:
                assert False, "Unknown task config, please specify" + \
                        " total_num_scenes"

            # Automatically configure number of training threads based on
            # number of GPUs available and GPU memory size
            total_num_scenes = args.total_num_scenes
            gpu_memory = 1000
            for i in range(num_gpus):
                gpu_memory = min(gpu_memory,
                    torch.cuda.get_device_properties(i).total_memory \
                            /1024/1024/1024)
                if i==0:
                    assert torch.cuda.get_device_properties(i).total_memory \
                            /1024/1024/1024 > 10.0, "Insufficient GPU memory"

            num_processes_per_gpu = int(gpu_memory/1.4)
            # num_processes_on_first_gpu = int((gpu_memory - 10.0)/1.4)
            num_processes_on_first_gpu = 1  # 改为1，只显示一个窗口

            if num_gpus == 1:
                args.num_processes_on_first_gpu = num_processes_on_first_gpu
                args.num_processes_per_gpu = 0
                args.num_processes = num_processes_on_first_gpu
            else:
                total_threads = num_processes_per_gpu * (num_gpus - 1) \
                                + num_processes_on_first_gpu

                num_scenes_per_thread = math.ceil(total_num_scenes/total_threads)
                num_threads = math.ceil(total_num_scenes/num_scenes_per_thread)
                args.num_processes_per_gpu = min(num_processes_per_gpu,
                                        math.ceil(num_threads//(num_gpus-1)))

                args.num_processes_on_first_gpu = max(0,
                        num_threads - args.num_processes_per_gpu*(num_gpus - 1))

                args.num_processes = num_threads

            args.sim_gpu_id = 1

            print("Auto GPU config:")
            print("Number of processes: {}".format(args.num_processes))
            print("Number of processes on GPU 0: {}".format(
                                      args.num_processes_on_first_gpu))
            print("Number of processes per GPU: {}".format(
                                      args.num_processes_per_gpu))

    if args.eval == 1:
        if args.train_global:
            print("WARNING: Training Global Policy during evaluation")
        if args.train_local:
            print("WARNING: Training Local Policy during evaluation")
        if args.train_slam:
            print("WARNING: Training Neural SLAM module during evaluation")

    assert args.short_goal_dist >= 1, "args.short_goal_dist >= 1"

    if args.use_deterministic_local:
        args.train_local = 0

    if args.num_mini_batch == "auto":
        args.num_mini_batch = max(1, args.num_processes // 2)  # 确保至少为1
    else:
        args.num_mini_batch = int(args.num_mini_batch)
        if args.num_mini_batch < 1:
            args.num_mini_batch = 1  # 确保至少为1

    return args
