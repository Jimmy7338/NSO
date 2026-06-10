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
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='当前进程使用的GPU ID（默认0，用于多GPU训练）')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='使用的GPU数量（默认1，设置为2可启用双GPU训练）')
    parser.add_argument('--sync_interval', type=int, default=1000,
                        help='多GPU训练时，模型参数同步间隔（全局步数，默认1000）')
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
    parser.add_argument('--paper_mode', action='store_true', default=False,
                        help='论文完整配置：语义+结构奖励+RPN+回环+SSC 一键启用')
    parser.add_argument('--semantic_reward_coeff', type=float, default=0.12,
                        help='语义奖励系数（论文 λ_sem=0.12）')
    parser.add_argument('--structural_reward_coeff', type=float, default=0.12,
                        help='结构内容奖励系数（论文 λ_struct=0.12）')
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

    parser.add_argument('--use_ssc_completion', action='store_true', default=False,
                        help='启用语义场景补全（SSC）功能，预测未观测区域的语义信息')
    parser.add_argument('--ssc_confidence_thresh', type=float, default=0.5,
                        help='SSC 补全结果的置信度阈值（默认0.5）')
    parser.add_argument('--ssc_update_interval', type=int, default=10,
                        help='SSC 更新的间隔步数（默认10，减少计算开销）')
    parser.add_argument('--ssc_model_path', type=str, default=None,
                        help='SSC 深度学习模型路径（如果为 None，使用基于规则的补全）')
    parser.add_argument('--ssc_max_distance', type=int, default=10,
                        help='SSC 最大补全距离（格子数，默认10）')
    parser.add_argument('--ssc_use_structural_prior', action='store_true', default=True,
                        help='使用结构先验（门框等）进行语义补全')
    parser.add_argument('--use_exploration_completion', action='store_true', default=False,
                        help='启用探索地图补全（预测未观测但可能存在的可探索区域，让覆盖地图更大）')
    parser.add_argument('--exploration_completion_thresh', type=float, default=0.5,
                        help='探索地图补全的置信度阈值（默认0.5）')
    parser.add_argument('--exploration_completion_distance', type=int, default=15,
                        help='探索地图补全的最大距离（格子数，默认15）')
    parser.add_argument('--exploration_completion_update_interval', type=int, default=10,
                        help='探索地图补全的更新间隔（步数，默认10）')
    
    parser.add_argument('--use_voxel_based_completion', action='store_true', default=False,
                        help='启用基于3D体素的语义补全（使用深度图+相机内参+语义分割，更精确）')
    
    # 全局目标可达性评估网络参数
    parser.add_argument('--use_goal_reachability', action='store_true', default=False,
                        help='启用全局目标可达性评估网络，在生成目标前评估可达性')
    parser.add_argument('--train_goal_reachability', action='store_true', default=False,
                        help='训练目标可达性评估网络')
    parser.add_argument('--goal_reachability_threshold', type=float, default=0.5,
                        help='目标可达性阈值，超过此值认为目标可达（默认0.5）')
    parser.add_argument('--goal_reachability_model_path', type=str, default=None,
                        help='预训练的目标可达性评估网络路径')
    parser.add_argument('--goal_reachability_lr', type=float, default=1e-4,
                        help='目标可达性网络学习率（默认1e-4）')
    parser.add_argument('--goal_reachability_collect_data', action='store_true', default=False,
                        help='收集目标可达性训练数据')
    parser.add_argument('--goal_reachability_data_dir', type=str, default='tmp/goal_reachability_data',
                        help='目标可达性数据保存目录')
    parser.add_argument('--goal_reachability_max_candidates', type=int, default=10,
                        help='生成候选目标点的最大数量，从中选择可达性最高的（默认10）')
    parser.add_argument('--reachability_mask_alpha', type=float, default=2.0,
                        help='论文式(2) 可达性掩码系数 α')
    parser.add_argument('--rpn_in_channels', type=int, default=0,
                        help='RPN 输入通道数；0=从 reach checkpoint 自动推断，否则 2/4')
    parser.add_argument('--paper_rewards', type=int, default=0,
                        help='1: 启用论文式(4)完整奖励（语义+结构+前沿+内在惩罚）')
    parser.add_argument('--use_structural_reward', type=int, default=0,
                        help='1: 启用结构/前沿奖励项')
    parser.add_argument('--use_intrinsic_goal_penalty', type=int, default=0,
                        help='1: 对已探索栅格目标施加内在惩罚')
    parser.add_argument('--intrinsic_reward_coeff', type=float, default=0.05,
                        help='内在目标惩罚系数')
    parser.add_argument('--loop_pose_correction', type=int, default=0,
                        help='1: 回环检测后轻量位姿校正（论文3.5.1）')
    parser.add_argument('--loop_pose_correction_weight', type=float, default=0.35,
                        help='回环位姿校正混合权重')
    parser.add_argument('--voxel_size', type=float, default=0.05,
                        help='体素大小（米，默认0.05m=5cm）')
    parser.add_argument('--voxel_grid_x', type=int, default=200,
                        help='体素网格X方向尺寸（默认200）')
    parser.add_argument('--voxel_grid_y', type=int, default=200,
                        help='体素网格Y方向尺寸（默认200）')
    parser.add_argument('--voxel_grid_z', type=int, default=50,
                        help='体素网格Z方向尺寸（默认50）')
    parser.add_argument('--use_semantic_segmentation', action='store_true', default=True,
                        help='使用语义分割（True）或目标检测（False）进行像素级标注')

    # ----------------------------------------------------------------
    # NSO 新增参数：OV-SDF、STGHP、RPN-UQ、IGCR
    # ----------------------------------------------------------------
    # OV-SDF：开放词汇语义密度场
    parser.add_argument('--use_open_vocab_semantic', action='store_true', default=False,
                        help='启用 CLIP+GroundingDINO 开放词汇语义密度场（替代固定类别 YOLOv8）')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        help='CLIP 模型名称（默认 ViT-B/32）')
    parser.add_argument('--ov_queries', type=str,
                        default='indoor furniture and appliances,doorway and passage',
                        help='开放词汇查询词，逗号分隔')
    parser.add_argument('--ov_query_weights', type=str, default='0.7,0.3',
                        help='查询权重，逗号分隔，与 ov_queries 对应')
    parser.add_argument('--gdino_config', type=str, default=None,
                        help='GroundingDINO 配置文件路径（None=降级到 YOLOv8）')
    parser.add_argument('--gdino_weights', type=str, default=None,
                        help='GroundingDINO 权重文件路径')
    # STGHP：语义拓扑图层次规划
    parser.add_argument('--use_topo_graph', action='store_true', default=False,
                        help='启用语义拓扑图层次规划（STGHP）')
    parser.add_argument('--topo_update_period', type=int, default=100,
                        help='拓扑图更新周期（步数，对应论文 N_topo=100）')
    parser.add_argument('--topo_lambda_F', type=float, default=1.0,
                        help='拓扑目标选择：前沿长度权重 λ_F')
    parser.add_argument('--topo_lambda_S', type=float, default=0.5,
                        help='拓扑目标选择：语义密度权重 λ_S')
    parser.add_argument('--topo_lambda_D', type=float, default=0.3,
                        help='拓扑目标选择：图距离惩罚权重 λ_D')
    # RPN-UQ：MC-Dropout 不确定性感知
    parser.add_argument('--use_rpn_uq', action='store_true', default=False,
                        help='启用 RPN 不确定性估计（MC-Dropout，对应 RPN-UQ）')
    parser.add_argument('--rpn_mc_samples', type=int, default=10,
                        help='MC-Dropout 采样次数 T_MC（默认10）')
    parser.add_argument('--rpn_dropout', type=float, default=0.1,
                        help='RPN MC-Dropout 概率（默认0.1）')
    parser.add_argument('--reachability_mask_beta', type=float, default=1.0,
                        help='RPN-UQ 方差惩罚强度 β（论文公式6）')
    parser.add_argument('--rpn_lambda_ece', type=float, default=0.1,
                        help='RPN 损失中 ECE 正则化系数（论文公式9）')
    # IGCR：信息增益覆盖奖励
    parser.add_argument('--use_igcr', action='store_true', default=False,
                        help='启用信息增益覆盖奖励（IGCR，替代面积增量）')
    parser.add_argument('--intrinsic_penalty', type=float, default=0.1,
                        help='重复目标内在惩罚幅度')

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
    parser.add_argument("--habitat_version", type=int, default=1, choices=[1, 2],
                        help="1=Habitat 0.1 (nso), 2=Habitat-Lab 0.2.4 (nso_h2)")
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

    if args.paper_mode:
        # 论文完整配置（NSO：四项核心创新全部启用）
        args.paper_rewards = 1
        args.use_structural_reward = 1
        args.use_intrinsic_goal_penalty = 1
        args.use_semantic = True
        args.semantic_reward_coeff = 0.12
        args.structural_reward_coeff = 0.12
        args.frontier_reward_coeff = 0.15
        # OV-SDF：开放词汇语义密度场
        args.use_open_vocab_semantic = True
        args.clip_model = getattr(args, "clip_model", "ViT-B/32")
        args.ov_queries = getattr(
            args, "ov_queries",
            "indoor furniture and appliances,doorway and passage"
        )
        # STGHP：语义拓扑图层次规划
        args.use_topo_graph = True
        args.topo_update_period = getattr(args, "topo_update_period", 100)
        args.topo_lambda_F = getattr(args, "topo_lambda_F", 1.0)
        args.topo_lambda_S = getattr(args, "topo_lambda_S", 0.5)
        args.topo_lambda_D = getattr(args, "topo_lambda_D", 0.3)
        # RPN-UQ：不确定性感知可达性预测
        args.use_goal_reachability = True
        args.train_goal_reachability = True
        args.use_rpn_uq = True
        args.rpn_mc_samples = getattr(args, "rpn_mc_samples", 10)
        args.rpn_dropout = getattr(args, "rpn_dropout", 0.1)
        args.reachability_mask_alpha = 2.0
        args.reachability_mask_beta = getattr(args, "reachability_mask_beta", 1.0)
        args.goal_reachability_max_candidates = 16
        # IGCR：信息增益覆盖奖励
        args.use_igcr = True
        # 回环检测（系统组件）
        args.use_loop_detection = True
        args.loop_pose_correction = 1
        args.loop_interval = 100
        # 禁用未完成的 SSC 深度网络（仅保留规则传播作为后处理）
        args.use_ssc_completion = False

    return args
