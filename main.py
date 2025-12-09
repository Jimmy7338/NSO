import time
from collections import deque

import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import gym
import logging
from arguments import get_args
from env import make_vec_envs
from utils.storage import GlobalRolloutStorage, FIFOMemory
from utils.optimization import get_optimizer
from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module

import algo

import sys
import matplotlib

# if sys.platform == 'darwin':
#     matplotlib.use("tkagg")
matplotlib.use("TkAgg")
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# plt.ion()
# fig, ax = plt.subplots(1,4, figsize=(10, 2.5), facecolor="whitesmoke")

from semantic_detector import SemanticDetector
from semantic.semantic_map import SemanticMap2D
from loop.semantic_vlad import SemanticVLADExtractor
from loop.loop_detector import LoopDetector

args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.use_loop_detection and not args.use_semantic:
    raise ValueError("启用回环检测需要同时开启 --use_semantic 以提供语义信息。")

def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1.gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]


def main():
    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    policy_loss = 0

    best_cost = 100000
    costs = deque(maxlen=1000)
    exp_costs = deque(maxlen=1000)
    pose_costs = deque(maxlen=1000)

    g_masks = torch.ones(num_scenes).float().to(device)
    l_masks = torch.zeros(num_scenes).float().to(device)

    best_local_loss = np.inf
    best_g_reward = -np.inf

    if args.eval:
        traj_lengths = args.max_episode_length // args.num_local_steps
        explored_area_log = np.zeros((num_scenes, num_episodes, traj_lengths))
        explored_ratio_log = np.zeros((num_scenes, num_episodes, traj_lengths))

    g_episode_rewards = deque(maxlen=1000)

    l_action_losses = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    print("[初始化] 开始初始化环境...")
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    print("[初始化] 环境初始化完成，正在重置环境...")
    obs, infos = envs.reset()
    print("[初始化] 环境重置完成，开始训练循环...")

    # Initialize map variables
    ### Full map consists of 4 channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area
    ### 3. Current Agent Location
    ### 4. Past Agent Locations

    torch.set_grad_enabled(False)

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), \
                       int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, 4, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, 4, local_w, local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Semantic modules (optional)
    semantic_detector = None
    semantic_map2d = None
    loop_vlad_extractor = None
    loop_detector = None
    loop_last_detection_step = [-1 for _ in range(num_scenes)]
    if args.use_semantic:
        print("[初始化] 开始初始化语义检测器...")
        # 根据参数决定是否使用类别映射
        use_mapping = not (hasattr(args, 'semantic_use_all_classes') and args.semantic_use_all_classes)
        # 决定是否启用室内场景过滤
        # 默认启用室内过滤（除非使用--semantic_use_all_classes且明确禁用）
        if hasattr(args, 'semantic_no_indoor_filter') and args.semantic_no_indoor_filter:
            indoor_only = False  # 明确禁用
        elif hasattr(args, 'semantic_indoor_only') and args.semantic_indoor_only:
            indoor_only = True  # 明确启用
        else:
            # 默认启用室内过滤（适合模拟器场景）
            indoor_only = True
        print(f"[初始化] 正在加载YOLO模型 (yolov8n.pt)，这可能需要一些时间...")
        semantic_detector = SemanticDetector(
            model_name="yolov8n.pt", 
            device=device,
            use_custom_mapping=use_mapping,
            indoor_only=indoor_only
        )
        print("[初始化] 语义检测器初始化完成")
        if not use_mapping:
            if indoor_only:
                print("[Semantic] 使用所有YOLO类别（共80类），但过滤掉不合理的室外物体（如飞机、火车、红绿灯等）")
            else:
                print("[Semantic] 使用所有YOLO类别（共80类），不进行类别映射过滤")
        semantic_map2d = SemanticMap2D(
            num_scenes=num_scenes,
            num_classes=semantic_detector.num_classes,
            full_w=full_w,
            full_h=full_h,
            local_w=local_w,
            local_h=local_h,
            map_resolution_cm=args.map_resolution,
            vision_range=args.vision_range,
            dump_dir=dump_dir
        )
        if semantic_detector.use_custom_mapping:
            semantic_priority = {
                'person': 1.5,
                'chair': 1.2,
                'couch': 1.2,
                'bed': 1.2,
                'dining table': 1.3,
                'cup': 1.15,
                'bottle': 1.1,
                'bowl': 1.1,
                'tv': 1.35,
                'laptop': 1.35,
                'mouse': 1.2,
                'keyboard': 1.2,
                'cell phone': 1.2,
                'remote': 1.1,
                'microwave': 1.2,
                'oven': 1.15,
                'toaster': 1.1,
                'refrigerator': 1.2,
                'sink': 1.1,
                'book': 1.05,
                'clock': 1.05,
                'vase': 1.1,
                'potted plant': 1.15,
                'toilet': 1.05,
            }
            semantic_weights = np.ones(semantic_detector.num_classes, dtype=np.float32)
            for idx, name in enumerate(semantic_detector.get_class_names()):
                semantic_weights[idx] = semantic_priority.get(name, 1.0)
            semantic_map2d.set_class_weights(semantic_weights)
        semantic_map2d.to_device(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    ### Planner pose inputs has 7 dimensions
    ### 1-3 store continuous global agent location
    ### 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                            torch.from_numpy(origins[e]).to(device).float()

    init_map_and_pose()

    # Global policy observation space
    g_observation_space = gym.spaces.Box(0, 1,
                                         (8,
                                          local_w,
                                          local_h), dtype='uint8')

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=(2,), dtype=np.float32)

    # Local policy observation space
    l_observation_space = gym.spaces.Box(0, 255,
                                         (3,
                                          args.frame_width,
                                          args.frame_width), dtype='uint8')

    # Local and Global policy recurrent layer sizes
    l_hidden_size = args.local_hidden_size
    g_hidden_size = args.global_hidden_size

    # slam
    nslam_module = Neural_SLAM_Module(args).to(device)
    slam_optimizer = get_optimizer(nslam_module.parameters(),
                                   args.slam_optimizer)

    # Global policy
    g_policy = RL_Policy(g_observation_space.shape, g_action_space,
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': g_hidden_size,
                                      'downscaling': args.global_downscaling
                                      }).to(device)
    g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.global_lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)

    # Local policy
    l_policy = Local_IL_Policy(l_observation_space.shape, envs.action_space.n,
                               recurrent=args.use_recurrent_local,
                               hidden_size=l_hidden_size,
                               deterministic=args.use_deterministic_local).to(device)
    local_optimizer = get_optimizer(l_policy.parameters(),
                                    args.local_optimizer)

    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                      num_scenes, g_observation_space.shape,
                                      g_action_space, g_policy.rec_state_size,
                                      1).to(device)

    slam_memory = FIFOMemory(args.slam_memory_size)

    # Loading model
    if args.load_slam != "0":
        print("Loading slam {}".format(args.load_slam))
        state_dict = torch.load(args.load_slam,
                                map_location=lambda storage, loc: storage)
        nslam_module.load_state_dict(state_dict)

    if not args.train_slam:
        nslam_module.eval()

    if args.load_global != "0":
        print("Loading global {}".format(args.load_global))
        state_dict = torch.load(args.load_global,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if not args.train_global:
        g_policy.eval()

    if args.load_local != "0":
        print("Loading local {}".format(args.load_local))
        state_dict = torch.load(args.load_local,
                                map_location=lambda storage, loc: storage)
        l_policy.load_state_dict(state_dict)

    if not args.train_local:
        l_policy.eval()

    # Predict map from frame 1:
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx
         in range(num_scenes)])
    ).float().to(device)

    _, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, local_pose = \
        nslam_module(obs, obs, poses, local_map[:, 0, :, :],
                     local_map[:, 1, :, :], local_pose)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, 8, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

    global_input[:, 0:4, :, :] = local_map.detach()
    global_input[:, 4:, :, :] = nn.MaxPool2d(args.global_downscaling)(full_map)

    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(global_orientation)

    # Run Global Policy (global_goals = Long-Term Goal)
    g_value, g_action, g_action_log_prob, g_rec_states = \
        g_policy.act(
            g_rollouts.obs[0],
            g_rollouts.rec_states[0],
            g_rollouts.masks[0],
            extras=g_rollouts.extras[0],
            deterministic=False
        )

    cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
    global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]
                    for action in cpu_actions]

    # Compute planner inputs
    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['goal'] = global_goals[e]
        p_input['map_pred'] = global_input[e, 0, :, :].detach().cpu().numpy()
        p_input['exp_pred'] = global_input[e, 1, :, :].detach().cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        # 添加语义密度图（如果启用）- 使用全局地图的对应窗口
        if args.use_semantic:
            try:
                # 获取全局语义密度图的对应窗口区域
                gx1, gx2, gy1, gy2 = int(planner_pose_inputs[e][3]), int(planner_pose_inputs[e][4]), \
                                     int(planner_pose_inputs[e][5]), int(planner_pose_inputs[e][6])
                semantic_density = semantic_map2d.get_full_density_window(e, gx1, gx2, gy1, gy2)
                p_input['semantic_density'] = semantic_density
            except Exception:
                p_input['semantic_density'] = None
        else:
            p_input['semantic_density'] = None

    # Output stores local goals as well as the the ground-truth action
    output = envs.get_short_term_goal(planner_inputs)

    last_obs = obs.detach()
    local_rec_states = torch.zeros(num_scenes, l_hidden_size).to(device)
    start = time.time()

    total_num_steps = -1
    g_reward = 0

    torch.set_grad_enabled(False)

    for ep_num in range(num_episodes):
        for step in range(args.max_episode_length):
            total_num_steps += 1

            g_step = (step // args.num_local_steps) % args.num_global_steps
            eval_g_step = step // args.num_local_steps + 1
            l_step = step % args.num_local_steps

            # ------------------------------------------------------------------
            # Local Policy
            del last_obs
            last_obs = obs.detach()
            local_masks = l_masks
            local_goals = output[:, :-1].to(device).long()

            if args.train_local:
                torch.set_grad_enabled(True)

            action, action_prob, local_rec_states = l_policy(
                obs,
                local_rec_states,
                local_masks,
                extras=local_goals,
            )

            if args.train_local:
                action_target = output[:, -1].long().to(device)
                policy_loss += nn.CrossEntropyLoss()(action_prob, action_target)
                torch.set_grad_enabled(False)
            l_action = action.cpu()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Env step
            obs, rew, done, infos = envs.step(l_action)
            if args.use_loop_detection:
                for e in range(num_scenes):
                    infos[e]['loop_detected'] = False
                    infos[e]['loop_match'] = None

            l_masks = torch.FloatTensor([0 if x else 1
                                         for x in done]).to(device)
            g_masks *= l_masks
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Reinitialize variables when episode ends
            if step == args.max_episode_length - 1:  # Last episode step
                init_map_and_pose()
                del last_obs
                last_obs = obs.detach()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Neural SLAM Module
            if args.train_slam:
                # Add frames to memory
                for env_idx in range(num_scenes):
                    env_obs = obs[env_idx].to("cpu")
                    env_poses = torch.from_numpy(np.asarray(
                        infos[env_idx]['sensor_pose']
                    )).float().to("cpu")
                    env_gt_fp_projs = torch.from_numpy(np.asarray(
                        infos[env_idx]['fp_proj']
                    )).unsqueeze(0).float().to("cpu")
                    env_gt_fp_explored = torch.from_numpy(np.asarray(
                        infos[env_idx]['fp_explored']
                    )).unsqueeze(0).float().to("cpu")
                    env_gt_pose_err = torch.from_numpy(np.asarray(
                        infos[env_idx]['pose_err']
                    )).float().to("cpu")
                    slam_memory.push(
                        (last_obs[env_idx].cpu(), env_obs, env_poses),
                        (env_gt_fp_projs, env_gt_fp_explored, env_gt_pose_err))

            poses = torch.from_numpy(np.asarray(
                [infos[env_idx]['sensor_pose'] for env_idx
                 in range(num_scenes)])
            ).float().to(device)

            _, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, local_pose = \
                nslam_module(last_obs, obs, poses, local_map[:, 0, :, :],
                             local_map[:, 1, :, :], local_pose, build_maps=True)

            locs = local_pose.cpu().numpy()
            planner_pose_inputs[:, :3] = locs + origins
            local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
            for e in range(num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                local_map[e, 2:, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
            # ------------------------------------------------------------------
            # Semantic Detection and Map Update (optional, minimal intrusion)
            if args.use_semantic and (total_num_steps % max(1, args.semantic_interval) == 0):
                try:
                    # obs: (num_scenes, C, H, W), assumed uint8 [0,255] or float in [0,255]
                    # Convert to uint8 if needed
                    obs_for_det = obs
                    if obs_for_det.dtype != torch.uint8:
                        obs_for_det = torch.clamp(obs_for_det, 0, 255).to(torch.uint8)
                    # 使用可配置的置信度阈值（默认0.2，平衡检测数量和准确率）
                    conf_thresh = args.semantic_conf_thresh if hasattr(args, 'semantic_conf_thresh') else 0.2
                    det_results = semantic_detector.detect_batch(obs_for_det, conf=conf_thresh)
                    
                    # 统计并记录检测到的语义类别和标签
                    class_names = semantic_detector.get_class_names()
                    total_detections = 0
                    for e in range(num_scenes):
                        det = det_results[e]
                        if det is not None and len(det.get("classes", [])) > 0:
                            total_detections += len(det.get("classes", []))
                            classes = det["classes"]
                            scores = det["scores"]
                            boxes = det.get("boxes", None)
                            # 统计每个类别出现的次数和平均置信度
                            class_counts = {}
                            class_scores = {}
                            for cls_id, score in zip(classes, scores):
                                cls_id = int(cls_id)
                                if cls_id < len(class_names):
                                    cls_name = class_names[cls_id]
                                    if cls_name not in class_counts:
                                        class_counts[cls_name] = 0
                                        class_scores[cls_name] = []
                                    class_counts[cls_name] += 1
                                    class_scores[cls_name].append(float(score))
                            
                            # 计算平均置信度
                            class_avg_scores = {name: np.mean(scores) for name, scores in class_scores.items()}

                            # 构建带框的检测信息
                            detection_overlays = []
                            if boxes is not None and len(boxes) == len(classes):
                                for box, cls_id, score in zip(boxes, classes, scores):
                                    cls_id = int(cls_id)
                                    if 0 <= cls_id < len(class_names):
                                        label = class_names[cls_id]
                                    else:
                                        label = f"class_{cls_id}"
                                    if hasattr(box, "tolist"):
                                        box_coords = box.tolist()
                                    else:
                                        box_coords = [float(b) for b in box]
                                    detection_overlays.append({
                                        "box": box_coords,
                                        "label": label,
                                        "score": float(score)
                                    })
                            
                            # 存储到infos中供可视化使用
                            infos[e]['detected_classes'] = list(class_counts.keys())
                            infos[e]['class_counts'] = class_counts
                            infos[e]['class_avg_scores'] = class_avg_scores
                            infos[e]['detection_overlays'] = detection_overlays
                        else:
                            infos[e]['detected_classes'] = []
                            infos[e]['class_counts'] = {}
                            infos[e]['class_avg_scores'] = {}
                            infos[e]['detection_overlays'] = []
                    
                    loop_detection_this_step = args.use_loop_detection and \
                        (total_num_steps % max(1, args.loop_interval) == 0)
                    # 延迟初始化：只在真正需要回环检测时才创建extractor
                    if args.use_loop_detection and loop_vlad_extractor is None and loop_detection_this_step:
                        print("[Loop] 初始化语义增强 NetVLAD 提取器...")
                        loop_vlad_extractor = SemanticVLADExtractor(
                            num_semantic_classes=semantic_detector.num_classes,
                            device=device,
                            lazy_load=True  # 延迟加载模型
                        )
                    if args.use_loop_detection and loop_detection_this_step and loop_detector is None:
                        # 先等会在循环内通过首个描述子初始化
                        pass

                    # 调试信息：每100步打印一次检测统计
                    if total_num_steps % 100 == 0:
                        print(f"[Semantic Detection] Step {total_num_steps}: "
                              f"Total detections={total_detections}, "
                              f"conf_thresh={conf_thresh:.2f}, "
                              f"interval={args.semantic_interval}")
                    
                    # Update per scene
                    for e in range(num_scenes):
                        semantic_map2d.update_with_detections(
                            scene_idx=e,
                            detections=det_results[e],
                            local_map=local_map[e],
                            local_pose=local_pose[e],
                            lmb_e=lmb[e],
                            full_pose=None  # 暂时不使用，使用局部位姿即可
                        )
                        if args.use_loop_detection:
                            infos[e]['loop_detected'] = False
                            infos[e]['loop_match'] = None

                        if args.use_loop_detection and loop_detection_this_step:
                            rgb_tensor = obs[e].detach()
                            # 使用轻量级模式（如果启用）可以显著提升速度
                            lightweight = hasattr(args, 'loop_use_lightweight') and args.loop_use_lightweight
                            desc = loop_vlad_extractor.encode(rgb_tensor, det_results[e], lightweight=lightweight)
                            if loop_detector is None:
                                loop_detector = LoopDetector(
                                    descriptor_dim=desc.shape[0],
                                    semantic_dim=semantic_detector.num_classes,
                                    sim_thresh=args.loop_sim_thresh,
                                    semantic_thresh=args.loop_sem_thresh,
                                    min_step_gap=args.loop_min_gap,
                                    top_k=args.loop_top_k
                                )
                            pose_np = full_pose[e].detach().cpu().numpy()
                            global_step = total_num_steps * num_scenes + e
                            match = loop_detector.detect_loop(desc, pose_np, global_step)
                            if match is not None:
                                infos[e]['loop_detected'] = True
                                infos[e]['loop_match'] = {
                                    'matched_step': match.matched_step,
                                    'current_step': match.current_step,
                                    'distance': match.distance,
                                    'semantic_sim': match.semantic_sim
                                }
                                if loop_last_detection_step[e] != match.current_step:
                                    loop_last_detection_step[e] = match.current_step
                                    print(f"[Loop] Env {e} step {match.current_step} "
                                          f"matched {match.matched_step} "
                                          f"(sim={match.distance:.3f}, sem={match.semantic_sim:.3f})")
                            loop_detector.add_keyframe(desc, pose_np, global_step)
                    # Optional visualization
                    if args.print_images:
                        for e in range(num_scenes):
                            semantic_map2d.save_visualizations(total_num_steps * num_scenes, e)
                except Exception as _:
                    pass
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Global Policy
            if l_step == args.num_local_steps - 1:
                # For every global step, update the full and local maps
                for e in range(num_scenes):
                    full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                        local_map[e]
                    full_pose[e] = local_pose[e] + \
                                   torch.from_numpy(origins[e]).to(device).float()

                    locs = full_pose[e].cpu().numpy()
                    r, c = locs[1], locs[0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                    int(c * 100.0 / args.map_resolution)]

                    lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                      (local_w, local_h),
                                                      (full_w, full_h))

                    planner_pose_inputs[e, 3:] = lmb[e]
                    origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                                  lmb[e][0] * args.map_resolution / 100.0, 0.]

                    local_map[e] = full_map[e, :,
                                   lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
                    local_pose[e] = full_pose[e] - \
                                    torch.from_numpy(origins[e]).to(device).float()

                locs = local_pose.cpu().numpy()
                for e in range(num_scenes):
                    global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
                global_input[:, 0:4, :, :] = local_map
                global_input[:, 4:, :, :] = \
                    nn.MaxPool2d(args.global_downscaling)(full_map)

                if False:
                    for i in range(4):
                        ax[i].clear()
                        ax[i].set_yticks([])
                        ax[i].set_xticks([])
                        ax[i].set_yticklabels([])
                        ax[i].set_xticklabels([])
                        ax[i].imshow(global_input.cpu().numpy()[0, 4 + i])
                    plt.gcf().canvas.flush_events()
                    # plt.pause(0.1)
                    fig.canvas.start_event_loop(0.001)
                    plt.gcf().canvas.flush_events()

                # Get exploration reward and metrics
                g_reward = torch.from_numpy(np.asarray(
                    [infos[env_idx]['exp_reward'] for env_idx
                     in range(num_scenes)])
                ).float().to(device)

                if args.eval:
                    g_reward = g_reward*50.0 # Convert reward to area in m2

                g_process_rewards += g_reward.cpu().numpy()
                g_total_rewards = g_process_rewards * \
                                  (1 - g_masks.cpu().numpy())
                g_process_rewards *= g_masks.cpu().numpy()
                per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

                if np.sum(g_total_rewards) != 0:
                    for tr in g_total_rewards:
                        g_episode_rewards.append(tr) if tr != 0 else None

                if args.eval:
                    exp_ratio = torch.from_numpy(np.asarray(
                        [infos[env_idx]['exp_ratio'] for env_idx
                         in range(num_scenes)])
                    ).float()

                    for e in range(num_scenes):
                        explored_area_log[e, ep_num, eval_g_step - 1] = \
                            explored_area_log[e, ep_num, eval_g_step - 2] + \
                            g_reward[e].cpu().numpy()
                        explored_ratio_log[e, ep_num, eval_g_step - 1] = \
                            explored_ratio_log[e, ep_num, eval_g_step - 2] + \
                            exp_ratio[e].cpu().numpy()

                # Add samples to global policy storage
                g_rollouts.insert(
                    global_input, g_rec_states,
                    g_action, g_action_log_prob, g_value,
                    g_reward, g_masks, global_orientation
                )

                # Sample long-term goal from global policy
                g_value, g_action, g_action_log_prob, g_rec_states = \
                    g_policy.act(
                        g_rollouts.obs[g_step + 1],
                        g_rollouts.rec_states[g_step + 1],
                        g_rollouts.masks[g_step + 1],
                        extras=g_rollouts.extras[g_step + 1],
                        deterministic=False
                    )
                cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
                global_goals = [[int(action[0] * local_w),
                                 int(action[1] * local_h)]
                                for action in cpu_actions]

                g_reward = 0
                g_masks = torch.ones(num_scenes).float().to(device)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Get short term goal
            planner_inputs = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(planner_inputs):
                p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                p_input['pose_pred'] = planner_pose_inputs[e]
                p_input['goal'] = global_goals[e]
                # 添加语义密度图（如果启用）- 使用全局地图的对应窗口
                if args.use_semantic:
                    try:
                        # 获取全局语义密度图的对应窗口区域
                        gx1, gx2, gy1, gy2 = int(planner_pose_inputs[e][3]), int(planner_pose_inputs[e][4]), \
                                             int(planner_pose_inputs[e][5]), int(planner_pose_inputs[e][6])
                        semantic_density = semantic_map2d.get_full_density_window(e, gx1, gx2, gy1, gy2)
                        p_input['semantic_density'] = semantic_density
                    except Exception:
                        p_input['semantic_density'] = None
                else:
                    p_input['semantic_density'] = None

            output = envs.get_short_term_goal(planner_inputs)
            # ------------------------------------------------------------------

            ### TRAINING
            torch.set_grad_enabled(True)
            # ------------------------------------------------------------------
            # Train Neural SLAM Module
            if args.train_slam and len(slam_memory) > args.slam_batch_size:
                for _ in range(args.slam_iterations):
                    inputs, outputs = slam_memory.sample(args.slam_batch_size)
                    b_obs_last, b_obs, b_poses = inputs
                    gt_fp_projs, gt_fp_explored, gt_pose_err = outputs

                    b_obs = b_obs.to(device)
                    b_obs_last = b_obs_last.to(device)
                    b_poses = b_poses.to(device)

                    gt_fp_projs = gt_fp_projs.to(device)
                    gt_fp_explored = gt_fp_explored.to(device)
                    gt_pose_err = gt_pose_err.to(device)

                    b_proj_pred, b_fp_exp_pred, _, _, b_pose_err_pred, _ = \
                        nslam_module(b_obs_last, b_obs, b_poses,
                                     None, None, None,
                                     build_maps=False)
                    loss = 0
                    if args.proj_loss_coeff > 0:
                        proj_loss = F.binary_cross_entropy(b_proj_pred,
                                                           gt_fp_projs)
                        costs.append(proj_loss.item())
                        loss += args.proj_loss_coeff * proj_loss

                    if args.exp_loss_coeff > 0:
                        exp_loss = F.binary_cross_entropy(b_fp_exp_pred,
                                                          gt_fp_explored)
                        exp_costs.append(exp_loss.item())
                        loss += args.exp_loss_coeff * exp_loss

                    if args.pose_loss_coeff > 0:
                        pose_loss = torch.nn.MSELoss()(b_pose_err_pred,
                                                       gt_pose_err)
                        pose_costs.append(args.pose_loss_coeff *
                                          pose_loss.item())
                        loss += args.pose_loss_coeff * pose_loss

                    if args.train_slam:
                        slam_optimizer.zero_grad()
                        loss.backward()
                        slam_optimizer.step()

                    del b_obs_last, b_obs, b_poses
                    del gt_fp_projs, gt_fp_explored, gt_pose_err
                    del b_proj_pred, b_fp_exp_pred, b_pose_err_pred

            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Train Local Policy
            if (l_step + 1) % args.local_policy_update_freq == 0 \
                    and args.train_local:
                local_optimizer.zero_grad()
                policy_loss.backward()
                local_optimizer.step()
                l_action_losses.append(policy_loss.item())
                policy_loss = 0
                local_rec_states = local_rec_states.detach_()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Train Global Policy
            if g_step % args.num_global_steps == args.num_global_steps - 1 \
                    and l_step == args.num_local_steps - 1:
                if args.train_global:
                    g_next_value = g_policy.get_value(
                        g_rollouts.obs[-1],
                        g_rollouts.rec_states[-1],
                        g_rollouts.masks[-1],
                        extras=g_rollouts.extras[-1]
                    ).detach()

                    g_rollouts.compute_returns(g_next_value, args.use_gae,
                                               args.gamma, args.tau)
                    g_value_loss, g_action_loss, g_dist_entropy = \
                        g_agent.update(g_rollouts)
                    g_value_losses.append(g_value_loss)
                    g_action_losses.append(g_action_loss)
                    g_dist_entropies.append(g_dist_entropy)
                g_rollouts.after_update()
            # ------------------------------------------------------------------

            # Finish Training
            torch.set_grad_enabled(False)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Logging
            if total_num_steps % args.log_interval == 0:
                end = time.time()
                time_elapsed = time.gmtime(end - start)
                log = " ".join([
                    "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                    "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                    "num timesteps {},".format(total_num_steps *
                                               num_scenes),
                    "FPS {},".format(int(total_num_steps * num_scenes \
                                         / (end - start)))
                ])

                log += "\n\tRewards:"

                if len(g_episode_rewards) > 0:
                    log += " ".join([
                        " Global step mean/med rew:",
                        "{:.4f}/{:.4f},".format(
                            np.mean(per_step_g_rewards),
                            np.median(per_step_g_rewards)),
                        " Global eps mean/med/min/max eps rew:",
                        "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_episode_rewards),
                            np.median(g_episode_rewards),
                            np.min(g_episode_rewards),
                            np.max(g_episode_rewards))
                    ])

                log += "\n\tLosses:"

                if args.train_local and len(l_action_losses) > 0:
                    log += " ".join([
                        " Local Loss:",
                        "{:.3f},".format(
                            np.mean(l_action_losses))
                    ])

                if args.train_global and len(g_value_losses) > 0:
                    log += " ".join([
                        " Global Loss value/action/dist:",
                        "{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_value_losses),
                            np.mean(g_action_losses),
                            np.mean(g_dist_entropies))
                    ])

                if args.train_slam and len(costs) > 0:
                    log += " ".join([
                        " SLAM Loss proj/exp/pose:"
                        "{:.4f}/{:.4f}/{:.4f}".format(
                            np.mean(costs),
                            np.mean(exp_costs),
                            np.mean(pose_costs))
                    ])

                if args.use_semantic:
                    # 收集所有有效的语义奖励值（包括0.0，因为0.0也是有效值）
                    sem_vals = [info.get('sem_reward', 0.0) for info in infos if 'sem_reward' in info]
                    if len(sem_vals) > 0:
                        # 计算平均值，即使有些值是0.0也要显示
                        sem_mean = np.mean(sem_vals)
                        log += " ".join([
                            " Semantic Reward:",
                            "{:.4f},".format(sem_mean)
                        ])
                    
                    # 输出检测到的语义类别统计
                    all_detected = []
                    for info in infos:
                        detected = info.get('detected_classes', [])
                        if detected:
                            all_detected.extend(detected)
                    if all_detected:
                        from collections import Counter
                        class_freq = Counter(all_detected)
                        log += " Detected classes: " + ", ".join([f"{name}({count})" for name, count in class_freq.most_common(5)])
                    
                    # 结构奖励（若可用）
                    if hasattr(args, 'structural_reward_coeff') and args.structural_reward_coeff > 0:
                        # exploration_env 已将结构奖励并入 total reward，不单列
                        pass

                print(log)
                logging.info(log)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Save best models
            if (total_num_steps * num_scenes) % args.save_interval < \
                    num_scenes:

                # Save Neural SLAM Model
                if len(costs) >= 1000 and np.mean(costs) < best_cost \
                        and not args.eval:
                    best_cost = np.mean(costs)
                    torch.save(nslam_module.state_dict(),
                               os.path.join(log_dir, "model_best.slam"))

                # Save Local Policy Model
                if len(l_action_losses) >= 100 and \
                        (np.mean(l_action_losses) <= best_local_loss) \
                        and not args.eval:
                    torch.save(l_policy.state_dict(),
                               os.path.join(log_dir, "model_best.local"))

                    best_local_loss = np.mean(l_action_losses)

                # Save Global Policy Model
                if len(g_episode_rewards) >= 100 and \
                        (np.mean(g_episode_rewards) >= best_g_reward) \
                        and not args.eval:
                    torch.save(g_policy.state_dict(),
                               os.path.join(log_dir, "model_best.global"))
                    best_g_reward = np.mean(g_episode_rewards)

            # Save periodic models
            if (total_num_steps * num_scenes) % args.save_periodic < \
                    num_scenes:
                step = total_num_steps * num_scenes
                if args.train_slam:
                    torch.save(nslam_module.state_dict(),
                               os.path.join(dump_dir,
                                            "periodic_{}.slam".format(step)))
                if args.train_local:
                    torch.save(l_policy.state_dict(),
                               os.path.join(dump_dir,
                                            "periodic_{}.local".format(step)))
                if args.train_global:
                    torch.save(g_policy.state_dict(),
                               os.path.join(dump_dir,
                                            "periodic_{}.global".format(step)))
            # ------------------------------------------------------------------

    # Print and save model performance numbers during evaluation
    if args.eval:
        logfile = open("{}/explored_area.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_area_log[e].shape[0]):
                logfile.write(str(explored_area_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        logfile = open("{}/explored_ratio.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_ratio_log[e].shape[0]):
                logfile.write(str(explored_ratio_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        log = "Final Exp Area: \n"
        for i in range(explored_area_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_area_log[:, :, i]))

        log += "\nFinal Exp Ratio: \n"
        for i in range(explored_ratio_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_ratio_log[:, :, i]))

        print(log)
        logging.info(log)


if __name__ == "__main__":
    main()
