import math
import os
import pickle
import subprocess
import sys
import time

import gym
import matplotlib
import numpy as np
import quaternion
import skimage.morphology
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms


# if sys.platform == 'darwin':
#     matplotlib.use("tkagg")
# else:
#     matplotlib.use('Agg')
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')

import habitat
from habitat import logger


def _get_sim_actions():
    try:
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        return HabitatSimActions
    except ImportError:
        return habitat.SimulatorActions


def _scene_name_from_sim(habitat_env):
    sim = habitat_env.sim
    if hasattr(sim, "config") and hasattr(sim.config, "SCENE"):
        return sim.config.SCENE
    if hasattr(sim, "curr_scene_name"):
        return sim.curr_scene_name
    ep = habitat_env.current_episode
    if ep is not None:
        return ep.scene_id
    return "unknown"


from env.utils.map_builder import MapBuilder
from env.utils.fmm_planner import FMMPlanner

from env.habitat.utils.noisy_actions import CustomActionSpaceConfiguration
import env.habitat.utils.pose as pu
import env.habitat.utils.visualizations as vu
from env.habitat.utils.supervision import HabitatMaps

from model import get_grid


def _preprocess_depth(depth):
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

    mask1 = depth == 0
    depth[mask1] = np.NaN
    depth = depth*1000.
    return depth


class Exploration_Env(habitat.RLEnv):

    _live_viewer_proc = None

    @property
    def original_action_space(self):
        """Habitat 2 VectorEnv 初始化时会查询该属性。"""
        return self.action_space

    @property
    def original_action_space(self):
        """Habitat 2 VectorEnv 初始化时会查询该属性。"""
        return self.action_space

    @classmethod
    def _start_live_viewer(cls, user_display, live_dir):
        if cls._live_viewer_proc is not None and cls._live_viewer_proc.poll() is None:
            return
        os.makedirs(live_dir, exist_ok=True)
        log_path = os.path.join(live_dir, "viewer.log")
        try:
            open(log_path, "w", encoding="utf-8").close()
        except OSError:
            pass
        script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scripts/nso_live_viewer.py"))
        env = os.environ.copy()
        env["DISPLAY"] = user_display
        env["NSO_VIEWER_PROCESS"] = "1"
        env.pop("__GLX_VENDOR_LIBRARY_NAME", None)
        env.pop("NSO_USE_XVFB_GPU", None)
        env.pop("MPLBACKEND", None)
        log_f = open(log_path, "a", encoding="utf-8")
        cls._live_viewer_proc = subprocess.Popen(
            [sys.executable, script, live_dir],
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(1.2)
        if cls._live_viewer_proc.poll() is not None:
            log_f.close()
            try:
                with open(log_path, encoding="utf-8") as f:
                    tail = f.read()[-800:]
            except OSError:
                tail = ""
            print("[可视化] 查看器启动失败，详见 {}:\n{}".format(
                log_path, tail or "(无日志)"))
            cls._live_viewer_proc = None
        else:
            print("[可视化] 查看器进程 pid={}，日志 {}".format(
                cls._live_viewer_proc.pid, log_path))

    def __init__(self, args, rank, config_env, config_baseline, dataset):
        self.figure = None
        self.ax = None

        # NSO Live 帧导出（仅 NSO_LIVE_VIS=1）；默认 NSO_VIS_NATIVE 走原始 TkAgg 窗口
        self._native_x11 = os.environ.get("NSO_VIS_NATIVE") == "1"
        self._user_x11_display = os.environ.get("NSO_X11_DISPLAY")
        self._sim_display = os.environ.get("DISPLAY")
        self._use_live_viewer = bool(
            args.visualize
            and not self._native_x11
            and os.environ.get("NSO_LIVE_VIS") == "1"
            and self._user_x11_display)

        self._live_fast_vis = False
        if args.print_images or args.visualize:
            if self._use_live_viewer:
                self._live_vis_dir = os.environ.get(
                    "NSO_LIVE_VIS_DIR", "/tmp/nso_vis_live")
                os.makedirs(self._live_vis_dir, exist_ok=True)
                os.environ["NSO_LIVE_VIS_DIR"] = self._live_vis_dir
                from env.habitat.utils import visualizations as _vu
                self._live_fast_vis = (
                    not args.print_images and _vu.live_vis_fast_enabled())
            need_mpl_figure = args.print_images or (
                args.visualize and not self._live_fast_vis)
            if need_mpl_figure:
                import matplotlib
                if self._native_x11 or (
                        args.visualize and not self._use_live_viewer):
                    matplotlib.use("TkAgg", force=True)
                else:
                    matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as plt
                self._plt = plt
                if args.visualize and (
                        self._native_x11 or not self._use_live_viewer):
                    plt.ion()
                self.figure, self.ax = plt.subplots(
                    1, 2, figsize=(6 * 16 / 9, 6),
                    facecolor="whitesmoke",
                    num="Thread {}".format(rank))
                if args.visualize and (
                        self._native_x11 or not self._use_live_viewer):
                    try:
                        self.figure.show()
                    except Exception:
                        pass
            if self._native_x11 and args.visualize and rank == 0:
                print("[可视化] 原始 matplotlib 窗口 (TkAgg, DISPLAY={})".format(
                    os.environ.get("DISPLAY", "")))
            elif self._use_live_viewer and rank == 0:
                mode = "OpenCV 快速" if self._live_fast_vis else "matplotlib"
                print("[可视化] NSO Live 帧导出（{}，DISPLAY={}）".format(
                    mode, self._user_x11_display))

        self.args = args
        self.num_actions = 3
        self.dt = 10

        self.rank = rank

        self.sensor_noise_fwd = \
                pickle.load(open("noise_models/sensor_noise_fwd.pkl", 'rb'))
        self.sensor_noise_right = \
                pickle.load(open("noise_models/sensor_noise_right.pkl", 'rb'))
        self.sensor_noise_left = \
                pickle.load(open("noise_models/sensor_noise_left.pkl", 'rb'))

        sim_actions = _get_sim_actions()
        sim_actions.extend_action_space("NOISY_FORWARD")
        sim_actions.extend_action_space("NOISY_RIGHT")
        sim_actions.extend_action_space("NOISY_LEFT")

        if hasattr(config_env, "defrost"):
            config_env.defrost()
            config_env.SIMULATOR.ACTION_SPACE_CONFIG = (
                "CustomActionSpaceConfiguration"
            )
            config_env.freeze()
        elif hasattr(config_env, "habitat"):
            from habitat.config import read_write
            with read_write(config_env):
                config_env.habitat.simulator.action_space_config = (
                    "CustomActionSpaceConfiguration"
                )

        # xvfb+GPU 模式：恢复 sim 的 DISPLAY，避免与 X11 转发混用
        if (not self._native_x11 and self._user_x11_display
                and self._sim_display):
            os.environ["DISPLAY"] = self._sim_display
        if os.environ.get("NSO_USE_XVFB_GPU") and os.uname().sysname == "Linux":
            if os.system("command -v nvidia-smi >/dev/null 2>&1") == 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "0")
                os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

        super().__init__(config_env, dataset)

        if self._use_live_viewer and rank == 0:
            if os.environ.get("NSO_VIEWER_EXTERNAL"):
                pid = os.environ.get("NSO_VIEWER_PID", "")
                alive = pid.isdigit() and os.path.exists("/proc/{}".format(pid))
                if alive:
                    print("[可视化] 使用 SSH 会话查看器 pid={}".format(pid))
                else:
                    print("[可视化] 警告: 外部查看器未运行，请重新 bash scripts/run_nso_vis.sh")
            else:
                self._start_live_viewer(
                    self._user_x11_display, self._live_vis_dir)

        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, args.frame_height,
                                                    args.frame_width),
                                                dtype='uint8')

        self.mapper = self.build_mapper()

        self.episode_no = 0

        self.res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((args.frame_height, args.frame_width),
                                      interpolation = Image.NEAREST)])
        self.scene_name = None
        self.maps_dict = {}
        self.semantic_bonus_acc = 0.0
        self.structural_bonus_acc = 0.0
        self.frontier_bonus_acc = 0.0
        self._last_semantic_density = None
        self._last_fresh_sem = None
        self._last_door_map = None
        self._last_frontier_map = None
        self._last_global_goal = None
        self._last_intrinsic_val = 0.0
        self._pending_global_goal = None
        self._pending_goal_start_rc = None
        self._last_path_unreachable = False
        self._last_embodied_success = None

    def randomize_env(self):
        self._env._episode_iterator._shuffle_iterator()

    def save_trajectory_data(self):
        if "replica" in self.scene_name:
            folder = self.args.save_trajectory_data + "/" + \
                        self.scene_name.split("/")[-3]+"/"
        else:
            folder = self.args.save_trajectory_data + "/" + \
                        self.scene_name.split("/")[-1].split(".")[0]+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = folder+str(self.episode_no)+".txt"
        with open(filepath, "w+") as f:
            f.write(self.scene_name+"\n")
            for state in self.trajectory_states:
                f.write(str(state)+"\n")
            f.flush()

    def save_position(self):
        self.agent_state = self._env.sim.get_agent_state()
        self.trajectory_states.append([self.agent_state.position,
                                       self.agent_state.rotation])


    def reset(self):
        args = self.args
        self.episode_no += 1
        self.timestep = 0
        self._previous_action = None
        self.trajectory_states = []
        self.semantic_bonus_acc = 0.0
        self.structural_bonus_acc = 0.0
        self.frontier_bonus_acc = 0.0
        self._last_sem_reward = 0.0
        self._last_area_reward = 0.0
        self._last_semantic_density = None
        self._last_fresh_sem = None
        self._last_door_map = None
        self._last_frontier_map = None
        self._last_global_goal = None
        self._last_intrinsic_val = 0.0
        self._pending_global_goal = None
        self._pending_goal_start_rc = None
        self._last_path_unreachable = False
        self._last_embodied_success = None

        if args.randomize_env_every > 0:
            if np.mod(self.episode_no, args.randomize_env_every) == 0:
                self.randomize_env()

        # Get Ground Truth Map
        self.explorable_map = None
        while self.explorable_map is None:
            obs = super().reset()
            full_map_size = args.map_size_cm//args.map_resolution
            self.explorable_map = self._get_gt_map(full_map_size)
        self.prev_explored_area = 0.

        # Preprocess observations
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        if self.args.frame_width != self.args.env_frame_width:
            rgb = np.asarray(self.res(rgb))
        state = rgb.transpose(2, 0, 1)
        depth = _preprocess_depth(obs['depth'])

        # Initialize map and pose
        self.map_size_cm = args.map_size_cm
        self.mapper.reset_map(self.map_size_cm)
        self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]
        self.curr_loc_gt = self.curr_loc
        self.last_loc_gt = self.curr_loc_gt
        self.last_loc = self.curr_loc
        self.last_sim_location = self.get_sim_location()

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))

        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
            self.mapper.update_map(depth, mapper_gt_pose)

        # Initialize variables
        self.scene_name = _scene_name_from_sim(self.habitat_env)
        self.visited = np.zeros(self.map.shape)
        self.visited_vis = np.zeros(self.map.shape)
        self.visited_gt = np.zeros(self.map.shape)
        self.collison_map = np.zeros(self.map.shape)
        self.col_width = 1

        # Set info
        self.info = {
            'time': self.timestep,
            'fp_proj': fp_proj,
            'fp_explored': fp_explored,
            'sensor_pose': [0., 0., 0.],
            'pose_err': [0., 0., 0.],
            'exp_reward': None,
            'exp_ratio': None,
            'sem_reward': 0.0,
            'area_reward': 0.0,
            'loop_detected': False,
            'loop_match': None,
            'detected_classes': [],
            'class_counts': {},
            'class_avg_scores': {},
            'detection_overlays': [],
            'path_unreachable': False,
            'embodied_goal_success': None,
        }

        self.save_position()

        return state, self.info

    def step(self, action):

        args = self.args
        self.timestep += 1

        self.info['loop_detected'] = False
        self.info['loop_match'] = None

        # Action remapping
        if action == 2: # Forward
            action = 1
            noisy_action = _get_sim_actions().NOISY_FORWARD
        elif action == 1: # Right
            action = 3
            noisy_action = _get_sim_actions().NOISY_RIGHT
        elif action == 0: # Left
            action = 2
            noisy_action = _get_sim_actions().NOISY_LEFT

        self.last_loc = np.copy(self.curr_loc)
        self.last_loc_gt = np.copy(self.curr_loc_gt)
        self._previous_action = action

        if args.noisy_actions:
            obs, rew, done, info = super().step(noisy_action)
        else:
            obs, rew, done, info = super().step(action)

        # Preprocess observations
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        if self.args.frame_width != self.args.env_frame_width:
            rgb = np.asarray(self.res(rgb))

        state = rgb.transpose(2, 0, 1)

        depth = _preprocess_depth(obs['depth'])

        # Get base sensor and ground-truth pose
        dx_gt, dy_gt, do_gt = self.get_gt_pose_change()
        dx_base, dy_base, do_base = self.get_base_pose_change(
                                        action, (dx_gt, dy_gt, do_gt))

        self.curr_loc = pu.get_new_pose(self.curr_loc,
                               (dx_base, dy_base, do_base))

        self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt,
                               (dx_gt, dy_gt, do_gt))

        if not args.noisy_odometry:
            self.curr_loc = self.curr_loc_gt
            dx_base, dy_base, do_base = dx_gt, dy_gt, do_gt

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))


        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
                self.mapper.update_map(depth, mapper_gt_pose)


        # Update collision map
        if action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, t2 = self.curr_loc
            if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                self.col_width = min(self.col_width, 9)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold: #Collision
                length = 2
                width = self.col_width
                buf = 3
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
                                        (j-width//2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
                                        (j-width//2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r*100/args.map_resolution), \
                               int(c*100/args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                    self.collison_map.shape)
                        self.collison_map[r,c] = 1

        # Set info
        self.info['time'] = self.timestep
        self.info['fp_proj'] = fp_proj
        self.info['fp_explored']= fp_explored
        self.info['sensor_pose'] = [dx_base, dy_base, do_base]
        self.info['pose_err'] = [dx_gt - dx_base,
                                 dy_gt - dy_base,
                                 do_gt - do_base]


        if self.timestep%args.num_local_steps==0:
            total_reward, ratio, sem_bonus, area_reward = self.get_global_reward()
            self.info['exp_reward'] = total_reward
            self.info['exp_ratio'] = ratio
            self.info['sem_reward'] = sem_bonus
            self.info['area_reward'] = area_reward
            # 保存当前的语义奖励值，以便在非全局奖励步骤时也能显示
            self._last_sem_reward = sem_bonus
            self._last_area_reward = area_reward
        else:
            self.info['exp_reward'] = None
            self.info['exp_ratio'] = None
            # 保留上一次的语义奖励值，而不是设置为0.0
            # 这样在日志记录时就能看到正确的语义奖励值
            if hasattr(self, '_last_sem_reward'):
                self.info['sem_reward'] = self._last_sem_reward
            else:
                self.info['sem_reward'] = 0.0
            # area_reward 也保留上一次的值
            if hasattr(self, '_last_area_reward'):
                self.info['area_reward'] = self._last_area_reward
            else:
                self.info['area_reward'] = 0.0
            # 保持语义检测信息（如果存在）
            if 'detected_classes' not in self.info:
                self.info['detected_classes'] = []
                self.info['class_counts'] = {}
                self.info['class_avg_scores'] = {}
                self.info['detection_overlays'] = []

        self.save_position()

        if self.info['time'] >= args.max_episode_length:
            done = True
            if self.args.save_trajectory_data != "0":
                self.save_trajectory_data()
        else:
            done = False

        return state, rew, done, self.info

    def get_reward_range(self):
        # This function is not used, Habitat-RLEnv requires this function
        return (0., 1.0)

    def get_reward(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        return 0.

    def get_global_reward(self):
        curr_explored = self.explored_map*self.explorable_map
        curr_explored_area = curr_explored.sum()

        reward_scale = self.explorable_map.sum()
        m_reward = (curr_explored_area - self.prev_explored_area)*1.
        m_ratio = m_reward/reward_scale
        m_reward = m_reward * 25./10000. # converting to m^2
        self.prev_explored_area = curr_explored_area

        m_reward *= 0.02 # Reward Scaling

        semantic_bonus = self.semantic_bonus_acc * self.args.semantic_reward_coeff
        structural_bonus = self.structural_bonus_acc
        frontier_bonus = self.frontier_bonus_acc * self.args.frontier_reward_coeff
        
        # 调试信息：每100步打印一次最终奖励值
        if self.timestep % 100 == 0 and self.args.use_semantic:
            print(f"[Semantic Reward Final] Step {self.timestep}: "
                  f"semantic_bonus_acc={self.semantic_bonus_acc:.6f}, "
                  f"semantic_bonus={semantic_bonus:.6f}, "
                  f"coeff={self.args.semantic_reward_coeff:.4f}")
        
        self.semantic_bonus_acc = 0.0
        self.structural_bonus_acc = 0.0
        self.frontier_bonus_acc = 0.0

        # 论文式 (4)：R_total = R_map + λ_sem R_sem + λ_struct R_struct + λ_front R_front
        use_paper = bool(getattr(self.args, 'paper_rewards', 0))
        struct_coeff = self.args.structural_reward_coeff
        if not (getattr(self.args, 'use_structural_reward', 0) or use_paper):
            structural_bonus = 0.0
            frontier_bonus = 0.0
        if not self.args.use_semantic:
            semantic_bonus = 0.0

        intrinsic_penalty = 0.0
        if getattr(self.args, 'use_intrinsic_goal_penalty', 0) or use_paper:
            coeff = getattr(self.args, 'intrinsic_reward_coeff', 0.05)
            intrinsic_penalty = coeff * float(self._last_intrinsic_val)

        total_reward = (m_reward + semantic_bonus
                        + struct_coeff * structural_bonus
                        + frontier_bonus
                        + intrinsic_penalty)

        return total_reward, m_ratio, semantic_bonus, m_reward

    def get_done(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        return False

    def get_info(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        info = {}
        return info

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def get_spaces(self):
        return self.observation_space, self.action_space

    def build_mapper(self):
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['fov'] =  self.args.hfov
        params['resolution'] = self.args.map_resolution
        params['map_size_cm'] = self.args.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.args.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['visualize'] = self.args.visualize
        params['obs_threshold'] = self.args.obs_threshold
        self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
                                             self.args.map_resolution)
        mapper = MapBuilder(params)
        return mapper


    def get_sim_location(self):
        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o


    def get_gt_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do


    def get_base_pose_change(self, action, gt_pose_change):
        dx_gt, dy_gt, do_gt = gt_pose_change
        if action == 1: ## Forward
            x_err, y_err, o_err = self.sensor_noise_fwd.sample()[0][0]
        elif action == 3: ## Right
            x_err, y_err, o_err = self.sensor_noise_right.sample()[0][0]
        elif action == 2: ## Left
            x_err, y_err, o_err = self.sensor_noise_left.sample()[0][0]
        else: ##Stop
            x_err, y_err, o_err = 0., 0., 0.

        x_err = x_err * self.args.noise_level
        y_err = y_err * self.args.noise_level
        o_err = o_err * self.args.noise_level
        return dx_gt + x_err, dy_gt + y_err, do_gt + np.deg2rad(o_err)


    def get_short_term_goal(self, inputs):

        args = self.args

        # Get Map prediction
        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        semantic_density = inputs.get('semantic_density')
        self._last_semantic_density = semantic_density
        self._last_fresh_sem = None
        
        # 存储语义密度图用于可视化
        if args.use_semantic and 'semantic_density' in inputs:
            self._last_semantic_density = inputs.get('semantic_density', None)

        grid = np.rint(map_pred)
        explored = np.rint(exp_pred)

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0/args.map_resolution - gx1),
                      int(c * 100.0/args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, grid.shape)

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0/args.map_resolution - gx1),
                 int(c * 100.0/args.map_resolution - gy1)]
        start = pu.threshold_poses(start, grid.shape)
        #TODO: try reducing this

        self.visited[gx1:gx2, gy1:gy2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1

        steps = 25
        for i in range(steps):
            x = int(last_start[0] + (start[0] - last_start[0]) * (i+1) / steps)
            y = int(last_start[1] + (start[1] - last_start[1]) * (i+1) / steps)
            self.visited_vis[gx1:gx2, gy1:gy2][x, y] = 1

        # Get last loc ground truth pose
        last_start_x, last_start_y = self.last_loc_gt[0], self.last_loc_gt[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0/args.map_resolution),
                      int(c * 100.0/args.map_resolution)]
        last_start = pu.threshold_poses(last_start, self.visited_gt.shape)

        # Get ground truth pose
        start_x_gt, start_y_gt, start_o_gt = self.curr_loc_gt
        r, c = start_y_gt, start_x_gt
        start_gt = [int(r * 100.0/args.map_resolution),
                    int(c * 100.0/args.map_resolution)]
        start_gt = pu.threshold_poses(start_gt, self.visited_gt.shape)
        #self.visited_gt[start_gt[0], start_gt[1]] = 1

        steps = 25
        for i in range(steps):
            x = int(last_start[0] + (start_gt[0] - last_start[0]) * (i+1) / steps)
            y = int(last_start[1] + (start_gt[1] - last_start[1]) * (i+1) / steps)
            self.visited_gt[x, y] = 1


        use_structural = (getattr(args, 'use_structural_reward', 0)
                          or getattr(args, 'paper_rewards', 0))

        # 语义新鲜度奖励 R_sem（论文式 6）
        semantic_bonus = 0.0
        if args.use_semantic and semantic_density is not None:
            semantic_density = semantic_density.astype(np.float32)
            if semantic_density.shape == (gx2 - gx1, gy2 - gy1):
                semantic_density = np.maximum(semantic_density, 0.0)
                visited_window = self.visited_vis[gx1:gx2, gy1:gy2].astype(np.float32)
                observed_window = self.explored_map[gx1:gx2, gy1:gy2].astype(np.float32)
                fresh_mask = np.clip(observed_window - visited_window, 0.0, 1.0)
                fresh_sem = semantic_density * fresh_mask
                sem_max = np.max(semantic_density)
                sem_sum = np.sum(semantic_density)
                if sem_max > 0:
                    fresh_sem_norm = fresh_sem / (sem_max + 1e-6)
                else:
                    fresh_sem_norm = fresh_sem
                active_cells = float(np.count_nonzero(fresh_mask))
                observed_cells = float(np.count_nonzero(observed_window))
                visited_cells = float(np.count_nonzero(visited_window))
                
                # 调试信息：每100步打印一次语义奖励诊断信息
                if self.timestep % 100 == 0 and args.use_semantic:
                    fresh_sem_sum = np.sum(fresh_sem) if active_cells > 0 else 0.0
                    fresh_sem_norm_sum = np.sum(fresh_sem_norm) if active_cells > 0 else 0.0
                    print(f"[Semantic Reward Debug] Step {self.timestep}: "
                          f"sem_max={sem_max:.4f}, sem_sum={sem_sum:.4f}, "
                          f"active_cells={active_cells:.0f}, observed={observed_cells:.0f}, "
                          f"visited={visited_cells:.0f}, fresh_mask_sum={np.sum(fresh_mask):.0f}, "
                          f"fresh_sem_sum={fresh_sem_sum:.4f}, fresh_sem_norm_sum={fresh_sem_norm_sum:.4f}")
                
                if active_cells > 0:
                    # 计算新鲜区域的语义密度平均值（归一化后）
                    fresh_sem_sum = np.sum(fresh_sem_norm)
                    semantic_bonus = float(fresh_sem_sum / (active_cells + 1e-6))
                    
                    # 如果归一化后的值太小，使用原始语义密度的相对值
                    # 这样可以确保即使归一化后值小，也能给予合理的奖励
                    if semantic_bonus < 0.01 and sem_max > 0:
                        # 使用新鲜区域的原始语义密度，按比例缩放
                        fresh_sem_raw_sum = np.sum(fresh_sem)
                        semantic_bonus = float(fresh_sem_raw_sum / (active_cells * sem_max + 1e-6))
                        # 限制在合理范围内
                        semantic_bonus = min(semantic_bonus, 1.0)
                else:
                    # 如果没有新鲜区域，但语义密度不为0，仍然给予少量奖励（鼓励探索有语义的区域）
                    if sem_max > 0:
                        # 使用已观测但可能已访问的区域，给予较小的奖励
                        semantic_bonus = float(sem_sum / (observed_cells * sem_max + 1e-6)) * 0.1
                        semantic_bonus = min(semantic_bonus, 0.1)
                self._last_fresh_sem = fresh_sem_norm
                
                # 调试信息：每100步打印一次语义奖励值
                if self.timestep % 100 == 0 and args.use_semantic:
                    print(f"[Semantic Reward Step] Step {self.timestep}: "
                          f"semantic_bonus={semantic_bonus:.6f}, "
                          f"semantic_bonus_acc={self.semantic_bonus_acc:.6f}")
            else:
                # 形状不匹配时的调试信息
                if self.timestep % 100 == 0 and args.use_semantic:
                    print(f"[Semantic Reward Debug] Step {self.timestep}: "
                          f"Shape mismatch! semantic_density.shape={semantic_density.shape if semantic_density is not None else None}, "
                          f"expected=({gx2 - gx1}, {gy2 - gy1})")
        else:
            # semantic_density为None时的调试信息
            if self.timestep % 100 == 0 and args.use_semantic:
                print(f"[Semantic Reward Debug] Step {self.timestep}: semantic_density is None")
        if args.use_semantic:
            self.semantic_bonus_acc += semantic_bonus

        # 结构内容奖励 R_struct、前沿 R_front（论文式 7 与 3.4.4）
        structural_bonus = 0.0
        structural_map = None
        frontier_bonus = 0.0
        frontier_map = None
        if use_structural:
            try:
                import cv2
                grid_bin = np.rint(map_pred).astype(np.uint8)
                free = (1 - grid_bin).astype(np.uint8)
                observed_window = self.explored_map[gx1:gx2, gy1:gy2].astype(np.float32)
                visited_window = self.visited_vis[gx1:gx2, gy1:gy2].astype(np.float32)
                fresh_mask = np.clip(observed_window - visited_window, 0.0, 1.0)

                left = np.roll(grid_bin, 1, axis=1)
                right = np.roll(grid_bin, -1, axis=1)
                up = np.roll(grid_bin, -1, axis=0)
                down = np.roll(grid_bin, 1, axis=0)
                door_h = (left == 1) & (right == 1) & (free == 1)
                door_v = (up == 1) & (down == 1) & (free == 1)
                door_mask = (door_h | door_v).astype(np.float32)

                dist = cv2.distanceTransform(
                    (free * observed_window).astype(np.uint8), cv2.DIST_L2, 3)
                narrow_thresh = max(1, int(self.args.narrow_width_cells // 2))
                narrow_mask = (dist > 0) & (dist < narrow_thresh * 1.5)
                door_mask_enhanced = np.maximum(
                    door_mask, narrow_mask.astype(np.float32) * 0.5)

                if np.any(door_mask_enhanced > 0):
                    door_dist = cv2.distanceTransform(
                        (1 - door_mask_enhanced).astype(np.uint8), cv2.DIST_L2, 3)
                    door_boost_radius = int(self.args.door_boost_distance)
                    door_proximity = np.clip(
                        1.0 - door_dist / max(door_boost_radius, 1), 0.0, 1.0)
                    door_mask_enhanced = np.maximum(
                        door_mask_enhanced, door_proximity * 0.3)

                door_mask = door_mask_enhanced
                self._last_door_map = door_mask
                narrow_score = np.clip(
                    (narrow_thresh - dist) / max(narrow_thresh, 1), 0.0, 1.0)

                k = max(3, int(self.args.open_kernel) | 1)
                free_float = (free * observed_window).astype(np.float32)
                open_score = cv2.blur(free_float, (k, k))
                if np.max(open_score) > 0:
                    open_score = open_score / (np.max(open_score) + 1e-6)

                structural_map = (
                    self.args.w_struct_door * door_mask
                    + self.args.w_struct_narrow * narrow_score
                    + self.args.w_struct_open * open_score
                ).astype(np.float32)
                structural_map *= fresh_mask

                active_cells_s = float(np.count_nonzero(fresh_mask))
                if active_cells_s > 0:
                    structural_bonus = float(
                        np.sum(structural_map) / (active_cells_s + 1e-6))

                frontier_mask = fresh_mask * free.astype(np.float32)
                if self._last_door_map is not None and np.any(self._last_door_map > 0):
                    door_dist = cv2.distanceTransform(
                        (1 - self._last_door_map).astype(np.uint8), cv2.DIST_L2, 3)
                    door_boost_radius = int(self.args.door_boost_distance * 2)
                    door_proximity = np.clip(
                        1.0 - door_dist / max(door_boost_radius, 1), 0.0, 1.0)
                    frontier_map = frontier_mask * (
                        1.0 + door_proximity * self.args.room_exploration_boost)
                else:
                    frontier_map = frontier_mask

                active_cells_f = float(np.count_nonzero(frontier_mask))
                if active_cells_f > 0:
                    frontier_bonus = float(
                        np.sum(frontier_map) / (active_cells_f + 1e-6))
                self._last_frontier_map = frontier_map
            except Exception:
                structural_map = None
                frontier_map = None
        self.structural_bonus_acc += structural_bonus
        self.frontier_bonus_acc += frontier_bonus

        # Get goal
        goal = inputs['goal']
        goal = pu.threshold_poses(goal, grid.shape)

        # 记录全局目标与内在惩罚（论文：避免选择已探索栅格）
        self._last_global_goal = [int(goal[0]), int(goal[1])]
        self._last_intrinsic_val = float(-exp_pred[goal[0], goal[1]])

        # 具身可达性监督：记录本周期长期目标起点
        if self._pending_global_goal is not None:
            from env.habitat.reachability_utils import embodied_goal_reached
            self._last_embodied_success = embodied_goal_reached(
                (start[0], start[1]),
                tuple(self._pending_global_goal),
                success_radius_cells=max(1, int(50 / args.map_resolution)),
            )
            self.info['embodied_goal_success'] = self._last_embodied_success
        else:
            self._last_embodied_success = None
            self.info['embodied_goal_success'] = None

        self._pending_global_goal = [int(goal[0]), int(goal[1])]
        self._pending_goal_start_rc = (int(start[0]), int(start[1]))

        # Get intrinsic reward for global policy
        # Negative reward for exploring explored areas i.e.
        # for choosing explored cell as long-term goal
        self.extrinsic_rew = -pu.get_l2_distance(10, goal[0], 10, goal[1])
        self.intrinsic_rew = -exp_pred[goal[0], goal[1]]

        # Get short-term goal
        stg, path_unreachable = self._get_stg(
            grid, explored, start, np.copy(goal), planning_window)
        self._last_path_unreachable = path_unreachable
        self.info['path_unreachable'] = path_unreachable

        # Find GT action
        if self.args.eval or not self.args.train_local:
            gt_action = 0
        else:
            gt_action = self._get_gt_action(1 - self.explorable_map, start,
                                            [int(stg[0]), int(stg[1])],
                                            planning_window, start_o)

        (stg_x, stg_y) = stg
        relative_dist = pu.get_l2_distance(stg_x, start[0], stg_y, start[1])
        relative_dist = relative_dist*5./100.
        angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                stg_y - start[1]))
        angle_agent = (start_o)%360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal)%360.0
        if relative_angle > 180:
            relative_angle -= 360

        def discretize(dist):
            dist_limits = [0.25, 3, 10]
            dist_bin_size = [0.05, 0.25, 1.]
            if dist < dist_limits[0]:
                ddist = int(dist/dist_bin_size[0])
            elif dist < dist_limits[1]:
                ddist = int((dist - dist_limits[0])/dist_bin_size[1]) + \
                    int(dist_limits[0]/dist_bin_size[0])
            elif dist < dist_limits[2]:
                ddist = int((dist - dist_limits[1])/dist_bin_size[2]) + \
                    int(dist_limits[0]/dist_bin_size[0]) + \
                    int((dist_limits[1] - dist_limits[0])/dist_bin_size[1])
            else:
                ddist = int(dist_limits[0]/dist_bin_size[0]) + \
                    int((dist_limits[1] - dist_limits[0])/dist_bin_size[1]) + \
                    int((dist_limits[2] - dist_limits[1])/dist_bin_size[2])
            return ddist

        output = np.zeros((args.goals_size + 1))

        output[0] = int((relative_angle%360.)/5.)
        output[1] = discretize(relative_dist)
        output[2] = gt_action

        self.relative_angle = relative_angle

        if args.visualize or args.print_images:
            dump_dir = "{}/dump/{}/".format(args.dump_location,
                                                args.exp_name)
            ep_dir = '{}/episodes/{}/{}/'.format(
                            dump_dir, self.rank+1, self.episode_no)
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)

            if args.vis_type == 1: # Visualize predicted map and pose
                # 获取语义密度图（如果提供）
                semantic_density = None
                if args.use_semantic and hasattr(self, '_last_semantic_density'):
                    semantic_density = self._last_semantic_density
                    # 确保尺寸匹配
                    if semantic_density.shape != (gx2-gx1, gy2-gy1):
                        semantic_density = None

                # 全局图 + 掩码裁切；默认用局部 planning window（与论文原始可视化一致）
                if os.environ.get("NSO_VIS_FULL_MAP", "0") == "1":
                    goal_full = (
                        int(np.clip(gx1 + goal[0], 0, self.map.shape[0] - 1)),
                        int(np.clip(gy1 + goal[1], 0, self.map.shape[1] - 1)),
                    )
                    vis_grid = vu.get_colored_map(
                        np.rint(self.map),
                        self.collison_map,
                        self.visited_vis,
                        self.visited_gt,
                        goal_full,
                        self.explored_map,
                        self.explorable_map,
                        self.map * self.explored_map,
                        semantic_density=None,
                        semantic_freshness=None,
                        structural_map=None,
                    )
                    pos = (
                        self.curr_loc[0] * 100.0 / args.map_resolution,
                        self.curr_loc[1] * 100.0 / args.map_resolution,
                        self.curr_loc[2],
                    )
                    gt_pos = (
                        self.curr_loc_gt[0] * 100.0 / args.map_resolution,
                        self.curr_loc_gt[1] * 100.0 / args.map_resolution,
                        self.curr_loc_gt[2],
                    )
                    if os.environ.get("NSO_VIS_MAP_ZOOM", "1") == "1":
                        activity = (
                            (self.visited_vis > 0)
                            | (self.explored_map > 0)
                            | (np.rint(self.map) > 0)
                        )
                        vis_grid, pos, gt_pos = vu.crop_map_and_poses_from_mask(
                            vis_grid, pos, gt_pos, activity)
                        os.environ["NSO_VIS_MASK_CROP_DONE"] = "1"
                else:
                    vis_grid = vu.get_colored_map(np.rint(map_pred),
                                    self.collison_map[gx1:gx2, gy1:gy2],
                                    self.visited_vis[gx1:gx2, gy1:gy2],
                                    self.visited_gt[gx1:gx2, gy1:gy2],
                                    goal,
                                    self.explored_map[gx1:gx2, gy1:gy2],
                                    self.explorable_map[gx1:gx2, gy1:gy2],
                                    self.map[gx1:gx2, gy1:gy2] *
                                        self.explored_map[gx1:gx2, gy1:gy2],
                                    semantic_density=semantic_density,
                                    semantic_freshness=self._last_fresh_sem,
                                    structural_map=structural_map)
                    pos = (
                        start_x - gy1 * args.map_resolution / 100.0,
                        start_y - gx1 * args.map_resolution / 100.0,
                        start_o,
                    )
                    gt_pos = (
                        start_x_gt - gy1 * args.map_resolution / 100.0,
                        start_y_gt - gx1 * args.map_resolution / 100.0,
                        start_o_gt,
                    )
                # 获取语义检测信息用于可视化
                detected_classes = self.info.get('detected_classes', [])
                class_counts = self.info.get('class_counts', {})
                class_avg_scores = self.info.get('class_avg_scores', {})
                detection_overlays = self.info.get('detection_overlays', [])
                
                vis_grid = np.flipud(vis_grid)
                try:
                    vu.visualize(self.figure, self.ax, self.obs, vis_grid[:,:,::-1],
                                pos,
                                gt_pos,
                                dump_dir, self.rank, self.episode_no,
                                self.timestep, args.visualize,
                                args.print_images, args.vis_type,
                                detected_classes=detected_classes,
                                class_counts=class_counts,
                                class_avg_scores=class_avg_scores)
                finally:
                    os.environ.pop("NSO_VIS_MASK_CROP_DONE", None)

            else: # Visualize ground-truth map and pose
                vis_grid = vu.get_colored_map(self.map,
                                self.collison_map,
                                self.visited_gt,
                                self.visited_gt,
                                (goal[0]+gx1, goal[1]+gy1),
                                self.explored_map,
                                self.explorable_map,
                                self.map*self.explored_map)
                # 获取语义检测信息用于可视化
                detected_classes = self.info.get('detected_classes', [])
                class_counts = self.info.get('class_counts', {})
                class_avg_scores = self.info.get('class_avg_scores', {})
                detection_overlays = self.info.get('detection_overlays', [])
                
                vis_grid = np.flipud(vis_grid)
                vu.visualize(self.figure, self.ax, self.obs, vis_grid[:,:,::-1],
                            (start_x_gt, start_y_gt, start_o_gt),
                            (start_x_gt, start_y_gt, start_o_gt),
                            dump_dir, self.rank, self.episode_no,
                            self.timestep, args.visualize,
                            args.print_images, args.vis_type,
                            detected_classes=detected_classes,
                            class_counts=class_counts,
                            class_avg_scores=class_avg_scores)

        return output

    def get_embodied_reach_label(self) -> float:
        """返回上一周期长期目标的具身回溯标签（论文式 3）。"""
        if self._last_embodied_success is None:
            return 0.0
        return float(self._last_embodied_success)

    def get_reachability_supervision(self, inputs):
        """为 RPN 提供 FMM + 具身回溯混合监督（论文 3.3.3）。"""
        from env.habitat.reachability_utils import (
            build_traversible_local,
            embodied_goal_reached,
            fmm_reachability_map,
            goal_reachable_label,
        )

        args = self.args
        map_pred = np.rint(inputs['map_pred'])
        explored = np.rint(inputs['exp_pred'])
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        goal = pu.threshold_poses(inputs['goal'], map_pred.shape)

        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        visited_window = self.visited_vis[gx1:gx2, gy1:gy2]
        collision_window = self.collison_map[gx1:gx2, gy1:gy2]
        traversible = build_traversible_local(
            map_pred, explored, visited_window, collision_window, self.selem)
        reach_map = fmm_reachability_map(traversible, tuple(start), dt=self.dt)
        fmm_label = goal_reachable_label(
            traversible, tuple(start), tuple(goal), dt=self.dt)
        embodied_label = embodied_goal_reached(
            tuple(start), tuple(goal),
            success_radius_cells=max(1, int(50 / args.map_resolution)),
        )
        if self._last_embodied_success is not None:
            embodied_label = float(self._last_embodied_success)
        label = max(float(fmm_label), embodied_label)
        return reach_map.astype(np.float32), float(label)

    def _get_gt_map(self, full_map_size):
        self.scene_name = _scene_name_from_sim(self.habitat_env)
        logger.error('Computing map for %s', self.scene_name)

        # Get map in habitat simulator coordinates
        self.map_obj = HabitatMaps(self.habitat_env)
        if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
            logger.error("Invalid map: {}/{}".format(
                            self.scene_name, self.episode_no))
            return None

        agent_y = self._env.sim.get_agent_state().position.tolist()[1]*100.
        sim_map = self.map_obj.get_map(agent_y, -50., 50.0)

        sim_map[sim_map > 0] = 1.

        # Transform the map to align with the agent
        min_x, min_y = self.map_obj.origin/100.0
        x, y, o = self.get_sim_location()
        x, y = -x - min_x, -y - min_y
        range_x, range_y = self.map_obj.max/100. - self.map_obj.origin/100.

        map_size = sim_map.shape
        scale = 2.
        grid_size = int(scale*max(map_size))
        grid_map = np.zeros((grid_size, grid_size))

        grid_map[(grid_size - map_size[0])//2:
                 (grid_size - map_size[0])//2 + map_size[0],
                 (grid_size - map_size[1])//2:
                 (grid_size - map_size[1])//2 + map_size[1]] = sim_map

        if map_size[0] > map_size[1]:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale) \
                             * map_size[1] * 1. / map_size[0],
                    (y - range_y/2.) * 2. / (range_y * scale),
                    180.0 + np.rad2deg(o)
                ]])

        else:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale),
                    (y - range_y/2.) * 2. / (range_y * scale) \
                            * map_size[0] * 1. / map_size[1],
                    180.0 + np.rad2deg(o)
                ]])

        rot_mat, trans_mat = get_grid(st, (1, 1,
            grid_size, grid_size), torch.device("cpu"))

        grid_map = torch.from_numpy(grid_map).float()
        grid_map = grid_map.unsqueeze(0).unsqueeze(0)
        translated = F.grid_sample(grid_map, trans_mat)
        rotated = F.grid_sample(translated, rot_mat)

        episode_map = torch.zeros((full_map_size, full_map_size)).float()
        if full_map_size > grid_size:
            episode_map[(full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size,
                        (full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size] = \
                                rotated[0,0]
        else:
            episode_map = rotated[0,0,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size]



        episode_map = episode_map.numpy()
        episode_map[episode_map > 0] = 1.

        return episode_map

    def _get_plain_semantic_map(self, gx1, gx2, gy1, gy2):
        """
        为语义窗口构建“原始地图”背景：
        - 未知区域：深灰
        - 可达区域：浅灰
        - 障碍/占用：深色
        不绘制轨迹、目标或其他叠加元素。
        """
        h = max(gx2 - gx1, 1)
        w = max(gy2 - gy1, 1)
        plain = np.zeros((h, w, 3), dtype=np.uint8)
        # 默认未知区域
        plain[:] = 35

        if hasattr(self, 'explorable_map') and self.explorable_map is not None:
            explorable_patch = self.explorable_map[gx1:gx2, gy1:gy2]
            explorable_mask = explorable_patch > 0
            plain[explorable_mask] = [215, 215, 215]

        if hasattr(self, 'map') and self.map is not None:
            map_patch = self.map[gx1:gx2, gy1:gy2]
            obstacle_mask = map_patch > 0.5
            plain[obstacle_mask] = [70, 70, 70]

        # 与主窗口保持一致的朝向
        plain = np.flipud(plain)
        return plain


    def _get_stg(self, grid, explored, start, goal, planning_window):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(20., dist)
        x1 = max(1, int(x1 - buf))
        x2 = min(grid.shape[0]-1, int(x2 + buf))
        y1 = max(1, int(y1 - buf))
        y2 = min(grid.shape[1]-1, int(y2 + buf))

        rows = explored.sum(1)
        rows[rows>0] = 1
        ex1 = np.argmax(rows)
        ex2 = len(rows) - np.argmax(np.flip(rows))

        cols = explored.sum(0)
        cols[cols>0] = 1
        ey1 = np.argmax(cols)
        ey2 = len(cols) - np.argmax(np.flip(cols))

        ex1 = min(int(start[0]) - 2, ex1)
        ex2 = max(int(start[0]) + 2, ex2)
        ey1 = min(int(start[1]) - 2, ey1)
        ey2 = max(int(start[1]) + 2, ey2)

        x1 = max(x1, ex1)
        x2 = min(x2, ex2)
        y1 = max(y1, ey1)
        y2 = min(y2, ey2)

        traversible = skimage.morphology.binary_dilation(
                        grid[x1:x2, y1:y2],
                        self.selem) != True
        traversible[self.collison_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                    int(start[1]-y1)-1:int(start[1]-y1)+2] = 1

        if goal[0]-2 > x1 and goal[0]+3 < x2\
            and goal[1]-2 > y1 and goal[1]+3 < y2:
            traversible[int(goal[0]-x1)-2:int(goal[0]-x1)+3,
                    int(goal[1]-y1)-2:int(goal[1]-y1)+3] = 1
        else:
            goal[0] = min(max(x1, goal[0]), x2)
            goal[1] = min(max(y1, goal[1]), y2)

        def add_boundary(mat):
            h, w = mat.shape
            new_mat = np.ones((h+2,w+2))
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        traversible = add_boundary(traversible)

        planner = FMMPlanner(traversible, 360//self.dt)

        reachable = planner.set_goal([goal[1]-y1+1, goal[0]-x1+1])
        path_unreachable = not np.any(reachable)

        stg_x, stg_y = start[0] - x1 + 1, start[1] - y1 + 1
        for i in range(self.args.short_goal_dist):
            stg_x, stg_y, replan = planner.get_short_term_goal([stg_x, stg_y])
        if replan or path_unreachable:
            stg_x, stg_y = start[0], start[1]
        else:
            stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), bool(path_unreachable or replan)


    def _get_gt_action(self, grid, start, goal, planning_window, start_o):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(5., dist)
        x1 = max(0, int(x1 - buf))
        x2 = min(grid.shape[0], int(x2 + buf))
        y1 = max(0, int(y1 - buf))
        y2 = min(grid.shape[1], int(y2 + buf))

        path_found = False
        goal_r = 0
        while not path_found:
            traversible = skimage.morphology.binary_dilation(
                            grid[gx1:gx2, gy1:gy2][x1:x2, y1:y2],
                            self.selem) != True
            traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
            traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                        int(start[1]-y1)-1:int(start[1]-y1)+2] = 1
            traversible[int(goal[0]-x1)-goal_r:int(goal[0]-x1)+goal_r+1,
                        int(goal[1]-y1)-goal_r:int(goal[1]-y1)+goal_r+1] = 1
            scale = 1
            planner = FMMPlanner(traversible, 360//self.dt, scale)

            reachable = planner.set_goal([goal[1]-y1, goal[0]-x1])

            stg_x_gt, stg_y_gt = start[0] - x1, start[1] - y1
            for i in range(1):
                stg_x_gt, stg_y_gt, replan = \
                        planner.get_short_term_goal([stg_x_gt, stg_y_gt])

            if replan and buf < 100.:
                buf = 2*buf
                x1 = max(0, int(x1 - buf))
                x2 = min(grid.shape[0], int(x2 + buf))
                y1 = max(0, int(y1 - buf))
                y2 = min(grid.shape[1], int(y2 + buf))
            elif replan and goal_r < 50:
                goal_r += 1
            else:
                path_found = True

        stg_x_gt, stg_y_gt = stg_x_gt + x1, stg_y_gt + y1
        angle_st_goal = math.degrees(math.atan2(stg_x_gt - start[0],
                                                stg_y_gt - start[1]))
        angle_agent = (start_o)%360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal)%360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > 15.:
            gt_action = 1
        elif relative_angle < -15.:
            gt_action = 0
        else:
            gt_action = 2

        return gt_action

    def close(self):
        if self.figure is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self.figure)
            except Exception:
                pass
            self.figure = None
            self.ax = None
        if (self.rank == 0
                and not os.environ.get("NSO_VIEWER_EXTERNAL")
                and Exploration_Env._live_viewer_proc is not None):
            if Exploration_Env._live_viewer_proc.poll() is None:
                Exploration_Env._live_viewer_proc.terminate()
            Exploration_Env._live_viewer_proc = None
        super().close()

    def close(self):
        if self.figure is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self.figure)
            except Exception:
                pass
            self.figure = None
            self.ax = None
        if (self.rank == 0
                and not os.environ.get("NSO_VIEWER_EXTERNAL")
                and Exploration_Env._live_viewer_proc is not None):
            if Exploration_Env._live_viewer_proc.poll() is None:
                Exploration_Env._live_viewer_proc.terminate()
            Exploration_Env._live_viewer_proc = None
        super().close()
