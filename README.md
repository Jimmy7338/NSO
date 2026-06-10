# NSO：语义-结构联合感知层次化主动覆盖探索

> **NSO**（Neural Semantic-Structure Active exploration）在 Active Neural SLAM 基础上扩展了语义密度奖励、拓扑结构感知、自监督可达性预测（RPN）、双重回环检测与语义场景补全（SSC）。论文见 `Semantic_Enhanced_Active_SLAM_Paper.tex`；一键复现论文配置：`bash scripts/run_nso_paper_train.sh` 或 `python main.py --paper_mode ...`。

---

# Active Neural SLAM 完整项目指南

> **本文档旨在为完全不了解项目的读者提供全面、深入的项目介绍，涵盖项目背景、技术原理、实现细节、使用方法等所有方面。**

---

## 📑 目录

1. [项目背景与概述](#1-项目背景与概述)
2. [技术原理详解](#2-技术原理详解)
3. [项目结构](#3-项目结构)
4. [核心模块详解](#4-核心模块详解)
5. [依赖与安装](#5-依赖与安装)
6. [运行参数详解](#6-运行参数详解)
7. [使用指南](#7-使用指南)
8. [技术细节](#8-技术细节)
9. [扩展功能](#9-扩展功能)
10. [训练与评估](#10-训练与评估)
11. [常见问题与故障排除](#11-常见问题与故障排除)
12. [参考文献](#12-参考文献)

---

## 1. 项目背景与概述

### 1.1 项目简介

本项目是一个基于深度学习的主动式同时定位与建图（SLAM）系统，用于在未知环境中进行自主探索和导航。本项目基于 ICLR 2020 论文《Learning To Explore Using Active Neural SLAM》的开源实现进行了大量改进和扩展。

**参考论文：**
- **标题：** Learning To Explore Using Active Neural SLAM
- **作者：** Devendra Singh Chaplot, Dhiraj Gandhi, Saurabh Gupta, Abhinav Gupta, Ruslan Salakhutdinov
- **会议：** ICLR 2020
- **论文链接：** https://openreview.net/pdf?id=HklXn1BKDH

**本项目的主要改进：**
- ✅ **语义增强探索**：集成 YOLOv8 进行实时语义对象检测，基于语义信息优化探索策略
- ✅ **结构感知奖励**：识别门框、狭窄通道、开阔区域等结构特征，引导智能体优先探索关键区域
- ✅ **前沿区域奖励**：鼓励探索可见但未访问的区域，提高探索效率
- ✅ **语义回环检测**：基于 NetVLAD 和语义信息的回环检测，提高定位精度
- ✅ **增强的奖励机制**：多层次的奖励函数，结合语义、结构和前沿信息

### 1.2 核心问题

本项目解决的核心问题：

1. **从 RGB 图像直接学习地图和位姿估计**（无需深度传感器或里程计）
2. **主动探索策略**（智能体自主决定探索方向）
3. **分层决策**（全局策略 + 局部策略）
4. **语义引导探索**（利用环境中的语义对象信息优化探索路径）
5. **结构感知导航**（识别并优先探索门框、通道等关键结构）
6. **高效探索**（通过前沿区域检测和语义奖励提高探索效率）

### 1.3 项目特点

- ✅ **端到端学习**：从 RGB 图像直接学习地图和位姿
- ✅ **主动探索**：智能体自主决定探索方向
- ✅ **分层架构**：全局策略（长期目标）+ 局部策略（短期导航）
- ✅ **语义增强探索**：集成 YOLOv8 进行实时语义对象检测，基于语义密度优化探索策略
- ✅ **结构感知奖励**：识别门框、狭窄通道、开阔区域等结构特征，引导智能体优先探索关键区域
- ✅ **前沿区域奖励**：鼓励探索可见但未访问的区域，提高探索效率
- ✅ **语义回环检测**：基于 NetVLAD 和语义信息的回环检测，提高定位精度
- ✅ **多层次奖励机制**：结合基础探索、语义、结构和前沿信息的综合奖励函数

### 1.4 应用场景

- **机器人自主探索**：未知环境的探索和建图
- **室内导航**：智能家居、服务机器人的导航
- **虚拟环境训练**：在仿真环境中训练导航策略
- **研究平台**：SLAM、强化学习、视觉导航的研究

---

## 2. 技术原理详解

### 2.1 系统架构

Active Neural SLAM 由三个核心模块组成：

```
┌─────────────────────────────────────────────────────────┐
│                    RGB 图像输入                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Neural SLAM Module                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  地图预测    │  │  位姿估计    │  │  探索预测    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Global Policy (全局策略)                    │
│           输出长期目标 (Long-term Goal)                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Analytic Path Planner                         │
│        将长期目标转换为短期目标                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Local Policy (局部策略)                      │
│           输出动作 (Move Forward/Turn Left/Right)        │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Neural SLAM Module（神经 SLAM 模块）

**功能：** 从 RGB 图像预测地图和智能体位姿

**输入：**
- RGB 图像（256×256）
- 历史地图和位姿

**输出：**
1. **障碍物地图（Obstacle Map）**：预测环境中的障碍物位置
2. **探索地图（Exploration Map）**：已探索区域的标记
3. **位姿估计（Pose Estimation）**：智能体在地图中的位置和朝向

**网络结构：**
- **编码器：** ResNet-18（提取图像特征）
- **解码器：** 卷积层（生成地图）
- **位姿估计器：** 全连接层（预测位姿变化）

**损失函数：**
- **投影损失（Projection Loss）**：预测地图与真实地图的差异
- **探索损失（Exploration Loss）**：探索区域的准确性
- **位姿损失（Pose Loss）**：位姿估计的准确性

### 2.3 Global Policy（全局策略）

**功能：** 基于当前地图状态，输出长期探索目标

**输入：**
- 局部地图（8通道，包含障碍物、探索区域、当前位置等）
- 智能体朝向（72个方向的嵌入）

**输出：**
- 长期目标坐标（归一化的 (x, y) 坐标）

**网络结构：**
- **卷积编码器：** 4层卷积 + 最大池化
- **全连接层：** 256维隐藏层
- **输出层：** Actor-Critic 架构
  - **Actor：** 输出动作概率分布（长期目标位置）
  - **Critic：** 输出状态价值（用于 PPO 训练）

**训练方法：** PPO (Proximal Policy Optimization)

**奖励函数：**
```
total_reward = m_reward + semantic_bonus + structural_bonus + frontier_bonus
```

其中：
- `m_reward`：基础探索奖励（新探索的面积）
- `semantic_bonus`：语义奖励（探索包含语义对象的新区域）
- `structural_bonus`：结构奖励（门框、狭窄通道等）
- `frontier_bonus`：前沿奖励（可见但未访问的区域）

### 2.4 Local Policy（局部策略）

**功能：** 导航到短期目标

**输入：**
- RGB 图像（128×128）
- 目标方向（相对于智能体的角度）

**输出：**
- 动作：`MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`

**网络结构：**
- **编码器：** ResNet-18（提取图像特征）
- **LSTM：** 512维隐藏层（处理时序信息）
- **输出层：** 动作概率分布

**训练方法：** 模仿学习（Imitation Learning）或强化学习

### 2.5 训练流程

```
1. 初始化环境，重置智能体
2. 循环训练：
   a. 获取 RGB 图像
   b. Neural SLAM Module 预测地图和位姿
   c. 每 num_local_steps 步（默认25步）：
      - Global Policy 输出长期目标
      - Path Planner 转换为短期目标
   d. Local Policy 输出动作
   e. 执行动作，获取奖励
   f. 累积经验，更新策略
3. 定期保存模型
```

---

## 3. 项目结构

### 3.1 目录结构

```
Neural-SLAM/
├── main.py                    # 主程序入口
├── model.py                   # 神经网络模型定义
├── arguments.py               # 命令行参数定义
├── semantic_detector.py       # YOLOv8 语义检测器
│
├── algo/                      # 强化学习算法
│   └── ppo.py                 # PPO 算法实现
│
├── env/                       # 环境相关
│   └── habitat/               # Habitat 仿真环境
│       ├── __init__.py        # 环境初始化
│       ├── exploration_env.py # 探索环境实现
│       └── habitat_api/        # Habitat API 封装
│
├── utils/                     # 工具函数
│   ├── storage.py              # 经验回放缓冲区
│   ├── optimization.py         # 优化器
│   ├── distributions.py        # 动作分布
│   └── model.py               # 模型工具函数
│
├── semantic/                  # 语义相关模块
│   ├── semantic_map.py        # 2D 语义地图
│   ├── class_mapping.py        # 类别映射
│   ├── semantic_identifier.py # 语义标识
│   └── semantic_map_updater.py # 语义地图更新器
│
├── loop/                      # 回环检测模块
│   ├── semantic_vlad.py       # NetVLAD 特征提取
│   └── loop_detector.py        # 回环检测器
│
├── scripts/                   # 脚本文件
│   ├── train_with_semantic.sh  # 语义训练脚本
│   ├── compare_experiments.py # 对比实验脚本
│   └── run_comparison.sh      # 运行对比实验
│
├── docs/                      # 文档
│   ├── COMPLETE_PROJECT_GUIDE.md  # 本文档
│   ├── SEMANTIC_REWARD_EXPLANATION.md  # 语义奖励说明
│   ├── TRAINING_WITH_SEMANTIC.md      # 语义训练指南
│   └── ...
│
├── data/                      # 数据目录
│   └── scene_datasets/        # 场景数据集
│
├── pretrained_models/         # 预训练模型
├── tmp/                       # 临时文件
└── requirements.txt           # Python 依赖
```

### 3.2 核心文件说明

| 文件 | 功能 |
|------|------|
| `main.py` | 主程序，包含训练循环、环境交互、模型更新 |
| `model.py` | 定义 Global Policy、Local Policy、Neural SLAM Module |
| `arguments.py` | 定义所有命令行参数，包括训练、环境、模型参数 |
| `semantic_detector.py` | YOLOv8 语义检测器封装 |
| `env/habitat/exploration_env.py` | 探索环境实现，包含奖励计算、地图更新 |
| `algo/ppo.py` | PPO 算法实现 |
| `utils/storage.py` | 经验回放缓冲区，用于存储训练数据 |

---

## 4. 核心模块详解

### 4.1 Neural SLAM Module

**文件：** `model.py` 中的 `Neural_SLAM_Module` 类

**功能：**
- 从 RGB 图像预测障碍物地图
- 预测探索区域
- 估计智能体位姿变化

**网络结构：**
```python
# 编码器：ResNet-18
encoder = models.resnet18(pretrained=True)

# 地图解码器
map_decoder = nn.Sequential(
    nn.ConvTranspose2d(...),  # 上采样
    nn.ReLU(),
    ...
)

# 位姿估计器
pose_estimator = nn.Sequential(
    nn.Linear(...),
    nn.ReLU(),
    nn.Linear(3)  # (x, y, theta)
)
```

**损失函数：**
```python
total_loss = (
    proj_loss_coeff * proj_loss +    # 投影损失
    exp_loss_coeff * exp_loss +      # 探索损失
    pose_loss_coeff * pose_loss      # 位姿损失
)
```

### 4.2 Global Policy

**文件：** `model.py` 中的 `Global_Policy` 类

**功能：**
- 基于局部地图输出长期目标
- 使用 PPO 算法训练

**网络结构：**
```python
# 输入：8通道局部地图 (local_w × local_h)
# 1. 障碍物地图
# 2. 探索地图
# 3. 当前位置
# 4. 历史位置
# 5-8. 其他特征

# 卷积编码器
main = nn.Sequential(
    nn.MaxPool2d(2),
    nn.Conv2d(8, 32, 3, stride=1, padding=1),
    nn.ReLU(),
    ...
)

# 全连接层
linear1 = nn.Linear(out_size * 32 + 8, hidden_size)
linear2 = nn.Linear(hidden_size, 256)

# Actor-Critic
actor_linear = nn.Linear(256, action_space_size)
critic_linear = nn.Linear(256, 1)
```

### 4.3 Local Policy

**文件：** `model.py` 中的 `Local_IL_Policy` 类

**功能：**
- 基于 RGB 图像和目标方向输出动作
- 使用模仿学习或强化学习训练

**网络结构：**
```python
# 编码器：ResNet-18
encoder = models.resnet18(pretrained=True)

# LSTM（处理时序）
lstm = nn.LSTM(input_size, hidden_size)

# 动作输出
action_linear = nn.Linear(hidden_size, num_actions)
```

### 4.4 环境（Exploration Environment）

**文件：** `env/habitat/exploration_env.py`

**功能：**
- 管理 Habitat 仿真环境
- 计算奖励
- 更新地图
- 处理语义信息

**关键方法：**
- `step()`: 执行动作，返回观察和奖励
- `get_short_term_goal()`: 将长期目标转换为短期目标
- `get_global_reward()`: 计算全局奖励（每 `num_local_steps` 步）
- `update_map()`: 更新地图和位姿

---

## 5. 依赖与安装

### 5.1 系统要求

- **操作系统：** Linux (推荐 Ubuntu 18.04+)
- **Python：** 3.8+
- **CUDA：** 10.0+ (GPU 训练)
- **GPU：** NVIDIA GPU (推荐 8GB+ 显存)

### 5.2 核心依赖

**Python 包（requirements.txt）：**
```
matplotlib              # 可视化
tensorboard             # 训练日志
seaborn==0.9.0          # 数据可视化
imageio==2.6.0          # 图像处理
scikit-fmm              # Fast Marching Method
scikit-image            # 图像处理
scikit-learn==0.22.2.post1  # 机器学习工具
ultralytics>=8.0.0      # YOLOv8 语义检测
opencv-python>=4.7.0    # 计算机视觉
faiss-cpu>=1.7.2        # 相似度搜索（回环检测）
```

**主要依赖：**
- **PyTorch：** 1.2.0+ (推荐 1.8.0+)
- **Habitat-Sim：** 特定版本（见安装说明）
- **Habitat-API：** 特定版本（见安装说明）

### 5.3 安装步骤

#### 步骤 1：安装 Habitat-Sim

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout 9575dcd45fe6f55d2a44043833af08972a7895a9
pip install -r requirements.txt
python setup.py install --headless  # Linux
python setup.py install              # Mac OS
```

#### 步骤 2：安装 Habitat-API

```bash
git clone https://github.com/facebookresearch/habitat-api.git
cd habitat-api
git checkout b5f2b00a25627ecb52b43b13ea96b05998d9a121
pip install -e .
```

#### 步骤 3：安装 PyTorch

```bash
# Linux with GPU
conda install pytorch==1.2.0 torchvision cudatoolkit=10.0 -c pytorch

# Mac OS
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
```

#### 步骤 4：安装项目依赖

```bash
cd Neural-SLAM
pip install -r requirements.txt
```

#### 步骤 5：下载数据集

按照 [Habitat-API 数据下载说明](https://github.com/facebookresearch/habitat-api#data) 下载 Gibson 或 Matterport3D 数据集。

**数据目录结构：**
```
Neural-SLAM/
  data/
    scene_datasets/
      gibson/
        Adrian.glb
        Adrian.navmesh
        ...
    datasets/
      pointnav/
        gibson/
          v1/
            train/
            val/
            ...
```

#### 步骤 6：验证安装

```bash
python main.py -n1 --auto_gpu_config 0 --split val
```

如果运行成功，说明安装正确。

---

## 6. 运行参数详解

### 6.1 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--seed` | int | 1 | 随机种子 |
| `--num_processes` | int | 4 | 并行训练进程数 |
| `--num_episodes` | int | 1000000 | 训练回合数 |
| `--eval` | int | 0 | 1=评估模式，0=训练模式 |
| `--exp_name` | str | "exp1" | 实验名称 |
| `--dump_location` | str | "./tmp/" | 模型和日志保存路径 |

### 6.2 训练控制参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_global` | int | 1 | 是否训练全局策略 |
| `--train_local` | int | 1 | 是否训练局部策略 |
| `--train_slam` | int | 1 | 是否训练 SLAM 模块 |
| `--log_interval` | int | 10 | 日志记录间隔 |
| `--save_interval` | int | 1 | 模型保存间隔 |

### 6.3 环境参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--env_frame_width` | int | 256 | 环境图像宽度 |
| `--env_frame_height` | int | 256 | 环境图像高度 |
| `--max_episode_length` | int | 1000 | 最大回合长度（步数） |
| `--split` | str | "train" | 数据集划分（train/val） |
| `--task_config` | str | "tasks/pointnav_gibson.yaml" | 任务配置文件 |
| `--priority_scene` | str | None | 优先使用的场景名称 |
| `--camera_height` | float | 1.25 | 相机高度（米） |
| `--hfov` | float | 90.0 | 水平视野角度（度） |

### 6.4 全局策略参数（PPO）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--global_lr` | float | 2.5e-5 | 全局策略学习率 |
| `--global_hidden_size` | int | 256 | 隐藏层大小 |
| `--gamma` | float | 0.99 | 折扣因子 |
| `--num_global_steps` | int | 40 | 全局策略前向步数 |
| `--ppo_epoch` | int | 4 | PPO 更新轮数 |
| `--num_mini_batch` | str | "auto" | Mini-batch 数量（auto=进程数/2） |
| `--clip_param` | float | 0.2 | PPO 裁剪参数 |
| `--entropy_coef` | float | 0.001 | 熵系数 |
| `--value_loss_coef` | float | 0.5 | 价值损失系数 |

### 6.5 局部策略参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--local_optimizer` | str | "adam,lr=0.0001" | 局部策略优化器 |
| `--num_local_steps` | int | 25 | 局部策略步数（每25步更新全局目标） |
| `--local_hidden_size` | int | 512 | LSTM 隐藏层大小 |
| `--short_goal_dist` | int | 1 | 短期目标最大距离 |
| `--use_recurrent_local` | int | 1 | 是否使用 LSTM |

### 6.6 SLAM 模块参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--slam_optimizer` | str | "adam,lr=0.0001" | SLAM 优化器 |
| `--slam_batch_size` | int | 72 | SLAM 批次大小 |
| `--slam_iterations` | int | 10 | SLAM 迭代次数 |
| `--slam_memory_size` | int | 500000 | SLAM 经验回放缓冲区大小 |
| `--proj_loss_coeff` | float | 1.0 | 投影损失系数 |
| `--pose_loss_coeff` | float | 10000.0 | 位姿损失系数 |
| `--exp_loss_coeff` | float | 1.0 | 探索损失系数 |

### 6.7 地图参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--map_size_cm` | int | 2400 | 地图大小（厘米） |
| `--map_resolution` | int | 5 | 地图分辨率（厘米/像素） |
| `--vision_range` | int | 64 | 视野范围（像素） |
| `--global_downscaling` | int | 2 | 全局地图下采样倍数 |
| `--obs_threshold` | float | 1.0 | 障碍物阈值 |
| `--collision_threshold` | float | 0.20 | 碰撞阈值 |

### 6.8 语义相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_semantic` | flag | False | 启用语义检测 |
| `--semantic_conf_thresh` | float | 0.2 | 语义检测置信度阈值 |
| `--semantic_use_all_classes` | flag | False | 使用所有 YOLO 类别 |
| `--semantic_indoor_only` | flag | False | 仅使用室内类别 |
| `--semantic_interval` | int | 1 | 语义检测间隔（步数） |
| `--semantic_reward_coeff` | float | 0.1 | 语义奖励系数 |

### 6.9 结构奖励参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--structural_reward_coeff` | float | 0.1 | 结构奖励系数 |
| `--w_struct_door` | float | 2.0 | 门框权重 |
| `--w_struct_narrow` | float | 1.0 | 狭窄通道权重 |
| `--w_struct_open` | float | 0.5 | 开阔区域权重 |
| `--door_boost_distance` | float | 5.0 | 门框增强距离（格子数） |
| `--room_exploration_boost` | float | 1.5 | 房间探索增强系数 |

### 6.10 前沿奖励参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--frontier_reward_coeff` | float | 0.15 | 前沿奖励系数 |

### 6.11 回环检测参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_loop_detection` | flag | False | 启用回环检测 |
| `--loop_interval` | int | 100 | 回环检测间隔（步数） |
| `--loop_min_gap` | int | 200 | 最小回环间隔（步数） |
| `--loop_top_k` | int | 5 | 候选数量（FAISS top-k） |
| `--loop_sim_thresh` | float | 0.75 | NetVLAD 相似度阈值 |
| `--loop_sem_thresh` | float | 0.6 | 语义相似度阈值 |

### 6.12 可视化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-v, --visualize` | int | 0 | 1=启用可视化 |
| `--vis_type` | int | 1 | 1=预测地图，2=真实地图 |
| `--print_images` | int | 0 | 1=保存可视化图像 |

---

## 7. 使用指南

### 7.1 基础训练

**训练完整模型（所有模块）：**
```bash
python main.py
```

**仅训练特定模块：**
```bash
# 仅训练全局策略
python main.py --train_local 0 --train_slam 0

# 仅训练局部策略
python main.py --train_global 0 --train_slam 0

# 仅训练 SLAM 模块
python main.py --train_global 0 --train_local 0
```

### 7.2 使用语义信息训练

**启用语义检测和奖励：**
```bash
python main.py \
  --use_semantic \
  --semantic_use_all_classes \
  --semantic_conf_thresh 0.1 \
  --semantic_interval 1 \
  --semantic_reward_coeff 0.12 \
  --structural_reward_coeff 0.12 \
  --frontier_reward_coeff 0.15 \
  --w_struct_door 2.0 \
  --door_boost_distance 5.0 \
  --room_exploration_boost 1.5 \
  --exp_name training_with_semantic
```

**或使用脚本：**
```bash
bash scripts/train_with_semantic.sh
```

### 7.3 评估预训练模型

**下载预训练模型：**
```bash
mkdir pretrained_models
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1UK2hT0GWzoTaVR5lAI6i8o27tqEmYeyY' -O pretrained_models/model_best.global
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1A1s_HNnbpvdYBUAiw2y1JmmELRLfAJb8' -O pretrained_models/model_best.local
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1o5OG7DIUKZyvi5stozSqRpAEae1F2BmX' -O pretrained_models/model_best.slam
```

**评估：**
```bash
python main.py \
  --split val \
  --eval 1 \
  --train_global 0 \
  --train_local 0 \
  --train_slam 0 \
  --load_global pretrained_models/model_best.global \
  --load_local pretrained_models/model_best.local \
  --load_slam pretrained_models/model_best.slam
```

**带可视化：**
```bash
python main.py \
  --split val \
  --eval 1 \
  --train_global 0 \
  --train_local 0 \
  --train_slam 0 \
  --load_global pretrained_models/model_best.global \
  --load_local pretrained_models/model_best.local \
  --load_slam pretrained_models/model_best.slam \
  -v 1
```

### 7.4 指定场景

**优先使用特定场景：**
```bash
python main.py --priority_scene Cantwell
```

### 7.5 对比实验

**运行基础版本与本项目的对比：**
```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM
bash scripts/run_comparison.sh Cantwell 1
```

---

## 8. 技术细节

### 8.1 地图表示

**地图通道（8通道）：**
1. **障碍物地图**：0=可通行，1=障碍物
2. **探索地图**：0=未探索，1=已探索
3. **当前位置**：智能体当前位置（热力图）
4. **历史位置**：智能体历史轨迹
5-8. **其他特征**：语义密度、结构特征等

**地图坐标系统：**
- 原点：地图中心
- 单位：像素（每个像素 = `map_resolution` 厘米）
- 范围：`[-map_size_cm/2, map_size_cm/2]` 厘米

### 8.2 奖励函数详解

**基础探索奖励：**
```python
m_reward = (curr_explored_area - prev_explored_area) * 0.02
```

**语义奖励：**
```python
# 新鲜区域 = 已观测 - 已访问
fresh_mask = observed_window - visited_window

# 新鲜区域的语义密度
fresh_sem = semantic_density * fresh_mask

# 归一化
fresh_sem_norm = fresh_sem / (sem_max + 1e-6)

# 计算奖励
semantic_bonus = sum(fresh_sem_norm) / active_cells
final_semantic_reward = semantic_bonus_acc * semantic_reward_coeff
```

**结构奖励：**
```python
# 门框检测：左右或上下两侧为障碍，中间为可通行
# 狭窄通道：宽度 < narrow_width_cells
# 开阔区域：均值核 > open_kernel

structural_bonus = (
    w_struct_door * door_map +
    w_struct_narrow * narrow_map +
    w_struct_open * open_map
) * fresh_mask
```

**前沿奖励：**
```python
# 前沿 = 已观测但未访问的区域，且不是障碍物
frontier_mask = fresh_mask * free_space

# 如果前沿靠近门框，给予额外奖励
if near_door:
    frontier_bonus *= room_exploration_boost
```

### 8.3 训练流程详解

**单步流程：**
```
1. 获取 RGB 图像
2. Neural SLAM Module 预测地图和位姿
3. 更新全局地图和局部地图
4. 如果 timestep % num_local_steps == 0:
   a. Global Policy 输出长期目标
   b. Path Planner 转换为短期目标
   c. 计算全局奖励
5. Local Policy 输出动作
6. 执行动作，获取局部奖励
7. 存储经验到缓冲区
8. 定期更新策略
```

**PPO 更新：**
```
1. 收集 num_global_steps 步经验
2. 计算优势函数（GAE）
3. 对 num_mini_batch 个批次：
   a. 计算策略损失（带裁剪）
   b. 计算价值损失
   c. 计算熵损失
   d. 反向传播，更新参数
4. 重复 ppo_epoch 次
```

### 8.4 内存优化

**减少内存占用：**
```bash
# 减少批次大小和步数
python main.py \
  --num_global_steps 40 \
  --num_local_steps 25 \
  --slam_batch_size 72 \
  --num_processes 1
```

---

## 9. 扩展功能

### 9.1 语义检测

**功能：** 使用 YOLOv8 检测环境中的语义对象

**实现：**
- **文件：** `semantic_detector.py`
- **模型：** YOLOv8n（轻量级）
- **类别：** 80 个 COCO 类别或自定义类别映射

**使用：**
```python
detector = SemanticDetector(
    model_name="yolov8n.pt",
    device=device,
    use_custom_mapping=True,
    indoor_only=True
)

detections = detector.detect_batch(images, conf=0.2)
```

**语义地图：**
- **文件：** `semantic/semantic_map.py`
- **功能：** 将检测结果映射到 2D 地图
- **输出：** 语义密度图（每个像素的语义对象密度）

### 9.2 回环检测

**功能：** 检测智能体是否回到之前访问过的位置

**实现：**
- **文件：** `loop/loop_detector.py`
- **方法：** NetVLAD + 语义相似度
- **数据库：** FAISS（快速相似度搜索）

**使用：**
```bash
python main.py \
  --use_semantic \
  --use_loop_detection \
  --loop_interval 100 \
  --loop_min_gap 200
```

### 9.3 结构感知

**功能：** 识别环境结构特征（门框、狭窄通道、开阔区域）

**实现：**
- **文件：** `env/habitat/exploration_env.py`
- **方法：** 基于地图的几何分析

**参数：**
- `--w_struct_door`: 门框权重
- `--w_struct_narrow`: 狭窄通道权重
- `--w_struct_open`: 开阔区域权重

---

## 10. 训练与评估

### 10.1 训练指标

**训练日志包含：**
- **Losses：**
  - Local Loss（局部策略损失）
  - SLAM Loss（投影/探索/位姿损失）
  - Semantic Reward（语义奖励）
- **Rewards：**
  - Exploration Reward（探索奖励）
  - Total Reward（总奖励）
- **Metrics：**
  - Exploration Ratio（探索覆盖率）
  - Explored Area（探索面积）

### 10.2 模型保存

**自动保存：**
- 每 `save_interval` 步保存一次
- 每 `save_periodic` 步保存一次周期性检查点

**保存位置：**
```
{dump_location}/{exp_name}/
  ├── models/
  │   ├── model.global.{step}
  │   ├── model.local.{step}
  │   └── model.slam.{step}
  └── logs/
      └── train.log
```

### 10.3 评估指标

**探索覆盖率（Exploration Ratio）：**
```
exploration_ratio = explored_area / total_explorable_area
```

**探索面积（Explored Area）：**
```
explored_area = sum(explored_map) * (map_resolution^2) / 10000  # m²
```

### 10.4 TensorBoard 可视化

**启动 TensorBoard：**
```bash
tensorboard --logdir {dump_location}/{exp_name}/logs
```

**查看指标：**
- 打开浏览器访问 `http://localhost:6006`
- 查看训练曲线、损失、奖励等

---

## 11. 常见问题与故障排除

### 11.1 安装问题

**Q: Habitat-Sim 安装失败**
- **A:** 确保安装了所有系统依赖（CMake、Eigen3 等）
- 参考 [Habitat-Sim 安装文档](https://github.com/facebookresearch/habitat-sim)

**Q: CUDA 版本不匹配**
- **A:** 确保 PyTorch 和 CUDA 版本兼容
- 检查：`python -c "import torch; print(torch.version.cuda)"`

### 11.2 运行问题

**Q: `num_mini_batch cannot be zero`**
- **A:** 当 `num_processes=1` 时，确保 `num_mini_batch >= 1`
- 修复：已自动处理，`num_mini_batch = max(1, num_processes // 2)`

**Q: CUDA out of memory**
- **A:** 减少批次大小和进程数
- 使用：`--num_global_steps 40 --num_local_steps 25 --slam_batch_size 72 --num_processes 1`

**Q: 语义奖励一直为 0**
- **A:** 检查语义检测是否正常工作
- 查看日志中的 `[Semantic Detection]` 和 `[Semantic Reward Debug]` 信息
- 确保 `--use_semantic` 已启用

### 11.3 性能问题

**Q: 训练速度慢**
- **A:** 
  - 减少 `--num_processes`（但会降低并行度）
  - 减少 `--semantic_interval`（减少语义检测频率）
  - 使用 `--loop_use_lightweight`（轻量级回环检测）

**Q: 内存占用高**
- **A:**
  - 减少 `--slam_memory_size`
  - 减少 `--num_global_steps` 和 `--num_local_steps`
  - 减少 `--slam_batch_size`

### 11.4 数据问题

**Q: 找不到场景数据**
- **A:** 检查 `data/scene_datasets/` 目录
- 确保场景文件（.glb 和 .navmesh）存在

**Q: 数据集路径错误**
- **A:** 检查 `--task_config` 参数
- 确保配置文件中的路径正确

---

## 12. 参考文献

### 12.1 基础论文

1. **Active Neural SLAM（本项目基于此论文的实现）**
   - Chaplot, D.S., et al. "Learning To Explore Using Active Neural SLAM." ICLR 2020.
   - [论文链接](https://openreview.net/pdf?id=HklXn1BKDH)
   - **说明：** 本项目基于该论文的开源实现进行了大量改进和扩展

### 12.2 相关技术

- **PPO:** Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv 2017.
- **NetVLAD:** Arandjelovic, R., et al. "NetVLAD: CNN architecture for weakly supervised place recognition." CVPR 2016.
- **YOLOv8:** Ultralytics. "YOLOv8 Documentation." https://docs.ultralytics.com/

### 12.3 数据集

- **Gibson:** Xia, F., et al. "Gibson Env: Real-World Perception for Embodied Agents." CVPR 2018.
- **Matterport3D:** Chang, A., et al. "Matterport3D: Learning from RGB-D Data in Indoor Environments." 3DV 2017.

### 12.4 工具和框架

- **Habitat:** Savva, M., et al. "Habitat: A Platform for Embodied AI Research." ICCV 2019.
- **PyTorch:** Paszke, A., et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS 2019.

---

## 附录

### A. 快速参考

**常用训练命令：**
```bash
# 基础训练
python main.py

# 语义训练
python main.py --use_semantic --semantic_reward_coeff 0.12

# 评估
python main.py --eval 1 --load_global model.global --load_local model.local --load_slam model.slam

# 可视化
python main.py -v 1
```

**关键参数速查：**
- `--num_processes`: 并行进程数
- `--num_local_steps`: 局部策略步数（默认25）
- `--num_global_steps`: 全局策略步数（默认40）
- `--semantic_reward_coeff`: 语义奖励系数（默认0.1）

### B. 文件说明

| 文件/目录 | 说明 |
|-----------|------|
| `main.py` | 主程序入口 |
| `model.py` | 神经网络模型 |
| `arguments.py` | 参数定义 |
| `env/habitat/exploration_env.py` | 环境实现 |
| `semantic_detector.py` | 语义检测器 |
| `semantic/` | 语义相关模块 |
| `loop/` | 回环检测模块 |
| `algo/ppo.py` | PPO 算法 |
| `utils/` | 工具函数 |

### C. 相关资源

- **参考论文：** [Learning To Explore Using Active Neural SLAM](https://openreview.net/pdf?id=HklXn1BKDH)
- **基础实现（参考）：** https://github.com/devendrachaplot/Neural-SLAM

---

**文档版本：** 1.0  
**最后更新：** 2025-11-28  
**说明：** 本文档描述的是基于参考论文实现进行改进的版本，包含语义增强、结构感知等扩展功能

