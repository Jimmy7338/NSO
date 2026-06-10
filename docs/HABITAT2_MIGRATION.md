# NSO → Habitat 2 迁移方案

> **作者：** 李兆宇  
> 论文主实验在 Habitat 1.x（`nso` 环境）完成；Habitat 2 为可选升级路径。

## 目标

在**保留 NSO 算法代码**（`model.py`、`algo/`、`main.py` 训练循环）的前提下，将仿真与数据集从 **Habitat 0.1.7 + vendored habitat-api** 升级到 **Habitat-Lab v0.2.4 + habitat-sim 0.2.4**，并继续使用现有 `data/` 目录中的 Gibson / MP3D / 测试场景。

## 版本选型

| 组件 | 当前 (H0.1) | 目标 (H2) |
|------|-------------|-----------|
| Python | 3.8 (`nso`) | **3.9** (`nso_h2`) |
| habitat-sim | 0.1.7 display | **0.2.4** headless+bullet |
| habitat-lab | vendored 0.1.2 | **v0.2.4**（pip editable） |
| PyTorch | 1.13.1+cu117 | 1.13.1+cu117（同版本） |

不采用 Habitat 3.x：任务偏社交/人形，与 NSO PointNav 探索不匹配。

## 迁移阶段

### 阶段 0：并行环境（已完成基础）

- [x] 新建 conda 环境 `nso_h2`（Python 3.9）
- [x] 安装 `habitat-sim=0.2.4`（headless + bullet + NVIDIA 渲染验证）
- [ ] 安装 `habitat-lab` v0.2.4（需源码，见下）

### 阶段 1：源码与依赖

1. 在本机下载：<https://github.com/facebookresearch/habitat-lab/archive/refs/tags/v0.2.4.tar.gz>
2. SFTP 上传到服务器：`/home/ubuntu/NSO/third_party/habitat-lab-v0.2.4.tar.gz`
3. 执行：`bash scripts/fetch_habitat2_source.sh`
4. 执行：`bash scripts/install_habitat2.sh`

### 阶段 2：配置与数据

- NSO 自带配置：`configs/habitat2/pointnav_habitat_test.yaml`（与 H0.1 数据路径一致）
- 数据**无需重下**：继续用 `data/datasets/pointnav/...` 与 `data/scene_datasets/...`
- Gibson/MP3D 全量场景仍按原 `scripts/download_datasets.sh` 说明申请

### 阶段 3：环境适配层 `env/habitat2/`

| 文件 | 职责 |
|------|------|
| `compat.py` | 统一观测键 `rgb`/`depth`、sim 位姿、scene_id |
| `__init__.py` | `construct_envs`（对齐 H0.1 接口） |
| `exploration_env.py` | 继承/移植 `Exploration_Env`，接 H2 RLEnv |
| `noisy_actions.py` | 移植自定义噪声动作（H2 API） |
| `sync_vector_env.py` | 复用或软链 H0.1 版本 |

### 阶段 4：主程序接入

- `arguments.py` 增加 `--habitat_version {1,2}`（默认 `1`）
- `env/__init__.py` 按版本路由 `construct_envs`
- `scripts/run_nso_h2.sh`：H2 专用启动（`nso_h2` + xvfb + GPU）

### 阶段 5：验证清单

```bash
# 1. 仅仿真器
bash scripts/smoke_habitat2_sim.sh

# 2. Habitat-Lab PointNav（需阶段 1 完成）
conda activate nso_h2
python scripts/smoke_habitat2_lab.py

# 3. NSO 评估
bash scripts/run_nso_h2.sh --num_episodes 1 --max_episode_length 100 -v 0
```

### 阶段 6：可视化（弃用 X11 实时窗）

- H2 推荐：`--print_images 1` + 本地查看 PNG
- 或使用 `habitat.utils.visualizations` / 视频导出（待接入）

## API 对照（开发时查阅）

| H0.1 (NSO 现状) | H2 适配 |
|-----------------|---------|
| `cfg_env(paths)` + `defrost/freeze` | `habitat.get_config()` + `habitat.config.read_write` |
| `PointNavDatasetV1` | `habitat.datasets.pointnav.pointnav_dataset`（注册名可能仍为 V1） |
| `habitat.RLEnv` | `habitat.core.env.RLEnv`（接口相近） |
| `obs['rgb']`, `obs['depth']` | 可能为 `observations['rgb']` 或 sensor uuid，由 `compat` 归一化 |
| `sim.config.SCENE` | `episode.scene_id` / `sim.curr_scene_name` |
| `habitat.SimulatorActions` | `habitat.tasks.nav.nav` 中 action 定义 |
| `SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID` | `habitat.simulator.habitat_sim_v0.gpu_device_id` |
| `SensorSpec` | **`CameraSensorSpec`**（0.2 必须） |

## 风险与回退

- **H0.1 保留**：`--habitat_version 1` 或环境 `nso` 可继续跑现有流程
- **网络**：服务器无法访问 GitHub 时，必须本机下载 tarball
- **自定义 VectorEnv `get_short_term_goal`**：需在 H2 中重新挂接 `SyncVectorEnv`

## 当前进度（自动更新）

见 `scripts/install_habitat2.sh` 运行末尾的状态检查。
