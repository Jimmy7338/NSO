# 场景选择与数据集使用指南

## 一、当前项目中的场景

### 1. Gibson数据集（当前使用）

**特点**：
- 88个室内场景
- 场景大小：中等（主要是住宅和办公室）
- 所有场景都是室内环境

**验证集中的场景**（14个）：
- Cantwell, Denmark, Eastville, Edgemere, Elmira, Eudora, Greigsville, Mosquito, Pablo, Ribera, Sands, Scioto, Sisters, Swormville

**查看所有可用场景**：
```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM
ls data/scene_datasets/gibson/*.glb | xargs -n1 basename | sed 's/.glb//' | sort
```

### 2. 场景大小对比

Gibson数据集中的场景都是室内场景，没有室外场景。如果您需要更大的场景，建议使用Matterport3D或HM3D数据集。

## 二、如何指定优先运行某个场景

### 方法1：修改代码指定场景（推荐）

修改 `env/habitat/__init__.py` 文件，在场景列表前添加您想要的场景：

```python
# 在 construct_envs 函数中，找到 scenes 列表
scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

# 如果指定了优先场景，将其移到列表最前面
if hasattr(args, 'priority_scene') and args.priority_scene:
    priority = args.priority_scene
    if priority in scenes:
        scenes.remove(priority)
        scenes.insert(0, priority)
        print(f"[Scene] 优先使用场景: {priority}")
```

然后在 `arguments.py` 中添加参数：
```python
parser.add_argument('--priority_scene', type=str, default=None,
                    help='优先使用的场景名称（如：Cantwell）')
```

### 方法2：直接修改配置文件（简单但不够灵活）

修改 `env/habitat/__init__.py` 的 `construct_envs` 函数：

```python
scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

# 强制将某个场景放在第一位
if len(scenes) > 0:
    target_scene = "Cantwell"  # 改为您想要的场景名
    if target_scene in scenes:
        scenes.remove(target_scene)
        scenes.insert(0, target_scene)
        print(f"[Scene] 优先场景: {target_scene}")
```

### 方法3：使用单进程模式确保运行指定场景

```bash
python main.py \
  --num_processes 1 \
  --auto_gpu_config 0 \
  --split val \
  --task_config tasks/pointnav_gibson.yaml
```

由于验证集只有14个场景，使用单进程时第一个场景会被优先加载。

## 三、下载更大的数据集

### 1. Matterport3D数据集（更大室内场景）

**特点**：
- 90个大型室内场景
- 场景比Gibson更大、更复杂
- 包含多层建筑、大型商场等

**下载步骤**：

1. **注册并下载**：
   ```bash
   # 访问 Matterport3D 官网注册
   # https://niessner.github.io/Matterport/
   
   # 下载场景数据（需要注册）
   # 场景文件较大，每个约500MB-2GB
   ```

2. **下载数据集配置**：
   ```bash
   cd /home/ubuntu/lzy/ANS/Neural-SLAM/data
   
   # 下载 Matterport3D 的 pointnav 数据集
   # 从 habitat-api 仓库获取
   git clone https://github.com/facebookresearch/habitat-api.git temp_habitat
   cp -r temp_habitat/data/datasets/pointnav/mp3d data/datasets/pointnav/
   rm -rf temp_habitat
   ```

3. **使用Matterport3D**：
   ```bash
   python main.py \
     --task_config tasks/pointnav_mp3d.yaml \
     --split val \
     --num_processes 1 \
     --auto_gpu_config 0
   ```

### 2. HM3D（Habitat-Matterport 3D）数据集

**特点**：
- 1,000个建筑级3D重建
- 比Matterport3D更大、更多样
- 包含多层住宅、商店、办公室等

**下载步骤**：

1. **使用habitat-sim下载工具**：
   ```bash
   # 安装 habitat-sim
   pip install habitat-sim
   
   # 下载HM3D数据集
   python -m habitat_sim.utils.datasets_download \
     --uids hm3d_minival \
     --data-path data/scene_datasets/
   ```

2. **下载数据集配置**：
   ```bash
   # HM3D使用与Matterport3D类似的数据集格式
   # 需要下载对应的pointnav数据集配置
   ```

### 3. Replica数据集（高质量室内场景）

**特点**：
- 18个高质量室内场景
- 包含精确的语义标注
- 场景质量高但数量较少

**下载**：
```bash
# 访问 Replica 官网
# https://github.com/facebookresearch/Replica-Dataset

# 下载场景数据
# 需要申请访问权限
```

## 四、室外场景数据集（需要额外处理）

### 1. BotanicGarden数据集

**特点**：
- 48,000平方米的植物园
- 室外自然环境
- 需要转换为Habitat格式

**注意**：Habitat-Sim主要设计用于室内场景，室外场景可能需要额外的配置和转换。

### 2. 使用室外场景的挑战

- Habitat-Sim的物理引擎主要针对室内环境优化
- 需要自定义配置来支持室外场景
- 可能需要修改导航网格生成

## 五、快速实现：添加场景优先级功能

我已经为您创建了修改代码，让您可以指定优先场景：

### 步骤1：修改 arguments.py

在 `get_args()` 函数中添加：

```python
parser.add_argument('--priority_scene', type=str, default=None,
                    help='优先使用的场景名称（如：Cantwell, Denmark等）')
```

### 步骤2：修改 env/habitat/__init__.py

在 `construct_envs()` 函数中，找到 `scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)` 这一行，在其后添加：

```python
# 如果指定了优先场景，将其移到列表最前面
if args.priority_scene and args.priority_scene in scenes:
    scenes.remove(args.priority_scene)
    scenes.insert(0, args.priority_scene)
    print(f"[Scene] 优先使用场景: {args.priority_scene}")
```

### 步骤3：使用

```bash
python main.py \
  --split val \
  --eval 1 \
  --priority_scene Cantwell \
  --num_processes 1 \
  --auto_gpu_config 0 \
  --train_global 0 --train_local 0 --train_slam 0 \
  --load_global pretrained_models/model_best.global \
  --load_local pretrained_models/model_best.local \
  --load_slam pretrained_models/model_best.slam \
  -v 1
```

## 六、推荐的大场景

### Gibson数据集中的较大场景：

根据经验，以下场景相对较大：
- **Cantwell** - 多层住宅
- **Denmark** - 大型住宅
- **Scioto** - 复杂室内布局
- **Sisters** - 多层结构

### Matterport3D中的大场景：

- **17DRP5sb8fy** - 大型办公室
- **1pXnuDYAj8r** - 多层住宅
- **29hnd4uzFmX** - 大型商场

## 七、检查场景大小

您可以使用以下Python脚本检查场景大小：

```python
import habitat_sim
import os

scene_path = "data/scene_datasets/gibson/Cantwell.glb"
if os.path.exists(scene_path):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim = habitat_sim.Simulator(sim_cfg)
    
    # 获取场景边界
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    
    bounds = sim.pathfinder.get_bounds()
    size = bounds[1] - bounds[0]
    print(f"场景大小: {size}")
    print(f"场景体积: {size[0] * size[1] * size[2]:.2f} 立方米")
    
    sim.close()
```

## 八、总结

1. **当前项目**：使用Gibson数据集，88个室内场景
2. **指定场景**：使用 `--priority_scene` 参数（需要添加代码支持）
3. **更大场景**：下载Matterport3D或HM3D数据集
4. **室外场景**：需要额外处理，Habitat主要支持室内

**建议**：
- 如果需要更大的室内场景，优先考虑Matterport3D
- 如果需要测试特定场景，使用 `--priority_scene` 参数
- 室外场景需要更多配置工作，建议先使用室内大场景

