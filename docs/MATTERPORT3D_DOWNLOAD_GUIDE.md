# Matterport3D数据集下载与配置详细指南

## 一、Matterport3D数据集简介

**Matterport3D (MP3D)** 数据集包含：
- **90个大型室内场景**
- 包括大型商场、多层住宅、办公室等
- 场景比Gibson数据集更大、更复杂
- 每个场景文件约500MB-2GB

**大型商场场景示例**：
- `17DRP5sb8fy` - 大型办公室/商场
- `1pXnuDYAj8r` - 多层住宅/商场
- `29hnd4uzFmX` - 大型商场
- `2azQ1b91cZZ` - 大型室内空间

## 二、下载步骤

### 步骤1：注册并申请访问权限

1. **访问Matterport3D官网**：
   - 官网：https://niessner.github.io/Matterport/
   - 或直接访问：https://github.com/niessner/Matterport

2. **填写使用协议**：
   - 需要填写Matterport3D数据集使用协议
   - 提供研究用途说明
   - 等待审核通过（通常1-3个工作日）

3. **获取下载链接**：
   - 审核通过后会收到下载链接和访问凭证

### 步骤2：下载场景数据

**方法1：使用官方下载脚本（推荐）**

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM

# 创建下载目录
mkdir -p data/scene_datasets/mp3d

# 下载官方下载脚本（如果有）
# 通常Matterport3D会提供下载脚本或wget链接列表
```

**方法2：手动下载（如果提供了直接链接）**

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM/data/scene_datasets/mp3d

# 下载大型商场场景示例（需要替换为实际下载链接）
# wget [下载链接] -O 17DRP5sb8fy.glb
# wget [下载链接] -O 17DRP5sb8fy.navmesh
```

**方法3：使用habitat-sim下载工具（如果支持）**

```bash
# 检查habitat-sim是否支持MP3D下载
python -c "import habitat_sim; print(habitat_sim.__version__)"

# 如果支持，使用以下命令
python -m habitat_sim.utils.datasets_download \
  --uids mp3d \
  --data-path data/scene_datasets/
```

### 步骤3：下载PointNav数据集配置

Matterport3D的场景文件需要配合PointNav数据集配置文件使用：

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM

# 方法1：从habitat-api仓库下载（推荐）
# 创建临时目录
mkdir -p temp_download
cd temp_download

# 克隆habitat-api（只获取数据集配置）
git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/habitat-api.git
cd habitat-api
git sparse-checkout init --cone
git sparse-checkout set data/datasets/pointnav/mp3d

# 复制数据集配置到项目
cp -r data/datasets/pointnav/mp3d ../../data/datasets/pointnav/

# 清理
cd ../..
rm -rf temp_download
```

**或者手动创建数据集目录结构**：

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM

# 创建目录结构
mkdir -p data/datasets/pointnav/mp3d/v1/train
mkdir -p data/datasets/pointnav/mp3d/v1/val
mkdir -p data/datasets/pointnav/mp3d/v1/test
```

### 步骤4：下载数据集JSON文件

PointNav数据集需要JSON配置文件，可以从以下来源获取：

**方法1：从habitat-api仓库下载**

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM

# 下载完整的数据集配置
git clone --depth 1 https://github.com/facebookresearch/habitat-api.git temp_habitat_api
cp -r temp_habitat_api/data/datasets/pointnav/mp3d/* data/datasets/pointnav/mp3d/
rm -rf temp_habitat_api
```

**方法2：手动下载（如果habitat-api有直接下载链接）**

访问：https://github.com/facebookresearch/habitat-api/tree/main/data/datasets/pointnav/mp3d

下载以下文件到对应目录：
- `v1/train/train.json.gz`
- `v1/val/val.json.gz`
- `v1/test/test.json.gz`

## 三、配置项目使用Matterport3D

### 步骤1：验证场景文件

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM

# 检查场景文件
ls -lh data/scene_datasets/mp3d/*.glb | head -5

# 应该看到类似输出：
# 17DRP5sb8fy.glb
# 1pXnuDYAj8r.glb
# ...
```

### 步骤2：验证数据集配置

```bash
# 检查数据集配置文件
ls -la data/datasets/pointnav/mp3d/v1/val/

# 应该看到：
# val.json.gz
```

### 步骤3：测试加载场景

创建一个测试脚本验证场景可以正常加载：

```bash
cat > test_mp3d_scene.py << 'EOF'
import habitat_sim
import os

# 测试加载一个MP3D场景
scene_path = "data/scene_datasets/mp3d/17DRP5sb8fy.glb"
if os.path.exists(scene_path):
    print(f"测试加载场景: {scene_path}")
    
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.default_agent_id = 0
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    
    print("✓ 场景加载成功！")
    print(f"场景路径: {sim.semantic_scene.scene_id}")
    
    # 获取场景边界
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    
    bounds = sim.pathfinder.get_bounds()
    size = bounds[1] - bounds[0]
    print(f"场景大小: {size}")
    print(f"场景体积: {size[0] * size[1] * size[2]:.2f} 立方米")
    
    sim.close()
else:
    print(f"✗ 场景文件不存在: {scene_path}")
    print("请先下载Matterport3D场景文件")
EOF

python test_mp3d_scene.py
```

## 四、运行项目使用Matterport3D

### 基本运行命令

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM

python main.py \
  --task_config tasks/pointnav_mp3d.yaml \
  --split val \
  --eval 1 \
  --num_processes 1 \
  --auto_gpu_config 0 \
  --train_global 0 --train_local 0 --train_slam 0 \
  --load_global pretrained_models/model_best.global \
  --load_local pretrained_models/model_best.local \
  --load_slam pretrained_models/model_best.slam \
  -v 1
```

### 指定特定场景（大型商场）

首先查看可用的场景：

```bash
# 查看验证集中的场景
python -c "
import gzip, json
f = gzip.open('data/datasets/pointnav/mp3d/v1/val/val.json.gz', 'rt')
data = json.load(f)
scenes = set([ep['scene_id'].split('/')[-1].split('.')[0] for ep in data['episodes']])
print('可用场景:')
for s in sorted(scenes):
    print(f'  - {s}')
f.close()
"
```

然后使用 `--priority_scene` 参数指定场景：

```bash
python main.py \
  --task_config tasks/pointnav_mp3d.yaml \
  --split val \
  --priority_scene 17DRP5sb8fy \
  --eval 1 \
  --num_processes 1 \
  --auto_gpu_config 0 \
  --train_global 0 --train_local 0 --train_slam 0 \
  --load_global pretrained_models/model_best.global \
  --load_local pretrained_models/model_best.local \
  --load_slam pretrained_models/model_best.slam \
  -v 1
```

## 五、快速下载脚本（如果官方提供）

如果Matterport3D提供了下载脚本，可以创建自动化下载脚本：

```bash
cat > download_mp3d.sh << 'EOF'
#!/bin/bash

# Matterport3D下载脚本
# 需要先获得下载链接和访问凭证

MP3D_DIR="data/scene_datasets/mp3d"
mkdir -p $MP3D_DIR

# 大型商场场景ID列表（示例）
LARGE_SCENES=(
    "17DRP5sb8fy"  # 大型办公室/商场
    "1pXnuDYAj8r"  # 多层住宅/商场
    "29hnd4uzFmX"  # 大型商场
    "2azQ1b91cZZ"  # 大型室内空间
)

echo "开始下载Matterport3D场景..."
echo "注意：需要替换下面的URL为实际的下载链接"

for scene in "${LARGE_SCENES[@]}"; do
    echo "下载场景: $scene"
    # 替换为实际下载链接
    # wget [下载链接]/${scene}.glb -O ${MP3D_DIR}/${scene}.glb
    # wget [下载链接]/${scene}.navmesh -O ${MP3D_DIR}/${scene}.navmesh
done

echo "下载完成！"
EOF

chmod +x download_mp3d.sh
```

## 六、常见问题解决

### 问题1：场景文件找不到

**错误**：`FileNotFoundError: data/scene_datasets/mp3d/xxx.glb`

**解决**：
```bash
# 检查场景文件路径
ls -la data/scene_datasets/mp3d/

# 确保场景文件在正确位置
# 场景文件应该直接在 mp3d/ 目录下，不是子目录
```

### 问题2：数据集配置文件找不到

**错误**：`FileNotFoundError: data/datasets/pointnav/mp3d/v1/val/val.json.gz`

**解决**：
```bash
# 从habitat-api下载数据集配置
cd /home/ubuntu/lzy/ANS/Neural-SLAM
git clone --depth 1 https://github.com/facebookresearch/habitat-api.git temp_api
cp -r temp_api/data/datasets/pointnav/mp3d/* data/datasets/pointnav/mp3d/
rm -rf temp_api
```

### 问题3：场景加载失败

**错误**：场景加载时崩溃或报错

**解决**：
1. 检查场景文件是否完整下载
2. 检查habitat-sim版本是否兼容
3. 尝试使用其他场景测试

### 问题4：显存不足

**解决**：
```bash
# 降低分辨率
python main.py \
  --task_config tasks/pointnav_mp3d.yaml \
  --env_frame_width 256 --env_frame_height 256 \
  --frame_width 128 --frame_height 128 \
  ...
```

## 七、推荐的下载顺序

1. **先下载数据集配置**（文件小，下载快）
   ```bash
   git clone --depth 1 https://github.com/facebookresearch/habitat-api.git temp
   cp -r temp/data/datasets/pointnav/mp3d/* data/datasets/pointnav/mp3d/
   rm -rf temp
   ```

2. **查看需要哪些场景**
   ```bash
   python -c "
   import gzip, json
   f = gzip.open('data/datasets/pointnav/mp3d/v1/val/val.json.gz', 'rt')
   data = json.load(f)
   scenes = set([ep['scene_id'].split('/')[-1].split('.')[0] for ep in data['episodes']])
   print('\n'.join(sorted(scenes)))
   f.close()
   "
   ```

3. **选择性下载场景**（只下载需要的场景，节省空间）

4. **测试单个场景**（确保配置正确）

5. **运行完整测试**

## 八、文件结构检查清单

下载完成后，确保以下文件结构正确：

```
Neural-SLAM/
  data/
    scene_datasets/
      mp3d/
        17DRP5sb8fy.glb          # 场景文件
        17DRP5sb8fy.navmesh       # 导航网格
        1pXnuDYAj8r.glb
        1pXnuDYAj8r.navmesh
        ...
    datasets/
      pointnav/
        mp3d/
          v1/
            train/
              train.json.gz       # 训练集配置
            val/
              val.json.gz         # 验证集配置
            test/
              test.json.gz        # 测试集配置
```

## 九、验证安装

运行完整验证脚本：

```bash
cat > verify_mp3d_setup.py << 'EOF'
import os
import gzip
import json

print("=" * 50)
print("Matterport3D安装验证")
print("=" * 50)

# 检查场景目录
scene_dir = "data/scene_datasets/mp3d"
if os.path.exists(scene_dir):
    scenes = [f for f in os.listdir(scene_dir) if f.endswith('.glb')]
    print(f"✓ 找到 {len(scenes)} 个场景文件")
    if scenes:
        print(f"  示例: {scenes[0]}")
else:
    print(f"✗ 场景目录不存在: {scene_dir}")

# 检查数据集配置
dataset_file = "data/datasets/pointnav/mp3d/v1/val/val.json.gz"
if os.path.exists(dataset_file):
    print(f"✓ 数据集配置文件存在")
    try:
        with gzip.open(dataset_file, 'rt') as f:
            data = json.load(f)
            scenes_in_dataset = set([ep['scene_id'].split('/')[-1].split('.')[0] 
                                   for ep in data['episodes']])
            print(f"✓ 验证集中有 {len(scenes_in_dataset)} 个场景")
            print(f"  示例场景: {list(scenes_in_dataset)[:3]}")
    except Exception as e:
        print(f"✗ 无法读取数据集文件: {e}")
else:
    print(f"✗ 数据集配置文件不存在: {dataset_file}")

# 检查任务配置
task_config = "env/habitat/habitat_api/configs/tasks/pointnav_mp3d.yaml"
if os.path.exists(task_config):
    print(f"✓ 任务配置文件存在")
else:
    print(f"✗ 任务配置文件不存在: {task_config}")

print("=" * 50)
EOF

python verify_mp3d_setup.py
```

## 十、总结

1. **申请访问权限**：访问Matterport3D官网申请
2. **下载场景文件**：下载.glb和.navmesh文件到 `data/scene_datasets/mp3d/`
3. **下载数据集配置**：从habitat-api获取JSON配置文件
4. **验证安装**：运行验证脚本确保一切正常
5. **运行测试**：使用 `--task_config tasks/pointnav_mp3d.yaml` 运行

**注意事项**：
- Matterport3D场景文件较大，确保有足够的磁盘空间（建议至少50GB）
- 下载可能需要较长时间，建议使用稳定的网络连接
- 如果只需要测试，可以先下载1-2个场景验证配置


