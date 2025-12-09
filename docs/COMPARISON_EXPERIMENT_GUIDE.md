# 对比实验指南

## 一、概述

本指南说明如何运行基础版本和本项目的增强版本进行对比实验，评估语义检测等优化功能的效果。

## 二、项目对比

### 基础版本 (`/home/ubuntu/lzy/Neural-SLAM`)
- **特点**：基础Neural-SLAM实现
- **功能**：无语义检测、无回环检测
- **适用场景**：作为baseline对比

### 修改项目 (`/home/ubuntu/lzy/ANS/Neural-SLAM`)
- **特点**：增强版Neural-SLAM
- **新增功能**：
  - 语义检测（YOLO）
  - 语义地图构建
  - 语义奖励机制
  - 回环检测（可选）
  - 场景优先级选择

## 三、快速开始

### 方法1：使用自动化脚本（推荐）

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM

# 运行对比实验（默认场景：Cantwell，1个episode）
bash scripts/run_comparison.sh

# 指定场景和episode数量
bash scripts/run_comparison.sh Cantwell 3

# 指定场景、episode和超时时间（秒）
bash scripts/run_comparison.sh Cantwell 3 7200
```

### 方法2：使用Python脚本

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM

python scripts/compare_experiments.py \
  --scene Cantwell \
  --episodes 3 \
  --timeout 7200
```

### 方法3：手动运行（详细控制）

#### 步骤1：运行基础版本

```bash
cd /home/ubuntu/lzy/Neural-SLAM

python main.py \
  --split val \
  --eval 1 \
  --train_global 0 --train_local 0 --train_slam 0 \
  --load_global pretrained_models/model_best.global \
  --load_local pretrained_models/model_best.local \
  --load_slam pretrained_models/model_best.slam \
  --num_processes 1 \
  --auto_gpu_config 0 \
  --num_episodes 3 \
  --priority_scene Cantwell \
  --exp_name original_comparison \
  -v 0
```

#### 步骤2：运行修改项目

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM

python main.py \
  --split val \
  --eval 1 \
  --train_global 0 --train_local 0 --train_slam 0 \
  --load_global pretrained_models/model_best.global \
  --load_local pretrained_models/model_best.local \
  --load_slam pretrained_models/model_best.slam \
  --num_processes 1 \
  --auto_gpu_config 0 \
  --num_episodes 3 \
  --priority_scene Cantwell \
  --use_semantic \
  --semantic_use_all_classes \
  --semantic_conf_thresh 0.1 \
  --semantic_interval 1 \
  --semantic_reward_coeff 0.12 \
  --exp_name modified_comparison \
  -v 0
```

#### 步骤3：手动对比结果

```bash
# 查看基础版本结果
cat tmp/original_comparison/explored_ratio.txt

# 查看修改项目结果
cat tmp/modified_comparison/explored_ratio.txt
```

## 四、对比指标

### 主要指标

1. **探索覆盖率 (Exploration Ratio)**
   - 最终覆盖率
   - 平均覆盖率
   - 最大覆盖率
   - 覆盖率曲线

2. **探索面积 (Explored Area)**
   - 最终探索面积
   - 最大探索面积
   - 面积增长曲线

3. **奖励指标**
   - 平均episode奖励
   - 语义奖励（仅修改项目）

4. **性能指标**
   - 运行时间
   - 每步FPS

### 输出文件

对比实验会在 `comparison_results/results_YYYYMMDD_HHMMSS/` 目录下生成：

```
comparison_results/
  results_20241210_143022/
    original/              # 基础版本结果
      run.log
      output.txt
    modified/              # 修改项目结果
      run.log
      output.txt
    comparison.json        # JSON格式对比结果
    comparison_report.md   # Markdown格式报告
    comparison_plots.png   # 对比图表
```

## 五、查看结果

### 查看对比报告

```bash
# 查看最新的对比报告
ls -t comparison_results/results_*/comparison_report.md | head -1 | xargs cat

# 或直接打开
cat comparison_results/results_*/comparison_report.md
```

### 查看对比图表

```bash
# 查看对比图表（如果有图片查看器）
eog comparison_results/results_*/comparison_plots.png
```

### 查看JSON结果

```bash
# 使用jq美化查看（如果安装了jq）
cat comparison_results/results_*/comparison.json | jq .

# 或直接查看
cat comparison_results/results_*/comparison.json
```

## 六、实验配置建议

### 快速测试（验证功能）

```bash
# 单个场景，1个episode，快速验证
bash scripts/run_comparison.sh Cantwell 1 1800
```

### 标准对比（推荐）

```bash
# 单个场景，3-5个episodes，获得稳定结果
bash scripts/run_comparison.sh Cantwell 5 7200
```

### 完整评估（多场景）

```bash
# 对多个场景分别运行
for scene in Cantwell Denmark Scioto; do
    echo "运行场景: $scene"
    bash scripts/run_comparison.sh "$scene" 3 7200
done
```

## 七、结果解读

### 改进百分比

- **正数**：修改项目表现更好（绿色）
- **负数**：基础版本表现更好（红色）

### 关键指标

1. **最终探索覆盖率**
   - 表示episode结束时的探索完成度
   - 越高越好

2. **平均探索覆盖率**
   - 表示整个episode的平均探索水平
   - 反映探索的稳定性

3. **最大探索覆盖率**
   - 表示达到的最高探索水平
   - 反映探索的峰值能力

### 典型改进预期

- **探索覆盖率提升**：5-15%（语义引导帮助发现更多区域）
- **探索效率提升**：10-20%（语义奖励优化路径选择）
- **语义检测数量**：每个房间检测到5-20个对象（取决于场景）

## 八、常见问题

### Q1: 实验运行时间太长

**解决**：
- 减少episode数量：`--episodes 1`
- 减少最大步数：`--max_episode_length 500`
- 关闭可视化：`-v 0`

### Q2: 显存不足

**解决**：
- 使用单进程：`--num_processes 1`
- 降低分辨率：`--env_frame_width 256 --frame_width 128`

### Q3: 结果差异不明显

**可能原因**：
- episode数量太少（建议至少3个）
- 场景太小或太简单
- 语义检测效果不佳

**解决**：
- 增加episode数量
- 选择更大的场景
- 检查语义检测配置

### Q4: 如何对比多个场景

```bash
# 创建批量对比脚本
cat > batch_compare.sh << 'EOF'
#!/bin/bash
scenes=("Cantwell" "Denmark" "Scioto" "Sisters")
for scene in "${scenes[@]}"; do
    echo "对比场景: $scene"
    bash scripts/run_comparison.sh "$scene" 3 7200
done
EOF
chmod +x batch_compare.sh
./batch_compare.sh
```

## 九、高级用法

### 自定义对比参数

编辑 `scripts/compare_experiments.py`，修改：

```python
# 修改基础版本参数
original_args = base_args + [
    '--exp_name', f'original_{args.scene or "default"}',
    # 添加自定义参数
]

# 修改项目参数
modified_args = base_args + [
    '--use_semantic',
    # 添加自定义语义参数
    '--semantic_model', 'yolov8s.pt',
    '--semantic_use_augment',
]
```

### 添加自定义指标

在 `extract_metrics()` 方法中添加新的指标提取逻辑。

### 并行运行多个实验

```bash
# 使用GNU parallel（如果安装）
parallel -j 2 bash scripts/run_comparison.sh ::: Cantwell Denmark Scioto ::: 3 ::: 7200
```

## 十、总结

1. **快速开始**：使用 `bash scripts/run_comparison.sh`
2. **查看结果**：检查 `comparison_results/` 目录
3. **解读指标**：关注探索覆盖率和改进百分比
4. **优化实验**：根据结果调整参数和配置

通过对比实验，您可以量化评估语义检测等优化功能对探索性能的提升效果。

