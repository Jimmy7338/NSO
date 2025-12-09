# 使用语义信息进行训练

## 问题诊断：语义奖励为0

如果训练过程中语义奖励一直为0，可能的原因包括：

### 1. 语义检测没有检测到对象
- **原因**：置信度阈值 `semantic_conf_thresh` 过高（默认0.2）
- **解决**：降低置信度阈值，例如 `--semantic_conf_thresh 0.1` 或 `0.05`

### 2. 语义密度全为0
- **原因**：即使检测到对象，语义地图可能没有正确更新
- **诊断**：查看调试信息中的 `sem_max` 和 `sem_sum` 值

### 3. fresh_mask全为0（最常见）
- **原因**：所有已观测区域都已访问过，没有"新鲜"区域
- **解决**：已改进奖励计算，即使没有新鲜区域也会基于语义密度给予少量奖励

### 4. 语义检测间隔过大
- **原因**：`semantic_interval` 设置过大，导致某些步骤没有语义信息
- **解决**：确保 `--semantic_interval 1`（每步都检测）

## 训练命令

### 方法1：使用训练脚本（推荐）

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM
bash scripts/train_with_semantic.sh
```

### 方法2：直接运行命令

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM
python main.py \
  --split train \
  --eval 0 \
  --train_global 1 \
  --train_local 1 \
  --train_slam 1 \
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
  --num_processes 4 \
  --exp_name training_with_semantic
```

**注意**：如果使用多行命令，确保每行末尾有反斜杠 `\`，且反斜杠后不能有空格。

### 方法3：单行命令（避免换行问题）

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM
python main.py --split train --eval 0 --train_global 1 --train_local 1 --train_slam 1 --use_semantic --semantic_use_all_classes --semantic_conf_thresh 0.1 --semantic_interval 1 --semantic_reward_coeff 0.12 --structural_reward_coeff 0.12 --frontier_reward_coeff 0.15 --w_struct_door 2.0 --door_boost_distance 5.0 --room_exploration_boost 1.5 --num_processes 4 --exp_name training_with_semantic
```

## 调试信息

训练过程中，每100步会打印以下调试信息：

```
[Semantic Detection] Step 100: Total detections=15, conf_thresh=0.10, interval=1
[Semantic Reward Debug] Step 100: sem_max=0.0234, sem_sum=1.234, active_cells=45, observed=120, visited=75, fresh_mask_sum=45
```

### 诊断指南

- **如果 `sem_max=0.0000`**：语义密度全为0，需要检查：
  - 语义检测是否正常工作（查看 `Total detections`）
  - 置信度阈值是否过高
  - 语义地图是否正确更新

- **如果 `active_cells=0`**：没有新鲜区域，但改进后的代码仍会基于语义密度给予少量奖励

- **如果 `Total detections=0`**：没有检测到任何对象，需要降低 `semantic_conf_thresh`

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_semantic` | False | 启用语义识别 |
| `--semantic_conf_thresh` | 0.2 | 语义检测置信度阈值（建议0.1） |
| `--semantic_interval` | 1 | 语义检测间隔步数（建议1） |
| `--semantic_reward_coeff` | 0.1 | 语义奖励系数 |
| `--structural_reward_coeff` | 0.1 | 结构奖励系数 |
| `--frontier_reward_coeff` | 0.15 | 前沿奖励系数 |
| `--w_struct_door` | 2.0 | 门框区域权重 |
| `--door_boost_distance` | 5.0 | 门框增强距离（格子数） |
| `--room_exploration_boost` | 1.5 | 房间探索增强系数 |

## 常见问题

### Q: 为什么语义奖励一直是0？

A: 可能的原因：
1. 置信度阈值过高，没有检测到对象
2. 语义地图未正确更新
3. 所有区域都已访问过（但改进后的代码仍会给予奖励）

查看调试信息以确定具体原因。

### Q: 如何提高语义检测数量？

A: 降低置信度阈值：
```bash
--semantic_conf_thresh 0.05  # 检测更多对象，但可能有误检
```

### Q: 训练速度很慢怎么办？

A: 可以增加语义检测间隔：
```bash
--semantic_interval 5  # 每5步检测一次
```

但注意：间隔越大，语义奖励的更新频率越低。

