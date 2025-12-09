# 可视化颜色含义与收敛判断机制

## 一、地图颜色含义

### 1. 基础地图颜色（按绘制顺序，后绘制的会覆盖先绘制的）

#### **浅灰色 (0.9, 0.9, 0.9)**
- **含义**: 地面真值地图 (`gt_map`)
- **说明**: 表示环境中可探索区域的地面真值（Ground Truth），即理论上可以到达的所有区域

#### **浅蓝色 (235/255, 243/255, 1.0)**
- **含义**: 已探索区域 (`explored`)
- **说明**: 表示机器人已经通过传感器观测到的区域（但不一定到达过）

#### **绿色 (Paired调色板第2色)**
- **含义**: 障碍物/占用区域 (`mat`)
- **说明**: 表示检测到的障碍物或不可通行区域

#### **深灰色 (0.6, 0.6, 0.6)**
- **含义**: 已探索的地面真值区域 (`gt_map_explored`)
- **说明**: 地面真值地图中已经被探索过的部分

#### **蓝绿色 (Paired调色板第3色)**
- **含义**: 已探索区域中的障碍物 (`mat * gt_map_explored`)
- **说明**: 在已探索区域内检测到的障碍物

#### **灰色 (0.6, 0.6, 0.6)**
- **含义**: 地面真值中已访问的区域 (`visited_gt`)
- **说明**: 机器人实际到达过的区域（基于地面真值）

#### **橙色/红色 (Paired调色板第4色)**
- **含义**: 预测的已访问区域 (`visited`)
- **说明**: 根据SLAM预测，机器人已经访问过的区域

#### **深红色 (Paired调色板第5色)**
- **含义**: 预测与真值一致的已访问区域 (`visited * visited_gt`)
- **说明**: 预测的访问区域与地面真值访问区域的重叠部分

#### **红色路径线**
- **含义**: 机器人轨迹
- **说明**: 红色线条表示机器人的移动路径，颜色深浅可能表示时间或速度

#### **红色箭头**
- **含义**: 当前机器人位置和朝向
- **说明**: 红色三角形箭头指向机器人当前的朝向

#### **蓝色圆点**
- **含义**: 目标点 (`goal`)
- **说明**: 全局规划器选择的下一个长期目标位置

### 2. 语义叠加颜色（如果启用了语义检测）

#### **JET热力图（蓝→绿→黄→红）**
- **含义**: 语义密度 (`semantic_density`)
- **说明**: 
  - **蓝色**: 语义密度低（检测到的物体少）
  - **绿色**: 语义密度中等
  - **黄色**: 语义密度较高
  - **红色**: 语义密度很高（检测到大量物体）
- **用途**: 在全局规划中，高密度区域（红色/黄色）通常具有更高的探索价值

#### **WINTER热力图（蓝→青→白）**
- **含义**: 语义新鲜度 (`semantic_freshness`)
- **说明**: 
  - **蓝色**: 已观测但未到达的语义价值区域
  - **青色/白色**: 高价值的新鲜语义区域
- **用途**: 表示"已看到但还没去过"的区域，这些区域具有探索价值

#### **AUTUMN热力图（红→橙→黄）**
- **含义**: 结构内容价值 (`structural_map`)
- **说明**: 
  - **红色**: 门框、狭窄通道、开阔区域等高价值结构特征
  - **橙色/黄色**: 中等价值的结构特征
- **用途**: 突出显示具有探索价值的结构特征（门框、通道、开阔区）

## 二、收敛判断机制

### 1. 探索覆盖率计算

程序通过以下方式计算探索覆盖率：

```python
# 当前已探索区域 = 探索地图 × 可探索地图
curr_explored = self.explored_map * self.explorable_map

# 已探索面积（像素数）
curr_explored_area = curr_explored.sum()

# 总可探索面积
total_explorable_area = self.explorable_map.sum()

# 探索比例
exploration_ratio = curr_explored_area / total_explorable_area
```

### 2. 奖励机制

#### **面积奖励 (Area Reward)**
```python
# 每步新增的探索面积
m_reward = (curr_explored_area - prev_explored_area) * 1.0

# 转换为平方米（假设每个像素代表 0.05m × 0.05m = 0.0025 m²）
m_reward = m_reward * 25.0 / 10000.0  # 转换为 m²

# 奖励缩放
m_reward *= 0.02
```

- **正奖励**: 探索新区域时获得
- **零奖励**: 在已探索区域移动时获得
- **负奖励**: 选择已探索区域作为目标时获得（通过 `extrinsic_rew`）

#### **语义奖励 (Semantic Reward)**
- 在全局规划中，高语义密度区域提供正反馈
- 在局部规划中，低语义密度区域（通道）提供正反馈

#### **结构奖励 (Structural Reward)**
- 门框、狭窄通道、开阔区域提供额外的探索奖励

### 3. 收敛判断标准

#### **主要判断方式：时间限制**

```python
if self.info['time'] >= args.max_episode_length:
    done = True
```

- **默认最大步数**: `max_episode_length = 1000` 步
- **结束条件**: 达到最大步数时，episode自动结束

#### **探索完成度指标**

虽然程序没有明确的"100%覆盖"自动结束机制，但可以通过以下指标判断：

1. **探索比例 (`fp_explored`)**
   - 记录在 `info['fp_explored']` 中
   - 范围: [0, 1]，表示已探索区域占总可探索区域的比例

2. **探索增长率**
   - 如果连续多步 `m_reward ≈ 0`，说明没有探索到新区域
   - 可能表示已经覆盖了所有可达区域

3. **日志记录**
   - 程序会记录 `explored_ratio_log`，保存每个episode的探索比例历史
   - 可以分析探索曲线来判断收敛情况

### 4. 实际收敛判断建议

由于程序使用**固定步数**作为结束条件，实际应用中可以通过以下方式判断是否成功覆盖：

#### **方法1：查看探索比例**
```python
# 在日志中查看 fp_explored 值
# 如果接近 1.0（如 > 0.95），说明已覆盖大部分区域
```

#### **方法2：观察探索奖励**
```python
# 如果连续多步 area_reward ≈ 0
# 说明没有探索到新区域，可能已经收敛
```

#### **方法3：可视化观察**
- 观察地图中是否还有**浅灰色未探索区域**
- 如果所有可探索区域都变成了**浅蓝色**，说明已基本覆盖

#### **方法4：修改代码添加自动结束**
可以添加以下逻辑实现自动结束：

```python
# 在 exploration_env.py 的 step() 方法中添加
exploration_ratio = curr_explored_area / total_explorable_area
if exploration_ratio >= 0.95:  # 95%覆盖率
    done = True
    print(f"Exploration complete! Coverage: {exploration_ratio:.2%}")
```

### 5. 探索效率优化

程序通过以下机制提高探索效率：

1. **全局规划器**
   - 选择未探索区域作为长期目标
   - 避免重复访问已探索区域（负奖励）

2. **局部规划器**
   - 在局部窗口内寻找最优路径
   - 考虑语义信息（低密度通道优先）

3. **语义引导**
   - 高密度区域：全局规划优先
   - 低密度区域：局部规划优先（通道）

4. **结构引导**
   - 门框、通道、开阔区域优先探索

## 三、关键参数

### 探索相关参数
- `--max_episode_length`: 最大episode长度（默认1000步）
- `--num_local_steps`: 局部规划步数（默认25步）
- `--map_resolution`: 地图分辨率（默认5cm/像素）

### 语义相关参数
- `--use_semantic`: 启用语义检测
- `--semantic_reward_coeff`: 语义奖励系数（默认0.1）
- `--semantic_conf_thresh`: 检测置信度阈值（默认0.15）

### 结构相关参数
- `--structural_reward_coeff`: 结构奖励系数（默认0.1）
- `--w_struct_door`: 门框权重（默认1.0）
- `--w_struct_narrow`: 狭窄通道权重（默认1.0）
- `--w_struct_open`: 开阔区域权重（默认1.0）

## 四、总结

1. **颜色含义**：
   - 基础颜色表示地图状态（探索、障碍、访问等）
   - 叠加颜色表示语义和结构信息

2. **收敛判断**：
   - 主要通过时间限制（固定步数）结束
   - 可通过探索比例、奖励变化等指标判断完成度
   - 建议添加自动结束机制（如95%覆盖率）

3. **探索策略**：
   - 全局规划：选择未探索区域
   - 局部规划：考虑语义和结构信息
   - 奖励机制：鼓励探索新区域，避免重复访问


## 五、运行指令与参数说明

### 1. 推荐运行指令

```shell
python main.py --split val --eval 1 --train_global 0 --train_local 0 --train_slam 0 --load_global pretrained_models/model_best.global --load_local pretrained_models/model_best.local --load_slam pretrained_models/model_best.slam -v 1 --use_semantic   --semantic_use_all_classes   --semantic_conf_thresh 0.1   --semantic_interval 1   --semantic_reward_coeff 0.12   --structural_reward_coeff 0.12   --w_struct_door 1.5   --w_struct_narrow 1.2   --w_struct_open 0.6   --use_loop_detection   --loop_interval 100   --loop_min_gap 200   --loop_top_k 5   --loop_sim_thresh 0.75   --loop_sem_thresh 0.6   --visualize 1   --vis_type 1   --exp_name semantic_overlay_debug --loop_use_lightweight


### 2. 参数逐项解释

- **CUDA_VISIBLE_DEVICES=0**：锁定使用第0块GPU，避免多卡环境中线程绑定到不期望的GPU。
- **--task_config tasks/pointnav_gibson.yaml**：指定Habitat任务配置，Gibson训练集提供丰富的室内结构，适合观察语义/结构奖励效果。
- **--split val**：在验证集运行，便于与训练阶段区分，同时验证泛化能力。
- **--auto_gpu_config 0**：关闭自动进程/GPU推断，避免程序根据显存再调整进程数。
- **--num_processes 1 / --num_processes_on_first_gpu 1**：确保仅开启单进程，方便调试，也能保证仅弹出一组可视化窗口。
- **--use_semantic**：开启语义检测、语义地图、语义奖励等整套功能。
- **--semantic_indoor_only**：使用室内类别白名单，过滤交通灯、飞机等离谱检测结果。
- **--semantic_conf_thresh 0.2**：当前实验中兼顾召回与精度的阈值；如果检测过少可下调至0.18。
- **--semantic_interval 1**：每一步都处理语义输入，保证语义密度/新鲜度实时更新。
- **--semantic_reward_coeff 0.12**：略高于默认0.1，使全局策略更关注语义丰富且未踏足的区域。
- **--structural_reward_coeff 0.12**：与语义奖励同量级，突出门框/狭窄通道/开阔视野的重要性。
- **--w_struct_door 1.5 / --w_struct_narrow 1.2 / --w_struct_open 0.6**：对门框赋予最高奖励，其次是狭窄通道；开阔区保持中等权重，防止过度偏好空旷区域。
- **--use_loop_detection**：启用语义强 NetVLAD 回环检测，辅助长期一致性。
- **--loop_interval 100**：每100步执行一次回环匹配，权衡实时性与开销。
- **--loop_min_gap 200**：要求查询帧与候选帧在时间上至少间隔200步，抑制短周期伪回环。
- **--loop_top_k 5**：FAISS返回前5个候选，后续再依据语义/空间过滤。
- **--loop_sim_thresh 0.75 / --loop_sem_thresh 0.6**：NetVLAD与语义向量的相似度阈值，只有两者都满足才认为可能回环。
- **--loop_use_lightweight**：仅使用语义直方图进行快速筛选，避免NetVLAD加载造成卡顿；若显存/算力允许，可去掉此项以获取更稳健描述子。
- **--visualize 1 / --vis_type 1**：开启双窗口可视化（主窗口+语义窗口），`vis_type=1` 显示预测地图。
- **--exp_name semantic_overlay_debug**：日志、模型将存到 `./tmp/semantic_overlay_debug`，方便与其它实验区分。

> 如需切换至纯训练模式，可移除 `--visualize`、`--use_loop_detection`、`--use_semantic` 等调试项，并恢复 `--num_processes` 为自动配置。增

