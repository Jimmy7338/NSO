# V7 全部 96 条原路线最新地图可行性审计

共检查 2892 个已记录分支动作。原 92 条成功往返、4 条碰撞分支全部覆盖。
最新地图守卫会改变 10 条原路线，其中 6 条原本成功、4 条原本碰撞。

用冻结 sources.zip 中 mapper 重融每个原 prefix 和动作后的 RGB-D/雷达；独立实现同口径 footprint 与动作前检查，并逐动作核对新 V8 guard。首次拒绝处额外独立 BFS 核对朝向恢复和剩余 48 动作预算内的返程最短成本，未执行任何返程。没有构造世界、读取 GT/outcome 表或生成新传感器。

| context/family | 候选 | 原成功 | 首次拒绝动作 | 原因 | 最新地图返程 |
| --- | ---: | --- | ---: | --- | --- |
| T1/storage_shelves | 5 | False | 16 | next_footprint_not_known_safe | known_safe_return_within_budget (15) |
| T1/ventilation_baffles | 5 | False | 16 | next_footprint_not_known_safe | known_safe_return_within_budget (15) |
| T2/storage_shelves | 2 | True | 13 | next_footprint_not_known_safe | known_safe_return_within_budget (14) |
| T2/storage_shelves | 5 | True | 13 | next_footprint_not_known_safe | known_safe_return_within_budget (14) |
| T2/ventilation_baffles | 2 | True | 13 | next_footprint_not_known_safe | known_safe_return_within_budget (14) |
| T2/ventilation_baffles | 5 | True | 13 | next_footprint_not_known_safe | known_safe_return_within_budget (14) |
| T4/storage_shelves | 5 | False | 18 | next_footprint_not_known_safe | known_safe_return_within_budget (17) |
| T4/ventilation_baffles | 5 | False | 18 | next_footprint_not_known_safe | known_safe_return_within_budget (17) |
| T5/storage_shelves | 5 | True | 16 | current_footprint_not_known_safe | current_footprint_not_known_safe (0) |
| T5/ventilation_baffles | 5 | True | 14 | next_footprint_not_known_safe | known_safe_return_within_budget (15) |

首次拒绝前的观测是新守卫与旧执行器共享的实际观测；首次拒绝后的继续记录只描述旧轨迹，不能代替守卫介入后的反事实传感器、重建收益或新碰撞率。通过静态检查的 return plan 也不保证在未来更新后的地图仍可执行。

因此不应只修补 4 条失败并保留其他收益：守卫是否影响成功分支由上表完整计数决定。需要使用独立新版本重放/执行整套固定候选，保留旧 V7 冻结数据与拟合结果；本审计没有改动它们。复用 mapper 不等于独立 SLAM，地图不确定性与保守栅格边界仍是适用范围限制。

## 对已封存训练选择的影响

将上述分支按 history 和 candidate_id 与独立 SVD 复核的训练选择表连接，T2 两 family 的候选 2、T5 两 family 的候选 5 均受影响；9 个 scorer 在这 4 个 history 都选择了相应路线，共 36/144 个选择、4/16 个 history。它们原本全部成功。原 4 条碰撞没有被训练 bank 选择，仍不能据此认为执行修复不影响训练选择结果。该连接只诊断版本敏感性，没有重训或修改原选择，输入哈希见 training_selection_impact.json。
