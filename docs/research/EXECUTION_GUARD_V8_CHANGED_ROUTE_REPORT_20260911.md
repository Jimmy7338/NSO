# V8 执行守卫：全部 10 条受影响路线实际验证

日期：2026-09-11。此前 4 条与新增 6 条均已执行并通过全部独立回放；新增回放核验 147 个动作、6 个案例，其中 5 条返程、1 条按真实原因停机。精确来源、哈希和逐案例结果见 [结果附件](/root/NSO/docs/research/execution_guard_v8_changed_route_results_20260911.json)。

## 结果

固定的 6 条原成功路线已全部执行同一 `ObservedExecutionGuard`：5 条在拒绝原路线动作后返程，1 条因当前位置完整 footprint 不再满足已知安全条件而停机；没有碰撞。6 条原完整往返路线均未完成，其中 T5/storage 候选 5 已到达原目标姿态，随后无法启动符合硬规则的返程。

加上此前 4 条原碰撞路线，共覆盖全 96 路线事后地图审计确定的全部 10 条受影响路线：**9 条完成守卫返程，1 条停机；0 次实际碰撞；0 条完成原完整往返路线；1 条到达原目标姿态。** 分支共付费 275 动作，另有 10 次原 prefix 各 20 动作，共 200 个 prefix 动作。不能把“0 条完成原完整路线”写成“所有目标都没有到达”，也不能把“9 条返程成功”写成“9 个原候选任务完成”。

| context / family | 候选 | 旧路线成功 | 原计划成本 | 原路线实际动作 | 守卫返程动作 | 原目标已到达 | 终止结果 |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| T1 / storage_shelves | 5 | 否 | 34 | 15 | 15 | 否 | 返回 prefix 锚点及朝向 |
| T1 / ventilation_baffles | 5 | 否 | 34 | 15 | 15 | 否 | 返回 prefix 锚点及朝向 |
| T4 / storage_shelves | 5 | 否 | 38 | 17 | 17 | 否 | 返回 prefix 锚点及朝向 |
| T4 / ventilation_baffles | 5 | 否 | 38 | 17 | 17 | 否 | 返回 prefix 锚点及朝向 |
| T2 / storage_shelves | 2 | 是 | 30 | 12 | 14 | 否 | 返回 prefix 锚点及朝向 |
| T2 / storage_shelves | 5 | 是 | 36 | 12 | 14 | 否 | 返回 prefix 锚点及朝向 |
| T2 / ventilation_baffles | 2 | 是 | 30 | 12 | 14 | 否 | 返回 prefix 锚点及朝向 |
| T2 / ventilation_baffles | 5 | 是 | 36 | 12 | 14 | 否 | 返回 prefix 锚点及朝向 |
| T5 / storage_shelves | 5 | 是 | 30 | 15 | 0 | 是 | 当前 footprint 不满足已知安全条件，停止 |
| T5 / ventilation_baffles | 5 | 是 | 30 | 13 | 15 | 否 | 返回 prefix 锚点及朝向 |

T5/storage 的目标到达动作是第 14 个付费动作；第 15 个动作完成后的最新地图使当前位置不满足守卫条件，因而第 16 个原动作被拒绝。返程规划同时返回 `current_footprint_not_known_safe`，系统没有放宽安全条件、清除未知栅格或免费转动取得新观测，直接记录 `return_unavailable:current_footprint_not_known_safe`。它没有发生物理碰撞，但也没有回到起点，最终 footprint 不是已知安全，必须作为未完成返程保留。

## 固定选择、真实执行与保留内容

新入口 [eval_execution_guard_cases_v8.py](/root/NSO/scripts/eval_execution_guard_cases_v8.py) 固定执行 T2 两 family 的候选 2/5，以及 T5 两 family 的候选 5。清单来自 [全 96 路线地图审计](/root/NSO/eval_results/response_v7_all_routes_latest_map_review_20260911/summary.json) 中**所有原成功且会被守卫改变**的分支；启动前检查没有遗漏或添加其他分支。入口不读取旧重建收益，不按收益选择案例，不计算语义优越。

执行前封存清单、原 route、prefix、原采集来源、守卫与 runner 源码。每个案例从原世界初始状态实际执行 20 动作 prefix，逐字段核对原 RGB-D、雷达和位姿账本，随后使用同一 48 动作预算运行。守卫仅接收最新已观测地图；首次拒绝切换返程，每个返程动作前重新规划并检查。无法返程时停机，停机决策不会产生新传感器帧。

新输出 [execution_guard_v8_success_impacts_20260911](/root/NSO/eval_results/execution_guard_v8_success_impacts_20260911) 保留全部 prefix、147 个分支动作的 RGB-D/雷达、逐步 assessment / return plan / mode switch / terminal decision、真实状态、最终 mapper 哈希、mesh 和栅格。原 prefix 使用内容相同的硬链接，原路径保持可读；没有删除原数据。输出逻辑文件量约 18.2 MB，实际新增更小；配置保留 400 MiB 磁盘余量。新执行耗时约 11.78 秒。

| 证据 | SHA-256 |
| --- | --- |
| 新 runner | `e1922fb6185808fe256bf303b6bd8d92c4edfa53f652d7f0e59b2a86822c1eb8` |
| 共同 guard | `94affb7c6e513471aae87d6ffddd94c2d1678f663f9dd1fc67b50c2c51eb17ca` |
| 新 6 案例执行前 seal | `7d1fbe6f3fcc864c33eb4aa2cf9371192ed27c6cfcd117a01f2224246dcd130f` |
| 新 6 案例 artifact 清单 | `30bc39c0c3eb314d4a55caa6897a284ec870a2ee5cae0507c7b27696313d8cda` |
| 旧 4 案例 artifact 清单 | `61bd1065c3ea3965de063632d3fcf2ee649b38ea53a73acd29ac7501c3d682ad` |

## 独立回放接口

封存 schema 为 `execution_guard_v8_fixed_success_impacts/1`。`pre_execution_seal.json.cases` 每项提供 `context_id`、`family`、`candidate_id`、`path`，例如 `T2/storage_shelves/candidate_002`。独立回放应从 seal 读取清单，验证集合恰为固定 6 条且无重复，不能再硬编码候选 5。

每个 `case.path` 下有 `fixture.json`、`route.json`、`candidates.json`、`prefix/records.json`、prefix `frames/scans`；执行输出位于 `guarded_candidate_{candidate_id:03d}`，内部字段沿用旧 4 案例结构。`result.json` 新增 `case_path`，其余 action/decision/state/mapper/mesh 结构一致。目录不是新的物理场景；同 family 的两个候选分别从完全一致的 prefix 重置执行。

独立回放已核对源归档与 seal、全部原始文件哈希、prefix、每个动作的传感器与位姿、独立 footprint 和带朝向 BFS 返程成本、所有付费动作/停机决策，以及最终地图与网格。**停机案例通过了忠实回放，但未被算为成功返程。** 整体状态现为已执行且已独立回放；原 metadata 的执行结束状态和 artifact 清单仍原样保留，实际回放状态由另存的 verification.json 绑定。

新增 [独立回放结果](/root/NSO/eval_results/execution_guard_v8_success_impacts_20260911/verification.json) 为 `passed_full`，耗时约 11.32 秒；[回放脚本](/root/NSO/scripts/replay_execution_guard_cases_v8.py) SHA-256 为 `3c60386312e1bad3cb5ce6e3726db56aa11b6746d03c08004a5aaf0dd4af7552`。它复用封存物理世界和 mapper，自行计算 footprint、返程 BFS 与停机依据，没有调用生产 guard 或生产路线规划器。T5/storage 的最终 footprint 与不可返程原因均已核实；该核验结果不把未完成任务转为成功。

## 对下一步的约束

这 10 条来自已看过轨迹的开发诊断，不能当作随机测试集或无碰撞概率保证。它们证明守卫在这些案例中确实拒绝了部分旧动作，并实际执行了记录中的返程或停机。它们没有评估重建效用，更没有证明语义优势。

旧训练 bank 的 144 个选择中，有 36 个选择、4/16 个 history 落在这些原成功但会改变的路线中。因此应保留原 V7 训练结果及旧指标，新增执行版本单独验证；不能仅删除 4 条旧碰撞、补上守卫后继续沿用全部旧收益。当前硬守卫仍有无法返程的停机情形，不能宣称完整方案已解决安全执行或准许 C 自动继续。
