# V28 编号关联与运行成本独立审核（2026-09-17）

**结论：本批 case01/03 的 24 个反馈样本没有发现 cue 与实测实例错配；直接拆分 cue 编号作为 measurement 槽号只对这两条已核历史成立，不是通用接口契约。** 本次未构建世界、未运行规划器、未重建 TSDF、未提取网格或重新评价 Q，不能视作中间 Q 标签的第二次独立复算。

## 对应关系的具体证据

[`diagnose_option_feedback_v28.py:233`](/root/NSO/scripts/diagnose_option_feedback_v28.py:233) 和第 242 行直接将 `cue-XXXX` 数字解释为实测槽。这个解释未由原脚本验证。两套跟踪器确有区别：

- [`observed_state_v26.py:295`](/root/NSO/nso/observed_state_v26.py:295) 限制深度上界，第 303 行选离坐标中位数最近的真实像素，第 312 行执行全局距离排序的一对一匹配。
- [`probe_facility_choice_v24_prefix.py:74`](/root/NSO/scripts/probe_facility_choice_v24_prefix.py:74) 使用有限正深度，第 85 行使用坐标中位数，第 86 行逐分量匹配最近已有 track，不实施一对一排除。
- [`facility_measurement_v26.py:79`](/root/NSO/nso/facility_measurement_v26.py:79) 按第二套 track 编号确定槽，首次真实 RGB/depth 像素写入固定 seed。双方均不以类别作位置关联，但这不保证发现顺序、聚合或编号始终一致。

独立脚本对每例全部 401 包重新执行**仅标记关联**，逐包核原 SHA、逐帧精确比较已保存 cue 审计，并精确复现终态二值 marker track 摘要。case01 的 93 次、case03 的 88 次真实组件观测全部对应同一槽。所有 24 个诊断样本的 cue、类别、开端坐标及既有评价 reference ID 均吻合。

| 历史 | cue→实测槽→已封存评价 ID | 首次动作 / 像素 | 实际颜色类别 |
|---|---|---|---|
| case01 | cue-0000→0→1（B） | 7 / [33,76] | 2 |
| case01 | cue-0001→1→0（A） | 94 / [32,84] | 3 |
| case03 | cue-0000→0→1（B） | 7 / [33,76] | 3 |
| case03 | cue-0001→1→0（A） | 88 / [32,84] | 2 |

首次 cue 点与固定 seed 的差异小于 1e-12 m。全历史同身份运行中心最大偏差 0.066931 m，远低于 0.75 m 关联距离；最近异身份距离分别 6.899595 m、6.898502 m。上述 ID 映射只读取已封存的评价收据，没有向跟踪器输入 GT。这里仍是人工 RGB 标记，不是自然图像语义网络的验证。

## 冻结与成本边界

V27 的 179 个源码/配置、9 项输入、源码包内全部 179 项逐 SHA 通过；完整 V27 根目录清单 3247 项，以及 V28 原失败根与恢复根清单均通过。V28 **恢复版**脚本、协议、恢复说明与其 manifest 一致。首次失败版脚本 SHA `7382f201…` 与恢复版 `fce3d661…` 不同，这是既有恢复说明声明的修改，不能表述为首次与恢复之间零改动。旧失败根只有 manifest、failure、inventory，没有原脚本字节快照。

- 恢复结果直接记录 case01/03 各 401 个已保存包，共 **802 次 mapper update / measurement.observe**。
- 首次失败发生于首例的终点一致性检查，按既定 `[1,3]` 顺序及恢复说明可归计 **401 次**，故原诊断总成本应包含 **401+802=1203 次已有包映射**。但旧 failure 收据没有 case/帧计数，401 是控制流和协议证据支持的归计，不能伪称从该收据直接读取。
- 本次关联审核另读 802 包，执行两套轻量跟踪器各 802 次，耗时 3.568 s；**新增 TSDF 更新、网格提取、Q 评价、世界构建、规划和物理动作全部为 0**。这 802 次读取不能再加进原 TSDF 映射成本。

后续新版应使用真实首次观测记录或显式身份映射关联 cue 与测量槽，并保留冲突/缺失门；无需修改此次已冻结结果，也无需为本审计重跑网格。

## 可复查产物

- 脚本：[`audit_option_mapping_v28_provenance_20260917.py`](/root/NSO/scripts/audit_option_mapping_v28_provenance_20260917.py)，SHA `850f308655ba901f3582c90415d53ccae6c2912353472561ae30fdcfc6683629`。
- 紧凑证据：[`result.json`](/root/NSO/audit_results/v28_provenance_20260917/result.json)，SHA `7c09b7c6293e00a6c3c5fc97ba0f33bb11dced63a6babdacb5cb2a34756b7c91`。
- 清单：[`artifact_hashes.json`](/root/NSO/audit_results/v28_provenance_20260917/artifact_hashes.json)，SHA `ff3a278d0b0413954b6dc4cb6e5514faa0c1ab066bf6f5773dca98d7f942791c`；该目录 28,825 B。manifest 是执行前来源记录，最终状态在已封存 result 中。
