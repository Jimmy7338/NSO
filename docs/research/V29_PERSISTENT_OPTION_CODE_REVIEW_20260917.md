# V29 持久观察选项：独立代码审查

2026-09-17。结论：下列最终版本未发现阻断生命周期合同验收的问题。仅审查源代码、运行手写观测测试；没有实例化世界、实际 mapper，没有新传感、TSDF 融合、网格或 Q 计算。此结论不等于自主策略有效或语义收益成立。

## 已核准的行为

1. **目标与起始支持保持。** `persistent_observation_option_v29.py:96` 打开时复制目标、观察序列、cue 位置/朝向、实际类别与规划干预类别、置信度及预测代理；起始全量支持为不可变记录。`active` 返回深拷贝。`observed_runtime_v29.py:103` 只在没有活动选项且未锁定返航时全局选目标；跨 5 步仅记录 `preserve_option_at_periodic_check`，不会替换 ID、目标或起始支持。

2. **完整费用与逐步安全。** `persistent_observation_option_v29.py:44` 按当前已观测图逐段求有朝向路径，依次扣除剩余观察路径动作，并保留最后位姿到原锚点及朝向的返航动作。底层 `facility_candidates_v20.py:82` 验证路径长度与带转向图成本一致，未知格不允许通行。每次 `next_action` 重算完整剩余任务，再由实际动作安全门检查第一步；不执行旧路径。返航预留与已付动作分别记录，不把预留计为已执行。

3. **每次动作只有一个归属。** `consume:202` 先验证连续 action、已签发动作、非碰撞位姿、锚点、预算减一、传感/地图合同与全量支持，再记账。最后到达帧归属于旧选项，下一选项只能在消费该帧后打开。碰撞动作仍记账。关闭仅发生一次，关闭后的无选项返航动作单列。完整序列需实际逐个达到位姿和朝向；不会用预测路径代替真实到达。

4. **取消后不复活。** 剩余观察或返航不可行、安全门拒绝都会取消并锁定返航。返航重新读取当前 belief，不能清除未知格以逃离。当返航返回 `None`，适配器在 `observed_runtime_v29.py:119` 设置终态；不继续选新目标。日志包含返航可用性、成本与具体拒绝原因。

5. **传感合同先于融合。** `observed_runtime_v29.py:45` 先调用 SensorPacket 校验，再拒绝重复 frame、外来 episode、非递增时间、错误 action/位姿、跳号及超预算，随后才更新 mapper。action-zero 初始化不支持隐形免费前缀。解析反例验证这些拒绝不消费待执行动作，也不增加 mapper 更新次数。

## 审查中发现并已修复的问题

初稿先处理 `sensor_end/budget_end`，会把恰在最后付费帧实际完成的观察误标为取消。最终 `consume:227–242` 改为：先归账，碰撞优先取消；无碰撞时先推进本帧实际达到的观察位姿，全部到达即完成，尚未完成才因传感或预算终止取消。两者均进入终态。观察完成与返回锚点是两个独立判断，因此传感终止时已到目标但未返航不会被写成任务成功。另已补齐返航拒绝原因/成本、固定传感与地图合同检查，以及 `observed_class_id`/`planning_class_id` 分列。

## 执行证据

- 独立运行 `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:tests/virtual3d .venv-3d/bin/python -B -m unittest tests/virtual3d/test_persistent_observation_option_v29.py -v`：16 项通过，unittest 报告 0.837 秒；工具 session 67989，exit 0。使用手写地图/packet 和 `NonFusingMapper`，不创建真实 mapper。
- 额外内存解析检查（session 15172，exit 0）：实际消费一个手写 `forward` 后，将原锚点置为未知，剩余路线不可行；选项只关闭一次，保留 1 次合成动作，返航日志为 `current_footprint_not_known_safe`，运行时终态为 `observed_return_unavailable`。重复请求不会增加计划/日志，尝试重新打开被拒绝。此项为工具终端检查，未伪装成保存物理轨迹或正式主任务。
- 本审查未重复封存执行器；正式封存由根任务的 `scripts/verify_persistent_observation_option_v29.py` 独立执行。报告仅对本节实际运行负责。

## 适用边界

`completed` 目前表示已经到达约定观察位姿，不能证明相机获得了充分的新信息、三维外形完成或 Q 提升。选项类支持多 waypoint，但当前适配器打开的是既有评分器选中的单目标位姿，尚未加入保证多视角采集的策略。重复位姿、原地转向和不同底盘位置也不可混称独立观察基线；日志已单列平移基线。

整段 outcome 是起止全量支持的净变化，非中途变化累计，也不是因果质量收益。新增 key、方向 bit、距离改善仍是未校准几何描述；标量学习奖励、Q 残差与校准均为空，未训练 RPN-UQ 或 IGCR。V28r1 原评分及已失败机会门不因本次合同通过而得到性能验证。安全保证仅针对当前观测网格，不能外推为定位误差、传感漏检下的真实零碰撞保证。

低层 `GeometryStateV26` 须由 builder 或等价校验产生；真正的 episode、时间、frame 身份检查在运行时适配器完成。不要绕过适配器将任意手工 state 视为真实传感历史。外部公开 `closed/events` 只应作为日志读取，不是安全可信持久存储。

## 审查版本 SHA-256

| 文件 | SHA-256 |
| --- | --- |
| `nso/persistent_observation_option_v29.py` | `4ecedc8a45bbd4b8a09080e9f68a6c93f81592c961080614a4b346a9a138358a` |
| `nso/observed_runtime_v29.py` | `89d53c331651df414f93681ff4f7be78f13c2bcb2b621bb40c3b24e4228e07d2` |
| `tests/virtual3d/test_persistent_observation_option_v29.py` | `7571325cae266215814ddd62d89cd122e93474e4ad72169141adbac524c19afe` |
| `docs/research/V29_PERSISTENT_OPTION_PROTOCOL_20260917.md` | `5e749a93586ae2116d08fa0f67fc408a4d8a70d8c46b4d96a35ca85c83c5da93` |

旧 V26/V27/V28 源及证据均未由本审查修改；新主任务额度仍为 12/36。
