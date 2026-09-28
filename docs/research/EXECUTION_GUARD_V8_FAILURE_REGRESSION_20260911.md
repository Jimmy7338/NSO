# V8执行保护：全部四条已知失败路线的开发回归

2026-09-11。仅针对事前指定的T1/T4、两物理family、原candidate5。没有根据本轮收益挑路线，没有读取旧candidate outcomes，也没有修改任何原V7资产、旧mapper/env或guard源码。**实际执行与独立回放均已完成；不宣称系统安全或语义性能通过。**

## 固定执行规则

先封存四条完整原路线、全部原prefix观察与源码。独立子进程从原T1/T4相同源码归档加载世界和mapper，仅加入V8 guard及新执行器；先手动执行原20动作prefix，21帧RGB-D/scan及位姿、动作、计数逐字段匹配原始记录。guard与return_plan只接收最新实测belief、当前位置与朝向、原prefix锚点及剩余预算，没有GT输入。

随后按原route逐动作调用guard.assess。第一次拒绝即进入return模式，每个返程动作前依据最新地图重新求return_plan并再次assess；前进和左右转都通过world.step真实执行并付费。当前足迹不安全、锚点不安全、断连或预算不足则明确停机，不伪造可行返回。真实碰撞或状态不一致时立即停止。terminal stop是控制决策，不调用world.step(stop)，也不免费取得新观察。每条分支上限48动作，原20动作prefix独立列账。

## 实际执行记录

|父上下文|物理家族|拒绝前付费动作|付费返程动作|总分支动作|新碰撞数|恢复原位置及朝向|原路线完成|
|---|---|---:|---:|---:|---:|---|---|
|T1|storage_shelves|15|15|30|0|是|否|
|T1|ventilation_baffles|15|15|30|0|是|否|
|T4|storage_shelves|17|17|34|0|是|否|
|T4|ventilation_baffles|17|17|34|0|是|否|

四条的首次拒绝原因均为next_footprint_not_known_safe；T1在分支已付15动作、绝对step35时拒绝下一次forward，T4在已付17动作、绝对step37时拒绝下一次forward。四条均有真实付费返程，终态footprint在最新地图上被判为已知安全；没有把停在原地记成执行了返程。合计128个分支动作，另有80个原前缀重放动作。

这些轨迹没有到达原candidate5目标；本轮验证的是在新观测否定旧路线局部可行性时终止继续前进并返回。不得把减少碰撞写成同等覆盖/重建收益下的改善，当前没有计算guard后面积、F1或联合指标，也不能用新回归直接替换原V7训练标签。

四条均来自已知失败的开发子集，不能估计一般场景碰撞率、故障检出率或完整ANS安全性。实验没有实际触发当前足迹已不安全、断连或返程超预算分支，这些仍是guard的明确停机路径及单元测试范围。观察地图、机器人足迹模型、定位精度有误差时仍可能发生碰撞；这里没有概率安全保证。guard本身仅使用几何可行性，其结果不能作为细类别语义创新证据。

## 复核材料

结果目录：[execution_guard_v8_failures_20260911](/root/NSO/eval_results/execution_guard_v8_failures_20260911/results.json)。每case包含全部prefix原始RGB-D/雷达、固定route、prefix精确回放检查；guarded_candidate_005下保存逐次before-action决定、belief hash、预算和理由、模式切换、实际actions/states、全部分支RGB-D/scan、初末mapper hashes、末网格与地图。

原始V7观察以带hash的只读硬链接留档，没有覆盖原文件。before-action判断先写decisions.jsonl，再允许动作进入world.step。所有返回也有对应判断与实际动作记录。根目录pre_execution_seal.json封存源码和四条路线，artifact_hashes.json记录492项输出（本报告位于docs，不改该封存目录）。

执行器：[eval_execution_guard_v8.py](/root/NSO/scripts/eval_execution_guard_v8.py)，SHA256 `0a6c631f6ffcbbde59fc5fd59727eb6634af376d16cd3d9080c0ebfeabec5e36`。guard SHA256 `94affb7c6e513471aae87d6ffddd94c2d1678f663f9dd1fc67b50c2c51eb17ca`，4项已有单元测试复跑通过，源码未改。

源归档SHA256 `b3987e4848844d9bdea13488441870f4da7f71f3e0ad505d10ffda1903765a2b`；执行前封存SHA256 `dc4cf65c52d6e8c051e79f22a5b834b9f58761939e0a213cf92d0eba18bc2c03`。

独立回放已覆盖全部prefix和128个guarded动作，独立实现footprint检查与BFS最短返回，不调用生产guard、路线求解器或执行循环；逐项核对每次判断、预算、模式、轨迹、传感器、地图与末网格全部一致，耗时9.51秒。[verification.json](../../eval_results/execution_guard_v8_failures_20260911/verification.json)为passed_full。执行元数据在采集时冻结，仍保留complete_execution_pending_independent_replay，后续验证状态以独立文件为准。

首次验证入口误将执行完成状态要求为普通complete，尚未重放即拒收；已保留[适配失败记录](../../audit_results/execution_guard_v8_replay_adapter_20260911/failed_status_check.json)，修正为本执行器实际的完成状态后完整通过，未重跑或改写物理数据。

## 全候选副作用检查

[全部96条原路线的独立审计](../../eval_results/response_v7_all_routes_latest_map_review_20260911/LATEST_MAP_REVIEW.md)检查2,892个已记录动作，发现守卫会改变10条路线：上述4条碰撞及另6条原成功路线。9个首次拒绝点具有当前地图下的预算内返回；T5/storage的候选5则当前位置footprint已不安全，按当前硬规则不能启动返回。

受影响的原成功路线包含训练模型所选的4/16个历史、36/144个模型—历史选择。因此不能用本4条回归替换旧训练失败标签，再沿用另外92条的全部成绩。下一阶段须单列新版本验证守卫改变的全部路线，并保留当前位置不安全时的停止与任务未完成；目前另6条只有原轨迹上的判断审计，没有新守卫下的物理执行或重建收益。
