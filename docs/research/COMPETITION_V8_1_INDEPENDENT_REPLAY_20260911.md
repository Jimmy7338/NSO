# V8.1 双设备有限开发实验：独立结构放行与完整物理回放

本轮 24 条共用候选分支均通过独立回放。752 个分支付费动作全部复现，24 条原始往返路线全部完成，24 次返回包含原始朝向，碰撞与执行失败均为 0。回放用时 150.94 秒。该结论证明执行记录与指标计算可复现，不等于语义收益门槛通过、真实机器人安全保证、独立泛化验证或完整 ANS 四模块运行证据。

## 范围与状态

| 检查 | 结果 |
| --- | --- |
| 原 V8 四历史结构审计 | 独立审计完成，但整版不放行：P0 两种布局各缺 east_entry/deep；原失败记录保留 |
| 新 V8.1 四历史结构审计 | 放行；每组均为六角色，两个观测簇分别对应真实不同设备位置 |
| 候选边界修正 | 只按观测 occupied 栅格完整单元边界修正导航目标；独立按四个格角投影计算一致 |
| 完整原始传感器回放 | 604 对唯一前缀 RGB-D/雷达；752 对分支 RGB-D/雷达，所有字段一致 |
| 独立执行门禁 | 当前/后继完整已知自由 footprint、带朝向的 BFS 最短返程、剩余动作预算均一致 |
| 网格与地图 | 各分支 prefix、arrival、final mesh 顶点/三角形及 final belief/camera_seen 数组精确一致 |
| 真值参考 | 独立按 seed 2026、32,000 初始样本和固定可达位置格点重新构造，与封存点/类别/权重精确一致 |
| 指标 | 独立 P/R/F1@2/5cm、固定旧面误差、二维覆盖、真实新可见面积、两位置面积及阶段去重复算一致 |
| 联合 AUC | 按原 prefix/arrival/final 稀疏检查点，在 48 动作窗口插值并对早停后末值保持，复算一致 |
| 原始资产保护 | 回放前后逐文件重新核验 SHA；未改原始传感器、指标、网格、metadata 或封存源代码 |

四组原始前缀共 600 个付费动作，因此开发实验唯一历史的记账为 600 个前缀动作加 752 个分支动作。为避免复用执行状态掩盖差异，回放为每条分支从初始状态重新执行同一前缀，共作 3,624 次前缀传感器对比；这些重复回放没有扩增实验样本量，也没有保存重复 raw 或网格。

## 独立实现与复用边界

新增脚本 `scripts/replay_competition_v8_1.py` 从通过哈希验证的 `sources.zip` 在临时独立目录加载世界与映射器。它没有调用生产 `ObservedExecutionGuard`、候选/路径规划器、评分器、执行器、重建评价器或表面可见性辅助函数。

脚本独立实现执行决策状态机、EDT footprint、反向带朝向 BFS、真实相机位姿矩阵、三角面面积采样、可达格点可观测参考过滤、当前有效深度像素与首交点可见性、P/R/F1 与旧面误差、覆盖率、位置面积、阶段去重和联合 AUC。对返程最短路存在并列的情况，验证记录路线的安全性、动作总数和最短距离，不强制采用同一并列路径。

复用部分是封存的物理传感器世界、`SemanticHistoryMapperV3`/TSDF、RGB-D/雷达契约，以及 Open3D 距离/射线引擎、NumPy 随机数和 SciPy EDT。这不是独立 SLAM 实现。模拟深度/位姿、人工设备颜色标识及精确位姿条件也没有被回放消除。

物理面积权重仍为唯一外表面总面积除以**过滤前** 32,000 个参考样本，无类别加权，无按可见子集重新归一化。连续首交点加最近有效深度像素是原冻结的有限图像可见性定义，不应写成真实重建精度本身。AUC 仍是稀疏检查点插值，未声称每一步都重建并测量了 F1。

## 可追溯绑定

- 运行目录：`eval_results/competition_v8_1_execution_20260911`
- 严格回放结果：`verification.json`，`status=passed_full`、`passed_full=true`、`partial=false`、`max_branches=null`、`branches_checked=branches_total=physical_branches=24`。
- 运行 manifest SHA256：`0803ca34680dfca0e160b67d1124f4480874034809aedde06eb906fe6ce049ab`
- 源代码归档 SHA256：`0fad4760922dbe2a6362cf5cb0d1a4d72e903b802338e49f11cea0ed05c32549`
- 预执行 seal SHA256：`c930250aa35a9f787ccb908e17528229ee050828fb5aa84c12050d53cb184e3a`
- 独立结构审查 SHA256：`5b2676e8c4a973522fdd4d8a344a08bf2bc365844427b359481fbf2dcaa7df3f`
- 独立回放脚本 SHA256：`aa09e7f321e914d1c4d7dea728461a6baf6e92c8219ab7c359c4ff514420a85f`

`verification.json` 没有重复抄写完整 `source_sha256` 字典。验证链为：预执行 seal 的源字典与原 metadata 严格相等，归档文件清单与字典严格相等，每个归档 payload 的 SHA 与字典相等，完整运行 manifest 在回放前后重查；报告同时绑定源归档、预执行 seal 和运行 manifest。回放脚本原样副本及结果副本保存在独立 `eval_results/competition_v8_1_replay_review_20260911` 中，不追加至物理运行的封存清单。

复现命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-3d/bin/python scripts/replay_competition_v8_1.py --run eval_results/competition_v8_1_execution_20260911
```

现有数值结果的语义/几何对照判断仍由预先冻结协议及独立结果分析负责。回放通过不会把排他性或 F1 门槛的失败改为通过。
