# 两个遮挡摆位的失败诊断与独立修复试验

## 东侧遮挡的 Open3D 崩溃

原 G、S 各自第二次在第 180 步检查点之后以 SIGSEGV 退出。故障定位到 Open3D 0.19.0 的 `RaycastingScene.add_triangles`。最后一次登记的模板是有限、无退化的八顶点十二三角形盒体。相同网格复制到新进程独立登记 2,100 次均成功。现有证据把嫌疑缩小到长行程内反复重建模板缓存时的对象生命周期，尚不足以定位 Open3D 内部的具体错误。

独立诊断版本在模板缓存失效、需要重建时先回收不再可达的对象，再调用同一个原缓存函数。G 的既有诊断和新增 S 的诊断都执行 192 步、成功返航，且零模拟碰撞。新增 S 在起点、共同分叉及第 48、96、144 步的检查点 JSON、融合网格和栅格与 S 原崩溃版本逐字节一致。诊断 G/S 的 192 条执行与传感器摘要完全相同，终点栅格和三维网格也逐字节一致。原冻结矩阵仍按原来的两个缺测终点报告。 [S 恢复回执](east_screened_S_gc/receipt.json)

## 西侧遮挡的返航断连

G、S、X 的 61 个动作和原始传感器摘要一致。固定动作逐帧重放逐字复现了全部 62 张占据栅格。第 60 步的地图对同一原生保护提供 42 个动作的返航路径。第 61 步在同一位置右转后，RGB-D 将栅格 `[23,16]` 从自由改为占据。这是该帧唯一的标签变化，当前车体和锚点仍安全。邻近的 `[24,15]` 成为唯一新增的不可通过足迹中心，切断了观察图中的返航路径。 [原帧和投影回执](west_screened_passive/receipt.json)

栅格 `[23,16]` 没有产生激光终点。44 个深度点的独立渲染标签都来自真实隔墙 `wall_09` 的西侧面。该面的源网格 x 恰为 3.4 m，深度重投影到该处的 x 略小于 3.4 m，按下取整于是被写入左侧相邻列。真实隔墙仍然存在。用连续源几何核验第 60 步计算的返航路径，所有扫掠均有效，中心线距真实障碍的最小距离为 0.36056 m，机器人半径为 0.20 m。

第 24 步还有一次更早的去程目标保护。初选路线的下一格 `[34,24]` 从第 6 步起已受 `[35,25]` 影响。该格来自隔墙 `wall_04` 端面上的八个 RGB-D 采样，激光没有在这里记录终点。第 24 步仍有 26 动作返航路径，重规划后继续运行至第 61 步。 [早期去程拒绝的观测来源](west_screened_rejection_provenance/receipt.json)

一种单精度舍入诊断在距离栅格整数边界仅一个 float32 ULP 时按整数边界解释，改动仅作用于 RGB-D 障碍端点写格。相同 62 帧重放中，44 个样本都保留并落在隔墙一侧的 `[23,17]`。同一原生保护重新找到 41 动作的路径，其扫掠均通过独立连续几何核验。逐帧检查发现原来已占据的真实障碍中心没有丢失，也没有新获准的足迹中心违反连续几何。该版从首帧起出现两格差异，属于独立仿真映射诊断。 [成对精度回放](west_screened_precision_shadow/receipt.json)

独立从首帧自主运行时，RGB-D 单端点修正于第 50 步改变实际转向，第 51 步再次因返航断连停止。这次是三束**激光**，没有深度样本落在致断连的 `[22,16]` 与 `[23,16]`。三个激光端点反算 x 坐标也略小于隔墙 x=3.4 m 的整数格边界。两次闭环均未发生模拟碰撞，均未返航。 [独立闭环记录](west_screened_G_precision_native/receipt.json) · [逐激光端点回执](west_screened_G_precision_native/failure_observation_provenance.json)

随后将同样的 ULP 归一规则同时作用于 RGB-D 障碍端点和有效激光回波端点。用于标记自由空间的激光射线保持原有投影。两条中止轨迹在各自中止帧均重新获得 41 动作的返航路径，独立连续几何核验的中心线最小间距分别为 0.30 m 与 0.36 m。不过，两条轨迹在第 13 至 36 步均有一个原本占据的真实障碍中心丢失，且在第 34 至 36 步各出现一个新获准但独立连续几何判为无效的足迹中心。第 13 步，来自隔墙边 x≈2.0 m 的激光端点被推到另一侧，丢失真实墙内的 `[22,9]`。因此双端点的通用舍入只完成等输入诊断，没有运行独立闭环。精确边界处从哪侧看到的证据是下一版需要处理的变量，当前保护继续按原观察栅格处理。 [历史轨迹结果](west_screened_dual_endpoint_shadow/archived_detail.json) · [仅 RGB-D 修正版轨迹结果](west_screened_dual_endpoint_shadow/rgb_only.json)

## 新版长走廊对照的独立进程恢复

长走廊 `rev26_corridor_receiving` 的原区域策略两次在第 180 步检查点后出现同一 Open3D 堆栈崩溃。只加入东侧诊断用过的缓存回收，按原 `FORMAL_FREEZE.json` 重新运行原区域策略。新增诊断完成 192 步、返航成功、零碰撞。旧两次留存的起点及第 48、96、144 步检查点 JSON、栅格和融合网格全与恢复轮逐字节相同。原两轮第 180 步仅有进程心跳。 [缓存恢复回执](rev26_corridor_region_S_gc/receipt.json)

冻结评价器重新计算恢复轮，正确自由区覆盖为 **98.29%**、全场景 F1@5 cm 为 **93.25%**、联合指标 J 为 **91.65%**，第 **135** 步达到 95% 覆盖。无上限面积策略第 **82** 步达到 95% 覆盖，比恢复轮早 53 步，但终点 J 为 **86.91%**，低 **4.74 个百分点**。严格前沿终点 J 为 **93.07%**，在第 **103** 步达到 95%。这是原对照技术中止后的独立敏感性结果，原两次失败仍保留在正式矩阵。 [冻结评价与配对端点](rev26_corridor_region_S_gc/measurement.json)

## 复跑

在中控原目录使用 `/home/ouc/nso_robot/envs/science312/bin/python`，设置 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`。脚本使用只能新建的输出目录，运行前将同名已生成目录归档。

```bash
python west_replay.py
python west_replay.py --out west_screened_rejection_provenance
python west_precision_shadow.py
python east_recovery.py
python west_closed_loop.py
python west_new_cut.py
python west_dual_shadow.py archived
python west_dual_shadow.py archived --details
python west_dual_shadow.py rgb_only
python revision_corridor_gc.py
python measure_corridor_gc.py
```

独立源几何只用于断连后的技术核验，规划和地图更新仍读取当前仿真传感器。批量实车运行、投影误差更大的传感器或其他布局的安全效果尚无这批数据支持。
