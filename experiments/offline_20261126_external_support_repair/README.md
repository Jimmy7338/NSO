# Replica 和 HM3D 异源场景地面支撑复核

在同一份原始网格上复核既有失败，再冻结仿真器的支撑几何修正并重复原配对矩阵。本批使用 Replica Apartment 1 与 HM3D 00770，每个场景含两个原起点。每个起点运行原区域策略 `region_S`、使用目标区域归属与沿途收益修订的 `combined_S`、最近可达前沿 `F_strict`，每轮最多 192 个计费动作。两个区域策略仍沿用原始设置，场景没有可用的结构类别输入，运行中采用几何回退。

## 支撑判定

原适配器仅保留整片落入地面高度带并通过法线阈值的三角面。两处 HM3D 失败出现在同层接缝，四处 Replica 失败涉及被排除的有支撑斜面。冻结修正保留穿过高度带的近水平三角面在带内的部分。对完全位于已提取地面外轮廓内的局部遗漏，则要求原始几何的投影覆盖和近地高度连续性。单处扫掠允许补认的总面积上限为 0.006 m²。扫掠半径为 0.20 m，原车身碰撞高度带为 0.15–1.20 m，感知仍直接对原网格采样。[实现](repair_geometry.py) · [校验记录](VALIDATION.json)

对六条原失败边分别核验了正反向，并逐步回放 11 条旧动作前缀校验位置、深度、RGB 和激光。独立构造的用例包含缓坡、接缝、无观测区、0.20 m 落差和车身障碍。原支撑带内 0.07 m 凹坑在新旧版均可通过。这些测试描述仿真模型的通过条件，实际环境的防跌落需单独验证。[独立用例](validation/repaired_controls.json)

## 配对与复核入口

- [冻结试验协议](FORMAL_PROTOCOL.json)与[执行记录](execution/formal/phase_receipt.json)列出全部 12 轮。
- [结果](RESULTS.md)与[逐轮数据](results/formal/episodes.csv)使用修正后可达网格计算覆盖，表面评价点集与原评估相同。覆盖与场景 F1@5 cm 相乘得到联合指标。固定步数的缺失结果留空。
- [原可达网格的逐步对照](results/formal/original_mask_matched_paid.csv)使用共同旧分母，仅比较双方实际到达的相同步数。
- [重跑停止点核验](RERUN_COLLISION_AUDIT.json)将新的停止定位到原始网格上的扫掠边。
- [修正前结果](../offline_20261126_ras_external_scans/RESULTS.md)保留原始失败。扫描场景本身只有两个独立来源，同一来源的两个起点按场景内重复处理。

数据源见 [HM3D 官方仓库](https://github.com/matterport/habitat-matterport-3dresearch)和 [Replica 官方测试场景发行](https://huggingface.co/datasets/ai-habitat/habitat_test_scenes)。数据资产从各自上游获取，本归档提供配置、记录、校验值和脚本。
