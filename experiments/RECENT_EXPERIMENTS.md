# NSO 近期实验归档

更新于 2026 年 11 月 26 日。各实验入口给出对应的协议、评估数据和复跑材料。货架遮挡实验及西侧返航修复的结果来自仿真。

| 日期 | 实验 | 成果入口 |
| --- | --- | --- |
| 11 月 26 日 | RAS 四楼因子消融（8 轮新增、8 轮复用） | [协议与结果](offline_20261126_ras_regional_factorial/README.md) · [交互与逐楼结果](offline_20261126_ras_regional_factorial/factor_audit/A_FACTOR_RESULTS.md) · [原生地图图件](offline_20261126_ras_regional_factorial/paper/checkpoints_a96.pdf) |
| 11 月 26 日 | Replica／HM3D 地面支撑修正复跑（12 轮） | [原始证据与协议](offline_20261126_external_support_repair/README.md) · [修正前后对照](offline_20261126_external_support_repair/FINDINGS.md) · [新停止点核验](offline_20261126_external_support_repair/RERUN_COLLISION_AUDIT.json) |
| 11 月 26 日 | Replica／HM3D 异源楼层验证（12 轮） | [场景与复跑](offline_20261126_ras_external_scans/README.md) · [逐楼终点与失败](offline_20261126_ras_external_scans/FAILURE_ANALYSIS.md) · [核验](offline_20261126_ras_external_scans/results/formal/verification.json) |
| 11 月 24 日 | TARE 共用传感器对照（四楼） | [协议与复跑](offline_20261124_tare_four_floor/README.md) · [逐楼结果](offline_20261124_tare_four_floor/RESULTS.md) · [逐帧核验](offline_20261124_tare_four_floor/results/formal/transport_audit.json) |
| 11 月 24 日 | Explore Lite 和 ARiADNE 同楼层对照（8 轮仿真） | [协议、结果与复跑](offline_20261124_external_exploration_baselines/README.md) · [Explore Lite 配对](offline_20261124_external_exploration_baselines/results/explore/pairs.csv) · [ARiADNE 配对](offline_20261124_external_exploration_baselines/results/ariadne/pairs.csv) · [TARE 接口验证](offline_20261124_external_exploration_baselines/work/tare/PROBE.md) |
| 11 月 23 日 | 四个独立楼层的区域探索验证（12 轮仿真） | [协议、原始记录与复跑](offline_20261123_region_independent_floors/README.md) · [逐楼对照](offline_20261123_region_independent_floors/paper/TABLES.md) · [覆盖与联合质量图](offline_20261123_region_independent_floors/paper/coverage_joint_quality.pdf) · [实验简报](offline_20261123_region_independent_floors/paper/BRIEF.md) |
| 11 月 23 日 | 货架观察路线的时间代价与新增表面 | [协议与复跑](offline_20261123_shelf_time_opportunity/README.md) · [配对实验简报](offline_20261123_shelf_time_opportunity/paper/BRIEF.md) · [核验记录](offline_20261123_shelf_time_opportunity/completion_audit.json) |
| 11 月 22 日 | 六组货架遮挡配对实验 | [实验协议与复跑](offline_20261122_shelf_occlusion_pairs/README.md) · [逐回合](offline_20261122_shelf_occlusion_pairs/results/formal/episodes.csv) · [逐布局](offline_20261122_shelf_occlusion_pairs/results/formal/pairs.csv) · [失败记录](offline_20261122_shelf_occlusion_pairs/results/formal/failures.csv) · [结果核验](offline_20261122_shelf_occlusion_pairs/completion_audit.json) |
| 11 月 22 日 | 货架实验补充诊断 | [全局表面差距的传感器回放](offline_20261122_shelf_occlusion_pairs/followup/global_residual/SENSOR_RESULTS.md) · [局部候选短分支](offline_20261122_shelf_occlusion_pairs/followup/local_witness/RESULTS.md) · [D+ 隐面收益](offline_20261122_shelf_occlusion_pairs/followup/local_scoring_diagnostic/RESULTS.md) · [西侧返航复核](offline_20261122_shelf_occlusion_pairs/followup/west_projection_native/RESULTS.md) |
| 11 月 22 日 | 区域全局选点修订 | [修订与留出楼层](offline_20261122_region_global_revision/README.md) · [目标区域归属与沿途收益对照](offline_20261122_region_global_revision/next/global_attribution/results/RESULTS.md) · [旧回合技术诊断](offline_20261122_region_global_revision/next/local_technical/RESULTS.md) |
| 11 月 22 日 | 完整楼层探索与局部巡检 | [楼层实验](offline_20261122_region_structure_floor/README.md) · [局部配对数据](offline_20261122_region_structure_floor/local/results/paired_result.md) · [轨迹与表面图 PDF](offline_20261122_region_structure_floor/local/results/local_inspection_trajectories_and_surfaces.pdf) · [可编辑 SVG 压缩包](offline_20261122_region_structure_floor/local/results/local_inspection_trajectories_and_surfaces.svg.zip) |
| 11 月 22 日 | 路线干预与几何评分 | [固定分叉干预](offline_20261122_route_intervention/README.md) · [几何首选路线](offline_20261122_route_top_geometry/README.md) · [加权几何对照](offline_20261122_weighted_geometric_control/README.md) |
| 11 月 21 日 | 候选路线诊断 | [路线收益与排序](offline_20261121_candidate_route_yield/README.md) · [路线执行与收益保留](offline_20261121_candidate_route_persistence/README.md) |

## 论文与图表

- [当前小论文主文 PDF](../docs/thesis/submission_focus_20261108/main.pdf) · [LaTeX 源稿](../docs/thesis/submission_focus_20261108/main.tex) · [补充材料](../docs/thesis/submission_focus_20261108/supplement.pdf)
- [货架遮挡实验 Results & Discussion](offline_20261122_shelf_occlusion_pairs/paper/manuscript_results_discussion.md) · [图注](offline_20261122_shelf_occlusion_pairs/paper/captions.md)
- 配对结果图 [PDF](offline_20261122_shelf_occlusion_pairs/paper/paired_shelf_area.pdf) · [SVG](offline_20261122_shelf_occlusion_pairs/paper/paired_shelf_area.svg) · [PNG](offline_20261122_shelf_occlusion_pairs/paper/paired_shelf_area.png)
- 机制图 [PDF](offline_20261122_shelf_occlusion_pairs/paper/mechanism_case.pdf) · [SVG](offline_20261122_shelf_occlusion_pairs/paper/mechanism_case.svg) · [PNG](offline_20261122_shelf_occlusion_pairs/paper/mechanism_case.png)
