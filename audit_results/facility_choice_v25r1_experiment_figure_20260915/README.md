# P00 首批实验示意与首条观测结果

本目录包含两张用于论文草稿或交流的静态图。图中英文标签采用米制等比例坐标；两图均已目视检查。新增 PNG 与 PDF 共 1,209,881 字节，小于 2 MiB。绘图没有构造世界、调用传感器、执行动作、融合 TSDF、重新求成本或计算质量。

## 固定采集设计

[p00_experiment_design.png](/root/NSO/audit_results/facility_choice_v25r1_experiment_figure_20260915/p00_experiment_design.png) 与 [PDF](/root/NSO/audit_results/facility_choice_v25r1_experiment_figure_20260915/p00_experiment_design.pdf) 展示 P00 的两种实际类别／外形排列，避免把两个复杂设施画进同一真实场景。实体柜体、外露附件、原有墙和背板、10 面 V25 r1 隔板均来自冻结声明或静态元数据。俯视图不表达高度：原有四面工位背板继承 V24.1 的 4.8 m 高度，新隔板为 2.4 m。

灰色虚线是两臂共享的 234 动作前缀；A 蓝色、B 橙色表示两个可替代的补看选项，并非同一任务依次执行两臂。实线到达两个声明视点，虚线为返航，黑星是原锚点。原地转向和重复经过的位置在图上重叠，但全部计入动作数。A 的总动作是 354（额外 120），B 为 340（额外 106），两者预算均为 400。坐标和路径直接读取冻结 `route_catalog`，未重新规划；附件从对应排列的静态证据读取。

建议图注：**P00 固定采集设计。两种排列共享实体隔板、公共前缀及各选项的付费路线，复杂外形在设施 A、B 之间交换。本图说明 GT 设计采集条件，不代表自主探索或语义策略结果；额外信息也可能通过主动几何探看获得。声明视点的射线可见条件不保证建档质量。**

## 首条任务的观测网格

[case00_observed_projections.png](/root/NSO/audit_results/facility_choice_v25r1_experiment_figure_20260915/case00_observed_projections.png) 与 [PDF](/root/NSO/audit_results/facility_choice_v25r1_experiment_figure_20260915/case00_observed_projections.pdf) 仅读取 case 00 已封存的四份 `observed_mesh.npz`：两实例槽位分别在公共前缀结束（234 动作）及 A 补看并返航结束（354 动作）的观测网格。该条物理任务及独立进程回放均已完成；本图没有读取其他分支或判断整个矩阵。

图绘制全部顶点，不抽样、不裁剪、不补面，前后采用同一标尺。它是观测表面的点投影，**不是评价器的填充外轮廓，也不等于原始深度点云或完整设施模型**。槽位图中还有柜体附近的结构；这值得后续检查，但此图不能单独证明实例关联、开口、碎片或低分的具体根因。已经记录的评价窗内 XY IoU 为：复杂设施 0.125731→0.216698，简单设施 0.164558→0.166585；这里只引用原成绩，不由图重算。完整槽位显示范围与评价窗不同，不能直接从点图读出这些分数。

建议图注：**同一实际任务中，公共前缀与补看返航后的观测网格三向投影。蓝色为前缀，橙色为终点；全部保存顶点按统一米制标尺叠加。单条固定采集的可视化不构成语义选择收益或低分根因的验证。**

## 来源与复现

`source_and_zero_action_receipt.json` 记录冻结服务成本、静态 r1 清单及三份纯几何源文件的 SHA；`case00_read_only_receipt.json` 记录首条结果、回放回执及四份网格的 SHA。输入均核对原封存清单，绘图后再次核对。`verification.json` 记录零新增物理操作、图片体积和读取后输入完整性；`artifact_hashes.json` 封存本目录除自身外的所有产物。

在 `/root/NSO` 执行下列命令可重绘。仅使用 Matplotlib / NumPy 和标准库；不导入模拟器、策略或评价器。PDF 的生成日期可能随重绘变化，不应以输出文件 SHA 代替输入一致性核验。不要覆盖本封存目录；需要重绘时先复制整个目录到项目中新的同层审计目录。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-3d/bin/python -B audit_results/facility_choice_v25r1_experiment_figure_20260915/plot_experiment.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-3d/bin/python -B audit_results/facility_choice_v25r1_experiment_figure_20260915/plot_case00_observed_projections.py
```
