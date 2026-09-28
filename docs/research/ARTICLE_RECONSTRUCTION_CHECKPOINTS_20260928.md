# 原轨迹中间重建与三维质量演化（2026-09-28）

本轮按事先写入的[固定协议](../../audit_results/article_stage_20260928/reconstruction_checkpoints/protocol.json)，完成原V36全部八条G/S轨迹在18、24、30、36、42动作处的离线重融合与表面评价，共40个检查点。没有创建World、查询新传感、调用规划器或改变路径。检查点属于八条旧轨迹的过程测量，不是40条新实验。

## 交付

- [五分量过程曲线](../thesis/figures/article_reconstruction_20260928/checkpoint_quality_components.pdf)：四条件分别展示C_map、P@5cm、R@5cm、F1@5cm与J5；付费动作横轴下列出真实累计平移距离。
- [实际TSDF阶段网格](../thesis/figures/article_reconstruction_20260928/checkpoint_actual_meshes.pdf)：固定18/30/42步，全部四个G/S配对，共24个面板；没有补洞、平滑、模板补面或网格简化。
- [40检查点完整数值CSV](../../audit_results/article_stage_20260928/reconstruction_checkpoints/checkpoint_metrics.csv)、[执行结果](../../audit_results/article_stage_20260928/reconstruction_checkpoints/result.json)、[封存清单](../../audit_results/article_stage_20260928/reconstruction_checkpoints/seal.json)。各case目录保存五个时刻的实际提取网格、二维地图、裁剪记录和完整2/5/10 cm评价。
- [图源与输出清单](../thesis/figures/article_reconstruction_20260928/manifest.json)、[相机和全三角面片显示记录](../thesis/figures/article_reconstruction_20260928/display_geometry.json)、[双语图注](../thesis/figures/article_reconstruction_20260928/captions.json)。两图均有PDF、SVG及260 dpi PNG。

重建入口为`scripts/replay_article_reconstruction_20260928.py run`。该科学执行源码按freeze绑定，不因后续排版改动而改变。最终绘图入口为`scripts/plot_article_reconstruction_20260928.py`，仅在原绘图函数导出前调整最后一行距离文字位置和网格数字的白色底衬；不运行重建或评价。首次排版图位于`tmp/article_reconstruction_layout_draft_20260928/`，不用于论文。

## 一致性与资源

344个原始保存传感包在任何重融合前均通过旧封存SHA核验。每条从第0帧开始按原顺序更新一次，使用原`ObservedRuntimeMapperV10`、4 cm体素、12 cm截断、4 m轴向深度上限、原RGB-D、扫描与精确保存位姿。每一步的二维地图摘要均与原trace一致。各case的五份预测先封存，再读取原场景的评价几何；覆盖分母直接复用原`evaluation_floor.npz`的reachable掩码。

使用相同`SurfaceMeasurementV34`、原完整外表面精确率参考和完整垂直表面召回参考，固定32768点与原采样种子。预先声明原18/42步的二维地图、canonical几何SHA、参考几何严格一致，数值绝对容差1e-10、相对容差0。实际16个已有检查点全部通过，最大指标差为 **0.0**；没有调整容差或用旧分数代替新测量。

实际计数为344次保存包重融合、40次网格提取和40次离线表面评价，1969项一致性检查通过；八case累计墙钟91.90秒。单线程、顺序执行，资源上限250 MiB且至少保留1 GiB自由空间。最终重建与图包约28.3 MiB，显著低于上限。

预检曾因当前捆绑Open3D没有distribution metadata而在版本记录处退出。当时尚未生成freeze、创建case或执行重建/评价，保留[预检失败记录](../../audit_results/article_stage_20260928/reconstruction_checkpoints/preflight_metadata_failure.json)。随后按原V36 runner已有做法读取模块`__version__`并记录来源；实验协议、轨迹、评价器与阈值未变。这不是重跑科学失败样本。

## 过程结果

所有G/S配对在全部五个检查点的C_map均相同；两个h0配对的全部分量也相同。到第30步检查点，二维覆盖已经达到该轨迹的终点值，之后的J5变化来自表面质量，而非新增二维覆盖。h1两个配对的三维召回和F1逐步分开。

| 条件 | 动作 | G的J5 | S的J5 | S−G |
|---|---:|---:|---:|---:|
| P00/h1 | 18 | 0.322016 | 0.322016 | 0 |
| P00/h1 | 24 | 0.353838 | 0.353018 | −0.000819 |
| P00/h1 | 30 | 0.583138 | 0.637744 | +0.054606 |
| P00/h1 | 36 | 0.673204 | 0.759918 | +0.086714 |
| P00/h1 | 42 | 0.672703 | 0.760456 | +0.087753 |
| P01/h1 | 18 | 0.302146 | 0.302146 | 0 |
| P01/h1 | 24 | 0.333881 | 0.349640 | +0.015759 |
| P01/h1 | 30 | 0.599072 | 0.632079 | +0.033007 |
| P01/h1 | 36 | 0.638276 | 0.723778 | +0.085503 |
| P01/h1 | 42 | 0.638118 | 0.723586 | +0.085468 |

上述正负差按原值计算后四舍五入。P00/h1在第24步仍有小幅负差，第30步检查点才捕获到正差；P01/h1在第24步检查点已有正差。检查点之间没有测量，不能精确声称优势在某个未测量动作开始。曲线连线只是阅读辅助，不插值生成新成绩；末段轻微回落也原样保留。

最终两张PNG已实际打开目视核查，最后一行距离文字与坐标标签已分开；网格数值使用白色文字底衬保证可读。排版调整后再次核验全部重建源码、封存输出和图文件哈希，均通过。

P00/h1终点R@5cm由G的0.595795增至S的0.724945，F1由0.702575增至0.794224；P01/h1召回由0.555450增至0.684509，F1由0.673401增至0.763595。P00/h1的终点累计平移为G 28 m、S 26 m，P01/h1均28 m，总付费动作均42。结果表明当前条件下方向分配能够改变已观察三维表面的完整性，不需要S执行更多动作或更长路径。

中间地图可能尚未返航或未满足覆盖门，因此它们在CSV中明确标为`intermediate_checkpoint`，不能计作合格终点。18步虽然回到共同起点，覆盖不足仍不合格；24/30/36步通常尚未返航。所有42步终点资格保持原结果。

## 可直接用于中文文章的段落

为分析重建收益的形成过程，我们对原实验的八条保存轨迹在18、24、30、36和42个付费动作处进行离线重融合，保持深度观测、位姿、TSDF参数和表面评价器不变。图中分别报告二维覆盖、表面精确率、召回率、F1及联合指标，并同时列出累计平移距离。所有配对在五个检查点的二维覆盖均相同，且在第30步检查点已达到各自终点覆盖值；两个h1条件的差异主要表现为后续表面召回和F1提高。P00/h1中，第24步S的J5仍略低于G，但第30、36和42步检查点出现正差；P01/h1在第24步检查点已有正差。两个h0条件全程持平。实际阶段网格进一步显示，路径方向的不同改变了侧部结构被观测的程度；这一变化没有依靠模板补面、更多付费动作或更长路径。该过程分析复用原轨迹，不增加独立实验样本；所有中间时刻分数均与终点任务资格分开解释。

## Paragraph for the English article

To examine how reconstruction gains develop, we reintegrated the eight original saved trajectories at 18, 24, 30, 36, and 42 paid actions while retaining the recorded depth observations, poses, TSDF settings, and surface evaluator. The checkpoint analysis separates planar coverage, surface precision, recall, F1, and joint quality, and reports the corresponding cumulative translation distances. Planar coverage is identical within every pair at all five checkpoints and reaches its endpoint value by the 30-action checkpoint. In the two h1 conditions, subsequent differences are primarily associated with improved surface recall and F1. For P00/h1, S has a slightly lower joint score at action 24 but a higher score at checkpoints 30, 36, and 42; P01/h1 already shows a positive difference at checkpoint 24. Both h0 pairs remain tied throughout. The actual checkpoint meshes show different observation of the lateral structure without template completion, additional paid actions, or longer routes. This analysis reuses the original trajectories and adds no independent trials; intermediate reconstruction scores are interpreted separately from terminal task qualification.

这些段落描述当前受控布局、人工类别提示和精确位姿下的过程证据，不改变既有论文的适用范围，也不替代六父开发分支的持平/负结果。
