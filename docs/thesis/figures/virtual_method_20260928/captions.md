# Publication-sized method and evidence figures

## Method flow / 方法信息流

**English.** CPU implementation of category-conditioned active observation.
Public templates, a common safe pose graph and the category–structure relation
support the four interfaces. Only already executed RGB-D, scan and pose packets
enter the online ledger and measured belief correction. The finite-budget
planner chooses the next atomic action; a legal-edge and return-budget guard
precedes execution. Actual depth and scan data build the TSDF and occupancy map;
public template surfaces are not fused into the measured mesh. Reference geometry
enters only the offline evaluator after outputs have been sealed. Dashed upper
arrows denote public knowledge, solid arrows executed information and control.
This diagram describes the CPU atomic-action controller used in the controlled
virtual experiments. The four interfaces separate observation representation,
belief correction, action planning and execution safeguards.

**中文。** 类别条件主动观测的 CPU 信息流。四模块共享公开模板、安全姿态图及类别—结构关系，
仅使用已执行动作取得的 RGB-D、扫描和位姿更新账本与实测构型权重。有限预算规划器选择
下一原子动作，经图边与返航预算检查后执行。实际深度与扫描构建 TSDF 和占据地图，
公开模板不补入预测网格；参考几何仅进入封存后的离线评价。上方虚线表示公开知识，
实线表示实际观测与控制信息。本图对应受控虚拟实验所用的 CPU 原子动作控制器，
四个接口分别承担观测表示、信念纠正、动作规划与执行约束检查。

## Publication main evidence / 主结果

**English.** Saved controlled virtual results, replotted at publication width.
(A) All four G/S paired conditions from two seen parent layouts; labels show
absolute differences in J5 and retain both ties. (B) The same four conditions
under four CPU mechanisms. SWAP-I and VISTA-I are mechanism adaptations, not
original complete systems. (C,D) Coverage and surface F1 separately: small
markers show all four conditions and horizontal colored lines show means;
light bars extend from zero to the mean and are not uncertainty intervals.
G/S in B–D reproduce A and do not add independent samples. Mean coverage is
identical across all four methods. No confidence interval or significance claim
is made; parent layouts are seen, related design layouts.

**中文。** 按正文宽度重排的已保存虚拟结果。（A）两个已见父布局、各两个构型的完整 G/S
配对，数字为 J5 绝对差，保留两组零差。（B）共同四条件下的 CPU 机制比较；SWAP-I、VISTA-I
为机制适配，非原作者完整系统。（C、D）分别展示覆盖与表面 F1；小点保留全部条件，
彩色横线为均值，浅色柱从零延伸至均值，不表示置信区间。B–D 中的 G/S 与 A 复用相同
证据，不增加独立样本，四种方法平均覆盖完全相同。图中不作统计显著性或未见布局泛化声明。

## Misleading-prior correction / 错误先验纠错

**English.** Complete saved development results on P00. (A) G, S, X (swapped
class), and Xnf (swapped class without correction) for both configurations.
(B) X minus Xnf at all three distance thresholds. Xnf disables both measured
geometric correction and anticipated future correction, so the contrast measures
the overall correction policy. The h1 negative effect at 2 cm is retained.
Neutral line colors distinguish configurations, not methods. These development
tasks are separate from the confirmation cohort and are not pooled with it.

**中文。** P00 开发批次的完整错误先验对照。（A）两个构型下 G、S、错误类别 X 及 Xnf
的全部终点。（B）2、5、10 cm 下 X−Xnf；Xnf 同时关闭实测几何更新和未来纠错预期，
比较对象为整体纠错政策。保留 h1 在 2 cm 下的负值，中性灰线及不同符号表示构型，
不表示方法。该开发批次不与确认批次混池。

## Recorded belief timeline / 已保存信念时间线

**English.** Complete saved P00/h1 development timelines for all four methods,
including observations 0–42. (A) G/S; (B) X/Xnf. Weights at index t follow the
observation obtained after executing t paid actions and precede action t+1.
Thus observing 18 precedes action 19, and observing 28 precedes action 29;
the horizontal axis is action count, not physical time. Curves are unsmoothed
post-update steps of uncalibrated belief weights. The true h1 designation is
used only for offline interpretation. This mechanism illustration adds no
independent tasks or counterfactual terminal outcomes.

**中文。** P00/h1 开发批次四方法的完整已保存时间线，保留观测 0–42。（A）G/S；
（B）X/Xnf。索引 t 的权重在执行 t 次付费动作并消费观测后产生，随后选择动作 t+1；
因此观测 18 对应下一动作 19，观测 28 对应下一动作 29。横轴为动作计数而非秒，
曲线为未平滑、未校准的更新后构型权重，真实 h1 标识仅用于离线解释。
该图不增加独立任务或反事实终点。

All four figures are 7.05 inches wide with 9 pt body labels and external captions.
Method illustration and saved-data presentation only: zero new experiments,
sensor packets, planning calls, TSDF integrations or surface evaluations.

Reproduce: `python3 scripts/plot_virtual_paper_method_20260928.py --output NEW_DIRECTORY`.
Inputs are checked against the saved source manifest. The new manifest hashes
the implementation descriptions, plot script, saved CSVs and generated outputs.
