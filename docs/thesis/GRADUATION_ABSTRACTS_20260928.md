# 中文摘要

## 摘要

工业生产中的设备布置、工位和物料位置调整，会带来重新获取当前环境几何的需求。机器人需要在有限运动预算内主动观察通行区域与设施表面，支持布局调整后的建档。受遮挡和相机视场限制，较高的二维覆盖率并不意味着设备开口、背面和凹入区域已经得到充分重建。本文聚焦单次建图期间环境静止的设施任务，研究类别知识如何帮助机器人选择有效观察方向，以及实际几何证据如何纠正错误先验。

本文构建分层决策与执行框架，通过OV-SDF、STGHP、RPN-UQ、IGCR四个模块接口，在无独立显卡环境下连接类别先验、几何反馈、预算规划和返航检查。类别信息形成候选构型权重，首次姿态的实测深度与扫描残差修正该权重，有限诊断前瞻综合考虑当前观察和后续信息获取的价值。规划器每次仅执行首个原子动作，再根据新观测重新决策。公开模板用于预测观察价值，实际三维表面仅由取得的深度融合，预测曝光与实测地图分别维护。

本文构建CPU虚拟实验平台，以二维覆盖率与设施表面F1@5 cm的乘积评价联合建档质量，并单独报告预算、碰撞和返航资格。主实验在两个已见父布局、每布局两种构型上进行配对确认。类别方法与共同主动几何对照使用相同规划器及重建后端，四组配对取得两胜两平，平均联合指标提高6.16%，平均覆盖率相同，收益体现为表面质量改善。错误类别开发实验中，同时包含实测修正与未来纠错预期的整体政策使平均联合指标提高2.62%，但2 cm严格距离阈值下仍存在负例。共同CPU条件下的SWAP-I、VISTA-I机制比较及保存的轨迹、信念与重建网格，进一步展示了观察分配与结果之间的联系。

另一个预先固定的128项预算、噪声和同族结构扩展中，96项完成合格，32项为当前规划约束下的低预算不可行。42步原布局和结构变体的平均类别收益分别为6.25%和4.23%；54步原布局的平均差为−0.70%，明确了类别收益对预算条件的依赖。

结果表明，在公开模板、安全姿态图、受控类别输入和准确位姿条件下，类别信息能够帮助减少观察方向歧义，几何反馈能够参与纠正错误决策。本文完成了静态配置内设施主动观测与三维建档的虚拟闭环验证，为工业环境布局调整后的再次建图提供观察策略基础。

**关键词：**工业设施；主动建图；类别先验；几何反馈；预算约束规划

## Abstract

Changes to equipment arrangements, workstations, and material locations in industrial production create a need to acquire current environmental geometry. A robot must actively observe accessible regions and equipment surfaces within a limited motion budget to support documentation after layout changes. Occlusion and limited camera fields of view mean that high two-dimensional coverage does not necessarily imply adequate reconstruction of openings, rear surfaces, or recessed areas. This thesis considers facility tasks whose geometry remains static within each mapping episode, studying how category knowledge can guide useful observation directions and how measured geometry can correct misleading priors.

The proposed framework separates planning from execution and connects category priors, geometric feedback, budgeted planning, and return checks through the OV-SDF, STGHP, RPN-UQ, and IGCR interfaces. Its implementation runs without a dedicated graphics processing unit. Category information establishes configuration weights, while measured depth and scan residuals at newly visited poses update those weights. Limited diagnostic lookahead considers the value of immediate observation and subsequent information acquisition. The planner executes only the first atomic action before replanning from a new observation. Public templates support observation prediction, whereas reconstructed surfaces contain only fused measured depth. Predicted exposure and measured maps are maintained separately.

A CPU virtual platform evaluates documentation quality using the product of two-dimensional coverage and facility surface F1 at 5 cm, with budget compliance, collision avoidance, and return validity reported separately. The main confirmation experiment uses two previously seen parent layouts, each with two configurations. The category-conditioned method and an active geometric control share the same planner and reconstruction backend. Across four matched conditions, the category method achieves two wins and two ties, improving the mean joint score by 6.16% at identical mean coverage; the gain therefore comes from surface quality. In a development experiment with incorrect categories, the complete correction policy, including measured updates and anticipation of future correction, improves the mean joint score by 2.62%, although a negative case remains at the strict 2 cm distance threshold. SWAP-I and VISTA-I mechanism comparisons under common CPU conditions, together with saved trajectories, beliefs, and meshes, further illustrate the relationship between observation allocation and outcomes.

A further fixed matrix of 128 budget, noise, and same-family geometry trials yields 96 qualified completions and 32 low-budget planning-feasibility failures. At budget 42, the mean category gain is 6.25% on the original layouts and 4.23% on the variants; at budget 54, the original-layout difference is −0.70%, identifying the dependence of the category benefit on the available budget.

The results show that category information can reduce ambiguity in observation direction and that geometric feedback can contribute to correcting decisions under public templates, a safe pose graph, controlled category inputs, and exact poses. The thesis provides virtual closed-loop validation of active observation and three-dimensional facility documentation within static configurations, offering an observation-planning component for remapping after industrial layout changes.

**Keywords:** industrial facilities; active mapping; category priors; geometric feedback; budget-constrained planning
