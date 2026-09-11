# 语义覆盖规划的研究依据与验证边界

## 研究结论

截至 2026-09-10，最有希望与 NSO 既有四模块架构兼容的方向是：**以二维覆盖路线为约束，在沿途可达位置安排语义条件的三维观测任务，并用实际新增深度证据修正后续观测需求。** 适用对象应是具有明确部件、遮挡结构和检查要求的设施，例如多房间实验室的开放货架、仪器台、配电柜和工具架。语义的作用是辨别“仍欠缺哪种有效观测”，而不是令某类区域永久获得高奖励。这是待验证的研究主张，现有开发结果尚未证明完整系统优于主流方法。

近年的研究已经覆盖语义物体多视角检查、形状补全驱动 NBV、覆盖路径指导、语义成本调制和任务相关贝叶斯选视角。因此，“语义 + 多视角 + 层次规划”不能作为独立的新颖性声明。可争取的增量是：在**全向单线雷达与窄视场 RGB-D 不同的可观测范围、有限行驶预算、语义误差和模型失配**条件下，如何把语义需求转化为可兑现的三维收益，同时给覆盖损失一个明确的约束。下面将原文事实与本项目的分析、建议分开。

本文阅读范围包括各论文的方法、实验和限制；个别文献的全文读取受限，在条目中明确列出。开源状态指可访问的官方页面或仓库，不代表本机已成功构建。来源信息的机器可读版本见 [source_inventory.json](/root/NSO/docs/research/source_inventory.json)，论文可使用已核实的 [BibTeX 文献库](/root/NSO/docs/research/semantic_coverage_references.bib)。

## 原始文献证据

### 1. TARE 与双分辨率探索

**Chao Cao、Hongbiao Zhu、Howie Choset、Ji Zhang，RSS 2021；2023 Science Robotics 扩展加入 Zhongqiang Ren。** 已读 2021 方法、理论和实验；2023 已核实出版信息及作者仓库，全文后续读取超时，不借此增加理论声明。[^1][^2]

TARE 用局部密集视点和全局稀疏区域路线，按视野重叠扣除重复面积，再求访问路线。2021 定理 3 给出层次化的长度界

\[
L_{\rm hierarchy}\le L^*+4D_H+2mD_G.
\]

其推导忽略动力学，并假定相关路线子问题求到精确最优；不能移植给 NSO 的有限束搜索。实验包含地面与空中平台。官方地面代码支持 ROS Melodic/Noetic。[^1][^2]

**对 NSO 的含义：** 应保留强全局覆盖骨架和边际去重；“房间很多”本身不会令纯几何 TARE 失效。可声称受其层次组织启发，不能把同类区域束搜索命名为 TARE 复现。

### 2. FUEL

**Boyu Zhou、Yichen Zhang、Xinyi Chen、Shaojie Shen，RA-L/ICRA 2021。** 已读 FIS、全局 ATSP、局部视点精化、轨迹生成及实验。[^3]

FUEL 维护增量前沿结构，先确定全局顺序，再在局部多候选中做图搜索。其时间下界综合平移与转向，首段另含运动方向一致性成本，避免相近路线间摇摆。官方仓库明确自 2021-08 起有无需 CUDA 的 CPU 仿真，支持 ROS Noetic。[^3][^4]

**对 NSO 的含义：** 它是可落地的强几何基线来源；地面运动约束适配须公开。FUEL 的原生飞行轨迹和本项目离散小车不可直接比较实际速度。

### 3. RACER

**Boyu Zhou、Hao Xu、Shaojie Shen，T-RO 2023。** 已读 hgrid、双机交互、容量约束路由及单机覆盖路线消融。[^5]

RACER 的核心是分布式多机任务分配；容量约束控制负载，路线指导减少未完成区域间的反复迁移。论文也做单机消融。官方仓库支持 ROS Noetic，并依赖 LKH-3。[^5][^6]

**对 NSO 的含义：** 可借鉴“保持区域访问顺序”，但单机小车不应通过取消 RACER 的多机优势来声称战胜协同探索；单机版本必须单独标注。

### 4. FAR Planner

**Fan Yang、Chao Cao、Hongbiao Zhu、Jean Oh、Ji Zhang，IROS 2022。** 已读多边形提取、两层可见图更新和指定目标导航实验。[^7]

FAR 从实测障碍点形成多边形，增量更新可见图并搜索到给定目标的路线。实验比较 A*、D* Lite、RRT*、BIT*、SPARS；校园场景中 A* 的行驶时间略好。它是路线规划器，不是自行定义下一探索目标的完整覆盖策略。官方有 ROS Noetic 实现。[^7][^8]

**对 NSO 的含义：** 可作为底层或区域间导航组件；不能把无探索目标的 FAR 单独列为较弱的覆盖 baseline。

### 5. FALCON

**Yichen Zhang、Xinyi Chen、Chen Feng、Boyu Zhou、Shaojie Shen，T-RO，2024 接收/DOI、2025 卷 41。** 已读覆盖路径、顺序约束 SOP、统一模拟器、六场景基准及消融。[^9]

它以未知/自由子区域路线指导局部前沿顺序，SOP 的先后约束保证局部决策服从全局访问意图。基准对照共享 80°×60°视场、5m 深度和动力学；包括 FUEL、单机 RACER 等。官方仓库是 ROS Noetic；默认仿真仍涉及 CUDA 点云渲染和 Open3D 资源配置。[^9][^10]

**对 NSO 的含义：** 优先约束区域顺序，比将所有补看目标与前沿反复混排更有依据；窄视场本身也不是 FALCON 未考虑的问题。

### 6. Finding Things in the Unknown / Semantic Eight

**Sotiris Papatheodorou、Nils Funk、Dimos Tzoumanikas、Christopher Choi、Binbin Xu、Stefan Leutenegger，ICRA 2023。** 已读地图、深度失效历史、三种收益及仿真/实机实验。[^11]

其选视角收益为背景熵、背景近距离观察、物体更近观察的加权和，再除移动时间；已有 `frontier + object` 候选。物体观测距离要求比背景严格。实机用 D455、Jetson Xavier NX 和 Vicon，仍出现物体可见却未检出的情况。官方 ROS Noetic 仓库存在，README 同时保留 work-in-progress 提示。[^11][^12]

**对 NSO 的含义：** “语义对象多靠近一些”已是直接前作，必须作为强语义对照；类别条件隐藏结构应超过仅物体/背景二值信息。

### 7. NBV-SC

**Rohit Menon、Tobias Zaenker、Nils Dengler、Maren Bennewitz，2022 预印本、2023 修订。** 已读超椭球拟合、未知表面视点合成、效用及全部三类实验。[^13]

NBV-SC 已用预测缺失形状引导补看，并在已知自由/占据空间排除不成立的预测。其效用把缺失面、方向、新视角程度与移动成本结合。三种果实场景使用相同 120s 轨迹预算，并比较加入同样视角去重的加强对照。原文承认集中补看可延迟发现新果实。[^13]

**对 NSO 的含义：** 手工货架/箱体形状先验不具有单独新颖性；新增实测几何似然、相同候选预算和覆盖开销控制才是需要验证的区别。没有核实到官方完整 NBV-SC 仓库，不能把搜索到的第三方同名代码当原实现。

### 8. 3D Active Metric-Semantic SLAM

**Yuezhan Tao、Xu Liu、Igor Spasojevic、Saurav Agarwal、Vijay Kumar，RA-L 2024。** 已读语义因子图、相关定向越野问题、主动语义回环及实机结果。[^14]

它在预算内最大化去除观测相关性的收益，并评估虚拟回环因子对位姿与地标不确定性的影响。原文用多层建筑实飞检验语义回环；系统异步运行各模块。官方 ROS Noetic 仓库提供带位置扰动的最小仿真。[^14][^15]

**对 NSO 的含义：** 语义创新也可以位于“观测如何改变定位/地标可信度”，但当前完美位姿 TSDF 实验不构成这类 active SLAM 证据。保持回环作为既有后端组件，不应临时增加未经实现的联合优化主张。

### 9. 2025 全景 LiDAR–Camera 语义探索

**Xiaoyang Zhan、Shixin Zhou、Qianqian Yang、Yixuan Zhao、Hao Liu、Srinivas Chowdary Ramineni、Kenji Shimada，RA-L 2025。** 已读物体融合、解耦采样、ATSP、安全状态机和实验。[^16]

其任务已同时要求几何覆盖和物体各可达扇区有效观察；先放语义视点，再补足未覆盖几何。仿真使用真值 3D 检测以隔离规划，实机传感器为全景相机与 128 线雷达。地图覆盖与物体视点完成率分别报告；实机另评估点云误差。全文和作者项目页未给出可确认的完整规划仓库。[^16][^17]

**对 NSO 的含义：** 这是最接近的完整方案前作。可以用其视点扇区策略作可复算对照，但本项目的窄视场 RGB-D、单线雷达和反馈式观测停止是需实测的差异；不能称首次联合语义多视角与全局覆盖。

### 10. Quality-guided UAV Surface Exploration

**Benjamin Sportich、Kenza Boubakri、Olivier Simonin、Alessandro Renzaglia，2025 预印本；ICUAS 2026 官方议程收录。** 已读质量函数、距离推导、视点生成及实验。[^18][^19]

其饱和 TSDF 权重代理为

\[
Z_{\rm sat}(M)=\xi(M)|M|^{-1}\sum_i\min(w_i,w_{\max}).
\]

按特定距离噪声模型产生观察距离目标，ROS1/Gazebo 两场景、全向 3D 雷达、15 分钟预算。原文解释了高精度与高覆盖之间的取舍。该 TSDF 权重是依赖传感器假设的代理，不等于通用网格误差。[^18]

**对 NSO 的含义：** 不能把观测次数直接改名为精度；ZED 的距离、入射角、纹理和相关误差仍需实测校准。代码公开状态未确认。

### 11. SCOUT

**Junyu Mao、Sara Ayoubi、Vishnu D. Sharma、Ilija Hadžić、Matthew Andrews，2026-06 预印本/ICRA workshop。** 已读 UGT、语义信念更新及初步实验。[^20]

它假定已有二维占据图，以物体标签熵和可见性/视角新颖性估计补看价值：

\[
\Delta H_i=H_i(1-e^{-\alpha\,Vis_i Nov_i}).
\]

实验为两个小场景、每条件三次、最多 50 观测，比较 lawnmower 的图节点/边精确率和召回率。其新颖性函数在相反方向衰减至零，服务于外观语义而非背面重建。未确认官方代码发布。[^20]

**对 NSO 的含义：** 可参考语义置信反馈；不能引用它证明未知场景全覆盖或三维几何精度提升，更不能无验证照搬其角度函数。

### 12. SGE

**Christopher Tatsch、Yu Gu，2026-08-29 预印本。** 已读图像空间采样、TSP、失败禁忌区、仿真和实地实验。[^21]

SGE 依据可通行语义、障碍距离、任务对象与深度给像素效用，按归一化效用采样后投影。语义影响目标生成，路线管理另处理失败和回迁。校园基准中 SGE 的覆盖为 44423m³、路径 3690m，TARE 为 45875m³、2341m；前沿论文也明确存在几何效率代价。未核实其完整官方仓库。[^21]

**对 NSO 的含义：** 可主张任务领域优势，同时完整报告几何代价；SGE 的自定义语义偏好不是无类别加权重建指标提高的证据。

### 13. SAGE

**Nitin Vegesna、Avideh Zakhor，2026-05-22 预印本/CVPR 非存档 workshop。** 已读 TSDF 语义记忆、缓存、物体前沿、代价公式及实验。[^22]

它在 FALCON 上加入 CLIP，把路线成本写为 \(d'_{ij}=m_jd_{ij}\)，并限制调制范围；物体前沿不再重复加语义偏置。五张 MP3D 地图、每图两个查询、30 起点试验。正文承认 FALCON 在时间、路长和体积上更好，SAGE 的优势是对象发现；与 FTU 使用不同原生仿真/感知栈。未核实官方完整代码。[^22]

**对 NSO 的含义：** 有界成本不自动推出有限预算覆盖保证；重复语义加权的警告直接适用于当前“形状先验 + 类别近看奖励”。对象发现距离也不能替代重建精度。

### 14. BayesianNBV

**Jingsen Zhu、Silvia Sellán、Alexander Terenin，SIGGRAPH 2026。** 已读贝叶斯决策公式、蒙特卡洛实现、分类/部件任务和限制。[^23]

原文基于随机曲面后验模拟未来扫描，选择预期正效用增量最大的相机；主要实验为物体扫描。论文指出熵下降可能被“自信但错误”的分类困住。官方 MIT 仓库已经有代码，但配置、数据脚本和快速开始尚待补齐，安装要求包含 GPU PyTorch 与 PyTorch3D。[^23][^24]

**对 NSO 的含义：** 它支持从任务损失推导效用，而不支持把未校准手工拟合分数称为严格后验，亦不支持宣称当前 CPU 直接复现整套算法。

### 15. Informative Object-centric NBV

**Seunghoon Jeong、Eunho Lee、Jeongyun Kim、Ayoung Kim，ICRA 2026。** 已读 3DGS 对象监督、信息矩阵、物体任务、基准及消融。[^25]

该方法按实例概率与不透明度重加权 \(H=J^TCJ\)，同时改变重建表示和选视角。实验含 GraspNet、合成堆叠与实采物体，几何误差只在物体像素上计算。消融显示对象向量监督已贡献大量误差改善；再加 NBV 置信权重并非所有数据条件都继续改善。实现涉及定制 CUDA rasterizer，未核实官方完整代码。[^25]

**对 NSO 的含义：** 不可将它报告的最大误差降幅全部归因于语义规划；我们的对照必须共享重建后端，并同时报告全场景与预先规定的对象区域。

## 与既有四模块架构的对应

以下是本项目的方案分析，不是任何单篇论文已经证明的 NSO 结论。保留 ANS 的感知建图、全局决策、局部导航层次，并保留四个模块的职责边界；内部表示和目标函数可以修正。

| 原模块 | 建议保留的职责 | 需要改进的内部对象 | 必须有的独立证据 |
|---|---|---|---|
| OV-SDF | 开放词汇语义输入影响覆盖决策 | 从永久吸引强度改为带置信度、可消耗的对象/部件观测需求；完整形状只是潜在收益分布 | 正确/缺失/置换标签；二值对象/背景；跨类别与先验失配；真实视觉输出 |
| STGHP | 房间/区域路线与局部导航分层 | 覆盖路线给出当前和下一地区；只在预算内插入补看，记录插入成本及区域欠账 | 对同候选同成本的全局路线基线；回迁次数、未完成区域和覆盖进度 |
| RPN-UQ | 目标可达性与风险控制 | 对碰撞自由、视场完整、观测成功分别估计，置信不足时回退几何检查 | 校准曲线与到达失败率；不能以 MC-Dropout 方差直接宣称概率保证 |
| IGCR | 信息增益/覆盖驱动决策与训练接口 | 分开二维未知面积、三维未知表面、已见表面误差改善；收益扣重且能被实测证据消耗 | 同历史反事实选视角排名、在线预测/实际增益曲线、闭环主指标 |

四模块保留不等于旧神经权重已有效。CPU 原型中替代某个神经组件的规则、模型或几何检查必须在方法表中披露；系统有效性、规划机制有效性和旧 ANS 训练系统有效性应分别验证。

## 可成立的特定应用域

### 设施巡检与可操作地图

建议主任务为“未知多房间设施的覆盖式建图与关键对象表面检查”。实验室仪器、工具架、储物格和设备柜需要能用于检查、识别或后续接近的模型；只从走廊扫到障碍外轮廓并不足够。语义先验的潜在价值是可见局部几何尚不足以推断部件布局时，用类别或部件提示改善视角分配；当几何已经充分揭示结构时，预期语义增量应缩小。

场景应改变**观测信息结构**，而不是给某方法更差的传感器。推荐三个具有可解释因果变量的场景族：

1. **柜体与开放储物结构混合。** 初始视角具有相近外轮廓，但内部层板/容纳面不同；全部几何方法也允许多模型拟合和相同接近候选。逐步增加部分遮挡，检验语义先验何时有信息价值。
2. **长通道连接稀疏工作单元。** 大量易建图背景与少量需要较完整观测的工作台共存；考察沿途补看和离开区域后的回迁开销。保留低物体密度、低回迁成本条件，确认边界而不是只选最有利点。
3. **语义先验失配控制。** 货架空/满、柜门开/关、异常部件或错标签；用实测几何及时削弱错误先验。未失配的模板集合上获胜不能证明语义泛化。

场景的真实应用理由、对象集合、观测要求和评分区域均应在确认实验前冻结。合成类别真值只支持规划上限；实际部署还需要 RGB 检测或可见标签接口。不能把本方案使用的隐藏形状预测送入 TSDF，也不能让在线规划访问参考网格或未来帧。

## 理论主张应如何限定

### 语义信息价值的必要条件

令 \(G\) 为当前实测几何历史，\(S\) 为当前可见语义，\(Y\) 为未见表面或未来有效观测，\(a\) 为候选行动。只有语义能改变条件分布且这种改变影响决策时，才有可验证的增量：

\[
p(Y\mid G,S)\ne p(Y\mid G),\qquad
\arg\max_a\mathbb E[U(a,Y)\mid G,S]
\ne \arg\max_a\mathbb E[U(a,Y)\mid G].
\]

从决策集合包含关系，可以直接证明**知道更多信息的最优期望效用不会更低**：有语义策略总可忽略语义并复制无语义策略。但是这不保证一个有限候选、错误先验、未校准预测的实现不退化。工程验证必须测量条件增益，而不是只展示语义改变了轨迹。这一推导是一般决策论结果在本任务中的表达，不作为新的基础理论。

### 固定历史下可推导的去重收益

在一次规划周期固定预测概率 \(p_{jk}\in[0,1]\)，令第 \(k\) 个视点获得第 \(j\) 个表面证据的概率为 \(p_{jk}\)。若采用条件独立近似，可用

\[
F(A)=\sum_j w_j\left[1-\prod_{k\in A}(1-p_{jk})\right],\quad w_j\ge0
\]

表示至少得到一次有效证据的期望价值。其边际

\[
\Delta(k\mid A)=\sum_jw_jp_{jk}\prod_{\ell\in A}(1-p_{j\ell})
\]

非负，且当集合增大时不增，因此为单调次模。该结论只约束**固定历史、固定概率的候选集合**；在线语义后验更新、相关测量和路径长度预算可能破坏相应的自适应性质。不能直接宣称带转向/路线约束的滚动束搜索具有 \(1-1/e\) 保证。

### 有限补看开销可以证明到哪一步

对已经可执行的几何覆盖骨架 \(P_0\)，插入第 \(k\) 个视点的增量路程为

\[
\delta_k=d(a_k,v_k)+d(v_k,b_k)-d(a_k,b_k)\ge0.
\]

若每次用同一碰撞自由距离度量精确核算、保留原骨架节点与访问顺序，并维护 \(\sum_k\delta_k\le\rho L(P_0)\)，则修改后路线长度满足 \(L(P)\le(1+\rho)L(P_0)\)。加入转向/停留后，应对总时间重新定义并核算插入成本。这个界只保证所承诺骨架的附加成本，**不会自动保证未知环境、不断重规划情况下同一截止时间的覆盖不下降**。

因此 NSO 可先实现可审计的局部承诺：补看完成后返回已承诺的下一覆盖视点，或在共享预算内继续原序列；检测预算不足时回退。全局覆盖进度仍以闭环实验检验。这里的插入界是初等路径代数，不是优于 TARE 的全局最优性定理。

### 联合指标的解释

保留 \(J(t)=C_{2D}(t)F_{3D,\tau}(t)\) 和时间积分可以衡量速度与几何质量的共同表现，但 F1 同时混合精确率与完整率，不应单称“三维精度”。应同时报告 \(C_{2D}\)、F1、表面误差及共同支持区域误差。若另使用任务对象质量 \(Q_{\rm task}\)，对象列表和阈值必须在实验前来自应用需求，且同样提供给任务适配基线；它不能替代全场景指标，也不能掩盖全场景退化。

## 基线实施与验证顺序

| 层级 | 必须比较的对象 | 公平性要求 | 当前可允许的表述 |
|---|---|---|---|
| 相同后端机制 | 去细类别、仅物体性、缺失/置换标签、去反馈、去路线约束 | 同输入历史/候选池/融合与运动预算；所有标签干预不改变 RGB-D 与场景 | 机制开发或独立机制验证 |
| 相同平台策略 | 纯覆盖、强几何路线、质量 NBV、固定扇区物体检查、距离阈值语义检查 | 任务需求、传感器、动作、导航与三维后端共享；参数预算匹配 | 明确说明“依据某论文机制实现的适配对照” |
| 官方方法 | TARE 优先；FUEL 或 FALCON 作为可运行补充；FTU 是直接语义对照 | 冻结官方 commit、依赖和适配 diff；先复现官方 demo，再接共同传感器 | 仅通过对应版本复现后称“与 TARE/FUEL/FTU 比较” |
| 实体小车 | ROS1 + 已能发布深度/位姿的 ZED + 单线雷达 | 统一速限、帧率、里程计、地图后端；标定、实际感知和失败均记录 | 实际运行后才能声称实车有效 |

建议先以同一历史的候选反事实实验确认“类别确实预测了可获得的新证据”，随后固定候选、后端和代码做闭环确认。主场景和普通控制场景应同时保留，统计单元使用独立几何种子/真实场景，不把重复帧或同场景多个版本计为独立样本。主结论必须同时满足：细类别相对相同几何拟合有增量；完整系统相对最强适配对照有净收益；二维覆盖代价在预定上限内；预测面积没有写入重建；失败和先验失配不被删去。

### 官方基线最小准备清单（已核实版本，尚未构建）

2026-09-10 通过 GitHub 的分支、提交、完整 tree 和原始文本接口核实下表，没有下载代码包、模型或场景。当前主环境是 Ubuntu 24.04.2，未发现 `/opt/ros`、`roscore`、`roslaunch` 或 `catkin_make`；有编译器、CMake 和 Docker 客户端。仅客户端存在不能证明有可用 Docker daemon 或 Noetic 镜像。检查时磁盘可用约 2.19GiB。本次只准备版本和步骤，不在该空间内强装 ROS。

| 原作者组件 | 冻结版本 | 已实际读取的证据 | 最小运行入口与缺项 |
|---|---|---|---|
| 地面 TARE | `melodic-noetic`，`44500592b86138257273e0cab264e6a847ccefc7`，2024-06-01 | README、未截断 tree、`package.xml`、CMake、garage launch | Ubuntu20.04/Noetic；C++17、PCL/Eigen、glog、仓内 OR-Tools 动态库；先启动下一行环境，再 `roslaunch tare_planner explore_garage.launch` |
| CMU 地面开发环境 | `noetic`，`bf0cba71365271ebff09831a05afd78578150300`，2026-02-18 | garage launch、vehicle simulator 依赖、场景下载脚本 | `roslaunch vehicle_simulator system_garage.launch`；Gazebo/ROS、地形分析、本地跟踪、`ps3joy`/USB 等；官方场景包约500MB，未下载 |
| FUEL CPU 原版 | HEAD，`662dd23c7b52b258d3c4a0155ff6632118e8984f`，2024-11-19 | README、未截断 tree、exploration package、local sensing CMake | Ubuntu20.04/Noetic；NLopt2.7.1、Armadillo、ROS/PCL/Eigen/OpenCV、仓内 LKH 组件；先 `roslaunch exploration_manager rviz.launch`，再 `roslaunch exploration_manager exploration.launch`，按原演示触发 |

固定源码证据：[TARE README](https://github.com/caochao39/tare_planner/blob/44500592b86138257273e0cab264e6a847ccefc7/README.md)、[TARE 构建依赖](https://github.com/caochao39/tare_planner/blob/44500592b86138257273e0cab264e6a847ccefc7/src/tare_planner/CMakeLists.txt)、[CMU 地面启动文件](https://github.com/HongbiaoZ/autonomous_exploration_development_environment/blob/bf0cba71365271ebff09831a05afd78578150300/src/vehicle_simulator/launch/system_garage.launch)、[开发环境官方说明](https://www.cmu-exploration.com/)、[FUEL CPU 构建开关](https://github.com/HKUST-Aerial-Robotics/FUEL/blob/662dd23c7b52b258d3c4a0155ff6632118e8984f/uav_simulator/local_sensing/CMakeLists.txt)、[FUEL 运行依赖](https://github.com/HKUST-Aerial-Robotics/FUEL/blob/662dd23c7b52b258d3c4a0155ff6632118e8984f/README.md)。FUEL 的 `ENABLE_CUDA=false` 已核实为该提交实际配置；不能由此推断主环境现在能编译全部依赖。

本机包数据库有 OpenCV 开发包4.6，未发现 Eigen、PCL、glog、Armadillo、NLopt、libusb 开发包；这不排除某处存在源码安装。后续在独立 Ubuntu20.04/Noetic 工作区准备依赖，先查已有镜像与空间，再只取固定提交和一个场景。首次原生演示应记录提交、完整环境、启动参数、返回/停止条件、地图与实际轨迹、规划耗时、退出日志和场景校验值。成功启动一次只能确认执行链，不能称复现了论文的比较结果。

**哪些能保持原版：** TARE 地面规划器和官方地面开发环境可以按上述提交保持源码不变，使用官方 3D 扫描、地形分析和跟踪器完成原生演示；FUEL 可保持原生 UAV 运动、深度渲染、FIS/路线和轨迹模块完成 CPU 演示。两者尚未在本机运行。

**哪些必须另行标注适配：** 接入单线雷达 + 窄视场 ZED 时，TARE 的覆盖预测传感器模型、注册扫描/地形输入和相机朝向处理需要审计；不能给它虚构全向 3D 扫描，也不能只限制其真实输入却保留错误的全向收益模型。FUEL 若改成地面 SE(2)，其可达候选高度、碰撞/地形约束、路线时间成本、轨迹/控制接口均发生改变，结果应称“FUEL 地面传感器适配版”。把 UAV 输出轨迹裁成二维不是原版复现。统一动作预算、速度、相机视野或地图后端的每个修改均应有独立 diff 和参数表；原生演示作为忠实性检查，统一平台实验作为主比较，分别报告。

## 最强的反对证据

这些限制决定论文应如何措辞。第一，当前 NSO 开发记录中新增对象假设的净收益仍未超过去假设版本，局部语义收益不能抵消这个系统级问题。第二，若实际目标只是未知平面空间覆盖，语义很可能没有足够独立信息，不能从某种物体数量较多推导覆盖优势。第三，最近的公开工作已经正面处理语义多视角和覆盖路径，需清楚说明超越其机制的部分，而不以名称差异建立创新。

第四，规划公式写成熵、后验或信息矩阵并不意味着估计已校准；错误而自信的语义先验可使机器人少看关键面。第五，尺寸固定的小型解析场景不能证明大尺度优势；规模增加必须保留连通、遮挡与通行条件的合理变化，并测量计算、存储和路径成本。第六，主流几何系统在普通覆盖上很强；公平比较应允许它们使用相同对象要求的任务适配，而不只用默认纯覆盖目标来衬托语义方法。

## 来源

[^1]: Cao, C.; Zhu, H.; Choset, H.; Zhang, J. **TARE: A Hierarchical Framework for Efficiently Exploring Complex 3D Environments.** RSS, 2021. [作者机构全文](https://publications.ri.cmu.edu/storage/publications/2021/06/RSS_2021.pdf)。
[^2]: Cao, C.; Zhu, H.; Ren, Z.; Choset, H.; Zhang, J. **Representation granularity enables time-efficient autonomous exploration in large, complex worlds.** Science Robotics 8(80), 2023. [DOI](https://doi.org/10.1126/scirobotics.adf0970)；[作者存档](https://biorobotics.ri.cmu.edu/papers/paperUploads/scirobotics.adf0970.pdf)；[官方 TARE 仓库](https://github.com/caochao39/tare_planner)。
[^3]: Zhou, B.; Zhang, Y.; Chen, X.; Shen, S. **FUEL: Fast UAV Exploration using Incremental Frontier Structure and Hierarchical Planning.** RA-L, 2021. [机构全文](https://repository.hkust.edu.hk/ir/bitstream/1783.1-108720/1/033635_1.pdf)。
[^4]: HKUST Aerial Robotics. **FUEL official implementation.** [仓库与 CPU/Noetic 说明](https://github.com/HKUST-Aerial-Robotics/FUEL)。
[^5]: Zhou, B.; Xu, H.; Shen, S. **RACER: Rapid Collaborative Exploration with a Decentralized Multi-UAV System.** T-RO 39(3), 2023. [原始全文](https://arxiv.org/pdf/2209.08533)。
[^6]: Robotics STAR Lab. **RACER official implementation.** [仓库](https://github.com/Robotics-STAR-Lab/RACER)。
[^7]: Yang, F.; Cao, C.; Zhu, H.; Oh, J.; Zhang, J. **FAR Planner: Fast, Attemptable Route Planner using Dynamic Visibility Update.** IROS, 2022. [全文 v3](https://arxiv.org/html/2110.09460v3)。
[^8]: Yang, F. **FAR Planner official implementation.** [仓库](https://github.com/MichaelFYang/far_planner)。
[^9]: Zhang, Y.; Chen, X.; Feng, C.; Zhou, B.; Shen, S. **FALCON: Fast Autonomous Aerial Exploration using Coverage Path Guidance.** T-RO, DOI 2024, volume 41, 2025. [全文 v2](https://arxiv.org/html/2407.00577v2)；[DOI](https://doi.org/10.1109/TRO.2024.3522148)。
[^10]: HKUST Aerial Robotics. **FALCON official implementation.** [仓库](https://github.com/HKUST-Aerial-Robotics/FALCON)。
[^11]: Papatheodorou, S.; Funk, N.; Tzoumanikas, D.; Choi, C.; Xu, B.; Leutenegger, S. **Finding Things in the Unknown: Semantic Object-Centric Exploration with an MAV.** ICRA, 2023. [全文 v2](https://arxiv.org/html/2302.14569v2)。
[^12]: ETHZ MRL / Smart Robotics Lab. **Semantic exploration ICRA 2023.** [官方仓库](https://github.com/ethz-mrl/semantic-exploration-icra-2023)。
[^13]: Menon, R.; Zaenker, T.; Dengler, N.; Bennewitz, M. **NBV-SC: Next Best View Planning based on Shape Completion for Fruit Mapping and Reconstruction.** 2022/2023. [全文 v2](https://arxiv.org/html/2209.15376v2)。
[^14]: Tao, Y.; Liu, X.; Spasojevic, I.; Agarwal, S.; Kumar, V. **3D Active Metric-Semantic SLAM.** RA-L 9(3), 2024. [全文 v3](https://arxiv.org/html/2309.06950v3)。
[^15]: Kumar Robotics. **3D Active Metric-Semantic SLAM implementation.** [官方仓库](https://github.com/KumarRobotics/kr_3d_active_ms_slam)。
[^16]: Zhan, X.; Zhou, S.; Yang, Q.; Zhao, Y.; Liu, H.; Ramineni, S. C.; Shimada, K. **Semantic Exploration and Dense Mapping of Complex Environments using Ground Robot with Panoramic LiDAR-Camera Fusion.** RA-L, 2025. [全文 v3](https://arxiv.org/html/2505.22880v3)；[DOI](https://doi.org/10.1109/LRA.2025.3609216)。
[^17]: Yang, Q. **Semantic Exploration and Dense Mapping project.** [作者项目页](https://qq-yang.com/portfolio/semantic_exploration/)。
[^18]: Sportich, B.; Boubakri, K.; Simonin, O.; Renzaglia, A. **Quality-guided UAV Surface Exploration for 3D Reconstruction.** 2025. [全文 v2](https://arxiv.org/html/2511.20353v2)。
[^19]: ICUAS. **2026 program and book of abstracts.** [官方议程](https://uasconferences.com/2026_icuas/wp-content/uploads/ICUAS-2026-Program-BofA.pdf)，TuA3.5, pp.143–150。
[^20]: Mao, J.; Ayoubi, S.; Sharma, V. D.; Hadžić, I.; Andrews, M. **SCOUT: Semantic scene COverage via Uncertainty-guided Traversal.** 2026-06-04, workshop/preprint. [全文 v1](https://arxiv.org/html/2606.06721v1)。
[^21]: Tatsch, C.; Gu, Y. **SGE: Semantically-Guided Exploration for Unstructured Environments via Image-Space Waypoint Sampling.** 2026-08-29, preprint. [全文 v1](https://arxiv.org/html/2608.29315v1)。
[^22]: Vegesna, N.; Zakhor, A. **Semantic-Aware Guided Drone Exploration for Language-Conditioned 3D Indoor Mapping.** 2026-05-22, non-archival workshop/preprint. [全文 v1](https://arxiv.org/html/2605.23160v1)；[出版状态](https://arxiv.org/abs/2605.23160)。
[^23]: Zhu, J.; Sellán, S.; Terenin, A. **A Bayesian Approach for Task-Specific Next-Best-View Selection with Uncertain Geometry.** SIGGRAPH, 2026. [全文 v1](https://arxiv.org/html/2605.05095v1)；[DOI](https://doi.org/10.1145/3799902.3811119)。
[^24]: Zhu, J. **BayesianNBV official implementation.** [仓库与未完成事项](https://github.com/jingsenzhu/BayesianNBV)。
[^25]: Jeong, S.; Lee, E.; Kim, J.; Kim, A. **Informative Object-centric Next Best View for Object-aware 3D Gaussian Splatting in Cluttered Scenes.** ICRA, 2026. [全文 v1](https://arxiv.org/html/2602.08266v1)；[接收声明](https://arxiv.org/abs/2602.08266)。

以上来源只支持其原始条件下的研究结论；它们不是 NSO 的实验成绩。2023 TARE 扩展全文、未发布实现的完整性以及所有官方代码在当前机器的构建状态仍需在复现阶段进一步核实。
