# V38：相关工作、正式出版元数据与外部基线选择

核查日期：2026-09-20。范围为与当前设施建档任务直接相关的原论文、作者网站及作者代码；没有安装新环境、执行外部规划器回合或复制他人的性能数字。本页供论文正文和后续对比实验使用，不能当作外部基线结果。

## 1. 论文应回答的具体问题

现有 V35/V36 证据支持的研究问题是：**在相同已观测几何、公共安全信息、传感器、融合后端和有限任务预算下，类别条件先验能否改变定向观察决策，并改善覆盖与三维外表面重建质量的联合结果？** 原实验使用人工语义线索和预先公开的类型—结构关系。它检验该信息及其规划使用方式的作用，尚未检验真实识别网络、未知设施类别或原神经 ANS 系统的整体泛化。

相关工作不应以“以往方法只做几何、不会补看”作为立论起点。几何规划器可以利用已观测深度、缺口、视向历史和不确定性补看；语义方法应证明在这些共同信息之外的增量。

## 2. 直接相关的方法，而非按年份堆叠引用

|研究线及代表方法|已核验的技术事实|与本文的关系及比较边界|
|---|---|---|
|层次几何探索：TARE|局部密集表示与全局粗粒度表示联合规划，作者发布地面机器人实现。|最优先的完整地面系统对照。它有主动选视点能力，不能等同随机或最近前沿。本文当前 G 是内部对照，不能改名为 TARE。[RSS 正式论文页](https://www.roboticsproceedings.org/rss17/p018.html)、[作者代码](https://github.com/caochao39/tare_planner)|
|增量前沿与层次探索：FUEL|维护增量前沿结构，依次规划访问顺序、局部视点及轨迹；官方已提供不要求 CUDA 的 CPU 仿真路线。|可作为外部几何补充，但原任务是 UAV。限制高度和替换控制器后应标为地面适配版。[作者说明及正式引用](https://github.com/HKUST-Aerial-Robotics/FUEL)|
|覆盖路径引导探索：FALCON|以连通性分解和全局覆盖路线引导局部前沿顺序，提供共同无人机仿真环境。|是应讨论的较新强几何方法。官方默认运行说明仍含 CUDA 渲染配置，本机未验证无 GPU 原生演示；不把“C++ 规划器”直接等同“完整 CPU 仿真可用”。[论文 v2](https://arxiv.org/abs/2407.00577)、[作者依赖与运行说明](https://github.com/HKUST-Aerial-Robotics/FALCON)|
|语义探索与巡检：SWAP|结合体积探索、目标补洞和满足分辨率／入射角要求的表面检查。|设施建档的直接前作；“语义让机器人多看目标”不是本文独有贡献。其观察质量条件与噪声 TSDF 后的外表面 F1 不同。[IEEE 正式记录](https://ieeexplore.ieee.org/document/10160469/)、[作者原论文](https://arxiv.org/abs/2303.07236)|
|语义预测巡检：SPP|从语义场景图中的重复结构预测未见目标，并将预测用于探索／检查；处理不完美重复模式。|与“语义预测—主动观察—反馈纠正”直接重叠。本文只能论证更具体的信息对照和预算下质量增量，不能宣称首次预测语义结构。SPP 需要重复结构关系，当前双类型构型不能自动充当其完整任务。[原论文](https://arxiv.org/abs/2506.06560)、[作者代码](https://github.com/ntnu-arl/predictive_planning_ros)|
|语义与几何不确定性主动建图：ActiveSGM|在 3D Gaussian Splatting 后端上联合利用语义和几何不确定性选择观察。|是最新相关语义主动建图方法之一；原神经后端、语义前端和当前 CPU TSDF 不同，完整系统对照与共同表示上的评分适配必须分开。[NeurIPS 2025 正式论文页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2fa561fb73e30a2396c268253eeca191-Abstract-Conference.html)、[作者代码](https://github.com/lly00412/ActiveSGM)|
|任务相关视向覆盖：VISTA|结合开放词汇查询相关性与视向覆盖规划滚动轨迹，并在线建立语义高斯地图。|其主要任务包含搜索，并非全部设施等权建档。将语义相关性和视向多样性迁入共同候选池，只能称机制适配。官方流程要求 CUDA GPU 与 ROS 2 Humble。[原论文](https://arxiv.org/abs/2507.01125)、[作者代码与依赖](https://github.com/StanfordMSL/VISTA)|

TARE 官方库同时列出 2023 年 Science Robotics 的扩展工作 *Representation Granularity Enables Time-Efficient Autonomous Exploration in Large, Complex Worlds*。本项目固定的是该库指定提交；不能因检索到“planner 2”脚注或 ROS 2 分支就额外虚构一个 TARE2 算法。

## 3. 可写入正文的定位

建议按三段组织相关工作：先交代强几何层次规划为何是必要基线；再承认 SWAP/SPP 已建立语义巡检和结构预测；最后讨论 ActiveSGM/VISTA 中语义不确定性、查询相关性与重建质量的联系及其信息条件。

可用表述：

> 几何探索与语义巡检已经分别形成成熟的层次规划和目标观察机制。本文研究有限预算设施建档中的一个具体条件：当已观测几何不足以区分后续观察方向，而类别线索携带公开结构知识时，该线索是否仍能提供可复核的建图收益。我们采用同一闭环、同一融合与评价后端，在移除或改变语义信息及反馈机制时保留共同几何信息，检验该收益的来源和适用边界。

这里“研究一个条件”是当前工作的定位，不是领域首创证明。“覆盖 × 三维质量”是透明任务效用，不能仅凭相乘形式宣称新的算法理论；必须同时报告两个组成项和成本。论文也不应把引用高水平工作本身当成与其完成公平比较的证据。

## 4. 2026-09-20 元数据核查与建议引用

正式出版状态优先查会议论文集、出版者和作者正式出版目录；arXiv Comments 未更新不能推出论文仍未发表。搜索摘要中的抓取时间不作为发表时间。

|键名建议|当前可确认记录|尚需谨慎的内容|
|---|---|---|
|`cao2021tare`|C. Cao, H. Zhu, H. Choset, J. Zhang. *TARE: A Hierarchical Framework for Efficiently Exploring Complex 3D Environments*. RSS 2021. DOI `10.15607/RSS.2021.XVII.018`。[RSS 作者与 BibTeX](https://www.roboticsproceedings.org/rss17/p018.html)|不要使用检索到的错误 DOI 尾号 029。|
|`zhou2021fuel`|B. Zhou, Y. Zhang, X. Chen, S. Shen. *FUEL: Fast UAV Exploration Using Incremental Frontier Structure and Hierarchical Planning*. IEEE RA-L 6(2):779–786, 2021。[作者给出的期刊 BibTeX](https://github.com/HKUST-Aerial-Robotics/FUEL)|ICRA option 不是另一篇独立实验论文。|
|`zhang2025falcon`（可保留旧键）|Y. Zhang, X. Chen, C. Feng, B. Zhou, S. Shen. *FALCON: Fast Autonomous Aerial Exploration Using Coverage Path Guidance*. IEEE T-RO 41:1365–1385. DOI `10.1109/TRO.2024.3522148`。[作者 BibTeX](https://github.com/HKUST-Aerial-Robotics/FALCON)|官方 BibTeX 写 2024，而卷年常记 2025；这是 early access／编卷年份差异待最终出版导出核对。IEEE 网页本次未返回完整元数据，不伪称已校对出版者卷年。可在当前稿注明作者引用年 2024，后续统一期刊格式。|
|`dharmadhikari2023swap`|M. Dharmadhikari, K. Alexis. *Semantics-aware Exploration and Inspection Path Planning*. ICRA 2023. DOI `10.1109/ICRA48891.2023.10160469`。[IEEE 正式记录](https://ieeexplore.ieee.org/document/10160469/)|本次 IEEE 返回内容确认会议、日期、DOI，未返回页码；不依靠二手页码补字段。|
|`dharmadhikari2025spp`|M. Dharmadhikari, K. Alexis. *Semantics-Aware Predictive Inspection Path Planning*. IEEE Transactions on Field Robotics, 2025. DOI `10.1109/TFR.2025.3578402`。[作者正式出版目录](https://www.autonomousrobotslab.com/publications.html)、[arXiv 的录用说明](https://arxiv.org/abs/2506.06560)|不再只记“未审预印本”；IEEE 正文页本次未提供可读元数据，卷页先不填。方法核对仍基于 v1，不能称逐字核对最终期刊版。|
|`chen2025activesgm`|L. Chen, H. Zhan, H. Yin, Y. Xu, P. Mordohai. *Understanding while Exploring: Semantics-driven Active Mapping*. Advances in Neural Information Processing Systems 38, 2025. DOI `10.52202/085713-1113`。[NeurIPS 正式论文集](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2fa561fb73e30a2396c268253eeca191-Abstract-Conference.html)|当前 arXiv 没有 venue 字段，但已有正式会议记录；不能沿用旧“仅预印本”标签。|
|`nagami2026vista`（或保留原键）|K. Nagami, T. Chen, J. Yu, O. Shorinwa, M. Adang, C. Dougherty, E. Cristofalo, M. Schwager. *VISTA: Open-Vocabulary, Task-Relevant Robot Exploration With Online Semantic Gaussian Splatting*. IEEE RA-L 11(3):3150–3157, 2026。[Stanford 作者官方档案，第 3 页](https://cap.stanford.edu/profiles/frdActionServlet?choiceId=printerprofile&profileId=70847&profileversion=full)|DOI `10.1109/LRA.2026.3653276` 由[作者 ORCID 中的 Crossref 导入记录](https://orcid.org/0000-0002-5552-8780)确认；本次 DOI/IEEE 页面未成功取得正文。方法核对仍须标明 arXiv v1 与固定代码版本，不能把旧源码检查改称最终出版版验证。|

推荐保留 ANS、经典前沿和 TSDF 等基础引用，再加入以上直接相关工作即可。不应为了“最新”将未读的 2026 论文堆入创新论证，也不承诺 SCI 分区或录用结果。

## 5. 本地实际准备状态与优先级

2026-09-20 只读检查发现 `/opt/ros` 不存在，`roscore`、`catkin_make` 不可用；`docker` 客户端在 `/usr/bin/docker`，本次未重新探测 daemon 或容器。工作盘当次空闲 109,195,264 字节。该空间不支持重新安装完整 ROS／Gazebo 环境。没有下载、安装、删除资产或运行官方方法。

|资产|固定版本／证据|可声称的状态|
|---|---|---|
|TARE 源码|`44500592b86138257273e0cab264e6a847ccefc7`；`third_party/official_baselines/tare_official`|本地已准备。历史完整节点构建、存活与接口注册通过；没有传感器驱动公平回合。|
|FUEL 源码|`662dd23c7b52b258d3c4a0155ff6632118e8984f`；`third_party/official_baselines/fuel_official`|本地已准备，未验证完整构建／运行。|
|CMU 地面环境源码|`bf0cba71365271ebff09831a05afd78578150300`|源码已准备；原生场景资产与可运行环境不能由此推定存在。|
|SPP/SWAP、ActiveSGM、VISTA|已有固定提交的静态机制审查，见下述旧报告|静态研究，未形成公平性能结果。FALCON 本轮为在线文献及 README 检查。|

推荐顺序是：（1）共同 CPU 闭环上的强内部 G/S 与公开机制适配，验证观测价值归因；（2）恢复固定 TARE 完整节点后做有记录补丁的地面窄视场系统比较；（3）若资源允许，加入 FUEL 的地面适配或任务条件合适的 SPP 原作者系统。不能用“改了几项收益公式”的版本替代完整作者系统名称。具体输入、统计与停止规则见 [V38 外部基线协议](/root/NSO/docs/research/V38_EXTERNAL_BASELINE_PROTOCOL_20260920.md)。

## 6. 本地来源与可追溯性

本轮阅读以下文件，未改写其历史状态：

|文件|SHA-256|
|---|---|
|`docs/research/official_baseline_source_manifest.json`|`ff702f088a13d8c5f39bd18b8287bc94f4154d65bf04babe0d2e1be3a72fa447`|
|`docs/research/TARE_NATIVE_BUILD_FEASIBILITY.md`|`16bf4bcb084b6d505708976c158b28bdddab8b1fa04e0c30a21d475f361a6cc9`|
|`docs/ROS1_LIVE_PROBE_RUNTIME_STATUS_20260914.md`|`af546af9d92af7c645b71e67fc043fb7decfa0578206bba24dcde308dbca7eea`|
|`docs/research/SEMANTIC_DOCUMENTATION_RELATED_WORK_GAP_20260915.md`|`b5d0407619509a957e484aea87cb9ea8b5beddbbfeada2d42c4dd5da92039ba2`|
|`docs/research/SEMANTIC_DOCUMENTATION_SPP_OVERLAP_REVIEW_20260915.md`|`fd2ec5e7603a7d61d75b2acd601db26a97fe00907449909994bcb24b33d1f28a`|

旧官方源清单写的是初次准备时 `build_executed=false`；之后 TARE 构建成功由 `audit_results/tare_native_node_attempt_20260911` 独立记录。本页保留时间顺序，不以旧静态字段否认后来历史构建，也不以历史构建冒充当前可启动环境。

## 主稿引用定稿补充

本轮主稿复核时补充核对 [FALCON 作者 STAR 出版目录](https://robotics-star.com/publication)，其 2025 栏明确列 T-RO 2025；主稿采用卷年 2025，41:1365–1385，保留前述早期在线发表年差异的审计记录。[FUEL 的 HKUST 正式研究记录](https://researchportal.hkust.edu.hk/en/publications/fuel-fast-uav-exploration-using-incremental-frontier-structure-an/)确认 DOI `10.1109/LRA.2021.3051563`，主稿引用改为该 DOI。
