# 虚拟文章相关工作与核验文献

核验日期：2026-09-28。以下正文片段可接在引言之后；保留现稿的 [1]–[10] 编号，补充体积融合、信念规划、带预算信息采集及表面评价的直接出处[11]–[14]。本轮仅核验文献与编写正文，没有新增实验、修改旧文献库或扩张论文性能主张。

## 工业应用背景与任务范围

具身机器人通过感知、决策与物理行动参与工业作业。世界经济论坛的工业Physical AI白皮书讨论制造与物流中的应用案例及规模化挑战[9]，可用于说明应用正在拓展，不宜据此声称全行业已经成熟普及。本文由工业设备布局、工位和物料放置变化引出主动获取当前环境几何的需求。单次实验保持静态，所验证的是类别先验和几何反馈如何帮助选择有效观察；跨任务变化检测、历史地图增量更新及动态避障不属于已执行实验。

主动SLAM通过机器人运动选择改善环境建模，相关综述从模块化和信念空间等角度整理其方法[10]。本文聚焦观察规划与实际建图质量，以准确模拟位姿隔离决策作用，不将表面质量改善解释为SLAM定位精度提升。

## 可直接纳入正文的相关工作

自主探索通常需要把环境表示、目标选择和局部运动协调起来。主动SLAM的研究综述将运动决策与地图估计联系起来，并讨论模块化和信念空间规划等组织方式[10]。早期代表Active Neural SLAM采用学习式建图和层次策略[1]；TARE通过局部精细表示与全局稀疏表示协调探索路径[5]。本文采用面向设施观察的分层决策与局部执行设计，在同一安全图和行动预算内研究观察分配。内部几何对照同样可以补看、获取诊断信息并重新规划，因此待检验的增量是类别信息对共同主动规划能力的作用。

主动三维重建进一步关注传感器应从何处获取下一次观测。GenNBV通过强化学习与几何、语义和动作表示，在五维自由空间选择观察位姿，研究跨场景的下一最佳视点策略[6]。设施建档也需要处理自遮挡和视向不足，但本文聚焦已知类别与结构关系下的有限构型歧义，采用离散地面动作和显式预算规划。为使观察策略与重建后端的作用能够区分，各对照使用相同的Open3D深度融合流程[4]，仅将实际取得的深度写入三维模型，并在任务结束后评价表面质量。公开模板参与决策预测，不替代真实观测填补模型。

语义巡检已将“应观察什么”与“怎样观察得足够好”联系起来。SWAP将体积探索、语义目标补洞以及满足分辨率和入射角要求的表面巡检结合[2]；SPP进一步利用语义场景图中的重复结构预测未见目标，并面向工业巡检组织路径[7]。因此，本文的动机并非首次提出语义补看或结构预测，而是研究一种较具体的条件：当已观测几何尚不能区分有效观察方向时，类别所携带的结构线索能否减少诊断后的改向代价，以及错误先验能否被实际几何证据修正。通过保留共同主动几何能力的有无类别对照，以及错误类别和纠错政策对照，可以分别观察这两个作用。

近期语义主动建图也直接结合任务相关性与观察价值。VISTA以在线语义高斯地图支持开放词汇任务，通过语义相关性和视向覆盖评价滚动轨迹[3]；ActiveSGM联合语义与几何不确定性，预测候选观测的信息性[8]。本文采用类别条件构型权重和实测残差反馈，在CPU环境实现预算受限的观察闭环；其应用价值是检验类别知识在共同几何信息之外的条件收益，而非宣称通用贝叶斯更新或信息价值原则为原创。实验中的SWAP-I与VISTA-I只是在共同候选与融合条件下实现相关规划机制，用来比较观察分配方式，不等同于原作者完整系统。由此，本文以受控配对、实际重建终点和反馈行为记录共同支撑设施建档中的语义决策作用。

体积重建为观察策略提供了共同的实测终点。Curless和Levoy将距离图逐帧融合为加权有符号距离场并提取等值面[11]；本项目使用Open3D实现，既不改变这一重建原则，也不向模型填入公开模板。Tanks and Temples以距离阈值下的精确率、召回率及F-score区分准确性与完整性[14]。本文沿用该评价思想，但设施ROI、外竖面召回范围以及与二维覆盖相乘的联合指标是本任务另行规定的评价范围，不等同于直接运行该基准。

类别线索的用途还需要放在信息决策中解释。Kaelbling等以信念状态联系部分可观测历史和策略选择[12]；Hollinger与Sukhatme在预定资源预算下规划信息采集轨迹[13]。本方法借鉴这些已有决策原则，将实际几何更新、有限的未来诊断和返航条件结合在离散姿态图内。一次诊断前瞻是计算上的有限近似，0.9和0.99是本实现的固定参数，不能因引用这些文献而声称它们具有概率校准或理论最优保证。

## 可直接替换现稿的参考文献

1. Chaplot, Devendra Singh; Gandhi, Dhiraj; Gupta, Saurabh; Gupta, Abhinav; Salakhutdinov, Ruslan. **Learning to Explore using Active Neural SLAM.** International Conference on Learning Representations (ICLR), 2020. [作者项目页及正式引用](https://devendrachaplot.github.io/projects/Neural-SLAM)；[作者原论文](https://arxiv.org/abs/2004.05155)。
2. Dharmadhikari, Mihir; Alexis, Kostas. **Semantics-aware Exploration and Inspection Path Planning.** IEEE International Conference on Robotics and Automation (ICRA), 2023. [作者原论文及录用说明](https://arxiv.org/abs/2303.07236)；[作者正式出版目录](https://www.autonomousrobotslab.com/publications.html)。本轮未从IEEE页面取得可读元数据，引用采用已核验的作者来源，不补未核验页码。
3. Nagami, Keiko; Chen, Timothy; Yu, Javier; Shorinwa, Ola; Adang, Maximilian; Dougherty, Carlyn; Cristofalo, Eric; Schwager, Mac. **VISTA: Open-Vocabulary, Task-Relevant Robot Exploration With Online Semantic Gaussian Splatting.** IEEE Robotics and Automation Letters, **11**(3):3150–3157, 2026. DOI: [10.1109/LRA.2026.3653276](https://doi.org/10.1109/LRA.2026.3653276)。[Stanford作者正式档案](https://profiles.stanford.edu/mac-schwager)确认卷期、页码及DOI；方法描述核对[作者2025年预印本](https://arxiv.org/abs/2507.01125)和[作者代码说明](https://github.com/StanfordMSL/VISTA)，不声称逐字校对最终期刊全文。
4. Zhou, Qian-Yi; Park, Jaesik; Koltun, Vladlen. **Open3D: A Modern Library for 3D Data Processing.** arXiv:1801.09847, 2018. DOI: [10.48550/arXiv.1801.09847](https://doi.org/10.48550/arXiv.1801.09847)。[作者原论文](https://arxiv.org/abs/1801.09847)。该条用于软件后端归属，不将其作为本文重建算法创新。
5. Cao, Chao; Zhu, Hongbiao; Choset, Howie; Zhang, Ji. **TARE: A Hierarchical Framework for Efficiently Exploring Complex 3D Environments.** Robotics: Science and Systems (RSS), 2021. DOI: [10.15607/RSS.2021.XVII.018](https://doi.org/10.15607/RSS.2021.XVII.018)。[RSS正式论文及BibTeX](https://www.roboticsproceedings.org/rss17/p018.html)。
6. Chen, Xiao; Li, Quanyi; Wang, Tai; Xue, Tianfan; Pang, Jiangmiao. **GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction.** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp.16436–16445. [CVF正式论文页及BibTeX](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_GenNBV_Generalizable_Next-Best-View_Policy_for_Active_3D_Reconstruction_CVPR_2024_paper.html)。
7. Dharmadhikari, Mihir; Alexis, Kostas. **Semantics-Aware Predictive Inspection Path Planning.** IEEE Transactions on Field Robotics, 2025. DOI: [10.1109/TFR.2025.3578402](https://doi.org/10.1109/TFR.2025.3578402)。[作者正式出版目录](https://www.autonomousrobotslab.com/publications.html)确认期刊与DOI；方法核对[作者原论文](https://arxiv.org/abs/2506.06560)。本轮未核对卷页，不补齐推测字段。
8. Chen, Liyan; Zhan, Huangying; Yin, Hairong; Xu, Yi; Mordohai, Philippos. **Understanding while Exploring: Semantics-driven Active Mapping.** Advances in Neural Information Processing Systems **38**, 2025, Main Conference Track. DOI: [10.52202/085713-1113](https://doi.org/10.52202/085713-1113)。[NeurIPS正式论文集](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2fa561fb73e30a2396c268253eeca191-Abstract-Conference.html)。方法名为ActiveSGM。
9. World Economic Forum. **Physical AI: Powering the New Age of Industrial Operations.** White paper, 2025-09-04. [官方白皮书](https://www.weforum.org/publications/physical-ai-powering-the-new-age-of-industrial-operations/)。官方说明包含感知、推理和自主行动，以及制造、物流案例和规模化挑战；用于应用动机，不作为本项目的技术效果证据。
10. Placed, Julio A.; Strader, Jared; Carrillo, Henry; Atanasov, Nikolay; Indelman, Vadim; Carlone, Luca; Castellanos, José A. **A Survey on Active Simultaneous Localization and Mapping: State of the Art and New Frontiers.** IEEE Transactions on Robotics, 2023. [作者论文与出版说明](https://arxiv.org/abs/2207.00254)。已核验作者、摘要及期刊信息；用于主动运动选择、模块化与信念空间规划的研究背景，不补未核验卷页。

11. Curless, Brian; Levoy, Marc. **A Volumetric Method for Building Complex Models from Range Images.** Proceedings of SIGGRAPH, 1996, pp.303–312. [作者项目及论文](https://graphics.stanford.edu/papers/volrange/)。核验加权有符号距离表示、逐帧融合和等值面提取；不把本文的Open3D具体设置归因于该论文。
12. Kaelbling, Leslie Pack; Littman, Michael L.; Cassandra, Anthony R. **Planning and Acting in Partially Observable Stochastic Domains.** Artificial Intelligence, **101**(1–2):99–134, 1998. [作者论文](https://www.cassandra.org/arc/papers/aij98.pdf)。核验信念状态、状态估计与策略选择的关系；本文的一次未来诊断是应用中的有限近似。
13. Hollinger, Geoffrey A.; Sukhatme, Gaurav S. **Sampling-based Robotic Information Gathering Algorithms.** The International Journal of Robotics Research, **33**(9):1271–1287, 2014. DOI: [10.1177/0278364914533443](https://doi.org/10.1177/0278364914533443)。[出版方记录](https://journals.sagepub.com/doi/10.1177/0278364914533443)支持信息质量与固定预算下的轨迹规划问题，不表述为本文实现了其RIG算法。
14. Knapitsch, Arno; Park, Jaesik; Zhou, Qian-Yi; Koltun, Vladlen. **Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction.** ACM Transactions on Graphics, **36**(4), 2017. [作者项目与正式引用](https://www.tanksandtemples.org/)；[作者论文](https://qianyi.info/docs/papers/siggraph17_tanksandtemples.pdf)与[官方评价说明](https://www.tanksandtemples.org/tutorial/)核验双向距离、精确率、召回率及F-score。本文使用该评价思想，未声称在此公开数据集上进行了实验。

正式排版时只保留每条的作者、题名、出版信息及一个DOI或官方URL；本节附带的“本轮核验”说明用于写作交接，不需要放入论文参考文献。上述作者来源和出版者来源分别证明方法与出版记录，不能把查到出版元数据等同于完整复现方法。

## 每项文献与本实现的具体差别

|编号／方法|原研究的侧重|本稿应采用的对应关系|
|---|---|---|
|[1] Active Neural SLAM|学习式建图与全局、局部策略的层次组合|作为学习式层次探索的历史背景；本方法使用显式CPU信念规划及执行守卫，研究工业设施的观察分配。|
|[2] SWAP|探索、语义目标完整性与有质量要求的表面巡检|承认语义巡检的直接前作；SWAP-I是共同CPU机制适配，既不是只改类别输入的因果对照，也不是作者完整系统。|
|[3] VISTA|开放词汇查询、在线语义3DGS、任务相关视向覆盖|本文是人工类别、双模板、CPU TSDF；VISTA-I只迁入相关规划机制。原作者实现要求CUDA及ROS 2，不把后端替换隐藏为同一完整实现。|
|[4] Open3D|通用三维数据处理软件库|作为各组共享融合后端；主贡献位于观察分配，不能将采用TSDF作为新的建图算法。|
|[5] TARE|复杂环境的分层几何探索及路径效率|原生规划器迁移记录保留为工程证据；当前传感和任务适配条件不同，不纳入公平完整系统排名。|
|[6] GenNBV|学习式自由空间NBV及跨数据集泛化|作为主动重建研究背景；当前离散地面任务没有执行其原策略，不引用其数据作为本项目实测基线。|
|[7] SPP|从重复语义布局预测未见结构并规划巡检|本文检验人工类别与公开构型关系下的条件增量，不声称首次语义结构预测；两者知识来源和预测对象不同。|
|[8] ActiveSGM|语义／几何不确定性与3DGS主动建图|作为观察信息性研究背景；当前构型权重未校准、前瞻诊断为有限近似，不等同于其不确定性模型。|
|[9] 工业Physical AI白皮书|工业感知、推理与物理行动的应用及规模化挑战|用于说明应用需求，不作为算法贡献或全行业成熟普及的依据。|
|[10] 主动SLAM综述|运动选择与环境模型估计、模块化与信念空间方法|提供研究脉络；本实验只检验静态任务中的观察规划与建图质量，不检验定位改进。|
|[11] 体积距离图融合|加权有符号距离场与等值面提取|交代TSDF技术来源；当前后端使用Open3D，观察策略为变量。|
|[12] 部分可观测规划|信念状态及其策略价值|解释构型权重和未来观测，不声称精确POMDP求解。|
|[13] 带预算信息采集|信息质量、轨迹成本与运动约束|支持资源权衡动机；当前采用有限图动态规划，未实现RIG。|
|[14] Tanks and Temples|表面准确性、完整性与F-score|评价思想的直接来源；本文设施ROI和联合乘积另外定义。|

## 可选的引用键映射与本轮修正

|编号|建议引用键|既有库状态与处理|
|---|---|---|
|[1]|`chaplot2020learning`|`semantic_coverage_references.bib`未含此键；沿用作者项目页BibTeX键名，若采用数字引用无需新增库。|
|[2]|`dharmadhikari2023swap`|既有V38文献说明已建议此键，旧bib未含；本文件提供正式ICRA信息及作者来源。|
|[3]|`nagami2025vista`|保留旧键避免引用失效，但另建工作文献库时应将出版字段改为2026年RA-L、11(3):3150–3157及上述DOI；引用键中的2025不是出版年份。|
|[4]|`zhou2018open3d`|旧bib未含；本文件提供软件原论文，不虚构会议或期刊。|
|[5]|`cao2021tare`|沿用旧键；新工作文献库可补正式RSS DOI，尾号为018。|
|[6]|`chen2024gennbv`|沿用旧键；可补CVF已核验页码16436–16445及正式HTML地址。|
|[7]|`dharmadhikari2025spp`|既有V38文献说明已建议此键，旧bib未含；应引用正式IEEE TFR出版记录，不只标为预印本。|
|[8]|`chen2025activesgm`|既有V38文献说明已建议此键，旧bib未含；应引用NeurIPS 38正式主会记录，不标为未审预印本。|
|[9]|`wef2025physicalai`|工业应用背景，采用官方白皮书题名、日期和URL，不按学术方法论文处理。|
|[10]|`placed2023activeslam`|采用已核验的IEEE Transactions on Robotics期刊信息与作者论文URL，不推测卷页。|
|[11]|`curless1996volumetric`|本轮原始出处补充，数字编号与毕设统一，不修改历史bib。|
|[12]|`kaelbling1998planning`|作者存档核验；信念规划的理论背景。|
|[13]|`hollinger2014sampling`|出版方核验卷期、页码、DOI及问题定义。|
|[14]|`knapitsch2017tanks`|作者项目及指标说明核验；不构成新增公开基准实验。|

本轮不改写旧bib，以免历史材料的访问日期与元数据无记录变化。此前补充的GenNBV、SPP和ActiveSGM继续提供直接相关研究，[9]和[10]分别支持工业应用背景及主动建图研究脉络；新增[11]—[14]为正文中的具体技术概念提供直接出处。VISTA的正式出版字段、TARE DOI与GenNBV页码沿用已核验信息。相关工作中的“条件增量”依据本项目控制变量设计，不是文献检索自动证明的领域首次创新。
