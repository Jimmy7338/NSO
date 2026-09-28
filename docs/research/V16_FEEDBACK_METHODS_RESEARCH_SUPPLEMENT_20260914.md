# 语义增量预测与三维反馈的近期方法对照

现阶段应把“预测额外观察是否值得”与“已执行观察带来了什么”分开建模，并让两者使用一致的三维目标。已有 V16 观察器解决了后者的一部分输入问题；尚缺设施关联、候选条件预测、反馈校正和独立决策收益证据。本补充结合截至 2026 年 9 月 14 日可访问的原论文与作者代码说明，限定下一版方案的可借鉴部分及验证要求，延续项目既有文献综述。

## 方法依据与迁移边界

**PUN：预测与昂贵重建解耦。** 论文以单幅图像预测视点不确定性图，并汇聚历史预测抑制重复视点；训练图的监督来自合成图像与参考图像之间的误差。主要视点选择实验使用共同球面候选集和相同重建后端。因此，可借鉴前馈候选价值预测和共同候选控制，不能把图像误差监督直接称为本项目的三维联合回报。附录 S5 中 NeRFAssets 的部分几何／可见覆盖指标也没有随渲染指标一起占优，提示必须分别报告质量与覆盖。〔1〕[论文方法、实验及附录](https://arxiv.org/html/2506.14856v2)

作者仓库标注该工作被 ICLR 2026 接收，并提供代码、模型和数据入口；这不意味着已在本项目的 CPU 环境复现。〔2〕[作者代码说明](https://github.com/ZhangLab-DeepNeuroCogLab/PUN/blob/main/README.md)

**AREA3D：几何与语义场互补。** 所核对的 v1 将前馈几何置信度与 VLM 的区域提示结合，用可见性门控选择视点，并对已选视锥内的不确定性作固定比例衰减。该版本表 4 中，物体实验的组合模型 PSNR 高于仅几何模型，但 SSIM、LPIPS 没有同步改善。它支持研究语义对观察需求的提示作用，不能证明任何语义加权都改善全部重建指标。〔3〕[论文第 3 节及表 4](https://arxiv.org/html/2512.05131v1)

**BayesianNBV：按任务收益选择观测。** 该工作通过不确定表面的后验样本模拟可能的未来扫描，再计算任务相关的期望效用改善，主要评价对象级点云任务。这里有价值的是“观测价值由任务效用决定”的建模原则；本项目仍应围绕有限预算的二维覆盖与三维重建联合目标验证。〔4〕[论文第 3 节与结论](https://arxiv.org/html/2605.05095v1)

作者代码当前注明必要依赖包括 GPU 版 PyTorch，完整配置和数据生成说明仍有待补充项。因此，当前条件下不能把它列为已可直接运行的 CPU 外部基线，更不能把本地简化实现标成官方复现。〔5〕[作者仓库安装与发布状态](https://github.com/jingsenzhu/BayesianNBV)

**MACARONS：从在线观测学习覆盖预测。** 该方法联合学习占据场与下一视点预测，采用 RGB 在线自监督；论文讨论了已知位姿和简单局部路径策略的局限。其借鉴意义在于让后续观测检验此前的预测，而非仅按预定观察次数降低一个分数。〔6〕[论文与出版信息](https://openaccess.thecvf.com/content/CVPR2023/html/Guedon_MACARONS_Mapping_and_Coverage_Anticipation_With_RGB_Online_Self-Supervision_CVPR_2023_paper.html)

## 针对本项目的判断

上述方法不能替代本项目自己的语义归因实验。图像级不确定性、区域优先级、表面后验方差、最终 F1 与有限时间的覆盖回报是不同量。把其中一个名称改成“信息增益”不会使它成为另一个量，增加复杂后端也不能保证类别信息改变正确决策。

最适合当前 CPU 路径的方向是保留共享 TSDF 后端和安全候选生成，让小型预测器估计**相同后续执行策略下的真实增量**。在线三维反馈只作为预测条件或辅助监督，不能替代外部评价标签。以上是结合本地负结果形成的设计判断，不是这些论文已经证明的结论。

特别需要防止预期可见性与实际观测混淆。本地此前已修正“预计视野覆盖即消耗观察机会”的问题；新版本不能因借鉴视锥衰减或历史预测汇聚而重新消耗实际被遮挡、尚未测到的表面。保留实际测到的支持和方向证据作为反馈入口，预测遮挡区域继续保持为待检验假设。

## 四模块中的职责与剩余实现

| 接口 | 下一版职责 | 当前证据边界 |
|---|---|---|
| OV-SDF | 提供来源明确的类别概率、置信度、观测几何及关联不确定性 | 人工二类编码转换已核查；自然语义和稳定设施关联未验证 |
| STGHP | 在共同安全候选池中比较预算约束下的长期增量预测 | 候选历史、去程干预和共同延续已实现；新语义模型未获得训练资格 |
| RPN-UQ | 执行与返航约束保持公共；学习风险与确定性守卫分开报告 | 已验证确定性守卫；学习式风险校准仍待完成 |
| IGCR | 以实际三维观测检验候选预测，并有界更新设施／方向条件状态 | V16 全局三维辅助观察器通过核查；条件归因、预测校正和净收益未完成 |

候选预测的训练目标仍是原协议的长期带符号增量，例如 `Δ(二维覆盖率 × F1@5cm)`，并分别保存其各物理分量。对当前观测历史相同的候选，语义残差只能使用当时已经得到的类别信息，不能读取物理结构种子、世界布局编号或未来得分。

新反馈向量可包含首次支持数量、旧支持的新观察方向、动作前固定支持集合的带符号质量变化、缺失和重新出现的支持数量。它们都有明确单位或计数含义；目前不应直接以加权和充当精度奖励。既有可靠性代理可能因重复观察和残差平滑而改善，须检验它是否对应最终表面误差或 F1 的变化。

条件归因应在动作开始前记录候选所针对的观测设施及其稳定支持集合。若新观测无法可靠归属某个设施，保留为未分配证据；不要因为执行时持有某个候选编号就把整幅图像的收益全部分配给它。对结构合并、分裂、类别反转和坐标修正，分别处理关联失效与真正的低收益。

## 必须保留的验证链

第一步仍是完成已冻结 V15 候选的信息价值检查。它只涉及一个开发父布局，即使通过也不足以训练和宣称泛化。若不通过，下一步是单独冻结的多决策诊断，判断最初观察是否需要后续动作才能兑现价值；不能用新反馈指标替换原失败门槛。

进入具有足够父布局的数据阶段后，需要同时证明三件事：正确语义存在候选条件信息；等容量预测器利用了这些信息而非记住场景；反馈校正实际改变了选择且提高长期物理回报。预测误差更小而所有方法仍选同一路线，不构成新的语义决策收益。

关闭反馈、语义打乱、零置信度和固定语义对照须共享候选、安全约束、重建后端和总预算。还应保留几何侧同容量的反馈输入与预测头，防止把新增三维观测通道的公共收益归到语义。未来独立确认集只能在方法与阈值冻结后启用。

最终实验同时给出覆盖、precision、recall、F1、表面误差、有效新增表面积、运行时间、路程、碰撞和返航情况。渲染图更好、局部代理更高、模型预测更准及任务联合指标更高，应分别陈述。只有最后的决策与物理证据链闭合，才能把目前的接口实现提升为完整方案优势。

## 来源

1. Zhengquan Zhang, Feng Xu, Mengmi Zhang. *Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction*. [arXiv:2506.14856v2](https://arxiv.org/html/2506.14856v2)，2026 年修订版；核对方法、实验、附录 S5。
2. ZhangLab-DeepNeuroCogLab. [PUN 官方仓库 README](https://github.com/ZhangLab-DeepNeuroCogLab/PUN/blob/main/README.md)，访问日期 2026-09-14；核对作者声明的录用与发布状态。
3. Tianling Xu 等. *AREA3D: Active Reconstruction Agent with Unified Feed-Forward 3D Perception and Vision-Language Guidance*. [arXiv:2512.05131v1](https://arxiv.org/html/2512.05131v1)，2025 年预印本版本；核对第 3 节、表 4。此处不推断后续版本内容相同。
4. Jingsen Zhu, Silvia Sellán, Alexander Terenin. *A Bayesian Approach for Task-Specific Next-Best-View Selection with Uncertain Geometry*. [论文 v1](https://arxiv.org/html/2605.05095v1)，2026；正文标注 SIGGRAPH Conference Papers 2026。
5. Jingsen Zhu 等. [BayesianNBV 官方仓库](https://github.com/jingsenzhu/BayesianNBV)，访问日期 2026-09-14；核对 GPU 依赖及待发布内容。
6. Antoine Guédon, Tom Monnier, Pascal Monasse, Vincent Lepetit. *MACARONS: Mapping and Coverage Anticipation With RGB Online Self-Supervision*. [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Guedon_MACARONS_Mapping_and_Coverage_Anticipation_With_RGB_Online_Self-Supervision_CVPR_2023_paper.html)，940–951。

本地证据：`audit_results/semantic_v15_feedback_observability_20260914` 与 `audit_results/surface_feedback_v16_replay_20260914`。外部论文结果没有作为本项目成绩，也没有用于改动当前冻结实验。
