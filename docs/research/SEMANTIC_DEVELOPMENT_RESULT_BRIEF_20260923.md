# 语义机制开发实验：当前结果与未完成项

更新日期：2026-09-24。以进展汇总 010 为本次快照；正常关系组的六父 S/G 与关系失配组的六父 S/B 比较均已齐，12 条预定消融全部完成，固定先验／已知结构对照和边界仍未全部完成。这是冻结开发矩阵的阶段报告，不是独立确认结果。

**当前结论：真实的观察—信念—重规划闭环已有证据；正常关系组六父 S/G 为五平一负，未通过冻结的 5% 语义收益门；关系失配组六父 S/B 全部持平，新增算法门也未通过。** 失配组 B/S 的平均 J_nav 均为 0.521404，相对增量为 0，正增益父数为 0/6。这已是完整六父主比较的实测结论，不再只依赖缺失配对的计数上界；它仍不能扩展为语义方法普遍无效。正常关系组五个完整 S/B 配对同样全部持平，P03 B 中断缺失。

## 1. 已完成的实际工作

截至汇总 010，acquisition 阶段已有 **58 条轨迹完成独立复核，共 6845 个付费动作、6903 个保存传感包**。每条均完成独立策略与 TSDF 回放，58/58 成功返航，碰撞为 0。终点评价包括 16 次独立数值计算与 42 份严格同输入复用收据；复用不替代各自的策略与 TSDF 回放，也不增加独立场景数。另有 15 条完整轨迹尚未计入已复核终点。

来源：[进展汇总 010](../../audit_results/semantic_development_acquisition_20260923/progress_summary_010.json)，SHA-256 `ce26d8cc42464af8ebb981199dd48200e51d8c7588175f07c761624acbadfaff`，文件于北京时间 **2026-09-24 10:54:56** 生成。完整性错误为 0，`ledger_changed_during_summary=false`。本次账本为 73 条完整轨迹、3 条历史中断预约、20 条未启动；主队列已在下一槽创建前因冻结空间门槛停止，无新轨迹中断。汇总自身未运行 World、规划器、TSDF 融合或数值表面评价。下表包含全部六个正常关系开发父场景，缺失值明确保留。

| 正常关系父场景 | 强几何 G：J_nav | 普通贝叶斯 B：J_nav | 完整策略 S：J_nav | S 相对 G |
|---|---:|---:|---:|---:|
| P00 | 0.631784 | 0.631784 | 0.631784 | 0.00% |
| P01 | 0.354030 | 0.354030 | 0.354030 | 0.00% |
| P02 | 0.576491 | 0.565683 | 0.565683 | −1.87% |
| P03 | 0.434995 | 中断／缺失 | 0.434995 | 0.00% |
| P04 | 0.387321 | 0.387321 | 0.387321 | 0.00% |
| P05 | 0.889919 | 0.889919 | 0.889919 | 0.00% |

六父 G/S 平均 J_nav 分别为 **0.545757 / 0.543955**；按冻结的“均值相对差”计算，S/G 为 **−0.3301%**，低于预声明的 +5% 投入门。平均覆盖差为 +0.000301、返航率差为 0，覆盖与返航守卫通过，但收益门未通过。这是完整六父正常关系开发比较的结论，不是独立泛化或语义普遍无效的证明。

在 P00、P01、P02、P04、P05 五个完整 S/B 配对中，终点差均为 0，动作、位姿和网格均一致。由于 P03 B 缺失，预定六父正常关系非劣效比较仍不能完整判定。

关系失配组的 **六父 G/B/S 共 18 条轨迹已全部独立复核**。每个父场景内部，三种方法的覆盖、设施质量和联合终点均相同。

| 关系失配父场景 | G：J_nav | B：J_nav | S：J_nav | 已复核 S−B |
|---|---:|---:|---:|---:|
| P00 | 0.606915 | 0.606915 | 0.606915 | 0 |
| P01 | 0.350849 | 0.350849 | 0.350849 | 0 |
| P02 | 0.521766 | 0.521766 | 0.521766 | 0 |
| P03 | 0.420926 | 0.420926 | 0.420926 | 0 |
| P04 | 0.355879 | 0.355879 | 0.355879 | 0 |
| P05 | 0.872088 | 0.872088 | 0.872088 | 0 |

六父 B/S 的平均 C_nav、Q、J_nav 分别均为 **0.815458、0.626098、0.521404**；S/B 的均值相对差、平均覆盖差和返航率差均为 **0**。六父 B/S 均成功返航。完整比较在汇总 008 中已为 `numeric_thresholds_failed`：未达到 +2% 联合指标门，也未达到至少 4/6 正增益父场景门；覆盖与返航守卫通过。由于不存在正增益，单父正增益占比保持 null，不能写成“集中度为 0、已通过”。剩余固定对照、边界和消融仍用于确定作用条件与失败原因。

**历史必要条件判断：**[汇总 007](../../audit_results/semantic_development_acquisition_20260923/progress_summary_007.json)只有 P00/P01/P02 三个失配配对，全部持平；当时的[计数上界诊断](../../audit_results/semantic_development_acquisition_20260923/diagnostics/shift_positive_parent_gate_summary_007.json)已证明最多 3/6 为正，低于冻结的 4/6。该诊断 SHA-256 为 `f21cf0654bbbaa648b42b7670755de33c60be61138d7bfc3a519f4f6a953492f`，原三个缺失差值与 `insufficient_complete_pairs` 状态均保留。其后 [P03](../../audit_results/semantic_finite_queues_20260923/offline_P03_shift_20260924/result.json)、[P04](../../audit_results/semantic_finite_queues_20260924/offline_P04_shift_recovery/result.json)和 [P05](../../audit_results/semantic_finite_queues_20260924/offline_P05_shift/completion.json)正式完成，才形成上述完整六父结论。

另外，P00 已知结构参考 K 的 J_nav=0.676829，比同场景 G 高约 7.13%。真实结构信息在这一个案例中能够改善结果，但类别先验策略没有兑现该收益；K 仍是近似规划器的实际执行结果，不是最优上界。

正常关系的 **六父固定语义对照 F 已全部复核**。P00/P01/P03/P04 与 G 持平；P02 的 J_nav=0.497844，比 G 低 **13.64%**，P05 的 J_nav=0.890181，比 G 高 **0.0295%**。F 的六父平均 J_nav=0.532692，相对 G 均值为 **−2.3938%**。这是预定固定对照的描述性结果，没有另设或修改收益门。P02 的 B/S 比 F 好，却仍低于 G；由于 F 同时关闭实际信念纠错和未来信息效用，不能将 F→B 的改善单独归因于反馈。

正常关系已知结构参考 K 的 P00/P01/P03/P04 均已正式复核，后三父分别为 J_nav=0.354030 / 0.434995 / 0.387321，与各自 G 持平。P02/P05 的 K 原轨迹中断，继续保留缺失，不能用零分或其他策略替代。[固定对照批次一](../../audit_results/semantic_finite_queues_20260924/offline_nominal_FK_batch01/result.json)、[批次二](../../audit_results/semantic_finite_queues_20260924/offline_nominal_FK_batch02/result.json)及[后续三个固定对照](../../audit_results/semantic_finite_queues_20260924/offline_fixed_controls_batch03/completion.json)均已纳入汇总 009。失配组 F/K 尚未全复核，不能据此宣布五方法矩阵完成。

## 2. 机制确实运行，收益没有随之出现

P02 的三策略在第 70 个付费动作后的观测内容逐位相同。G 的最佳直接观察分数略高于诊断分数；B/S 的相对排序相反，实际执行了右转诊断。第 71 帧更新了设施结构信念，第 72 个付费 observe 完成宏动作并触发后续重规划。这是保存轨迹中的真实层次闭环，不是仅通过接口测试或离线假设分支。

该首次分叉时，每个类别只有一个已观测实例，`peer=[]`、`rho=0.5`。普通贝叶斯 B 和完整 S 因同样的类别先验与本实例信息价值选择同一诊断。因此这条证据不能归因于新增跨实例可靠性机制。

P02 的终点同时揭示了收益代理的限制：B/S 的覆盖比 G 增加 0.001808，但 Q 从 0.658652 降为 0.644971，联合指标下降 1.87%。固定评价设施 3 的 F1 从 0.824663 降为 0.771632，召回从 0.734003 降为 0.657466，是主要损失来源。后续路线也不同，不能把全部损失唯一归因于一次转向；可以确认的是，正的预测信息价值和实际信念更新并未保证更好的终点。

详见 [层次执行证据](SEMANTIC_HIERARCHY_EXECUTION_EVIDENCE_20260923.md) 与 [P02 实际付费诊断图](../../audit_results/semantic_development_figures_20260923/P02_closed_loop/P02_paid_diagnosis.pdf)。该图只使用保存的 RGB 帧、实际信念和决策记录。

## 3. 已定位的方案限制

**消融必须按实际开关解释。** `NC=shared_no_cross_future` 只移除跨实例的未来价值，仍然保留实际共享可靠性信念；不能将 NC 称为“关闭共享可靠性”。对应源码为 `nso/controller_semantic_mechanism.py` 的初始化开关及 `paid_diagnostic_value`，以及 `nso/semantic_reliability.py` 的同类实例更新。

| 保存的策略对照 | 可以检查的机制 | 解释边界 |
|---|---|---|
| S 与 NF | 规划结构信念是否吸收实测反馈 | NF 的前端地图仍更新；前端报告的反馈次数不是 NF 规划信念已更新的证据。 |
| S 与 NP | 未来信息价值是否参与排序 | NP 仍可记录正的信息价值诊断量，但排序使用观测前效用。 |
| S 与 NC | 跨实例未来效用 | NC 保留实际共享信念，不是可靠性整体消融。 |
| NC 与 B | 实际共享信念相对独立实例贝叶斯信念 | B 的字面跨实例未来开关虽为开，但没有共享更新，假想观测不改变其他实例的后验；应结合保存信念和动作说明作用。 |

P00 正常关系和失配条件的六条消融现已各自完成独立策略／TSDF 回放与严格数值复用。正常关系 NC/NF/NP 的 J_nav 均为 **0.6317842952**，失配条件均为 **0.6069153407**，分别与对应完整策略 S 和普通贝叶斯 B 持平。六次轨迹不是六个独立父场景；新增的四种成对差分也不是额外实验。完成凭据见 [P00 正常关系消融](../../audit_results/semantic_finite_queues_20260924/offline_ablation_P00_nom/completion.json)和 [P00 失配消融](../../audit_results/semantic_finite_queues_20260924/offline_ablation_P00_shift/result.json)。

[保存机制诊断](../../audit_results/semantic_development_acquisition_20260923/diagnostics/P00_ablation_saved_mechanisms.json)进一步区分了“机制没有激活”与“激活但未改变终点”：正常关系的反馈受支持点容量限制，NF 没有可移除的实际信念更新；失配条件下 S/NC/NP 吸收了证据并改变同类实例信念，NF 的规划证据仍为零，但各策略实际动作、路线和终点地图一致。NC 与 B 在共享信念上有区别，却未产生动作收益。所有这些 P00 保存轨迹均未实际选择诊断宏动作，因此零终点差不能证明未来信息项在其他条件下无效，也不能作为各模块有效的正面消融证据。

P04 的正常关系和失配条件六条消融也已正式复核：NC/NF/NP 的 J_nav 分别均为 **0.3873207515 / 0.3558791768**，与对应 B/S 同分。结合 P00，冻结的 12 条消融轨迹全部完成复核，两个开发父场景、两个条件下的四类成对差分全部为零。[P04 保存机制诊断](../../audit_results/semantic_development_acquisition_20260923/diagnostics/P04_ablation_saved_mechanisms.json)显示其原因与 P00 失配条件不同：P04 十条保存历史均没有通过资格检查的平面、没有实际反馈或共享可靠性更新、没有诊断候选；各方法之间的动作、信念和终点地图均相同。这里属于机制未激活，不能描述为“可靠性机制有效运转却无收益”，更不能将零差作为各模块独立有效的证明。正常关系每条轨迹的平面失败记录为 27 次标记深度样本不足、1 次姿态不确定、4 次视角倾斜；失配条件每条为 18/1/4 次。这些是当前保存历史的观测限制，不是对所有场景或原神经模块的结论。正式复核见 [P04 正常关系](../../audit_results/semantic_finite_queues_20260924/offline_ablation_P04_nom/result.json)与 [P04 失配](../../audit_results/semantic_finite_queues_20260924/offline_ablation_P04_shift/completion.json)。

- **反馈容量条件阻断纠错。** 支持点缓存到达 4096 后，新点不能插入，`not added` 会使该实例后续反馈不再具备更新资格。P00 G/F/B/S 各有 34 次反馈尝试、18 次有区分力残差，但实际应用为 0。不能把反馈函数调用次数写成有效纠错次数。
- **反馈激活仍不足以改变决策。** P01 G/B/S 各实际应用 7 次反馈；S 的共享可靠性也有变化，但三者动作、地图与终点一致。P00 的缓存原因不能解释所有布局。
- **有同类实例与语义预测，也可能没有可执行收益。** P03 的 S 从第 36 步起已有同类双实例互为 peer，但 rho 始终为 0.5；第 80/82/83/85 步四次反馈均被拒绝，支持点均已达 4096。首个可靠平面到第 80 步才出现，此时剩余 40 个动作。语义改变了 184 条保存的面积预测，但全部正设施预测均未进入路线与完整返航预算允许的直接候选集合；实际评分的 251 条直接候选、120 个动作及终点均与 G 相同。日志没有逐一拆分候选被路线或预算排除的原因，不能把它们全部归因于某一个条件。见 [P03 G/S 保存轨迹诊断](../../audit_results/semantic_development_acquisition_20260923/diagnostics/P03_GS_saved_behavior.json)。
- **共享更新的接收实例缺少可观测条件。** P05 在第 75 步出现真实跨实例信念传递，但接收实例没有可靠平面，其 240 条视点预测全部回退为零效用；G/B/S 的候选排序、目标、动作和终点相同。这说明共享机制确实更新了信念，却没有兑现为可评分的观察收益，不能归结为“共享模块完全没有运行”。见 [P05 机制诊断](../../audit_results/semantic_development_acquisition_20260923/diagnostics/P05_nominal_mechanism_brief.json)。
- **诊断效用代理与最终重建质量存在缺口。** P02 已发生真正付费诊断、纠错和后续规划，但结果更差。还没有证据说明共享可靠性能够弥补该缺口。

关系失配组的 [P00 G/B/S 保存日志诊断](../../audit_results/semantic_development_acquisition_20260923/diagnostics/P00_shift_GBS_mechanism_chain.json)显示：rho 发生变化，但动作和终点网格未变；现有独立复核进一步确认三者终点评分一致。第 66 步的[逐项代数复核](../../audit_results/semantic_development_acquisition_20260923/diagnostics/P00_shift_step66_rho_algebra.json)得到边缘证据比 1.102948、rho=0.524477，与冻结公式及保存值一致，该帧未发现公式计算错误。这一方向来自现有几何证据的边缘化，不能据此认定机制成功检测了类别—结构关系失配，也不能证明观测模型足够准确或经过校准。代数复核本身仍只作机制诊断，不能替代完整轨迹终点评价。

这些限制保留在当前结果中。唯一一次方法修正已经使用；没有通过放宽容量、改变先验、选择场景或换指标继续调出正结果。四模块接口与层次的运行证据，也不等于原神经网络、开放词汇前端或未知地图 SLAM 效能得到验证。

P02 失配组进一步排除了“全部同分都因为没有反馈”这一解释：G/B/S 各有 24 次反馈尝试、22 次实际应用，cabinet 的 88 条视点预测中有 54 条正值。但只形成 rack、cabinet 各一个观测实例，没有同类 peer，B/S 的结构信念逐帧相同。B/S 的 42 个诊断选项中有 9 个正信息价值，仍未超过最佳直接选项；最接近的第 75 步为直接 0.0751113、诊断 0.0735884，三者最终执行相同路线。正常关系组第 70 步则是诊断高出 0.0005461，实际付费执行后终点更差。该对照说明“信息价值为正”“被规划器选中”和“提高最终重建质量”是三个不同的实证环节。两种场景条件并非单因素干预，不能把结果差异唯一归因于某一因素。见[P02 失配组保存轨迹诊断](../../audit_results/semantic_development_acquisition_20260923/diagnostics/P02_shift_saved_behavior.json)，仅分析既有日志，没有新建轨迹或重评成绩。

对 P00 G 的保存终点另做了一次连续距离描述，保留全部 4 个设施、884 个固定参考点，使用原距离算法与采样，不改变主评分。等设施权重下，参考表面到完整预测网格的平均距离为 **18.45 cm**，中位数 **3.27 cm**，95 分位数 **74.42 cm**。误差集中程度与漏建范围不能仅靠一个 F1 阈值充分表达。

完整预测网格到含背景真值表面的平均距离只有 **0.47 cm**、95 分位数 **1.28 cm**，但其中 **88.39%** 预测面积最近的真值属于背景。因此“小的全网格误差”不能作为设施建档完整度高的证据。这两种方向、两种域的误差分别报告，不合称对称 Chamfer 距离。该补充是单条已有终点的描述，不新增方法胜负或独立实验样本；[距离收据与原始数组](../../audit_results/semantic_development_acquisition_20260923/continuous_distance/core_P00_nom_G_b120_lexicographic/receipt.json)可复核。

[连续误差与三维参考点着色图（PDF）](../../audit_results/semantic_development_acquisition_20260923/figures/continuous_distance_P00_G_v3/continuous_distance.pdf)保留全部 884 个设施参考点；固定 1 m 色标只截顶 3 个点的显示颜色，不删点、不裁统计分布。另有 [PNG](../../audit_results/semantic_development_acquisition_20260923/figures/continuous_distance_P00_G_v3/continuous_distance.png)和输入输出哈希。

## 4. 技术中断与剩余工作

**汇总 010 的账本为 73 条完整轨迹、3 条历史中断、20 条未启动，共 96 槽；已正式复核其中 58 条。** [服务观察 006](../../audit_results/semantic_finite_queues_20260924/service_observation_006.json)与[空间停止凭据](../../audit_results/semantic_finite_queues_20260924/primary_storage_guard_stop_006.json)确认 A/B 服务各完成 16/14 条新轨迹后正常触发冻结的 10 GiB 启动门槛，退出码均为 2、重启次数为 0。A 拒绝启动 P01 160 步 G，B 拒绝启动 P01 80 步 S；均在账本预约与 World 创建之前退出，不计为新增失败轨迹或零分。20 条未启动包含这两次资源拒绝及后续 18 条尚未调用的槽。

停止调查时持久可用空间为 10,715,348,992 字节，距单槽启动门槛尚缺 22,069,248 字节；若按每槽 64 MiB 上限为剩余 20 槽预留且结束后仍保留 10 GiB，需要再释放约 1.271 GiB，另有离线产物及其他任务增长。该估计不是磁盘空间承诺。当前不降低阈值、不自动重启、不删除其他项目的构建数据；继续离线复核已封存轨迹。边界实验仍缺项，整个 Goal 尚未完成。

**汇总 008 的历史快照**为 56 条完整、5 个预约行、35 条未尝试，已复核 38 条。当时的预约行包含 3 条历史中断和 2 条运行中任务，`ledger_changed_during_summary=true`，不能再用其数量代表当前库存。

**2026-09-24 启动前库存及汇总 007 的历史时点：43 条完整轨迹、3 条中断、50 条未尝试，共 96 槽。** 其中 29 条完整轨迹已有正式复核。9 月 23 日随后建立的 continuation A/B 队列也已停止；原 PID 均不存在，两队列都没有 terminal result/error，工具会话亦已不可用。因此新增两次中断的退出码与原因均记为 **unknown**，不能沿用更早一次的 143。

新增中断为 P02 正常关系 K（最后完整 paid_step 67，68 个步骤文件、68 组传感包）与 P05 正常关系 K（最后完整 paid_step 99，100 个步骤文件、101 组传感包）；两者均无完整终点或 manifest。最后产物时间均在北京时间 2026-09-23 16:13:14 左右，产物时间不等于确切退出时间。连同原 P03 正常关系 B，这三条都保留为缺失，不重试、不记零分。[本次只读库存](../../audit_results/semantic_queue_interruption_inventory_20260924/inventory.json)核对了原声明、slot_finished、账本及 59 个冻结源文件，未发现绑定错误。

原 A/B 顺序中剩余 **26/24 槽**已形成两份新声明，保留批次边界、互设同伴停止条件，并排除全部已尝试槽。见[精确分区凭据](../../audit_results/semantic_finite_queues_20260924/partition_receipt.json)。无损 Git 维护完成、原入口零 World 预检通过后，两条有限服务已启动，禁止自动重启；[首次运行观察](../../audit_results/semantic_finite_queues_20260924/service_observation_001.json)确认当时 43 完整、3 历史中断、2 运行中、48 未尝试。实际释放 503.47 MiB，59 个冻结源及全部 Git 历史保持原样；可用空间仍不足以保证全部 50 槽完成，每个新实验保留原 10 GiB 门槛。 维护不改变实验额度、方法或评价，不能将队列声明视为已执行实验。

**更早的历史中断：**剩余 85 槽原已明确排入两条有限队列。队列 A 的工具会话返回退出码 143；P03 B 留下 76 个传感包但没有完整终点或封存 manifest。终止信号的来源尚不明确，不能归因为算法任务失败、内存耗尽或 0 分。队列 B 在完成 P04 G/B/S 后于批次边界收到同伴停止信号并退出，没有继续开启新任务。

原始账本保留不变，其 `reserved_before_factory`/`world_created=false` 字段是未写完的预约状态；它不证明该轨迹没创建 World，也不证明进程仍然存活。工具确认的退出与保存传感包优先用于解释这一状态。见 [终止记录](../../audit_results/semantic_finite_queues_20260923/queue_A/error.json) 和 [同伴停止结果](../../audit_results/semantic_finite_queues_20260923/queue_B/result.json)。

**汇总 003 的历史快照**为 15 条完整并已复核、1 条中断、80 条未开始。其四父 S/G 相对差 −0.5544% 及[缺失值范围分析](../../audit_results/semantic_development_acquisition_20260923/diagnostics/nominal_effect_bounds_summary_003.json)仅反映当时尚缺 P03 S/P05 的状态；范围不是置信区间，也不再代替已经齐全的六父 S/G 比较。[原汇总 003](../../audit_results/semantic_development_acquisition_20260923/progress_summary_003.json)、[汇总 005](../../audit_results/semantic_development_acquisition_20260923/progress_summary_005.json)及其原始产物均作为历史快照保留。

随后确认旧队列进程已退出、没有未完成原子写入，59 个来源与账本绑定均正常，原入口 preflight 通过（当时持久可用 13.66GB，大于 10GiB 门槛）。因此在新审计目录明确接续未尝试的 P03 S，再沿原 A/B 顺序执行其余 79 槽；P03 S 与 P05 G/B/S 现已完成独立复核。未知 SIGTERM 原因继续保留，没有重新执行中断的 P03 B、扩大 96 槽或调整方法；新的技术或资源失败仍触发停止。

**汇总 006 时点**的 96 槽为：33 条完整轨迹，其中 22 条已独立复核、11 条待复核；3 个预约行，其中 2 条当时正在执行、1 条是上述 P03 B 中断；60 条未尝试，完整性错误为 0。预约状态必须结合监督记录解释，不能把 3 行都当作运行中或都当作未创建 World。原方案 5 条负结果另存，构造器失败另记；累计上限仍为 **101 World／102 attempts**，唯一一次方法修正已经用完。

[中断清单](../../audit_results/semantic_development_acquisition_20260923/diagnostics/interruption_inventory.json)保留当时的 96 槽分区、76 组三件传感包及 75 个步骤文件、监督退出证据。它记录的是原队列终止时点，不表示后续明确接续的任务仍暂停，也不授权重跑中断槽。

另一次中断发生在**离线复核进程**：会话 33882 在完成 P00 失配组的三条复核后，校验 P01 失配组 G 的保存数据时以 143 退出，原因未知，该次没有产出完整 review JSON。其[独立监督终止记录](../../audit_results/semantic_finite_queues_20260923/continuation_review/error.json)保留了已完成的六条复核和中断位置；退出时两个 World 队列仍在运行。后续以明确的小批任务重新校验已封存数据，保留原中断记录与既有收据，**不新增 World、不重跑原实验**。这与 P03 B 缺少完整轨迹终点的中断不同，不能计作新的算法失败或零分。

9 月 24 日 P04 失配 G 的一次离线复核也以 143 退出，未产出完整 review；[失败检查](../../audit_results/semantic_finite_queues_20260923/offline_P04_shift_20260924/termination_inspection.json)保留原尝试，信号来源未知。随后明确授权在独立有限 systemd 服务中继续校验同一封存轨迹，禁止自动重启，不新增 World；两个主实验服务在该次离线中断时仍正常运行。此次明确恢复的 G 与 B/S 服务现均成功退出，三条完整复核及同输入证明通过，[完成收据](../../audit_results/semantic_finite_queues_20260924/offline_P04_shift_recovery/result.json) SHA-256 为 `50105ba1f0ff496487073a7862711044cb7015d76c883874fa09f0b11e3c0949`。只有恢复成功后的成绩进入汇总 008，旧离线失败与三个原始轨迹中断记录均保留。

**Goal 尚未完成。** 正常关系组六父语义收益门与关系失配组六父算法增量门均已实测未通过；12 条预定消融均已完成；固定先验／已知结构对照及边界仍待完成预定比较，或依据事先声明的停止条件作出可复核的路线决定。正常关系组的 S/B 非劣效门还因 P03 B 中断缺失而不能完整判定。继续实验的作用是解释这一候选的有效范围与失败原因，不能另换门限或缺失处理来恢复“通过”。层次闭环证据、语义收益与新增机制增量分别验收，不能因闭环运行而宣称整体优势，也不能用技术中断代替科学结论。

当前主结果的 [PDF 图](../../audit_results/semantic_development_figures_20260923/progress_010/core_endpoints.pdf)、[逐父配对图](../../audit_results/semantic_development_figures_20260923/progress_010/paired_development_effects.pdf) 和 [全部 96 槽 CSV](../../audit_results/semantic_development_figures_20260923/progress_010/all_declared_endpoints.csv) 均绑定汇总 010。新增 [完整消融图](../../audit_results/semantic_development_figures_20260923/attribution_010/ablation_effects.pdf)和[成对差分 CSV](../../audit_results/semantic_development_figures_20260923/attribution_010/ablation_pairs.csv)显示 P00/P04 两条件的 16 个成对差分均为零。16 个展示差分来自 12 条冻结消融轨迹及已有核心对照；其中 NC/B 是检查开关含义后补充的机制比较，不是新增主验收门，也不是 16 次新实验。边界 CSV 保留全部 30 行，其中 6 行复用核心锚点，没有增加独立样本数。当前尚无已复核的新边界终点，因此没有生成仅重复核心锚点的边界图。

下一步先复核现有 **15 条已封存轨迹**：11 条失配 F/K 与 4 条预算实验；[待复核清单](../../audit_results/semantic_finite_queues_20260924/pending_saved_review_inventory.json)单列 20 条未启动主实验和 3 条历史中断，三者不能混算。主实验需要恢复冻结空间余量后才可接续未尝试槽；不进入独立留出确认，不重新调场景或验收门。汇总 010 是本轮结束的静态快照，尚不构成完整阶段验收。

历史[汇总 009](../../audit_results/semantic_development_acquisition_20260923/progress_summary_009.json)及[原消融图](../../audit_results/semantic_development_figures_20260923/attribution_009/ablation_effects.pdf)保留 52 条快照与 P04 缺失状态。

历史[汇总 008](../../audit_results/semantic_development_acquisition_20260923/progress_summary_008.json)及其[原图](../../audit_results/semantic_development_figures_20260923/progress_008/core_endpoints.pdf)保留 38 条快照；[汇总 007](../../audit_results/semantic_development_acquisition_20260923/progress_summary_007.json)、[汇总 006](../../audit_results/semantic_development_acquisition_20260923/progress_summary_006.json)及原图均按原始字节保留。
