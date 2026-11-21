# 全文整合建议与证据对应（工作稿）

原稿 `../GRADUATION_THESIS_20260928.md` 不覆盖。本目录章节是可审查的增补稿，尚未完成全文修订或学校格式排版。当前原标题与开题《基于语义增强与结构感知的未知大场景主动覆盖探索方法》不一致，整合时以开题题目为主，并核对学校规定。

## 插入位置与结构

1. 在原第3章后插入本章“未知环境联合探索方法及实车验证”，作为新第4章；原第4、5章顺延为第5、6章。原第3章和原第4章继续明确为公开图与模板的受控机制研究，其结果不替代未知探索。
2. 原第2章第5节第187行起的任务和信息权限只适用于V35。保留其限定，并在节首注明；增加本章第1节定义的在线未知地图与传感输入。原第3章第3节第274行公开安全图说明同样仅限原机制。
3. 原第1章第4节和第5节（第90、106行起）改为未知覆盖主线、表面观察联合目标，说明先有受控机制研究，再有不同版本的实车探索；原第106—108行五章组织应按六章重写。相关研究与指标需相应补充未知探索文献和覆盖定义，引用需逐项核验，不凭本工作稿新增文献。

## 可审查的摘要及结论替换建议

- 原中文摘要第3节及英文摘要第15节应整体复核，删除将受控设施建档当作全文唯一任务或将其性能迁移到未知探索的表述。可加入：“针对部分未知环境，构建以覆盖未知区域为主、兼顾实测表面观察的在线探索流程。基于明确标注的部署变体，在一个真实场景中完成40次动作及返航，内部观测面积代理由72.84 m²增至120.00 m²；后续7次动作后因当前轮廓安全判定停止，未完成返航。该案例验证了受限闭环运行，尚不能证明相对基线的效率或语义收益。”英文摘要应对应这一边界，不直接把代理写成ground-truth coverage。
- 原第5章第1、2节（第1024、1032行起）增加上述未知探索工作与失败，受控V35结果保留原条件及负结果；不得将两种方法的效果合并为一个性能结论。
- 原第5章第3节第1052行“尚不能推出……未知地图安全探索”可改为：“受控设施建档结果本身不能推出未知地图探索能力；另行完成的一次部分未知场景实车闭环及后续中止记录提供了部署可行性证据，尚不足以证明未知场景的效率优势或安全保证。”原第1054行不同评价定义不可合并的限定保留。
- 原第5章第4节（第1058行起）将匹配实车覆盖基线、自然语义对实际决策的作用、独立参考重建精度及多场景重复列为未完成工作；未知探索不再仅列为未来方向。

## 主要主张—证据

| 主张 | 直接证据 | 状态与限制 |
|---|---|---|
| 评分公式和部署变体 | `../../../nso/cpu_four_modules_v10.py` 的 `_scores_v10_1`；`/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge/nso_bridge/exploration_joint_policy.py` | 代码支持；部署与原版不等价，实际参数以轮次配置为准 |
| 40动作、返航、面积代理、12次S/G同分 | `/home/ouc/nso_robot/analyses/resume_20261104/episode_audit.json` | 单次闭环；无匹配G轨迹或独立面积真值 |
| 7动作追加及停止 | `/home/ouc/nso_robot/analyses/continuation_20261104/episode_audit.json` 和 `footprint_audit.json` | 非独立续行；黑矩形身份待确认，未返航 |
| 对象支持离线诊断 | `/home/ouc/nso_robot/analyses/continuation_20261104/object_only_continuation.json` | 仅离线反事实，不证明轨迹或收益 |
| 原版首帧八条矩阵 | `/home/ouc/nso_robot/analyses/reproduction_20261104/fresh_matrix/protocol.json` 与 `results.json` | 两个互换布置，不是独立自然场景；与部署版不同 |
| 语义提高效率、真实重建精度 | 无匹配证据 | 待补，不得声称 |

自检：数据流和候选分数均区分预测与观测；同状态评分不写成成对实验；保留S负结果、续行失败、前缀成本及非独立性。需进一步核对全文术语、参考文献、图表编号、学校模板和新增独立数据，不能宣称全文已完成。

- 部署策略固定矩阵：`/home/ouc/nso_robot/analyses/reproduction_20261104/deployed_policy_matrix/{protocol,results}.json`，共8条，物理适配器关闭；west 96 S高于G，其余三对持平。原版west 96为负差，不合并两版本，也不推出实车语义收益。
- F1标签的阈值由 `/home/ouc/nso_robot/workspaces/nso_research_20261029/NSO/utils/reconstruction_metrics.py` 中 `round(threshold*100)` 的命名约定及实际评价代码核验：`05cm` 对应0.05 m，即5 cm。模型型号查 `/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge/models/real_semantics_20261030/MODEL_MANIFEST.json`；实车配置另见 `config/real_semantic_frontend.json`。

本地PDF由 LibreOffice 渲染，公式作为图片嵌入；chapter.tex 保留可编辑数学源码。PDF为阅读稿，未按学校模板排版，当前机器未验证TeX编译。

- 同候选覆盖评分消融：`/home/ouc/nso_robot/analyses/reproduction_20261104/coverage_ablation_matrix/{protocol,results}.json` 及 `comparison.csv`；四条C，east 96实际94步。C不是标准frontier。west 96 G/S的F1差异主要来自召回，精确率及平均距离未提高。轨迹图来源 `/home/ouc/nso_robot/analyses/reproduction_20261104/comparison_artifacts/west96_trajectories.pdf` 及同名PNG；本目录复制供全文整合。
