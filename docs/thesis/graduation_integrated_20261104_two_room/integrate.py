from pathlib import Path
import re, shutil, json
base=Path(__file__).resolve().parent.parent; out=Path(__file__).resolve().parent; old=(base/'GRADUATION_THESIS_20260928.md').read_text(); chapter=(base/'unknown_exploration_revision_20261104/chapter.md').read_text()
title='基于语义增强与结构感知的未知大场景主动覆盖探索方法'
abstract='''## 摘要

面向未知大场景，移动机器人需要在地图不完整和运动预算有限的条件下持续探索，同时兼顾已经测得的设施表面。本文研究未知区域覆盖与实测表面观察的联合决策，并通过独立的受控设施建档研究分析类别先验与几何反馈对观察选择的条件作用。

未知探索部分从激光、RGB-D和位姿更新在线地图，在当前已知可行区域生成候选，并结合覆盖、表面质量与结构对象代理选择动作。基于原v10.3.1的部署变体改变了相机增益、候选组织与失效重规划，并适配传感范围。一个Q0受控场景的两种相关互换布局中，原版和部署策略各完成八条轨迹，另有四条同候选覆盖评分消融和四条frontier对照；24条均返航，但不构成24个独立场景。frontier在95%覆盖阈值更早，联合策略在96步预算下更早达到99%，两者结果混合；语义效应不能解释为未知覆盖效率提升。不同版本在west、96步条件的S/G差异方向相反，其他配对持平。实车部署变体在一个部分已知场景完成40次动作和返航，内部观测面积代理从72.84增至120.00 m²；复用历史的后续7次动作后因轮廓安全判定停止，未返航。实车面积不是独立覆盖真值，同状态的12次S/G评分均相等，尚未验证相对效率、语义收益或真实重建精度。

独立的V35受控机制研究采用公开姿态图、模板和准确模拟位姿。局部配对在两个已见父布局及两种构型中得到两胜两平，平均覆盖与表面F1乘积提高6.16%；新增背景布局四组为一胜三平，平均提高2.98%。预算扩展保留32次规划失败及54步条件下−0.70%的差异；多实例场景级九组S/B比较均无动作或质量差异。这些结果限定于受控设施任务，不能作为未知探索优势的证据。

**关键词：**未知环境探索；主动覆盖；结构感知；语义先验；实测表面

## Abstract

Mobile robots exploring large unknown environments must select observations with incomplete maps and limited motion budgets while also observing measured surfaces. This thesis studies joint unknown-space coverage and measured-surface observation, alongside a separate controlled facility study of the conditional effects of category priors and geometric feedback.

The exploration pipeline updates an online map using laser, RGB-D and pose inputs, generates candidates in currently known feasible space, and scores coverage, surface-quality and structural-object proxies. A deployment variant of v10.3.1 changes camera gain, candidate organization and replanning after invalidation, and adapts sensor ranges. In one controlled Q0 scene with two related swapped layouts, the original and deployment policies each produced eight trajectories, plus four same-candidate coverage-scoring ablations and four frontier comparisons. All 24 returned, but they are not 24 independent scenes. Frontier reached 95% coverage earlier, while the joint policy reached 99% earlier under the 96-action budget. These mixed outcomes do not establish a semantic improvement in unknown-coverage efficiency. The S/G difference at west with a 96-action budget reversed between versions; other pairs tied. In one partially known real scene, the deployment variant completed 40 actions and returned; its internal observed-area proxy rose from 72.84 to 120.00 m². A subsequent history-initialized continuation stopped after seven actions on a footprint safety judgment without returning. The area proxy is not independent coverage ground truth; S/G scores tied at all 12 audited same-state decisions. Relative real-world efficiency, semantic benefit and reconstruction accuracy remain unvalidated.

The separate V35 controlled mechanism uses a public pose graph and templates with exact simulated poses. Local paired tests on two previously seen parent layouts and two configurations yielded two wins and two ties, with a 6.16% mean increase in the coverage–surface-F1 product; four pairs on new backgrounds yielded one win and three ties, with a 2.98% mean increase. The budget extension retains 32 planning failures and a −0.70% difference at 54 steps. Nine scene-level S/B comparisons showed no action or quality difference. These results are confined to the controlled facility task and do not establish an unknown-exploration advantage.

**Keywords:** unknown-environment exploration; active coverage; structure awareness; semantic prior; measured surfaces

'''
old='# '+title+'\n\n'+abstract+old[old.index('## 1 绪论'):]
start=old.index('### 4 研究内容与主要工作'); end=old.index('## 2 相关技术与问题定义')
old=old[:start]+'''### 4 研究内容与主要工作

本文以未知区域主动覆盖为主目标，兼顾已测表面的观察。首先追溯v10.3.1的在线地图、候选和覆盖—表面联合评分，并区分原版与部署变体；基于虚拟传感的固定矩阵检查覆盖及表面指标，保留正差、负差、持平和提前停止。其次在真实RGB-D、激光及底盘反馈上连接语义前端、定位建图与导航，记录一次完整返航和一次续行中止。该部署属于显式修订，不能视为原策略等价复现。

独立的受控设施研究使用公开安全图与模板，分析类别先验、实测几何反馈和预算约束下的观察分配。局部、场景级和不同消融的任务与评价范围各自报告，不能把设施建档的收益迁移为未知探索的性能结论。所有表面地图只融合实测深度；候选预测与终点评价分开。

### 5 论文组织结构

全文共六章。第1章介绍问题、相关研究与研究内容；第2章介绍地图、观察和受控任务定义；第3章介绍公开图与模板条件下的V35受控机制；第4章给出在线未知探索、版本差异、仿真与实车记录；第5章报告独立V35受控实验；第6章总结证据边界与后续工作。附录列出复现材料。

'''+old[end:]
old=old.replace('### 5 任务数学定义与信息权限\n','### 5 任务数学定义与信息权限\n\n本节仅定义第3、5章V35受控设施任务的信息权限；第4章未知探索采用在线地图与实际传感输入，不预先提供全局安全姿态图。\n',1)
old=old.replace('### 3 观测接口与地图数据流\n','### 3 观测接口与地图数据流\n\n本节公开安全姿态图和模板属于V35受控机制的输入，不适用于第4章的在线未知探索。\n',1)
old=old.replace('## 3 系统设计与语义主动观测方法','## 3 受控设施系统设计与语义主动观测方法',1)
old=old.replace('## 4 仿真平台与实验分析','## 5 受控设施仿真平台与实验分析',1).replace('## 5 总结与展望','## 6 总结与展望',1)
chapter=chapter.replace('# 未知环境联合探索方法及实车验证（工作稿）','## 4 未知环境联合探索方法及实车验证',1)
chapter=re.sub(r'^## ([1-5]) ',r'### \1 ',chapter,flags=re.M)
chapter=chapter.replace('### 4 未知环境联合探索方法及实车验证','## 4 未知环境联合探索方法及实车验证',1)
chapter=chapter.replace('表1','表4-1').replace('表2','表4-2').replace('图1','图4-1')
chapter=chapter.replace('![](west96_trajectories.png)', '![](assets/west96_trajectories.png)')
chapter=chapter.replace('仿真使用自身虚拟传感、已知评价参考与原 v10.3.1，不能把其评分差异直接归因于实车部署。', '仿真使用虚拟传感和已知评价参考，分别运行原 v10.3.1 与部署策略变体；部署策略仿真关闭物理适配器，不能把仿真评分差异直接归因于实车部署。')
chapter=chapter.replace('| 证据 | 条件与独立单位 | 记录结果 | 可支持的范围 |', '**表4-1 未知探索仿真与实车记录的证据范围。**\n\n| 证据 | 条件与独立单位 | 记录结果 | 可支持的范围 |', 1)
chapter=chapter.replace('| 策略 | 布置 | 预算 | G覆盖率 / F1@5cm | S覆盖率 / F1@5cm |', '**表4-2 原版与部署策略的覆盖率和表面F1。**\n\n| 策略 | 布置 | 预算 | G覆盖率 / F1@5cm | S覆盖率 / F1@5cm |', 1)

frontier = """### 5 Frontier对照与覆盖进程

为补充外部形式的几何探索对照，在相同首帧、锚点和预算下加入四条本地 observed FrontierPolicy 轨迹。该实现以已观测空闲且对占据格膨胀后的可达格为基础，提取与未知格相邻的边界连通分量；每分量按距离确定性抽取至多四个代表点，加上当前位置后按距离排序，最多保留24个候选。四个朝向按视域内可见未知格计增益，nearest_frontier 优先距离，再按增益和格坐标破同分；无候选时原地扫描，最多四次后停止。它是已有的局部frontier实现，并非完整ROS explore_lite。其执行适配器的共同guard按未知格和占据格检查安全轮廓，并检查执行后的已知返航预算；探索动作首次被拒绝便永久转入返航，策略停止也转入返航。该契约不等价于NSO对候选的预筛和失效后的重规划，不能把差异仅归因于评分。四条件矩阵在west 48的smoke之后冻结且未调参；矩阵中的west 48精确复现该smoke，不是额外独立样本。四次运行各发生一次预算guard干预，分别在west 48/96的第23/76次和east 48/96的第23/53次已完成动作之后，均返航、无模拟碰撞，并耗尽预算；没有策略停止触发返航。

**表4-3 Frontier与部署联合策略G的固定矩阵对照。** 面积增量和表面面积单位均为m²；表面为相对首帧新增、使用仿真参考计算的可见表面，F1阈值为5 cm。四个条件共享同一Q0及相关互换布局，不是独立场景；联合S与G的对应差异见表4-2。

| 布置 | 预算 | Frontier覆盖增量 / 新可见表面 / F1 | 联合G覆盖增量 / 新可见表面 / F1 |
|---|---:|---:|---:|
| west | 48 | 15.00 / 84.12 / 0.8158 | 14.16 / 96.58 / 0.8909 |
| west | 96 | 16.88 / 107.23 / 0.9359 | 16.92 / 109.51 / 0.9456 |
| east | 48 | 15.00 / 84.02 / 0.8130 | 14.16 / 96.46 / 0.8883 |
| east | 96 | 16.76 / 96.56 / 0.8751 | 16.92 / 114.00 / 0.9638 |

在两种布局中，frontier第20动作达到95%覆盖，而96步联合G/S均在第24动作达到；96步下联合G/S第29动作达到99%，frontier第32动作达到。48步预算下frontier覆盖增量较大，但新可见表面和F1较低，结果体现不同目标和返航契约的权衡，不构成普遍排名。west、96步的联合S与G在第31动作首次出现表面差异，此时两者覆盖均为99.55%；两者到达90%、95%、99%的动作分别同为18、24、29。因此，该条件中的语义表面差异不能写成语义改善未知覆盖效率。全部24条轨迹包括原版8条、部署G/S 8条、同候选C消融4条和frontier 4条；它们共享Q0受控场景，均不是实车比较。

"""
chapter=chapter.replace('### 5 讨论与小结',frontier+'### 6 讨论与小结',1)
chapter=chapter.replace('原始与部署策略仿真矩阵提供受控首帧覆盖记录','原始、部署策略及frontier仿真矩阵提供受控首帧覆盖记录')

old=old.replace('## 5 受控设施仿真平台与实验分析',chapter+'\n\n## 5 受控设施仿真平台与实验分析',1)
old=old.replace('### 1 研究工作总结\n','### 1 研究工作总结\n\n本文已实现未知区域覆盖与已测表面观察的在线联合探索，并分别记录原版、部署变体和实车运行。单次实车完成40动作返航；历史初始化续行在7动作后停止，未返航。该案例支持受限集成闭环可行性，不能证明效率、语义收益或重建精度。Q0中的frontier与联合策略在早期覆盖和长预算覆盖阈值上各有较快条件，表面观察结果亦不同；共同guard下的返航契约差异限制归因，不能推出普遍效率排名。\n',1)
old=old.replace('尚不能推出自然设备识别能力、未知地图安全探索或未见大场景泛化','尚不能推出自然设备识别能力、未知地图探索优势或未见大场景泛化；另行完成的实车闭环和中止记录仅提供受限部署可行性证据，不构成安全保证')
old=old.replace('工程扩展可分步替换受控输入：先接入自然类别识别，再引入定位误差和实际传感缺失，最后利用现有自主导航小车进行有限场景的应用演示。实车重点检验接口、动作执行与观察闭环，不替代已经完成的虚拟实验。通过上述渐进扩展，可以在保留现有可复核成果的基础上检验适用范围。','未知探索后续应在相同初始信息、预算和返航契约下进行独立场景重复及匹配的实车覆盖基线比较，分别报告时间、路程、覆盖增量和失败；检验自然语义是否改变实际决策，并用独立参考评价重建精度。当前实车部署已连接自然语义及定位建图，但这些接口和单次运动完成并不证明其性能收益。')
old=old.replace('## 附录A 复现与材料索引','## 附录A 复现与材料索引\n\n未知探索材料：原版代码 `../../../nso/cpu_four_modules_v10.py`；部署策略 `/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge/nso_bridge/exploration_joint_policy.py`；原版、部署策略及覆盖消融矩阵分别位于 `/home/ouc/nso_robot/analyses/reproduction_20261104/fresh_matrix/`、`deployed_policy_matrix/`、`coverage_ablation_matrix/`；实车完整轮次审计 `/home/ouc/nso_robot/analyses/resume_20261104/episode_audit.json`，续行与轮廓审计位于 `/home/ouc/nso_robot/analyses/continuation_20261104/`。frontier协议、四条结果及覆盖进程见 `/home/ouc/nso_robot/analyses/frontier_baseline_20261104/matrix/{protocol.json,comparison.md,progress_metrics.csv,semantic_timing.json}`。详细证据对应见同目录 `integration_and_evidence.md`。这些路径为本机索引，迁移时需随数据调整。\n',1)
old=old.replace('Python及关键科学库版本见第4章平台说明','Python及关键科学库版本见第5章平台说明')
# Copy each referenced historical visual and linked PDF, retaining a portable relative tree.
paths=set(re.findall(r'\]\((/root/NSO/docs/thesis/figures/[^)]+)\)',old)); missing=[]
for p in paths:
 rel=p.split('/root/NSO/docs/thesis/')[1]; src=base/rel; dst=out/rel
 if src.exists(): dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst); old=old.replace(p,rel)
 else: missing.append(str(src))
(out/'assets').mkdir(exist_ok=True); shutil.copy2(base/'unknown_exploration_revision_20261104/west96_trajectories.png',out/'assets/west96_trajectories.png'); shutil.copy2(base/'unknown_exploration_revision_20261104/integration_and_evidence.md',out/'integration_and_evidence.md')
old=old.replace('详细证据对应见同目录 `integration_and_evidence.md`。', '双房间两条轨迹及预检与机制审计独立保存在 `/home/ouc/nso_robot/analyses/frontier_semantic_publication_stage/experiments/unknown_exploration_20261104/two_room_diagnostic/`；事后G×2控制见 `/home/ouc/nso_robot/analyses/geometric_weight_control_20261104/comparison.json`；frontier公平性回放见 `/home/ouc/nso_robot/analyses/frontier_fairness_20261104/audit.json`，均不增加24条矩阵样本。详细证据对应见同目录 `integration_and_evidence.md`。', 1)
old=old.replace('### 6 讨论与小结', '对四个frontier拒绝点的回放与原记录packet一致；剩余预算与完整最短返航代价分别为25/25、20/20、25/25和43/43。在相同已观测安全朝向图上枚举预算可负担的非零出发姿态，并按原baseline的4 m、90°、仅依赖belief的未知增益代理计算，四处正增益目标均为零；可行的原地转向可能是最短返航的一部分。这不能排除未来观测带来的收益，也不证明所有frontier或ROS实现的公平性，适配器契约差异仍然存在。\n\n'+(out/'two_room_section.md').read_text()+'### 7 讨论与小结',1)
(out/'thesis.md').write_text(old)
(out/'validation.json').write_text(json.dumps({'historical_figure_links':len(paths),'missing_historical_assets':missing,'note':'阅读工作稿，未使用学校模板；数据与来源需作者复核。'},ensure_ascii=False,indent=2))
print('assets',len(paths),'missing',len(missing)); print('\n'.join(missing))
