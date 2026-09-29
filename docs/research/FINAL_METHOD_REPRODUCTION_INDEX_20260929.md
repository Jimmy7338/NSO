# 方法—实现—证据与最终复现索引（2026-09-29）

本索引对应当前 CPU 虚拟主动观测实现、英文稿和毕业论文。用途是从论文中的式子追到实际执行符号，再追到已封存结果及图件。四模块表示可执行接口职责，不把旧神经网络、自然图像语义识别或真实 SLAM 定位精度写成已验证能力。

机器索引：[audit_results/final_delivery_20260929/reproduction_index.json](/root/NSO/audit_results/final_delivery_20260929/reproduction_index.json)。轻量入口：[scripts/verify_final_reproduction_index_20260929.py](/root/NSO/scripts/verify_final_reproduction_index_20260929.py)。

## 1. 一条命令检查交付材料

从仓库根目录执行；仅需要 Python 标准库。该命令不写入实验、不导入科学运行模块，也不重新运行已经通过的重放。

```bash
cd /root/NSO
python3 -B scripts/verify_final_reproduction_index_20260929.py
```

本次实际检查结果：**542/542 通过**，279 个去重后的文件，约 34.4 MB 字节哈希；包含 45 份执行源码、59 份历史保护源码、45 成员源码 ZIP、25 个公式所指代码符号和 14 份 manifest 的 114 个输出。两组源码有交集，不能把 45+59 当作 104 个不同源码文件。工具读取已通过的独立复核、重放和浏览器收据，不将收据校验宣称为重新进行逐帧审查或实验。

范围不包含整个历史仓库、全部原始传感数组或归档解压回读。完整帧审查已有独立报告：[docs/research/FINAL_VALIDATION_INDEPENDENT_REVIEW_20260929.md](/root/NSO/docs/research/FINAL_VALIDATION_INDEPENDENT_REVIEW_20260929.md)。正在排版的正文只按路径与章节索引；最终 PDF 的字节由各自构建 manifest 绑定。

## 2. 公式到代码及四接口

四接口均落在 [nso/cpu_four_modules_v35.py](/root/NSO/nso/cpu_four_modules_v35.py) 的 `CPUFourModuleControllerV35`。G/S 使用共同的安全图、两份公共模板、规划器和实际深度融合；G 同样能诊断和补看。

| 接口 | 实际输入与输出 | 执行符号 |
|---|---|---|
| OV-SDF | validated current paid RGB/depth/ranges/pose and map digest → category support and belief receipt | `CPUFourModuleControllerV35.update_semantic` |
| STGHP | belief, accumulated public-template masks, common safe graph, remaining budget → candidate values and selected next atomic observation action | `update_topo / compute_reward / select_target` |
| RPN-UQ | selected action, graph return distance, remaining budget → legal budget-feasible action or stop | `assess_action / next_action` |
| IGCR | acquired geometric residuals at distinct paid poses → bounded geometric log odds and next-step replanning evidence | `update_semantic -> ObservationBeliefV35.update` |

| 方法关系 | 代码符号（相对仓库根目录） | 论文与证据 |
|---|---|---|
| o_t=(RGB_t,D_t,ranges_t,T_t,map_hash_t), immutable arrays | `nso/cpu_four_modules_v35.py::ObservationV35.__post_init__`<br>`nso/cpu_four_modules_v35.py::CPUFourModuleControllerV35.accept` | Article 4.1; Table 1 / Figure 1；Thesis method 2–3；saved_replay_case0 exact per-frame/controller checks；new8 independent review, all 344 frames |
| L_s in {0,+log(9),-log(9)}; p(h0)=sigmoid(L_s+L_g) | `nso/observation_belief_v35.py::_rgb_class`<br>`nso/observation_belief_v35.py::ObservationBeliefV35.update`<br>`nso/observation_belief_v35.py::ObservationBeliefV35.probabilities`<br>`nso/cpu_four_modules_v35.py::CPUFourModuleControllerV35.update_semantic` | Article 4.2 / 5.2 / 5.7；Thesis method 4 / 7；Table 2 / Figures 2–3；new8 rear-partition h1: common prefix, decision18/action19 |
| rho(r)=min(r*r/2,12.5); per-pose log-likelihood evidence clipped to ±6; accumulated L_g clipped to ±24 | `nso/observation_belief_v35.py::_geometry_score`<br>`nso/observation_belief_v35.py::PublicTemplatesV35.geometry_evidence`<br>`nso/observation_belief_v35.py::ObservationBeliefV35.update`<br>`nso/cpu_four_modules_v35.py::CPUFourModuleControllerV35.update_semantic` | Article 4.2 / 5.3；Thesis method 4 / simulation 6；Figure 5; V35 original wrong-prior trace；new8 first measured geometry feedback step28 |
| U(p,M)=sum_h p_h U_h(M_h); masks accumulate by bitwise union; U_h is public coverage × visible-surface potential | `nso/pixel_information_v34.py::SavedPotentialModelV34._terminal`<br>`nso/cpu_four_modules_v35.py::CPUFourModuleControllerV35.update_topo`<br>`nso/cpu_four_modules_v35.py::CPUFourModuleControllerV35.compute_reward`<br>`nso/cpu_four_modules_v35.py::CPUFourModuleControllerV35.select_target` | Article 4.3；Thesis method 5；new8 candidate scores in independent review; three changed-score yet same-action pairs |
| lambda_0=p*r+(1-p)*(1-r); posterior branches p*r/lambda_0 and p*(1-r)/(1-lambda_0), r=.99; one future information event per lookahead | `nso/online_planner_v35.py::ForecastBeliefPlannerV35.select` | Article 4.3–4.4；Thesis method 5 / 5.1；old V35 G and S share diagnosis capability；all four new-layout G/S pairs retained |
| 1+d_return(v_next)<=remaining_budget; terminal at exact initial pose; frozen both-template coverage feasibility | `nso/cpu_four_modules_v35.py::CPUFourModuleControllerV35._return_distances`<br>`nso/cpu_four_modules_v35.py::CPUFourModuleControllerV35.assess_action`<br>`nso/cpu_four_modules_v35.py::CPUFourModuleControllerV35.next_action`<br>`nso/online_planner_v35.py::ForecastBeliefPlannerV35.select` | Article 4.3 / 5.1；Thesis method 5.2；new8 42 paid actions/exact return/no collisions；128: all 32 infeasible B30 attempts preserved |
| one saved paid RGBD observation -> one TSDF update; voxel .04m, truncation .12m; no template completion | `nso/observed_runtime_mapper_v10.py::ObservedRuntimeMapperV10.update`<br>`nso/mapping3d.py::SensorMapper.update`<br>`nso/mapping3d.py::SensorMapper.mesh`<br>`nso/surface_measurement_v34.py::extract_observed_asset_mesh` | Article 4.3 / Figure 4；Thesis method 3 / simulation 5.2；Figure 4; Figure 6; new8 actual meshes；case0 exact raw/ROI prefix and final meshes |
| F_tau=2*P_tau*R_tau/(P_tau+R_tau); J_tau=C_map*F_tau; main tau=.05m | `nso/surface_measurement_v34.py::SurfaceMeasurementV34.evaluate`<br>`scripts/run_final_local_layout_validation_20260929.py::run_case` | Article 3 / Tables 2–4 / 6；Thesis simulation 4 / 11；8 confirmation + 128 extension + 40 existing checkpoints + final new8 |

类别权重采用标准对数几率更新；当前 RGB 类别线索为受控可见信号。`IGCR` 的测量更新与 `OV-SDF` 在同一已验证观测入口执行，分别留下模块收据。规划势函数来自公共模板可见性，不是实际 TSDF 的 F1 或未观测表面的真值；实际全帧深度进入独立共同后端。

深度残差采用公开模板预测与当前付费测量，单视角证据和累计证据均有固定幅值限制；重复几何位姿不会重复计数。`ForecastBeliefPlannerV35.select` 内部的 `stop/quiet/continuation/value` 实现一次未来诊断事件的近似。论文中的简化代价条件用于解释，不能当作另一个实装规划算法。返航条件依赖给定安全图和精确仿真位姿，不外推为任意定位漂移或动态避障保证。

英文稿：[docs/thesis/ARTICLE_SUBMISSION_EN_20260928.md](/root/NSO/docs/thesis/ARTICLE_SUBMISSION_EN_20260928.md)；毕业方法章：[docs/thesis/GRADUATION_METHOD_CHAPTER_20260928.md](/root/NSO/docs/thesis/GRADUATION_METHOD_CHAPTER_20260928.md)；实验章：[docs/thesis/GRADUATION_SIMULATION_CHAPTER_20260928.md](/root/NSO/docs/thesis/GRADUATION_SIMULATION_CHAPTER_20260928.md)。

## 3. 完整数据组与主表来源

| 数据组 | 完整规模与解释 | 当前稿位置 | 封存汇总／表格 |
|---|---|---|
| 原确认 8 条 | 8 saved G/S routes, four pairs from two base layouts; 2 wins + 2 ties | Table 2, Figures 2–4, Section 5.2 | [汇总](/root/NSO/audit_results/v36_online_confirmation_review_20260918/result.json) / [完整表](/root/NSO/docs/thesis/figures/virtual_paper_20260928/v36_measurements.csv) |
| CPU 外部机制 | G/S records reused; SWAP-I/VISTA-I are common CPU mechanisms, not full original systems | Table 3, Section 5.4 | [汇总](/root/NSO/audit_results/v39_external_cpu_20260920/result.json) / [完整表](/root/NSO/docs/thesis/figures/virtual_paper_20260928/v39_method_means.csv) |
| 预算／噪声／同族扩展 128 项 | 128 declared; 96 qualified and 32 B30 failures; two noise seeds are within-layout repeats | Table 4, Figure 7, Section 5.5 | [汇总](/root/NSO/audit_results/thesis_expansion_20260928/analysis_review/summary.json) / [完整表](/root/NSO/audit_results/thesis_expansion_20260928/analysis_review/complete_endpoint_metrics.csv) |
| 多实例开发与两次消融 36 项 | 12 Original + 12 Ground + 12 Exposure; one original pipeline failure retains separately derived quality; J_nav is not local J5 | Table 5, Section 5.6; thesis simulation Section 10 | [汇总](/root/NSO/audit_results/article_stage_20260928/analysis_v1/article36_final_summary_v1/summary.json) / [完整表](/root/NSO/audit_results/article_stage_20260928/analysis_v1/article36_final_summary_v1/rows36.csv) |
| 既有轨迹 40 个检查点 | eight old V36 routes × five measured checkpoints 18/24/30/36/42; zero new independent samples; 16 existing endpoint/prefix parities exact | Figure 6 and full P/R/mesh supplement, Section 5.3 | [汇总](/root/NSO/audit_results/article_stage_20260928/reconstruction_checkpoints/result.json) / [完整表](/root/NSO/audit_results/article_stage_20260928/reconstruction_checkpoints/checkpoint_metrics.csv) |
| 最终新背景布局 8 项 | two fixed new background layouts, same public equipment family; eight qualified tasks/four pairs; 1 win, 3 ties; eight unused reserved tasks | Table 6, Figure 8, Section 5.7; thesis simulation Section 11 | [汇总](/root/NSO/audit_results/final_local_layout_validation_20260929/analysis/summary.json) / [完整表](/root/NSO/audit_results/final_local_layout_validation_20260929/analysis/episode_metrics.csv) |

各数据组不能合并成独立场景样本数。原确认的 G/S 路线在外部机制比较中复用；40 个检查点是 8 条旧路线的重复测量；噪声种子是布局内重复。场景级 `J_nav` 与局部 `J5` 评价域不同，不作混合平均。36 项中的原 CELL/G 评价失败保留，后来同版本派生质量另列。共享 S/B 的三布局必要机制条件未满足，48 条主矩阵未执行，不填零分，不写成失败实验。

最终新增布局保持同一设备模板族，只增加固定侧柱或后方隔断；不是新类别或新自然语义任务。8/8 合格、4 对结果为 1 胜 3 平 0 负；均值 G=0.7051877647、S=0.7262149057，相对差 +2.9818%。唯一正差是 rear-partition/h1：第18步决策、第19次实际动作分歧。side-column 两构型的类别分数确实变化，但候选排序和实际路线保持相同。详情由独立复核报告给出，不归因为未证实的可达性阻断。最终额度实际只用 8 条，另外 8 条保留未运行。

表1来自四接口合同；表2读取 `v36_measurements.csv`，表3读取 `v39_method_means.csv`；表4完整任务及分层均值来自 128 项 `analysis_review/`；表5读取 `rows36.csv`；表6读取最终 `episode_metrics.csv` 与 `paired_metrics.json`。以下只把保存表格打印到终端，不调用评价器：

```bash
python3 - <<'PY'
import csv,json
from pathlib import Path
index=json.loads(Path('audit_results/final_delivery_20260929/reproduction_index.json').read_text())
for name, cohort in index['cohorts'].items():
    print('DATASET',name,cohort['table'])
    with Path(cohort['table']).open(newline='') as stream:
        for row in csv.DictReader(stream): print(json.dumps(row,ensure_ascii=False))
PY
```

## 4. CPU 环境与代表性重放

实际科学环境为 Python 3.12.3、NumPy 1.26.4、SciPy 1.11.4、Open3D 0.19.0；CPU 为 Intel Xeon Processor (SapphireRapids)，可用逻辑核及 affinity 均为4。在线每次仅执行1条任务，三个 BLAS/OpenMP 线程变量均为1，无 GPU。版本来源：[真实 started.json](/root/NSO/audit_results/final_local_layout_validation_20260929/cases/L02_side_column_h0_b42_n350918_G/started.json)。科学解释器为 `.venv-3d/bin/python`，绘图解释器为 `.venv/bin/python`。

最小科学依赖见 [requirements-cpu.txt](/root/NSO/requirements-cpu.txt) 与 [requirements-3d.txt](/root/NSO/requirements-3d.txt)；完整记录见 [requirements-cpu.lock.txt](/root/NSO/requirements-cpu.lock.txt) / [requirements-3d.lock.txt](/root/NSO/requirements-3d.lock.txt)。Open3D 的安装分发名为 `open3d-cpu`，导入名为 `open3d`。该 CPU 过程不依赖 PyTorch、Habitat 或旧模型权重。当前 bundled Open3D 没有 dist-info 时，运行记录从模块 `__version__` 获取版本。

在全新机器建立环境可用以下命令；这里只提供步骤，本次未新装依赖。任意替换 NumPy/BLAS 后，不能预先声称位级一致。

```bash
python3.12 -m venv .venv-reproduce
.venv-reproduce/bin/python -m pip install -r requirements-3d.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

代表性任务事先固定为列表第0项 `L02_side_column_h0_b42_n350918_G`。已完成一次保存包重放，43帧、42次动作、耗时7.044秒：每帧地图和完整控制器计划/动作、前缀及终点原始/ROI网格均精确一致。该次重放确实重新融合了已保存帧，但没有新 World、传感查询、在线样本或质量评价。本轮封装只读取其 PASS 收据，不再次重放。

记录：[audit_results/final_delivery_20260929/saved_replay_case0/result.json](/root/NSO/audit_results/final_delivery_20260929/saved_replay_case0/result.json)；前置声明及源副本由同目录 `manifest.json` 绑定。将来需要自行重放时，使用尚不存在的新目录：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-3d/bin/python -B scripts/replay_final_local_saved_case_20260929.py --output tmp/reproduction_20260929/case0_replay
```

## 5. 两至三分钟离线演示

浏览器直接打开 [index.html](/root/NSO/docs/thesis/demos/v40_replay/index.html)，不需要服务端或外网。已有 Chromium 140.0.7339.16 的 **10/10 PASS**，覆盖关键动作、播放暂停、相机联动、h0零收益对照、移动布局、无外网请求和无未来网格泄漏；记录见 [audit_results/final_delivery_20260929/replay_browser/result.json](/root/NSO/audit_results/final_delivery_20260929/replay_browser/result.json)。

| 时间 | 演示操作与讲述 |
|---|---|
| 0–20秒 | 打开 HTML，说明真实保存的 RGB-D、位姿、付费动作与实际融合。页面用的是旧 P00 h0/h1 保存组，不是最终新8条。 |
| 20–60秒 | 选择 h1，定位共同18步前缀，点击“关键分歧”，对照决策18与动作19及类别信念。 |
| 60–95秒 | 播放后暂停；滑到42，打开离线评分，切换相机以同视角比较保存的终点网格。 |
| 95–120秒 | 切换 h0，展示路线及得分相同的对照，说明类别信息不保证每种布局都有额外增益。 |
| 120–150秒 | 打开最终新8条 `transfer_paths.pdf` / `transfer_quality.pdf`，展示全部4对及第0条精确重放 PASS。 |

中间19–41步明确显示日期为第18步的保存网格，不是每步重建动画；不得描述为连续实测精度曲线。评分为离线评价，必须主动勾选。浏览器检查仅说明界面与保存内容一致，不增加科学证据。

## 6. 图件的离线重绘命令

以下均从 `/root/NSO` 执行，只读取已有测量、轨迹或网格。每个输出使用 `tmp/reproduction_20260929/` 下不同的新目录，防止覆盖当前稿件图。入口已通过源码/参数检查；本轮没有为制作索引重复运行制图。图件的实际封存输出已由轻量检查逐个核对。渲染版本、字体或 PDF 时间戳可能改变新文件字节；不要求重新排版的 PDF 字节相同。

绘图命令不应替换为任何在线 runner 的 `run`，也不应运行检查点脚本的 `run`。若依赖原始数组已归档，先按附录恢复原字节到独立工作区并设置对应输入路径；缺输入应停止，不生成替代数据。仅看正文与现有图件无需恢复传感归档。

### figure1 — final_architecture.pdf

vector drawing and unchanged user photograph; no scientific data；封存记录：[manifest/provenance](/root/NSO/docs/thesis/figures/final_architecture_20260929/manifest.json)。

```bash
.venv/bin/python -B scripts/plot_final_architecture_20260929.py --output tmp/reproduction_20260929/architecture
```

### figure2 — local_paired_trajectories.pdf

saved V36 routes; full generator also renders earlier six-parent logs；封存记录：[manifest/provenance](/root/NSO/docs/thesis/figures/article_trajectories_20260928/manifest.json)。

```bash
.venv/bin/python -B - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import plot_article_trajectories_20260928 as p
p.OUT=Path('tmp/reproduction_20260929/trajectories').resolve()
assert not p.OUT.exists(), 'Choose a new output directory'
p.main()
PY
```

### figures3_5 — publication_main.pdf; publication_timeline.pdf

saved presentation CSVs; all four conditions retained；封存记录：[manifest/provenance](/root/NSO/docs/thesis/figures/virtual_method_20260928/manifest.json)。

```bash
.venv/bin/python -B scripts/plot_virtual_paper_method_20260928.py --output tmp/reproduction_20260929/method
```

### figure4 — mesh_comparison.pdf

saved original meshes and geometry; no fusion or evaluator；封存记录：[manifest/provenance](/root/NSO/docs/thesis/figures/scene_details_20260928/provenance.json)。

```bash
.venv/bin/python -B - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import plot_thesis_scene_details_20260928 as p
p.OUT=Path('tmp/reproduction_20260929/meshes').resolve()
assert not p.OUT.exists(), 'Choose a new output directory'
p.main()
PY
```

### figure6 — checkpoint_quality_submission.pdf

40 saved checkpoint rows, 120 plotted C/F1/J values; no new checkpoint measurement；封存记录：[manifest/provenance](/root/NSO/docs/thesis/figures/article_reconstruction_submission_20260928_final/manifest.json)。

```bash
.venv/bin/python -B scripts/plot_article_submission_checkpoints_20260928.py --output tmp/reproduction_20260929/checkpoint_submission
```

### figure7 — budget_effects.pdf

complete 128-row endpoint table including 32 failures；封存记录：[manifest/provenance](/root/NSO/docs/thesis/figures/expansion_20260928/provenance.json)。

```bash
.venv/bin/python -B - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import plot_thesis_expansion_20260928 as p
p.OUT=Path('tmp/reproduction_20260929/expansion').resolve()
assert not p.OUT.exists(), 'Choose a new output directory'
p.main()
PY
```

### figure8 — transfer_paths.pdf

all eight saved new-layout paths and meshes; also produces transfer_quality/transfer_meshes；封存记录：[manifest/provenance](/root/NSO/docs/thesis/figures/final_validation_20260929/manifest.json)。

```bash
.venv/bin/python -B - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import plot_final_validation_20260929 as p
p.OUT=Path('tmp/reproduction_20260929/new8').resolve()
assert not p.OUT.exists(), 'Choose a new output directory'
p.main()
PY
```

### supplement_mechanisms — mechanism_cases.pdf

whole AISLE and CELL saved paths; opposite signed outcomes retained；封存记录：[manifest/provenance](/root/NSO/docs/thesis/figures/article_mechanism_cases_20260928_final/manifest.json)。

```bash
.venv/bin/python -B scripts/plot_article_mechanism_cases_20260928.py --output tmp/reproduction_20260929/mechanisms
```

### supplement_checkpoints — checkpoint_quality_components.pdf; checkpoint_actual_meshes.pdf

only renderer.plot(), never checkpoint run/reconstruction；封存记录：[manifest/provenance](/root/NSO/docs/thesis/figures/article_reconstruction_20260928/manifest.json)。

```bash
.venv/bin/python -B - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import plot_article_reconstruction_20260928 as p
p.renderer.FIGURES=Path('tmp/reproduction_20260929/checkpoint_full').resolve()
assert not p.renderer.FIGURES.exists()
p.renderer.save_figure=p.publication_save
p.renderer.plot()
PY
```

### supplement_exposure — exposure_paired_quality.pdf; exposure_paired_routes.pdf

complete paired saved analysis; no quality evaluation；封存记录：[manifest/provenance](/root/NSO/docs/thesis/figures/article_exposure_20260928/complete12/manifest.json)。

```bash
.venv/bin/python -B scripts/plot_article_exposure_comparisons_20260928.py --analysis audit_results/article_stage_20260928/analysis_v1/exposure12_complete --output tmp/reproduction_20260929/exposure --render
```

### supplement_ground — ground_fixed_g_routes.pdf

fixed G, all three DEV layouts and actual saved paths；封存记录：[manifest/provenance](/root/NSO/docs/thesis/figures/article_ground_20260928/fixed_g_print_v1/manifest.json)。

```bash
.venv/bin/python -B scripts/plot_article_ground_g_strip_20260928.py --analysis audit_results/article_stage_20260928/analysis_v1/ground12_complete --output tmp/reproduction_20260929/ground
```

## 7. 归档和文稿重建

128项和最终新8条优先使用保存的 CSV/JSON 汇总。多实例原始大数组的原字节归档、失败 CELL/G 特殊包和 materialize 操作按 [复现附录 A.3](/root/NSO/docs/thesis/GRADUATION_REPRODUCTION_APPENDIX_20260928.md) 执行；普通 tar 解包不能替代会重建原 gzip 字节的专用恢复器。索引校验不重新解包已通过 roundtrip 的大归档。

需要重建当前文章或毕业全文时，以下只重新排版，会更新导出的文稿，不运行实验：

```bash
python3 -B scripts/build_writing_deliverables_20260928.py article --render --export
python3 -B scripts/build_writing_deliverables_20260928.py thesis --render --export
```

构建器需要已安装的文稿工具及字体资源；现有构建记录保存输出哈希。学校、学位等留空占位由实际模板补齐，不在复现索引中虚构。各稿只承接对应完整队列的已记录结果；自然设备实验仍是未来补充。

本索引、校验脚本和图表重绘说明不会改变冻结控制器、度量、配置、旧58/62/65闭包或最终45文件。没有新增世界、策略运行、传感包、TSDF融合或表面测量。可用机器索引中的 `scope_exclusions` 核对本次验收的实际范围。
