# 共同数值面预处理与离线补充评价方案（2026-09-28）

本文件说明已实现的独立适配器和待冻结执行的离线测量流程。**本子任务未运行任何真实新增表面评价**，没有重跑World、策略或TSDF；未修改旧58项执行源码、原episode或原ledger。

## 故障诊断

`dev_CELL_G_b160_n92801`的原ledger保持`attempt_failed`。旧评价器`nso/surface_evaluation_v40.py::_mesh`拒绝`twice_area <= 1e-12`，等价于面积不大于`5e-13 m²`。

对原封存网格只读复核：96,699个顶点、183,055个三角面中，只有原索引63,061的三角面触发该门，面积为`4.235695216559833e-13 m²`，最长边为`2.0228585607699087e-6 m`；全部顶点有限。原预测总面积约`127.06654743413263 m²`。`prediction_seal.json`在评价失败前已经钉住原mesh、occupancy和mapper三个文件；它仍保留原始SHA。

独立保存包运动检查1457项通过：160个付费动作、161帧RGB-D和扫描、22米平移、88次前进、58次转向、14次额外观察、零碰撞且完整姿态返航。该检查只读取原传感包、动作、返航成本和融合receipt，不属于新策略试验，也不把原端到端失败状态改为成功。

## 单一共同适配器

新增[nso/article_prediction_mesh_adapter_v1.py](../../nso/article_prediction_mesh_adapter_v1.py)，API：

```python
vertices, triangles, receipt = prepare_prediction_mesh_v1(raw_vertices, raw_triangles)
```

该适配器只移除面积不大于`5e-13 m²`的面，拒绝非有限值、越界索引、非整型索引和不合法数组。原顶点与其索引、剩余三角面的顺序和dtype保持不变。若没有移除任何面，直接返回原ndarray对象。没有GT、类别、ROI、误差分数或场景专用参数输入。

receipt schema为`article.prediction_numeric_face_adapter.v1`，记录原／剩余面数、原删除索引、删除总面积、最大面面积、最大边长，以及原／派生数组的逻辑SHA。原NPZ和原prediction seal均不得重写。新runner可在封存后的评价步骤中调用该函数，并另存receipt。

适配器源码SHA为：

`b1db10e1835912049c954dbc438749f47e5d40cbc978b750e7779efe25dd3892`

删除面积很小**不能直接保证分数变化很小**：旧评价器逐面消耗随机数；删除一面可能改变后续三角面内部的采样位置。只要存在删除，结果就必须标为新版本派生测量，不能冒称原分数完全等价。

## 所有旧12条任务的统一补充流程

新增[scripts/evaluate_article_common_numeric_20260928.py](../../scripts/evaluate_article_common_numeric_20260928.py)，测量版本固定为`article.common_numeric_face_evaluation.v1`。原参考、资产、传感和评价设置不变：阈值0.05米、采样间距0.3米、种子4002、样本上限1,000,000。

1. `draft`仅生成共同计划：复制旧protocol的全部12槽位，固定原始源码依赖、适配器、审查器及本脚本的SHA、参考pin、统一门限和输出根。按结果选择个别任务不在该计划中。
2. root审阅后，以精确计划SHA执行`prepare`：尚未终态的任务返回`pending_terminal_input`，不评价；终态任务在episode外保存原ledger快照、全部现存文件库存及SHA、原始metadata副本和运动检查。失败任务的`controller_final.json`也原字节复制，明确标记这是失败终态之后的保全，不伪装成在线原artifact manifest。
3. root使用`.venv-3d`及单线程执行`measure`：每槽位只允许一个不可覆盖、不可自动重试的测量尝试。全部输入再次核验后，零删除且旧独立审查、预测、参数、参考、覆盖及资格元数据完全一致的任务复用原分数；否则仅对内存中统一预处理后的完整预测调用旧评价器一次。

输出建议根为`audit_results/article_stage_20260928/common_evaluation_v1`。每条任务分别有`prepared/`和`measurement/`两个外部目录。

结果始终分列：

- `original_end_to_end_status`：原ledger终态，例如`attempt_failed`，不修改。
- `original_end_to_end_qualified`：原流程资格，不由新评价覆盖。
- `motion_completion_verified`：基于保存传感包的独立运动完成检查。
- `quality_measurement_available`：是否取得本版本的派生质量结果。
- `mode`：严格输入等价复用，或新增派生测量。
- `new_surface_evaluations`：实际新增调用次数，复用为0。

`prepared/input_snapshot.json`对失败episode的全部原文件进行终态后的外部SHA保全；其`original_artifact_manifest_available=false`与`forensic_capture_after_terminal_failure=true`会保留。原缺少的在线manifest不补写，原失败不擦除。

## 使用与验证

先生成计划，不进行评价：

```bash
.venv/bin/python -B scripts/evaluate_article_common_numeric_20260928.py draft \
  --protocol configs/virtual3d/article_development_v1_20260928.json \
  --reviews-root audit_results/article_stage_20260928/episode_reviews_v1 \
  --output-root audit_results/article_stage_20260928/common_evaluation_v1 \
  --plan <全新的计划文件.json>
```

之后`prepare`和`measure`均要求`--plan`、`--plan-sha256`和`--run-id`；没有默认自动测量入口。本子任务仅运行帮助入口、合成回归及CELL保存包运动检查，未实例化或执行真实测量计划。

11项合成测试通过，覆盖严格数值门、输入不修改、零删除数组等价、非有限/越界拒绝、CELL实际极小面fixture、原失败状态保留、测量失败不可重试、forensic输入变动拒绝，以及参考／预测／覆盖不一致时禁止复用旧指标。测试中的评价器是明确的合成stub，未调用真实表面评价。

这是一项所有方法共用的评价输入兼容修复，不能单独解释为语义规划改进。后续新旧前端对照应统一使用该数值规则，并保留原端到端失败率和派生质量的区别。
