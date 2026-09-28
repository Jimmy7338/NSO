# 附录A 复现与材料索引

## A.1 固定实现与观测数据

项目将实验运行与文稿构建分开保存。主确认、错误类别和外部机制批次分别位于`audit_results/v36_online_confirmation_20260918/`、`v35_online_development_20260918/`和`v39_external_cpu_20260920/`。每个批次保存配置、执行源码摘要、实际传感包、动作与信念记录、重建网格及评价结果。原开发与确认种子分别为1901和350918。完整科学依赖见仓库根目录`requirements-3d.lock.txt`；Python及关键科学库版本见第4章平台说明。

新增预算与几何扰动实验保存在`audit_results/thesis_expansion_20260928/`。其配置`configs/virtual3d/thesis_expansion_20260928.json`预先列出128项任务、两种子2026092801和2026092802、四策略及几何变体；`source_freeze.json`绑定实际执行源码和输入。原始任务输出与批后审阅汇总分别保存，失败记录不会由补充分析改写为成功。所有场景内的四策略共享几何与噪声规则，分析按场景、构型、预算和种子配对。

## A.2 保存结果的复核

发布清理移除了退役训练依赖及部分源码归档，科学结果和原摘要保持不变。当前检出不含全部旧源码闭包，严格历史复现须恢复清理前版本；新采集应重新封存，详见仓库发布迁移说明。

复核可以先从原始动作、评分和配对表开始，无需重新采集。确认批的只读审计入口为`scripts/verify_online_confirmation_v36.py`，指定原批次作为`--batch`，并用一个尚不存在的目录作为`--output`。启动科学Python前将`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`均设为1。审计检查已有文件和指标算术，不增加独立实验样本。

新增批次以`analysis_review/complete_endpoint_metrics.csv`提供完整任务长表，以同目录的`complete_paired_metrics.csv`提供条件配对。先检查任务状态、覆盖、碰撞和精确返航，再解释质量值。预算不可行任务的失败时刻部分重建另列为`partial_failure_endpoint`，不得与合格任务终点混合求均值。噪声重复用于展示同一几何的数值变化，不当作新增独立场景。

原始批次已封存，重采集时应使用独立目录并重新绑定源码、配置和种子，不能覆盖旧数据。两类语义对照的解释也应保持一致：S/G改变类别信息；X/Xnf同时改变实测纠错和未来纠错预期，代表整体政策比较。

## A.3 图件与文稿生成

场景与三维细节图由`scripts/plot_thesis_scene_details_20260928.py`读取已有路径与网格生成，来源摘要写入图件目录的`provenance.json`。实物平台图保留原始照片，仅增加矢量标注与接口说明，其文件来源保存在`figures/robot_platform_20260928/manifest.json`。照片不参与定量评分。

全文的可编辑源是分章Markdown，合并Markdown、LaTeX和PDF由构建脚本生成。在仓库根目录执行以下命令，可重建通用审阅稿及预览：

```bash
python3 -B scripts/build_writing_deliverables_20260928.py \
  thesis --render --export
```

构建记录绑定源文件与PDF/TeX的SHA256；排版不运行科学实验。当前为通用审阅稿，提交前须适配学校的封面、声明和引用格式。
