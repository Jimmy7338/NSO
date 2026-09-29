# 附录A 复现与材料索引

## A.1 固定实现与观测数据

项目将实验运行与文稿构建分开保存。主确认、错误类别和外部机制批次分别位于`audit_results/v36_online_confirmation_20260918/`、`v35_online_development_20260918/`和`v39_external_cpu_20260920/`。每个批次保存配置、执行源码摘要、实际传感包、动作与信念记录、重建网格及评价结果。原开发与确认种子分别为1901和350918。完整科学依赖见仓库根目录`requirements-3d.lock.txt`；Python及关键科学库版本见第4章平台说明。

新增预算与几何扰动实验保存在`audit_results/thesis_expansion_20260928/`。其配置`configs/virtual3d/thesis_expansion_20260928.json`预先列出128项任务、两种子2026092801和2026092802、四策略及几何变体；`source_freeze.json`绑定实际执行源码和输入。原始任务输出与批后审阅汇总分别保存，失败记录不会由补充分析改写为成功。所有场景内的四策略共享几何与噪声规则，分析按场景、构型、预算和种子配对。

独立多实例研究保存在`audit_results/article_stage_20260928/`。原开发12条与共同地面关联消融12条已经完成，分别使用`configs/virtual3d/article_development_v1_20260928.json`和`configs/virtual3d/article_ground_ablation_v2_20260928.json`。其后同光心曝光去重版本使用`article_exposure_ablation_v3_20260928.json`，也已完成全部12条。三个版本共36条完整尝试，原流程失败和统一派生质量分别保存，保留布局主矩阵未执行。

| 实现 | 源码归档（相对于多实例研究目录） | 对应原始记录 | 比较含义 |
|---|---|---|---|
| ArticleV1 | `execution_sources_v1.zip` | `development_v1/` | G为几何；B为固定类别先验；S另加同类共享；NBV为无诊断前瞻的几何机制 |
| GroundV2 | `execution_sources_ground_v2.zip` | `ground_ablation_v2/` | 四方法共同改变对象关联中的地面处理，完整深度融合不变 |
| ExposureV3 | `execution_sources_exposure_v3.zip` | `exposure_ablation_v3/` | 在Ground基础上共同改变同光心名义曝光去重，完整12条已复核 |

各协议的`source_sha256`、`source_archive_sha256`和数值环境记录绑定对应实现；当前源码不能追溯替代旧实验的冻结版本。局部实验S/G检验类别可用性，而场景级S/B检验共享；局部ROI下的\(J_5\)与全网格、四实例宏平均的\(J_{\rm nav}\)分别报告。共同数值面片适配后的CELL/G质量是派生测量，其原评价失败状态不变。

## A.2 保存结果的复核

当前NSO仓库已经恢复历史源码、依赖资料及131份原字节源码ZIP；清理后的SGAM独立快照与NSO研究归档用途不同。恢复文件按原对象及SHA核对，但旧神经训练所需的部分LFS本体、子模块和外部场景仍需另行获取，不能据此宣称旧Habitat环境全部可运行。范围与缺件见[迁移与恢复说明](/root/NSO/docs/research/REPOSITORY_SPLIT_AND_RECOVERY_20260928.md)。复现某一批次时应使用其原协议、源码包与输入清单，不以当前默认依赖替代历史数值环境。

复核可以先从原始动作、评分和配对表开始，无需重新采集。确认批的只读审计入口为`scripts/verify_online_confirmation_v36.py`，指定原批次作为`--batch`，并用一个尚不存在的目录作为`--output`。启动科学Python前将`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`均设为1。审计检查已有文件和指标算术，不增加独立实验样本。

新增批次以`analysis_review/complete_endpoint_metrics.csv`提供完整任务长表，以同目录的`complete_paired_metrics.csv`提供条件配对。先检查任务状态、覆盖、碰撞和精确返航，再解释质量值。预算不可行任务的失败时刻部分重建另列为`partial_failure_endpoint`，不得与合格任务终点混合求均值。噪声重复用于展示同一几何的数值变化，不当作新增独立场景。

原始批次已封存，重采集时应使用独立目录并重新绑定源码、配置和种子，不能覆盖旧数据。局部实验的两类对照也应保持原有解释：S/G改变类别信息；X/Xnf同时改变实测纠错和未来纠错预期，代表整体政策比较。

多实例原开发与Ground完整对照分别见`analysis_v1/development12_terminal_20260928/`和`analysis_v1/ground12_complete/`。Exposure完整配对见`analysis_v1/exposure12_complete/`，三版本36条汇总见`analysis_v1/article36_final_summary_v1/`。原开发的失败与共同数值补测分开读取；全部四设备保持在宏平均分母。独立审查位于`episode_reviews_v1/`、`episode_reviews_ground_v2/`，Exposure使用独立的`episode_reviews_exposure_v3/`。review通过表示记录符合其审查合同，不把原失败任务改写成原流程成功。

## A.3 无损原始数据归档与恢复

部分已审查任务为节省工作盘空间，将传感数组、压缩step日志及预测数组转为可逐字节恢复的归档，原manifest、ledger、review和压缩操作收据分别保留。合格任务的标准归档位于`episode_archives_v1/`及`episode_archives_ground_v2/`，文件名为`<run_id>.tar.xz`；配套`.report.json`记录原始和归档SHA、文件数及回读验证。归档内部会将step的gzip内容转换为可共享压缩字典的形式，因此标准tar解包不能代替专用恢复工具。恢复工具重建原gzip字节，并核验每个原始文件及原artifact manifest。

以下命令接口已经通过实际`--help`和源码核对；从仓库根目录运行。`verify`只流式检查归档，不写出解压数据，不运行策略、融合或评价。以已保存的原开发AISLE/B为例：

```bash
article_stage=audit_results/article_stage_20260928
episode=dev_AISLE_B_b160_n92801
archive_dir="$article_stage/episode_archives_v1"
python3 -B scripts/archive_article_episode_20260928.py \
  verify --archive "$archive_dir/$episode.tar.xz"
```

需要读取完整原始文件时，恢复到尚不存在的独立目录；该操作只产生原字节副本，不生成新实验样本：

```bash
article_stage=audit_results/article_stage_20260928
episode=dev_AISLE_B_b160_n92801
archive_dir="$article_stage/episode_archives_v1"
python3 -B scripts/archive_article_episode_20260928.py \
  restore --archive "$archive_dir/$episode.tar.xz" \
  --output "tmp/article_restore_example/$episode"
```

`restore`仅恢复episode文件，不创建对应阶段ledger。若原始文件已经按已审查压缩计划移出工作目录，而后续只读review需要完整阶段结构，可使用现存计划恢复原episode与其ledger快照：

```bash
plan_root=audit_results/article_stage_20260928/compaction_plans_v1
episode=dev_AISLE_B_b160_n92801
python3 -B scripts/compact_article_episode_20260928.py \
  materialize \
  --plan "$plan_root/development_complete_cohort/$episode" \
  --workspace tmp/article_review_workspace_example
```

该命令输出`materialization.json`及原阶段下的episode，要求workspace全新；原目录和归档均不覆盖。它只用于已存在、哈希匹配的压缩计划。当前报告中的默认review命令针对ArticleV1，Ground/Exposure复核必须选择相应版本reviewer并满足其基线输入要求，不能因为已恢复字节就混用审查器。恢复需保留至少1 GiB自由空间及单集允许量，不必一次展开全部数据。

原失败CELL/G另存为`episode_archives_v1/dev_CELL_G_b160_n92801_original_failed.tar.xz`，其回执schema为`article.failed_episode_original_byte_archive.v1`，包含658个文件的字节数与SHA。它是终态失败的原字节取证包，**不是上述标准合格episode包**，不使用专用`archive ... restore`冒充完整原manifest恢复；读取副本应按配套回执逐文件核验，保留失败记录及后续数值补测的区别。已有原始文件也不会因补测而被改写。

## A.4 图件与文稿生成

场景与三维细节图由`scripts/plot_thesis_scene_details_20260928.py`读取已有路径与网格生成，来源摘要写入图件目录的`provenance.json`。实物平台图保留原始照片，仅增加矢量标注与接口说明，其文件来源保存在`figures/robot_platform_20260928/manifest.json`。照片不参与定量评分。

全文的可编辑源是分章Markdown，合并Markdown、LaTeX和PDF由构建脚本生成。在仓库根目录执行以下命令，可重建通用审阅稿及预览：

```bash
python3 -B scripts/build_writing_deliverables_20260928.py \
  thesis --render --export
```

构建记录绑定源文件与PDF/TeX的SHA256；排版不运行科学实验。当前为通用审阅稿，提交前须适配学校的封面、声明和引用格式。

## A.5 最终验证、离线演示与复现入口

最终八次任务位于`audit_results/final_local_layout_validation_20260929/`，包括预登记协议、45文件源码包、准备封存、全部传感包与网格、配对汇总。独立复核位于`audit_results/final_delivery_20260929/validation_review.json`；原8条任务和结果全部保留，不与既有128项扩展混算。59份旧保护源码的摘要未改变。

代表路线采用预登记列表中的case0，即侧柱h0/G。对43帧已保存数据进行控制器及TSDF重放，动作、状态地图、前缀与终点的原始及评价域网格均精确一致；该过程不创建新世界、不查询新传感，也不重新计算质量指标。核验记录见`saved_replay_case0/`。交互演示`docs/thesis/demos/v40_replay/index.html`可离线打开，展示真实保存路径、已观测帧、信念和重建端点；浏览器十项检查均通过。播放与中间视图不能当作新采集或插值得到的三维性能。

最终统一索引`docs/research/FINAL_METHOD_REPRODUCTION_INDEX_20260929.md`给出公式、代码、数据、主表图及命令对应，包含2—3分钟演示顺序。所有重绘和文稿构建均不新增科研样本。学校、学位与专业资料按用户要求留空，当前通用封面预留替换位置，学校专属格式仍由实际模板确定。
