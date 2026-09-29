# 小论文与毕业论文交付入口

> 2026-09-29：已按固定最终目标进入收尾实施。用户要求学校、学位和专业资料留空，当前交付使用通用学位论文格式；这不等于已核验某一学校模板。最新状态见 [最终验收进度](../research/FINAL_DELIVERY_PROGRESS_20260929.md)。

2026-09-29 最终内容收敛更新。学校没有硬性发表要求，但发表文章是明确个人目标：先推进小论文投稿，毕业论文同步完善；真实小车验证作为后续补充。此处交付的是可审阅讨论稿，尚未向期刊或其他人员发送。

## 优先阅读

|用途|PDF|可编辑源|当前规模|
|---|---|---|---|
|目标期刊格式材料|[JIRS主文](submission_20260929/main.pdf)|[源码及补充ZIP](submission_20260929.zip)|20页，8图6表，待作者信息及声明签署|
|答辩材料|[19页PDF](defense_20260929/semantic_geometric_mapping_defense.pdf)|[PPT](defense_20260929/semantic_geometric_mapping_defense.pptx)|讲稿与18项问答同目录|
|聚焦的小论文投稿讨论稿|[Article](ARTICLE_SUBMISSION_EN_20260928.pdf)|[Markdown](ARTICLE_SUBMISSION_EN_20260928.md)|19页，8图，6表，14项文献|
|保留详细方法和全部版本结果的英文证据稿|[Evidence manuscript](VIRTUAL_PAPER_EN_20260928.pdf)|[Markdown](VIRTUAL_PAPER_EN_20260928.md)|36页，22图，9表|
|毕业论文通用全文审阅稿|[毕业论文](GRADUATION_THESIS_20260928.pdf)|分章见下表|57页，25图，18表，五章正文、中英文摘要及复现附录|
|多实例方法补充|[Supplementary Methods](ARTICLE_SUPPLEMENTARY_METHODS_20260928.pdf)|[源稿](ARTICLE_MULTI_INSTANCE_METHOD_APPENDIX_20260928.md)|5页，原场景级模型的明确公式与限制|

聚焦稿是当前小论文主要编辑入口，完整证据稿用于查阅详细方法和开发边界。不同稿件使用相同数据，不增加实验样本数。方法与实验核心PDF和中文文章PDF仍是较早检查点，不是本次36条场景结果的最新全文。

## 已完成的实验及结论

局部确认保留两布局、双构型的四组配对：两胜两平，平均联合指标相对提高6.1638%，平均覆盖相同。错误类别下完整纠错政策提高2.6241%，不能拆称某一个更新项的独立收益。固定128项扩展保留96项合格和32项低预算规划失败，42步类别收益为原布局6.2505%、同族变体4.2289%，54步为−0.6995%。另外完成40个保存轨迹检查点的真实重融合和评价；它们不是40条新路线。

本轮新增三布局、四方法、三个版本共36次场景级尝试：原版11项合格及1项原评价失败；共同地面版和曝光去重版各12项合格。36条均完成实际运动和返航，原失败的共同数值补测另列。曝光去重相对地面版有10/12个J正差，LOOP的B/S下降；三个版本九组S/B配对全部无实际动作差异，未建立额外共享贡献。类别改变路线但质量下降的AISLE案例，以及诊断改变路线但当帧没有新反馈的CELL案例，均已写入正文与真实路径图。

预声明要求出现至少一个S/B共享信息到实际动作的联系，才启动保留布局的48条主矩阵。该前提未满足，故主矩阵未执行，也没有追加第四轮开发。未执行的场景不能写为失败或零质量。全部结果、代价、失败与版本差分见[36条结果报告](../research/ARTICLE_EXPOSURE_ABLATION_RESULT_20260928.md)；停止依据见[最终机制门审查](../research/ARTICLE_FINAL_MAIN_GATE_VERDICT_20260928.md)。

## 毕业论文分章编辑

|位置|源文件|
|---|---|
|中英文摘要|[摘要](GRADUATION_ABSTRACTS_20260928.md)|
|第1章 绪论|[工业背景与研究问题](GRADUATION_INTRODUCTION_20260928.md)|
|第2章 相关技术与问题定义|[基础](GRADUATION_FOUNDATIONS_20260928.md)|
|第3章 系统方法|[四接口、信念、预算与理论](GRADUATION_METHOD_CHAPTER_20260928.md)|
|第4章 仿真实验|[局部确认、扩展、完整36条与机制](GRADUATION_SIMULATION_CHAPTER_20260928.md)|
|第5章 总结与展望|[已证实结论及边界](GRADUATION_CONCLUSION_20260928.md)|
|参考文献|[统一文献](GRADUATION_REFERENCES_20260928.md)|
|附录A|[环境、数据和恢复命令](GRADUATION_REPRODUCTION_APPENDIX_20260928.md)|

合并Markdown和TeX由分章生成，不直接作为修改入口。当前是通用审阅排版；学校模板、学位附页、导师及具体格式要求尚未提供，不能标为学校提交终稿。

## 图表与复现

科学图来自真实记录，包括全部配对路径、转向与观察成本、同视角实际TSDF、40个测量检查点和场景机制图。小车照片说明平台条件，架构使用原始小车图与程序矢量元素，不属于物理实验性能证据。ANS仅保留必要相关工作来源，四模块接口与分层结构保持。

三版科学源码分别固定58、62、65份文件，源码ZIP、原数据清单、审查、账本与无损压缩归档一并保留。每次pack已逐字节验证原文件恢复；原版失败使用单独取证归档，不混用合格episode恢复入口。不要改写冻结数据或自动重启旧队列。

[本轮最终内容审阅](../research/FINAL_CONTENT_REVIEW_20260929.md)及[最终交付](../research/FINAL_DELIVERY_20260929.md)记录当前PDF、来源与检查；旧9月28日审阅只对其当时版本有效。构建命令：

```bash
python3 -B scripts/build_writing_deliverables_20260928.py article --render --export
python3 -B scripts/build_writing_deliverables_20260928.py english --render --export
python3 -B scripts/build_writing_deliverables_20260928.py thesis --render --export
```

使用缓存的Pandoc/Tectonic、字体和文稿Python运行时；不会启动仿真。导出物的构建记录绑定源稿、图片和构建脚本。源稿变化后必须重新构建，不能沿用旧PDF校验状态。

## 最终布局补证与后续手续

最后预登记的两个背景布局、两构型及G/S共八次任务已经全部完成，8/8合格，四配对一胜三平，平均J5提高2.9818%。新增内容已进入本页三份当前全文和JIRS材料包。其设备族、公开模板与原确认噪声种子不变，不宣称新类别泛化。额外八次预留未使用，停止科研扩增。

科研内容已按[固定验收目标](../research/FINAL_PAPER_THESIS_TARGETS_20260929.md)收敛；建议把当前两稿交导师审阅，并完成作者声明和学校格式等外部手续。期刊录用、学校评审和实车展示不是本轮内部工作包的完成条件。完整交付与复现入口见[最终交付](../research/FINAL_DELIVERY_20260929.md)。

本轮只向原Jimmy7338/NSO提交推送，SGAM快照不改动，无PR/MR及外部投稿。
