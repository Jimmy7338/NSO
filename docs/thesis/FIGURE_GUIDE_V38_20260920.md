# V38 科研图表使用说明

所有定量图从既有封存结果读取，没有重新生成测试结果或重新计算表面质量。主稿使用 PDF；SVG 可用于后续编辑，PNG 为 300 dpi 预览。原生宽度 7.05 in（179.07 mm），适合常见双栏通栏排版；若改变期刊栏宽，应按最终印刷尺寸调整字体，而不是直接缩小到不可读。

|内容|文件|数据与含义|
|---|---|---|
|四模块实际信息流|[架构](/root/NSO/docs/thesis/figures/v38/fig01_architecture.pdf)|CPU 实现关系；规划输入与事后评价分开|
|全部配对与阈值|[确认结果](/root/NSO/docs/thesis/figures/v38/fig02_confirmation.pdf)|V36 全部四对，保留两个零差；无伪造总体误差棒|
|语义干预与纠错|[消融](/root/NSO/docs/thesis/figures/v38/fig03_ablation.pdf)|V35 G/S/X/Xnf；2 cm 的 h1 负反馈收益保留|
|学习型补充与边界|[学习型证据](/root/NSO/docs/thesis/figures/v38/fig04_learned_evidence.pdf)|8 个上下文真实点及既有组级 bootstrap 区间；压力层单列|
|真实信念与动作时序|[后验时间线](/root/NSO/docs/thesis/figures/v38/fig05_causal_timeline.pdf)|V35 P00/h1，每点为观测 t 后、动作 t+1 前；横轴不是秒|
|场景、轨迹与实测网格|[定性对照](/root/NSO/docs/thesis/figures/v38_qualitative/v38_saved_paths_and_tsdf.pdf)|V36 P00/h1 正例；共享坐标范围及相机；完整保存 TSDF，无补洞／配准|

图面采用白底、统一字体、细轴线、有限网格、a/b 面板标记及三线表。灰色 G、蓝色 S、橙色 X 与不同点形／线型配合，避免仅依赖颜色识别；这一做法与 [IEEE 作者图形可读性及无障碍建议](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/)一致。PDF 嵌入绘图字体；SVG 保留文本，编辑机器须具备 Liberation Sans。定性网格面片栅格化嵌入矢量容器以控制体积，文字和坐标仍为矢量；不把它称为所有面片均可矢量编辑。

绝对分数轴从零开始，差值轴保留零和负值。单个案例的复制执行不产生置信区间，采样点不作为独立任务。图注明确独立单位、主阈值、误差线定义、场景选择与评价范围；没有显著性星号或只选赢家的布局。

## 再生成与来源

```bash
.venv-3d/bin/python scripts/build_thesis_figures_v38.py
.venv-3d/bin/python scripts/build_thesis_qualitative_v38.py
python3 scripts/build_manuscript_v38.py
```

在论文发布验收后如需改图，保留已发布版本并另开输出目录，随后更新主稿和发布清单。当前脚本是论文制作工具，默认目录允许编辑迭代；它不会自动保护已发布稿件。

- [定量数值及输入清单](/root/NSO/docs/thesis/figures/v38/manifest.json)
- [完整在线分项 CSV](/root/NSO/docs/thesis/figures/v38/online_measurements.csv)
- [定性图来源与显示合同](/root/NSO/docs/thesis/figures/v38_qualitative/provenance.json)
- [实际轨迹 CSV](/root/NSO/docs/thesis/figures/v38_qualitative/trajectories.csv)
- [编译环境及固定版本](/root/NSO/docs/thesis/V38_BUILD_ENVIRONMENT_20260920.md)

颜色只表示方法及网格光照，未生成逐点误差热图。源网格和已存分数并排展示不构成新的质量计算，也不能以视觉外观取代全部构型定量结果。
