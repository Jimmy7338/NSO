# GroundV2 与 ExposureV3 配对分析接口

日期：2026-09-28。此文交付只读分析器；编写和测试未运行新的 World、策略、TSDF 或质量评价，也未生成真实实验快照。

`scripts/analyze_article_exposure_comparisons_20260928.py` 只接受冻结的开发布局消融协议：3 个布局 × G/B/S/NBV × GroundV2/ExposureV3，共 24 条记录。新增去重是一次诊断后的共同规划修正，结果只能作为开发阶段配对证据；本接口不处理正式测试集或 main 阶段。

协议冻结、实际运行终结且独立复核完成后，由 root 调用：

```bash
python3 scripts/analyze_article_exposure_comparisons_20260928.py \
  --protocol configs/virtual3d/article_exposure_ablation_v3_20260928.json \
  --snapshot \
  --output audit_results/article_stage_20260928/analysis_v1/exposure_comparison_NEW
```

输出目录必须不存在。默认复核目录是 `episode_reviews_exposure_v3` 和 `episode_reviews_ground_v2`，均位于 `audit_results/article_stage_20260928/`；可通过 `--exposure-reviews` / `--ground-reviews` 指定。未开始或 reserved 的记录不会打开逐帧文件，也不会标成失败。每份快照始终保留全部 24 个协议位置，后续进度需另建输出目录。

| 输出 | 内容 |
|---|---|
| `slots.csv` | 24 条状态记录；原流程资格、独立复核、运动完成、质量可用分别列出；质量、真实路程、转向、观测、前端实例数及首次 peer 等 |
| `paired_effects.csv` | 12 个 ExposureV3−GroundV2 配对位置，加两个版本各 3 布局 × S−B/B−G 的 12 个内部比较；质量和代价可用性分列 |
| `trajectories.csv` | 仅保存包中的实际位姿与实际传感动作，附累计 XY 路程；与独立复核的平移距离核对 |
| `selected_exposure.csv` | 每次重规划最终所选视点、每个预测实例的曝光前、抵扣、剩余和 exact-camera 门之后的有效名义面积 |
| `replans.csv` | 每次重规划选中项的分数、收益、成本、剩余预算，以及所选视点面积汇总与全部候选数量 |
| `macro_transitions.csv` | 宏动作编号、目标、初始化取消和路由原因的变化；编号 0 不算已提交宏动作 |
| `first_mechanism_witnesses.json` | 相同实际输入前缀内首个 peer 集变化、后验/分数/目标变化，以及实际执行分歧的紧凑证据 |
| `main_gate_witnesses.json` | 两版本、三个布局的全部 6 个 S/B 证据位置；首次实际动作分歧、共同前缀、当前决策及所选宏动作提交时的后验/peer、预算和守卫、支持文件 SHA。仅导出证据，不判门或启动实验 |
| `summary.json` / `findings.json` | 完整状态计数、各布局有符号差值及缺失/失败说明；三个布局未齐时不计算跨布局均值 |
| `captured_ledger_*.json` / `manifest.json` | 读取时账本快照，所有已读协议、源码、源归档、资产/参考清单、复核和原始文件的字节数与 SHA；所有输出 SHA |

分析器复用旧只读工具对 RGB-D/扫描的未压缩 NPY 数组作完整比较，仅忽略运行命名空间的 `frame_id`，保留时间、标定、实际位姿与付费步数。对不同运行中的实例，按当时实测标记锚点双向唯一匹配，禁止按类别或真值强制匹配。后验来自 `structure_belief`，即当帧反馈更新后的规划证据；不使用反馈前 association 中的旧后验。

“observed_instance_count” 指前端建立的实例身份数，不等同于真值设备发现率；“qualified_category_instance_count” 进一步要求类别支持数和无关联歧义/类别冲突。首次 informative peer 指已记录共享关系中的 peer 几何对数证据有非零结构对比，不表示所有物理上可能共享的机会。

曝光面积是名义模板机会面积，并非实际新增 TSDF 面积。`expected_before_exposure_m2 = expected_excluded_exposure_m2 + expected_after_exposure_m2`；最后再执行旧的完全相同相机姿态门。旧字段 `unexcluded_expected_new_surface_area_m2` 已经过曝光抵扣，只是尚未经过完全相同相机姿态门，不能将它标成曝光前面积。所有候选面积均检查，但只导出所选视点的明细以控制体积；这些面积不会冒充诊断动作的多步期望效用，也不跨重规划累加成重建收益。

共同质量版本固定为 `article.common_numeric_face_evaluation.v1`。原始失败不通过后处理改写为流程合格。质量差值要求相同评价版本、参考指纹与已核验运动完成；已有可信实际代价可独立报告。墙钟耗时来自原运行并发环境，不作为隔离性能基准。

主实验机制门的声明文件也绑定 SHA。门证据仅在双方都有保存的实际动作、动作分歧之前传感输入完全相同的情况下展开；没有分歧、某臂提前结束、传感先分歧、待运行或失败分别保留。诊断宏动作同样保留，不要求端点质量为正。逐帧原始日志没有 `return_latched` 字段，所以该值保持 null；不能将最终控制器快照中的 latch 倒填到过去。导出实际路由返航原因、首次记录返航步、宏动作提交时的重规划标志、预算守卫和返航余量，供结合冻结源码独立判断。geometry feedback 中已更新的实例用于反馈次数等字段，后验和共享权重仍取更新后的 structure belief。

程序不复制大型逐帧 JSON，输出总量设为 10 MiB 上限。只读数据绑定发生冲突时保留该行的错误状态，不给它分数或轨迹。历史 Ground CLI 可退出新运行器的依赖闭包，但其原始字节仍通过 Ground 源归档核验，其他共享执行源码必须一致。

纯夹具验证：

```bash
python3 -m unittest tests.test_analyze_article_exposure_comparisons_20260928 \
  tests.test_analyze_article_ground_comparisons_20260928 -v
```

本次 18 项新增测试与 10 项既有只读辅助测试全部通过；包括合成的完整 24 行快照、原始文件篡改拒绝、失败保留、负差保留、缺失/运行状态、版本/参考不一致、曝光面积守恒与两层门、post-feedback peer 证据、真实路程、相同前缀限制及不自动推断主实验许可的完整门证据位置。测试不读取真实实验作为结果样本。

## 新绘图适配器（等待实际快照发布）

`scripts/plot_article_exposure_comparisons_20260928.py` 接受上述已封存快照，保留完整 24 个条件位置。图中 Exposure Off 指 GroundV2，Exposure On 指 ExposureV3；两者都已使用共同地面关联修正，避免误写为有无地面模块。预定输出 3×4 配对路径图和 3×3 覆盖/F1/J 图，每图 PDF/SVG/PNG。所有路径都来自实际保存的位姿，标出起点、记录终点、方向、原地转向和额外付费观察；终点不自动等同成功返航。失败、运行中、未开始和待复核分别标明。

实际快照经 root 发布后才调用：

```bash
python3 scripts/plot_article_exposure_comparisons_20260928.py \
  --analysis audit_results/article_stage_20260928/analysis_v1/exposure_comparison_RELEASED \
  --output docs/thesis/figures/article_exposure_20260928/RELEASED \
  --render
```

此交付尚未渲染真实图，待首次实际渲染后再目视检查排版。5 项纯表格/选择器测试已通过：完整条件、可信但未完成返航的实际路径保留、输入篡改拒绝、累计路程核对及明确状态标签。转向标记使用保存传感动作名称 `turn_left` / `turn_right`；初始第 0 帧不计为额外付费观察。原绘图脚本和旧图均未修改。
