# 完整场景TSDF与设备局部图（2026-09-28）

图包固定读取`analysis_v1/aisle_gbs_association_audit_v4`比较快照：12个预声明开发槽位中，AISLE的G、B、S已完成并通过独立审查，其余9个在该快照中为`unstarted`。后来完成的任务不会自动进入这份图包；扩展图必须使用明确的新比较快照和新输出目录。

正式图包为[article_scene_meshes_20260928/aisle_gbs_v2](../thesis/figures/article_scene_meshes_20260928/aisle_gbs_v2)。初版`aisle_gbs_v1`保留为排版草稿，正式版本调整了色条与脚注间距，并补充可观测参考的说明；两版使用相同科学输入。

## 三类图

1. [AISLE GT／G／S预览](../thesis/figures/article_scene_meshes_20260928/aisle_gbs_v2/aisle_gt_g_s_preview.png)：共同正交视角、世界坐标范围及尺度，叠加实际执行路线。已严格核验G与S的顶点、三角面索引及XY路线完全相同，图中明确注明，不能据此声称语义改进。
2. [全部预声明场景与方法矩阵](../thesis/figures/article_scene_meshes_20260928/aisle_gbs_v2/scene_meshes_all_declared.png)：三行AISLE／CELL／LOOP，五列GT／NBV／G／B／S。保留全部未完成状态，不填零、不按优胜结果挑选场景或方法。
3. [AISLE全部四实例局部对照](../thesis/figures/article_scene_meshes_20260928/aisle_gbs_v2/art1_aisle_dev_b160_n92801_all_instance_closeups.png)：逐一展示全部设备，统一视角与物理尺度，使用离线GT包围盒外扩0.18米形成固定展示窗口。窗口内仅保留完整位于其中的原三角面，没有补洞、插值、表面修复或按预测语义标签挑选三角面。

以上均有PDF、SVG、PNG版本。网格作为高分辨率栅格层嵌入矢量PDF／SVG，文字仍保留字体。正式全场景图使用每份预测的全部205,566个三角面；没有抽样或抽稀。局部裁剪仅用于图示，不替换进入全场景指标计算的完整网格。

正式图包总量9,886,601字节（约9.43 MiB）；69条输入／输出SHA记录复核全部通过。三张正式PNG均已实际打开审图，色条、指标文字和图注没有互相遮挡。9个全网格面板（含预览中的重复展示）显示三角面数均等于原网格三角面数。

## 图中数字和颜色如何理解

颜色表示原三角面中心的世界高度，并由该三角面的真实几何法向计算明暗；它不是误差热图。路线是保存的XY轨迹投影到地面，为阅读清晰叠加绘制于遮挡之上，白色菱形表示共同起点与完整姿态返航位置。路线没有平滑或重规划。

GT展示完整几何，包括传感模型下不可观测的表面；原指标针对冻结的可观测参考。因而“GT有屋顶、预测无屋顶”与某设备取得很高可观测召回并不矛盾，不能把该图理解为整个GT表面的逐面误差比较。

局部图中的P／R／F1直接复制原独立review，不进行新评价。当前G、B、S三者均相同：

| 实例 | 类别 | 精度P | 召回R | F1 |
|---|---|---:|---:|---:|
| I1 | Cabinet | 0.997102 | 1.000000 | 0.998549 |
| I2 | Rack | 0.947067 | 0.557492 | 0.701843 |
| I3 | Cabinet | 0.965366 | 0.887814 | 0.924967 |
| I4 | Rack | 0.778184 | 0.831784 | 0.804092 |

这组图支持“完整场景建图流程已跑通，并且设备表面存在不同的重建难度”，不支持“新开发场景中语义策略优于对照”。当前三个方法的覆盖、F1、联合指标与路径一致，必须与后续独立场景结果分开叙述。

## 复用方式和溯源

新增生成器为[scripts/plot_article_scene_meshes_20260928.py](../../scripts/plot_article_scene_meshes_20260928.py)。它不导入World、规划器、TSDF融合器或表面评价器。输入必须明确声明`--analysis`、`--phase`、`--episodes`、`--reviews`和`--assets`；输出必须是全新目录。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  scripts/plot_article_scene_meshes_20260928.py \
  --analysis audit_results/article_stage_20260928/analysis_v1/aisle_gbs_association_audit_v4 \
  --episodes audit_results/article_stage_20260928/development_v1/episodes \
  --reviews audit_results/article_stage_20260928/episode_reviews_v1 \
  --assets audit_results/article_stage_20260928/scene_assets_v1 \
  --phase development \
  --output docs/thesis/figures/article_scene_meshes_20260928/NEW_SNAPSHOT
```

输入比较快照与review的manifest均重新校验；原artifact manifest绑定到比较快照和独立review，使用的预测网格、封存、结果与评价文件也核验原字节SHA。静态资产必须匹配冻结资产manifest。输出`manifest.json`记录实际用到的每个输入及所有输出的SHA，`generator_source.py`保存当次绘图代码，`display_geometry.json`逐图记录输入／显示三角面数、共同世界范围、相机参数、显示裁剪以及全部槽位状态。

后续任务完成后，可以在新快照中继续生成相同格式的三个开发场景×四方法对照，而无需重新运行传感、策略、TSDF或表面评价。失败或缺失槽位仍显示其原状态；若出现封存后评价失败，只有独立的新补充测量记录获得相应审查后，才可另外绘制并明确标为派生测量。

## 可并入英文稿的文字

> Figure X visualizes the complete saved TSDF meshes and executed routes under a shared orthographic view and metric scale. The AISLE development episode yields exactly identical G and S vertex arrays, triangle indices, and XY routes; this endpoint therefore demonstrates the reconstruction workflow without providing evidence of a semantic advantage. The instance panels include all four declared devices and report precision, recall, and F1 copied from the independent full-scene evaluation. Ground truth includes unobservable surfaces, whereas the reported scores use the frozen observable reference. Local bounding-box crops and height-based shading serve visualization only and do not enter planning or evaluation.
