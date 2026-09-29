# 最终局部布局验证：实现审查与执行入口

日期：2026-09-29。对应最终成果 P3。本文及本次静态工作不产生在线实验成绩。

**可以用现有科学实现完成 8 次补证，无需再改控制器或评价。** 新工作只增加两个背景布局、同步初始化世界与公共导航图的构造适配器，以及独立数据目录。设备的两个公开候选构型、V35 控制器、类别映射、B42、共同 18 步前缀、噪声、TSDF 和 V34 表面指标保持原样。这是已知设备族在两种新空间布置下的迁移验证，不是新设备类别或未知地图泛化。

## 一次性固定布局

坐标使用 P00 原局部米制坐标，box 顺序为 `[xmin,xmax,ymin,ymax,zmin,zmax]`。两个布局均保持设备双构型的全部原始 box、朝向、原点和类别关系，不旋转或改名已有路线；背景几何首次加入本轮协议。坐标按通道、物理间隙和原评价边界确定，未查看任何新策略成绩。

|布局|固定背景 box|物理任务含义|相对原安全整数中心的变化|
|---|---|---|---|
|L02_side_column|`[3.10,3.40,2.45,3.55,0.0,1.20]`|侧通道立柱使机器人需要从内侧连接绕过右侧局部阻挡|删除 `(3,3)`|
|L03_rear_partition|`[-0.70,0.70,3.80,4.20,0.0,1.10]`|设备后方隔离台切断后侧直通通道，两侧仍可从前方到达|删除 `(0,4)`|

两个背景不是同一场景的旋转或镜像：被删节点及其局部连接结构不同。两者仍来自同一设备族、共用公开设备模板，不能当作两个全新工业类别。它们是两个单独登记的布局单位；两个构型、G/S 配对和检查点不增加布局数。

原设施联合 ROI 的局部边界为 `[-2.75,2.75] × [1.1,3.4] × [0,1.8]`；冻结评价再向各方向扩展 0.20m。新立柱与 padded ROI 在 x 方向间隔 0.15m，新后侧隔离台在 y 方向间隔 0.20m。因此原指标可以原样使用，不另行删除背景三角形或更换分母。带噪融合仍可能产生几何误差，按原裁剪和精度计算保留，不承诺测量误差为零。

本次只做矩形距离与 BFS 的静态核查，每个布局得到 23 个完整安全整数中心、92 个朝向状态。两构型的节点和边完全相同；全部姿态可达且可返回；原 18 步前缀合法并恢复 `(0,0,0)`。最大最短返航成本分别为 16、14 个原语。两个布局中，从起姿到左右前侧示例观察姿态再回起姿均可用 16 个原语完成。此处是静态路径存在性，不证明 B42 下控制器一定满足覆盖门，也不预测收益。

## 已实现的独立文件

- 配置：`configs/virtual3d/final_local_layout_validation_20260929.json`。
- 运行入口：`scripts/run_final_local_layout_validation_20260929.py`。
- 静态夹具：`tests/virtual3d/test_final_local_layout_validation_20260929.py`。
- 新运行目录：`audit_results/final_local_layout_validation_20260929/`；本报告形成时尚未创建。

矩阵固定为两布局 × `h0/h1` × `G/S`，8 次任务。沿用 V36 的 `iid_025px` 和 `noise_seed=350918`，不是新增噪声水平。P00 继续作为冻结传感器的噪声父索引，两布局的真实身份由独立 case、完整场景配置和 preparation seal 记录；不能只凭 packet 的历史 `v34-P00` 命名推断布局。控制器仍只收到显式构造的 `ObservationV35`，不接收场景 ID、真实构型索引或评价对象。

G 保留相同公共导航图、两个完整公开候选模板、诊断性观察预测、几何反馈、补看和返航保护。G/S 的干预仍是 V35 原有类别先验开关。把无语义组改成不能补看会改变比较问题，本轮没有这样修改。

### 复用边界和必要适配

|实现|本轮用途|
|---|---|
|`nso/cpu_four_modules_v35.py` / `online_planner_v35.py` / `observation_belief_v35.py`|原样执行四接口、在线选择和观测信念更新|
|`nso/direction_information_v33.py` / `pixel_information_v34.py`|为两个构型生成共同完整公共图及可见表面预测；预测量不作为实测质量|
|`env/information_pixel_v34.py`|原样的像素渲染、噪声、单线扫描、运动和评价参照接口|
|`nso/observed_runtime_mapper_v10.py` / `surface_measurement_v34.py`|原样融合和测量；0.04m voxel、0.12m truncation，2/5/10cm 阈值|
|`scripts/run_thesis_expansion_20260928.py`|新收集循环的实现来源；原文件不变|
|`scripts/run_online_routes_v36.py::public_graph_audit/nonsemantic_fields`|复用纯公共图核查及原始观测字段哈希，不调用旧队列|

旧 expansion 的 `world_for()` 只更换 box；它没有同步刷新 `_nav_cells`、栅格、坐标变换及起点，不宜直接用于改变通道的布局。新 `configure_world()` 同步重置完整 parent、公共边界、shift、config 尺寸/B42、shape、GridTransform、合法位置集合、pose/start/heading、几何缓存、reference/background/ground boxes 和 primitives。渲染、运动、噪声及评价的方法本体继续继承冻结代码，没有覆盖 `step()`、`_packet()` 或指标。

公共模板准备为两个布局的两种公开构型各遍历全部 92 姿态，共 368 个 clean 查询、4 个模板 World。这是固定公开候选的准备过程，不执行策略、不选择保留哪种结果，也不计作新的闭环任务；查询次数单独记录。实际在线任务仍为 8 次，实际采集最多每次 42 付费观测加初始 1 帧。准备后会核对两构型的前缀深度/扫描完全一致；不得据策略成绩调整模板或场景。

## 数据保全和复核

`freeze` 在准备模板前保存 protocol、静态核查、45 个执行依赖的哈希及源码 ZIP，并验证历史 59 个保护文件仍未修改。新配置另列核心 9 个科学文件/场景的 SHA。`prepare` 保存每布局的 geometry、sensor_templates 和 metadata，再生成 preparation seal。

每条任务保留 `started.json`、所有原始 packet、实际 action/pose/nonsemantic 字段、全部后验和候选选择记录、四接口调用、prefix/final 的完整 raw mesh、按旧公共 ROI 裁剪的 mesh、地图、裁剪记录、预测先封存声明、评价分母、测量结果与任务 seal。完整 raw mesh 作为展示与保全资产，主指标仍按冻结公共 ROI 计算。成功、失败和未开始状态不能互换；已有目录拒绝重试，失败不会被同名成功覆盖。

`analyze` 只读取封存结果，核查文件 SHA、保存栅格的 C_map、P/R/F1/J 算术和资格条件；报告四个 S/G 配对、最早决策/动作差异及同前缀后验。它不重新融合或评价表面，也不冒充独立的传感物理复跑。结果中保留运行失败、不合格和未开始槽位；不以正收益作为完成条件。

旧 `verify_online_confirmation_v36.py` 固定了 P00/P01、旧目录及 V36 输出 schema；旧 `analyze_semantic_chain_v35.py::inspect_case` 还要求历史 replay 目录。不能把这两个旧入口直接指向新目录后声称已通过。独立复核应读取本次 packet、controller 和 preparation，复用其纯算术/哈希思路，保持“实际传感再执行”与“保存数据检查”的范围区别；这不需要另开策略试跑。

## CPU、磁盘与任务额度

读取此前 64 个已完成 B42 扩展记录：每次总墙钟 7.71–10.50s，中位 9.40s；原保存量每次 1,580,038–2,068,455 字节，中位 1,822,845 字节。其环境记录为 CPU 单线程、Open3D 0.19.0、NumPy 1.26.4、SciPy 1.11.4。此前两个新同族模板准备耗时 1.43/1.75s，模板资产约 0.33/0.37MB。上述均为历史实测，不能写成新任务实测耗时。

本轮额外保留 raw meshes，建议预留每条 3–6MiB、整批 64MiB；预计在线墙钟约 2–5 分钟，仍以实际记录为准。每条沿用 120s 完整任务上限；内部 V35 的每次规划 20s/150万状态上限不变。独立总输出硬限 256MiB、磁盘余量下限 1GiB，当前约 1.42GiB 空闲足以执行预计小批次，无需先删除旧成果。运行 `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`，一次一个任务。

本阶段总额度仍是 16 次新闭环；本脚本只开放固定的 8 个主槽，另 8 次预留没有可执行 case index。工程失败也占已启动任务名额，自动重试关闭；后续若需使用预留，另做明确 case 登记，不能清空本批计数。静态测试、公共模板准备、保存数据审查和重绘不作为新增独立闭环任务。

## 根线程执行顺序

以下命令是可执行入口，不表示本报告已运行这些阶段。使用与既有实验一致的 `.venv-3d` 科学环境；先由根线程复核、冻结，再执行一次全矩阵。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-3d/bin/python -B -m unittest tests.virtual3d.test_final_local_layout_validation_20260929 -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-3d/bin/python -B scripts/run_final_local_layout_validation_20260929.py freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-3d/bin/python -B scripts/run_final_local_layout_validation_20260929.py prepare
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-3d/bin/python -B scripts/run_final_local_layout_validation_20260929.py run
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-3d/bin/python -B scripts/run_final_local_layout_validation_20260929.py analyze
```

本代理实际仅执行了 `static`、13 项静态/行政/FakeWorld 夹具及源码 closure/旧哈希核查。均通过；未构造实际 World，未运行模板渲染、控制器选择、TSDF 或表面评价，未占在线名额。背景坐标和方法配置在这些静态检查中未调整。

根线程在冻结前发现 bundled Open3D 无 dist-info，运行元数据读取现已回退模块 `__version__`，并记录 CPU 型号、逻辑核数和可用 affinity 核数。元数据和源/准备 SHA 在占槽前获取；占槽后的 started 写入与操作异常进入失败保全路径。该修改没有执行任何在线任务。
