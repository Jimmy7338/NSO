# 同目标直接巡检对照

该局部机制实验在两个合成布局、四处预定摆位中运行。每个布局有一种侧向观测无遮挡和一种增加遮挡板的摆位。独立几何预检确认指定侧向位姿确实可达。东向无遮挡和西向无遮挡场景分别看见 5,058 和 5,104 个货架表面样本，共抽样 10,000 点。增加遮挡板后同一侧向位姿均看不见货架采样点。该条件只由离线评价器验证，在线策略没有这个位姿或遮挡条件输入。

历史预研已执行库房与展厅的 16 条候选路线。库房的正确、中性和错置类别对象得分方向与同代价路线净增面积方向分别匹配 4/4、2/4 和 0/4，展厅均为 1/2。链式 17 补齐了真正的 G_perf 首选，环式 20 跟踪了早期独有的柜体表面到实际终点。这些既有场景在本次独立摆位之前已查看，分别用于背景和一次校准。 [早期三种类别证据](https://github.com/Jimmy7338/NSO/blob/7fda64686ecd9b2adbc5aa35bc3e277eb720d00f/experiments/offline_20261121_candidate_route_yield/RESULTS_ZH.md) · [链式 17 真正几何首选](https://github.com/Jimmy7338/NSO/tree/a8c55b7b56e1d058a2e66558f18fbe143cb4311b/experiments/offline_20261122_route_top_geometry) · [环式 20 保留核验](https://github.com/Jimmy7338/NSO/tree/874bff785debcad6982059530cdeda892a26e7ba/experiments/offline_20261122_route_intervention)

四臂为中性结构可见性 G、同类别直接巡检 D、正确类别结构先验 S 和交换类别结构先验 X。D 读取与 S 相同的真实可见仿真类别与实例体素，按已测表面的视角欠覆盖、预测观测距离和入射角给已有点排序。遮挡检验使用当前融合网格。G、D、S、X 使用相同的合法候选池、未知面积权重、去程附加动作代价和平局规则。D 的单位为面积当量，不能当作已观测新增面积。

D 的系数只在已知环式 20 的原始首选候选池校准一次。该过程发布了所有三档系数的选择记录，机器人没有执行新动作，也没有访问任何新布局的实测指标。新布局的规则、可见类别、动作上限、参考表面和所有候选臂先封存。每个布局从各自相同的首帧开始，独立执行。共同状态首次出现冻结协议规定的可区分路线时，每臂自行在同一目标与返航预留下选路一次，随后恢复同一冻结 G_perf。报告所有摆位、同选、缺失、中止和失配。

第 96 个计费动作比较相对实际共同分叉融合网格的新增面积。评价用固定独立参考上的 5 cm 点到实测网格匹配，测量新增、分叉之前已见但现在失配、净增和共同与独有面积。终点保留核查用相同参考点。去程之后的动作仍可能增加观测，实际到达与所比较的共同计时时刻分别给出。启动故障在执行任何动作前由旧运行时的冻结文件名导致，日志保存在 `execution_attempt_001/`。修复是给已有的不可变封存文件增加原生文件名别名。

东侧遮挡 G/S 在第 180 步之后反复于继承的 Open3D 模板射线场景构建崩溃。两臂各一次完全等价重跑仍复现，原进程退出、诊断堆栈与中止前网格留在 `execution_attempt_002/`。`measure_aborted_checkpoints.py` 独立复算固定时刻网格。该诊断的早期面积与完整配对分开，等时网格相同的 G/S 终点和全程动作账仍缺测。

仪器化重放确认注册射线场景时崩溃，崩溃输入为有限、非退化的盒状网格。用相同版本 Open3D 在独立进程注册这份原始网格 2,100 次均完成。另一次工程诊断只在粒子缓存重建前回收不可达的 Python 对象，东侧遮挡 G 成功运行 192 步并返航，五个时刻的已有网格和传感数据链与原始中止运行一致。诊断路径单独归档，原始 G/S 中止与终点缺测继续保留。 [射线诊断](evidence_checks/raycast_diagnosis/isolated_mesh_check.json) · [缓存回收诊断](evidence_checks/gc_lifetime_diagnosis/completed.json)

## 复现

在中控原目录结构执行，所需资产和旧源码哈希均在 `FROZEN_INPUTS.json`。已有准备或执行输出要先归档，再重跑对应阶段。`prepare.py` 每次只处理一个预先登记的布局。所有四个布局均需要准备。

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for scene in east_visible east_screened west_visible west_screened; do
    "$NSO_PY" prepare.py "$scene"
done
"$NSO_PY" run.py --calibrate
"$NSO_PY" study.py seal
"$NSO_PY" study.py run --workers 2
"$NSO_PY" evaluate.py
"$NSO_PY" report.py
"$NSO_PY" archive.py
```

- `PROTOCOL.json`、`FROZEN_INPUTS.json` 和 `calibration.json` 记录规则、输入与一次校准。
- `results/arms.csv`、`results/paired.csv` 与 `results/measured.json` 记录每轮执行和真实配对差。
- `results/paired_result.md` 汇总每种摆位实际表面净增、真实路线分歧与终点仍独有的面积。
- `results/*_masks.npz` 记录固定参考空间与测量掩码。原始网格索引在测量 JSON，原始网格与动作均保留在 `runs/`。
- `report.py`、`results/local_inspection_trajectories_and_surfaces.svg` 和 PDF 给出可重绘图，`results/manuscript_zh.md` 和 `results/manuscript_en.md` 为可入稿段落。
- `EVIDENCE_INDEX.json` 索引全部阶段、摘要和原始网格哈希。压缩包 `local_native_evidence.tar.gz` 包含可分发的紧凑证据，原始融合 PLY 网格留在中控 `runs/`。

图中每行对应一个受控摆位，灰色底图和浅灰参考点仅作离线显示。轨迹从共同分叉至第 96 步，星标为分叉。黑外圈是第 96 步只被一臂恢复、且到实际终点仍只被该臂恢复的参考点。全程上限 192 个动作含返航。样本独立单位是建筑父布局，一栋中的遮挡与无遮挡是相关控制。
