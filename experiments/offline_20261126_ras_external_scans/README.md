# 异源扫描楼层验证

两个从真实扫描重建的楼层分别来自 Replica 和 HM3D。原始网格按米制刚性变换进入仿真，保留各自的墙面和家具几何。共享原四楼的 0.20 m 栅格、0.20 m 圆形足迹、192 步计费、RGB-D、平面激光、准确位姿、TSDF 和返航规则。转向和返航计费。两个起点分别运行完整区域规划、原区域规划及严格最近可达前沿。

| 来源 | 连通可达面积 | 几何房间／开口／独立环 | 来源标识 |
| --- | ---: | ---: | --- |
| Replica | 28.60 m² | 3／2／0 | apartment_1 |
| HM3D | 62.92 m² | 5／4／0 | 00770-NBg5UqG3di3 |

房间数是冻结净空核心分区的描述性统计。场景筛选只使用网格、扫掠净空、连通性和初始传感器数据。初版核心面积条件及排除记录保留。开发场景使用 HM3D 00337，正式矩阵不纳入该场景。 [预登记与几何接纳](scenes/STARTS.json)

扫描场景的语义通道按共同约定为空。两种区域策略都保留原局部 S 实现，在无已建模对象证据时沿用实现中的几何回退。评价检验跨来源的覆盖与观测建图。独立参考网格仅在几何准入和评价中读取，规划器从实际已采样观测估计未知空间。

公开归档提供数据源版本、合法访问入口、文件校验值、实验代码和派生统计。受许可的原始或转换三维网格保留在有授权的中控环境。 [Replica 源项目](https://github.com/facebookresearch/Replica-Dataset) · [HM3D 源项目](https://github.com/matterport/habitat-matterport-3dresearch)

## 运行

在中控恢复冻结网格、场景适配依赖和既有的科学 Python 环境后，从本实验目录运行。

```sh
nso_python=/home/ouc/nso_robot/envs/science312/bin/python
"$nso_python" study_external.py seal
"$nso_python" study_external.py run --workers 6
"$nso_python" evaluate_external.py
"$nso_python" report_external.py
```

每个起点首先按同一实际动作时钟配对。各源楼层内先平均其两个起点，再求楼层等权均值。两法都达到 95% 时才计算 A95 步数差。代码保留第 0、48、96、144 和 192 步及真实终点，未发生的检查点留空。每楼的两个起点属于同一个独立比较单位。

所有正式轨迹与实际终点见 [异源结果](RESULTS.md)。扫描地面支撑约束的独立重放与源三角面复核见 [失败分析](FAILURE_ANALYSIS.md)。第一次评价调用了仅适用于主实验家具分组的参考加载接口，故障和恢复保留在 [评价记录](results/evaluation_attempt_001/evaluation_failure.json)。恢复仅去除缺少实物标签的家具分组，场景的五厘米匹配、正确可达自由格与演员数据均沿用冻结版本。
