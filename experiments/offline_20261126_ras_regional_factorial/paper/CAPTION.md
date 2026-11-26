# A96 图与图注

**论文英文图注。** Observed occupancy, executed trajectories, and native TSDF surfaces at 96 paid actions on two simulated floors. Both methods start from the same sensor packet and use the same world frame, 0.20 m grid, map scale, and orthographic camera for the surface renderings. The hollow diamonds show the first targets selected from the common logged candidate set before action 1. The orange arrows show the realized heading after the first different executed action for each method. The offset-ring pair first differs at action 4 (forward versus left turn). The double-loop pair turns in opposite directions at action 1. At action 96, the full regional planner changes the joint score by +18.85 percentage points in the offset ring and −10.11 points in the double loop. Coverage C is the fraction of reference reachable free cells that the robot correctly observes as free. Surface F1 at 5 cm combines predicted-to-source accuracy with recall on the fixed first-hit visible reference, and J = C × F1. Surface panels render only native fused triangles. Background pixels show where the display rays find no fused triangle or leave the scene.

**中文图注。** 两个仿真楼层在第 96 个计费动作时的观测占据图、实际轨迹和原生 TSDF 表面。每楼两法共用首帧、场景坐标、0.20 m 栅格、作图比例与正交渲染视角。空心菱形标出第 1 步之前从同一候选集选出的目标，橙色箭头指向执行分歧后的实际朝向。错位环道在第 4 步分别直行、左转，双环在第 1 步分别左转、右转。相对原区域规划器，完整方法的 J 在错位环道提高 18.85 个百分点，在双环下降 10.11 个百分点。C 为正确观测到的可达自由格占比，F1 以 5 cm 为阈值，J 为两项指标之积。表面图直接渲染原始融合三角网格，背景指显示射线未与融合表面相交或离开绘制区域。

## 复画和判读

- 图中的原区域规划器对应 `region_S`，即 D0G0。完整区域规划器对应 `combined_S`，即 D1G1。两者共用局部类别策略 S。四个 A96 检查点均为实际捕获。轨迹从冷启动画至第 96 步，同一栅格内的转向由箭头或实际动作记录表达。
- 地图采用冻结的仿真世界坐标，栅格原点 (0, 0)，x 向右，y 向上。由行列号和 0.20 m 分辨率换算轨迹位置。两个方法的 PLY 使用同一相机的正交光线求交，采用相同的面光着色，无刚体配准、裁去原生三角面或填补未融表面。
- 定量表面评价的精确率将融合表面样点与独立场景三角面比较，召回率以预冻结的可见域表面点为分母。该可见域由布置规范决定的可达相机网格和四个朝向对独立场景网格进行首交投射，且独立于四条实走轨迹。错位环道有 1,647 个参考可达格，380 个参考相机位姿，51,838 个可见表面样点，面积约 277.44 m²。双环依次为 1,716、412、52,403 和约 294.27 m²。灰色未知栅格不等同于可见域缺失，表面面板中的空白也不作为 F1 计数。
- 两楼的初始候选评分由区域排序和全局选择日志相互核验。错位环道两法共 13 个有排序记录的候选，双环共 18 个。错位环道先改选后才出现执行分歧。菱形表示实际派发的位置，抵达该点由后续动作记录判定。
- 复画需要原档的 PLY、栅格、轨迹、日志及评价参考。PDF 适合直接排版，PNG 为 420 dpi，SVG 中的轨迹、文字和符号可编辑，融合表面是固定相机生成的栅格。Python 源码可重新生成表面视角和全部图层。

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/ouc/nso_robot/envs/science312/bin/python render_checkpoints.py \
  --study /home/ouc/nso_robot/analyses/region_independent_floors_20261123
```

[PDF 图](checkpoints_a96.pdf) · [PNG 图](checkpoints_a96.png) · [可编辑 SVG](checkpoints_a96.svg) · [渲染源码](render_checkpoints.py) · [分歧记录和 SHA-256](checkpoints_a96_receipt.json)
