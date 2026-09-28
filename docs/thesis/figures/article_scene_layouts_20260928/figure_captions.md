# New industrial-layout figures / 新工业布局图注

## Overview / 总览

**EN.** Nine predeclared industrial mapping layouts: one development layout and two test layouts for each of equipment aisles, screened workcells, and a production-island loop. All panels retain the same 0–8 m axes and world coordinate frame. Colored envelopes and identifiers locate the four facilities; arrows indicate their front sides. Gray segments and nodes are the shared coarse navigation prior, and H is the common start and required return pose. Classes, facility positions, and occluders are private asset information used here solely for offline explanation. These are 3 development + 6 predesigned test layouts, not nine completed runs; no executed trajectory or measured performance is shown.

**中。** 九个预先设计的工业主动建图布局：设备巷道、隔断工位与中央生产岛环路各包括一个开发布局和两个测试布局。全部子图保留统一的0–8 m坐标轴及世界坐标系。彩色包络与编号表示四台设备，箭头表示设备正面；灰色节点与线段为各方法共享的粗略导航先验，H为共同起点和规定返航位姿。类别、设施位置及遮挡信息仅用于离线解释，不作为控制器的隐藏真值输入。本图表示3个开发与6个预先设计的测试布局，不能解读为九次实验已经完成；图中不展示实际执行轨迹或实测性能。

## Development details / 开发布局详图

**EN.** Detailed views of the three development layouts under identical metre scaling. Equipment labels are offline annotations; marker-side arrows describe object orientation rather than robot motion. The equipment aisles require observation-direction and inter-device budget choices; workcell screens introduce detours; the central island couples circulation direction with observation order and return cost. The four costs are the minimum static primitive-action counts for a single close-front observation and full-pose return within the certified coarse candidate set. A valid pose is 0.35–1.30 m from the label in the horizontal plane, faces the label with cosine at least 0.5, and contains all nominal label corners in the field of view without intervening obstacle-envelope intersections. These conditional lower-bound costs exclude additional classification, diagnosis, revisits, and online sensing effects. They are neither executed trajectories nor proof of reconstruction quality.

**中。** 三个开发布局采用同一米制比例展示。设备类别为离线标注，标签面箭头表示物体朝向，不是机器人运动方向。巷道任务考查观察方向与跨设备预算分配；工位隔断引入绕行代价；中央生产岛把绕行方向、观察顺序与返航成本联系起来。每台设备旁的成本是经过认证的粗图候选集合内，单次近距离正面观察并按完整位姿返航所需原子动作数的静态最小值。候选到标签的水平距离为0.35–1.30 m、正面余弦不少于0.5，且标签四角处于名义相机视场、视线不与其他障碍包络相交。该条件下界不包含额外类别确认、诊断、重访及在线传感影响，不代表已执行轨迹或三维重建质量。

## Static cost matrix / 静态成本矩阵

**EN.** Static one-device close-view-and-return costs for all nine layouts, computed from the frozen geometry and common primitive navigation graph. Development rows use bold values. Orange outlines indicate costs above 120 actions for that single constrained visit; they do not mark an experiment failure. The task does not require visiting every device, and all four devices remain in the evaluation denominator. This table gives a geometric budget reference before outcome analysis; it is not a method ranking or a global optimum over unrestricted observation poses.

**中。** 九布局的单设备近观察与返航静态成本，由冻结几何和共享原子导航图计算。开发行使用粗体；橙框只标识相应受约束单次往返成本超过120动作，不表示实验失败。任务不以访问全部设备为资格要求，四台设备始终保留在评价分母中。该表用于结果分析前的几何预算说明，不构成方法排名，也不是任意观察位姿下的全局最优值。
