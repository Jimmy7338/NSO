# V25：case00 已付历史的一次表示检查（执行前协议，2026-09-15）

本任务只检查冻结 V25 在已见过的 V24.1 case00 历史上能否修复具体的混距外壳拟合问题。它是开发诊断，不是独立确认、语义增益实验或完整架构验证。只执行一次，失败保留，不修参数、不扩其他案例、不增加物理观察。

## 输入与冻结

- 原始证据：`audit_results/facility_choice_v24_paid_prefix_20260915/case_00`，D24-P00 / A_complex_B_simple，235个包0..234，234个已付动作。
- 旧表示与成绩：`audit_results/facility_choice_v24_shape_prefix_20260915/case_00.json`，原V24结果直接读取；不重跑旧后端、旧TSDF或旧指标。
- 新后端固定为 `nso/observed_shape_v25.py`，SHA256 `cb8685f30f9d5e9da5e5d2eacfdbf1d6d7ce8485a0f4d633e03908f4d5febe71`。它是整套解析门未通过的原型，不接默认runtime。
- 新driver：`scripts/replay_facility_choice_shape_v25_case00.py`。prepare冻结原147明确路径、新后端、新driver和本协议，并记录依赖版本及两旧封存根的清单、结果、源码包SHA。旧源码包引用，不复制旧传感包或原TSDF。
- 初始seed固定沿用原实际RGB二值marker深度像素：slot0动作24、像素(32,57)；slot1动作111、像素(33,3)。逐包核原packet SHA；在首次动作核实际像素、有效深度、去类别marker并集、连通组件规模及回投坐标。这里不重新挑seed，不用终点GT纠正关联。

## 固定执行顺序

1. prepare只核证据和写冻结元数据；不observe、snapshot、构建GT世界或评价。旧case JSON只提取seed与输入几何校验摘要；旧分数不参与新估计。
2. run唯一worker按0..234顺序加载所有已存包，两个后端都从frame0开始observe。对应首次动作才提供原seed，此前seed=None；以后不追加seed。必须235个包、470次observe、2次snapshot。
3. 读原final_mesh.npz核vertices/triangles/vertex_colors数组SHA；两次完整snapshot均使用这同一原TSDF的几何，零TSDF融合。
4. 两实例snapshot全部完成，并与原V24逐项核ground、measured/ground/cleaned/unassigned点数组计数与SHA、observed mesh数组SHA，之后才导入/构造参考世界与OutlineEvaluatorV23。后端不接GT边界、尺寸、类别、旧Q或参考网格。若任何共有输入或实测输出不相同，失败保留，不继续评分掩盖差异。
5. 两snapshot在评价前写出小审计检查点与新inferred mesh。raw引用原TSDF；observed mesh在SHA相同前提下引用旧保存文件；completed由observed+inferred顺序拼接，不另存副本。拒绝时inferred必须空，零推断时completed与observed逐数组相同。
6. 评价侧使用冻结V24.1 ReferenceWorld（step/sense/scan均禁止）和原V23 OutlineEvaluator；参数仍boundary spacing .01m、阈值.02/.05/.10、完成尺寸/边界容差.05m、IoU门.9。参考signature必须与原V24一致。
7. 原raw/measured分数在网格SHA一致后直接沿用，并明确标来源；新inferred和completed各评价一次，两实例completed各自对所有固定GT窗评价一次，总计4次评价。主读5cm Q及J，2/10cm和实例结果同时保存，不按最高成绩挑表示。
8. GT仅用于评价关联：沿用同一首次seed唯一窗口映射，不重新分配；再核与旧映射一致。种子关联与非空、非重复组件门分开；全局union+GT窗口分数不证明实例归属。全部预测保留，不用GT裁剪修改后端几何。

## 比较与解释

固定报告原V24与新V25的两个completion理由、free-ray统计、估计参数和边缘支持、raw/observed/inferred/completed四表示及每设施外形Q。raw与observed只复用已证明相同网格的旧评价；若新推断获准，其收益必须标成cuboid先验贡献，不归为新增实测精度。无推断时completed必须等于observed，评价也应相同。

覆盖、碰撞、返航与动作数全部继承原付费前缀，不产生新的策略收益。评价上下文仍是234动作的脚本前缀，不是后续两对象选臂的正式预算。只做case00不能说明两排列几何配对或跨场景泛化；此前已看过该历史，不能称独立测试。

“运行complete”只表示按协议完整保存；是否获准外壳、Q是否改善、外形完成门是否通过均按真实结果单列，拒绝也可构成完整运行。异常、超时、输入不变性失败则manifest标failed，保存进度和failure receipt，不重启。

## 资源与入口保护

- --prepare / --run分离；root只读审查通过后方可prepare/run。已有输出根不可prepare覆盖；run只接受未启动的prepared根，独占claim阻止重复启动。
- 单worker由父进程控制，实际计算最多600秒；超时终止唯一worker，父进程保留failure及最后进度，不重启。记录父/worker PID与真实计数。
- 新输出总计不超过4MiB（包括新源码包、结果、网格、进度、清单与失败记录）；每次先在内存序列化、按精确字节与块分配估计检查再写。预留64KiB失败元数据空间，实际磁盘空闲保持至少64MiB。源raw/observed及旧证据只引用；容量不足时失败，不删除或降采样几何。
- 输出：`audit_results/facility_choice_v25_case00_shape_20260915`。完成后记录源码、输入和产物SHA；本协议不授权参数修复、四案例扩展、新物理动作或接入规划器。
