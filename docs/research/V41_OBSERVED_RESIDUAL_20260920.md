# V41 R0/R1：当前已付 RGB-D 残差 caller

状态：**受控解析接口已实现并验证；未执行新的 World、物理轨迹或 TSDF 重建。** 日期：2026-09-20。

本次把 IGCR 的手填几何分数替换为由实际当前深度数组计算的残差。通过当前付费观测中的共有标签板像素拟合可见面，注册共享名义结构，再对当前唯一关联的测量像素预测深度。结构名称固定为 `planar / recessed / louvered / open_frame`。没有读取某个开发场景的真实结构、设施中心、尺寸、AABB、owner 或全节点 template depth。

这一步证明的是受控接口的数据来源与计算过程可运行，不是四模块整体优势、泛化能力或新的论文实验结果。不能把这里的解析前平面与凹槽数值计入 V35/V36/V39 的物理样本数、置信区间或性能曲线。

## 1. 实现位置与稳定接线

- `nso/observed_residual_v41.py`：`ObservedResidualV41`、共享名义库、观测标签面拟合和当前像素深度预测。
- `tests/test_observed_residual_v41.py`：19 项解析与真实接口集成测试。
- 依赖 `nso/observed_instances_v41.py` 的当前唯一几何关联输出；不修改 V40 已封存账本和 P1 几何。
- 复用严格的 `PaidRGBDObservationV40` 容器，其 `depth_m` 在本模块解释为 optical axial z。

接线顺序必须保持同一当前付费包，不得先用下一帧的几何更新当前决策：

```python
from nso.observed_residual_v41 import ObservedResidualV41

caller = ObservedResidualV41()  # 每个机器人/方法运行独立持有

# 每个 paid RGB-D packet 到达后；ledger 是 ObservedInstancesV41
associated = ledger.observe(observation)
evidence = caller.observe(observation, associated['accepted'])
feedback_receipts = []
for result in evidence['results']:
    if result['accepted']:
        feedback_receipts.append(ledger.apply_geometry_feedback(
            result['instance_id'],
            frame_id=result['frame_id'],
            observation_sha256=result['observation_sha256'],
            log_likelihoods=result['log_likelihoods']))
```

`accepted=True` 表示残差可计算；它**不等于**账本已经采用该证据。账本继续限制当前唯一实例关联、新几何支持、一帧/实例一次及累计上限。最终必须保存 `feedback_receipt['applied']` 和原因。重复支持即使能重新计算残差，也不能增加结构置信度。

调用者严格检查观察包类型、单调递增 paid step、不重复 frame ID、关联包 SHA、关联 frame ID/step、排序且不重复的当前像素、marker 像素为 support 子集、测量反投影点与 `support_sha256`。一帧重复实例 ID 会拒绝。账本负责从 action 0 开始连续的 paid step；合法静止或等待动作产生的相同 RGB-D/K/T payload 可以绑定新的当前帧，但标记重复测量、不扩展支持且不累加语义/几何证据。不能绕过账本直接把任意字典称为合法实例关联。

## 2. 使用字段与来源边界

当前 observation 仅有：`frame_id, paid_step, rgb, depth_m, intrinsic, world_from_camera`。

关联记录被使用的字段为：

| 字段 | 用途 |
|---|---|
| `instance_id` | 观测实例账本的 opaque ID；不是 renderer owner |
| `frame_id / paid_step / observation_sha256` | 绑定当前真实付费包 |
| `pixel_indices` | 当前唯一关联 support 的行优先 flat 像素 |
| `points_world_m / support_sha256` | 与当前 K/T/depth 反投影重新核对 |
| `marker_pixel_indices` | 当前实际可见的共有标签板像素，不提供真法向 |
| `geometry_feedback_eligible` | 记录账本的新支持资格；最终采用仍由账本决定 |

账本的 `marker_anchor_world_m`、首次 marker 元数据、类别和先验概率即使存在于关联记录，也不用于 caller 计算。平面锚点必须由 caller 自己从当前 marker 像素拟合。已知私有字段如 `world_aabb_m / local_solid_boxes / template_depths / true_structure` 会触发拒绝；没有读取私有资产、文件路径或 owner 的入口。

SHA 是可审计的数据一致性约束，不是一个能自动证明任意外部关联字典合法的安全边界。实例关联是否唯一仍要靠 `ObservedInstancesV41` 及其拒绝规则，不能仅凭一个与数据匹配的 SHA 推断关联正确。

## 3. 标签板面拟合及明确限制

设施假设保持竖直、world z 为已知重力方向、相机位姿为当前精确位姿。共有标签板安装约定来自通用开发原型：可见色块尺寸 0.32 × 0.30 m，安装于所有结构共有的实体标签板上。**这是受控人工标记先验，不是自然设施检测或通用六自由度物体位姿估计。**

流程如下：

1. 只取当前 marker 像素中轴向深度位于 `[0.1, 4.0]` m 的测量，反投影至 world。
2. 至少 16 个有效像素，且 marker 像素的有效深度比例至少 90%。对已测点 SVD 拟合平面；拒绝共线/太小、非平面、非竖直或过斜观测。
3. 平面法向朝向当前相机，结合 world z 构造平面内水平/竖直方向。检查可见范围与共有标签尺寸是否在像素分辨率容差内。
4. 使用可见矩形范围的中心作为**标签板面锚点**；它不是真设施中心。记录平面 RMS、测量范围、像素足迹上界、锚点/偏航不确定性与原 paid 包 SHA。
5. 第一个可靠拟合按实例缓存。后续没有 marker、但账本给出唯一已测几何关联时，可使用这份历史观测拟合和当前真实 support 深度。新的可靠 marker 若与缓存位置相差超过 0.15 m 或方向相差超过 20°，该帧拒绝评分，不静默重置实例。

拟合门包括：平面 RMS ≤0.015 m、最小/次小奇异值比 ≤0.15、法向竖直分量绝对值 ≤0.15、正面入射余弦 ≥0.35；锚点不确定性 ≤0.06 m、偏航不确定性 ≤12°。完整可见检查允许有限像素容差，**不能证明所有遮挡都被排除**，所以结果明确保存 `completeness_guaranteed=False`。噪声、远距离低分辨率、标签残缺、倾斜设施、安装位置改变或姿态不确定时，当前实现可能拒绝而不是产生证据。

此实现没有从设施实际 dimensions、world pose、中心、AABB 拟合或校正候选。也没有先读取标签私有 `markers.json` 再声称那是观测。

## 4. 共享名义结构与当前深度预测

`build_prototype_bank_v41()` 无场景参数输入。唯一名义宽/深/高是 `(1.2, 0.8, 1.6)` m，四结构共用尺度 `{0.85, 1.0, 1.15}`。通过公开通用 `structure_boxes(structure, dimensions)` 构造固体盒并集；不调用 `_profile`、开发资产生成器或 scene 文件。

每结构的 nuisance 候选数固定为 45：3 个尺度 ×3 个偏航 ×5 个平面内锚点偏移。偏航为 `{-u, 0, +u}`，其中 `u=max(5°, observed_yaw_uncertainty)` 且被拟合门限制到 ≤12°；锚点偏移为中心、水平 ±不确定性、竖直 ±不确定性。所有结构使用相同候选数和规则，没有按已知类别改变候选库或几何损失。

每个候选使用测得的标签面锚点、朝向，以及该名义原型的共同标签安装位置注册到 world。名义物体原点是计算出的 nuisance 参数，不被称为真实设施中心。

只对**当前相机、当前关联有效像素**生成射线。相机射线 z 归一到 1，所以交点参数就是光轴深度；没有径向 4 m 误筛。固体盒的进入/退出区间先排序合并，再求并集首个可见表面。near plane 落在重叠实体内部时取合并后的出口，不能返回某个盒子的内部面。这个问题由独立代码审查指出并已有专门回归测试。

默认最多 2048 个像素，超过时按排序后的 flat 像素等间隔确定性抽样；允许的参数上限为 6912。四结构使用同一有效像素集合，不能通过忽略难解释像素得到更低损失。

## 5. 残差口径

对结构 `s` 和候选 `q`，已测有效深度集合 `V`：

```text
error(s,q,i) = min(abs(predicted_axial_depth(s,q,i) - measured_depth(i)), 0.25 m)
              如果模型没有命中，则为 0.25 m
loss(s) = min_q mean_{i in V}(error(s,q,i))
score(s) = -(loss(s) - min_k loss(k)) / 0.05 m
```

输出键沿用 `log_likelihoods` 以匹配账本，但其科学含义是 **profiled robust pseudo-log-evidence，未校准概率似然**；输出 `calibrated_likelihood=False`。不是独立像素概率连乘，像素数也不是独立实验次数。这里的 0.25/0.05 m 是开发设置，尚未由标定数据估计。

深度为 0、低于 near 或超过 far 的像素全部忽略。缺测不能被当作“这里应该没有表面”，也不能降低平均损失或为 open frame 直接加分。少于 16 个当前有效像素时不输出 likelihood。观测有效但模型未命中，与传感器缺测是两个不同情况；前者可以产生几何矛盾。

返回每结构 `loss_m / valid_sample_count / predicted_hit_count / predicted_miss_count` 和最优尺度、偏航、锚点偏移；同时返回：

- 当前 paid packet SHA、support SHA、完整关联 mask SHA、实际评分 mask SHA；
- 观测标签面的原包 SHA 和原 marker mask SHA；
- 名义 prototype bank SHA、通用源文件/函数 SHA、每个名义几何的 SHA；
- caller 源码 SHA、当前 caller 配置 SHA；
- `ground_truth_used=False / future_observation_used=False / missing_depth_is_empty_space_evidence=False`。

这些布尔字段是实现边界声明，要结合调用图和 hash 审查，不能代替代码核验。

## 6. 实际验证与可用证据

运行环境：现有 `.venv-3d`，NumPy 1.26.4，CPU；由于依赖位于宿主 RAM，工具需使用 `sandbox_permissions=require_escalated`。本轮没有重新安装环境。可复用命令：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-3d/bin/python -m unittest discover -s tests -p 'test_observed_*v41.py' -v
```

结果：实例账本 17 项 + caller 19 项，**36/36 通过，联合运行 3.277 s**。覆盖：实际当前深度改变结构残差；测得 marker 面非真中心；部分、非平面、非竖直标签拒绝；无历史面拒绝；缺测不形成空面证据；源包、点、mask、hash 和当前帧约束；轴向边界；重叠固体内部面；真实前端 → caller → 账本反馈；真实前端在后续无 marker 帧继续用已测 support 关联并采用残差；G/S 的几何关联与几何证据完全相同；合法静止/连续全缺测新帧不增加证据。

两个解析正控制的当前残差如下，结构顺序固定。深度由独立的平面闭式表达式生成，没有调用预测 ray-box 函数来制造输入：

| 解析当前输入 | 有效评分像素 | planar loss (m) | recessed | louvered | open_frame |
|---|---:|---:|---:|---:|---:|
| 已测前平面 | 2048 | 0.000371094 | 0.151356268 | 0.112421396 | 0.157355102 |
| 已测凹槽内部平面 | 512 | 0.250000000 | ≈0 | 0.069335938 | 0.205105415 |

这是共享标签锚点下的两个**解析反事实输入**，不是一个静态场景中连续发生形变的合法轨迹。凹槽 fixture 的几何关联由解析测试显式构造；其内部与标签之间存在深度不连续，不能据此声称当前关联器已能跨越该间隔。额外真实 ledger 集成测试只证明满足唯一已测 support 条件时可接线，包括保持标签板几何但移除当前颜色的后续视角。louvered/open_frame 的真实正例与噪声压力验证仍未完成。

最终版本单次手动解析 profile：2048 像素约 0.614 s，512 像素约 0.232 s，进程 max RSS 42476 KiB。机器负载会影响这些数值；这不是规划方法的公平运行时间对比，也不含 renderer、TSDF、关联前端和候选全局搜索成本。

当前 prototype bank SHA：`61f7d11c699bcb775c1ad985ccfcacb93be084a2ce076ef1bfa975de21431657`。

当前 caller 源码 SHA：`63ed625cdcf00712b6ef9b39e4b4cb8ff951edb19f4a0cdb66eec5ff58a44b77`。

## 7. ANS / 四模块边界与下一步

该 caller 映射为 **IGCR：当前观测残差 → 每实例结构信念更新**。OV-SDF/实例层供给当前唯一测量关联；语义类别仅由账本决定结构先验；STGHP 与 RPN-UQ 将来读取更新后的 belief。G 与 S 共用 caller、相机、几何支持、候选库、损失和去重机制。

本轮没有实现从结构 posterior 预测未来候选的可见未观测表面收益，也没有用魔法类别权重替代该缺口。当前 API 没有未来候选 raycast 入口。action 0 自主语义质量调度、连续观测/真实重建残差、噪声与有限位姿误差、多设施接触/遮挡、不同安装位置、天然颜色/检测器、TSDF 与独立 Q 的一致性，需要后续独立开发验证。

R0/R1 仅解析数组工具，不改变资源协议。任何新的 R2–R4 物理实验批次仍须满足持久空间 `max(10 GiB, 2×batch_peak + 2 GiB)`，或由用户明确修订协议；当前小工具与测试通过不构成启动新物理批次的授权或空间豁免。
