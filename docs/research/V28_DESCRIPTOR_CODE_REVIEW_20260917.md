# V28 描述符代码独立审查（2026-09-17）

结论：r0 的观测来源与路线预算边界基本清楚，但存在一个可复现的方向支持计算错误；O 的名称也需要收窄。原 5 个测试通过不足以排除这些问题。本审查未修改算法或测试，未构造世界、读取传感包、融合 TSDF、提取网格或计算 Q。只执行原单元测试和手工观测状态的解析反例。

本文审查 r0，而不是尚待验收的修正版：

| 文件 | 审查版本 SHA256 |
|---|---|
| `nso/observed_debt_v28.py` | `594037aa62ea95ebb0e024e9c2a369fc2c8c90fda5b3e6a2105880b3d8c31ee1` |
| `tests/virtual3d/test_observed_debt_v28.py` | `83d1c5ac0c9ffbf0c420debee3e4cd9933ea125e18fcb6b6f0b6109bb718e74f` |
| `scripts/preflight_observed_debt_v28.py` | `d11eff72a1d8f0835e803f60b535abdab0a9d01d0256979ba5009717ec272cc2` |
| `docs/research/V28_DESCRIPTOR_PREFLIGHT_PROTOCOL_20260917.md` | `9e2e9872c23a166393aad0ff492f5e07b3cfa8a268384ee6afd5c329fca02478` |

主线程告知 r0 的 18 快照运行已结束；本审查不使用其排序结果决定反例或修正。r0 结果应原样保留，不能把受到下述支持错误影响的 S/O 变化解释为有效机制证据。

## 1. 阻断：patch 支持和 cue 方向使用不同原点

`nso/mapping3d_v2.py:37–41` 的方向位图按每个实测 patch 指向相机的水平向量写入。`nso/observed_planner_v26.py:112–113` 也逐 patch 查询该方向。方向扇区的世界坐标、角度增序和正负号一致，没有发现 180° 符号翻转。

r0 在 `describe_neighborhood` 第 34–40 行平均所有邻近 patch 的第 k 位；第 147–157 行却用 cue 中心指向候选相机的向量产生一个公共 k；第 187–188 行把该 k 的缺口作为候选收益。两种原点不能互换。1.5 米邻域、1.6 米候选环的尺度下，视差尤其不能忽略。

解析反例中，六个 patch 全部已经从候选相机的**同一个位置**观测过。真实 raw pool 含该安全候选，逐 patch 几何检查得到 visible=6、novel=0，但 r0 O 仍给正方向奖励：

| 项目 | 实际输出 |
|---|---|
| cue 中心 | `(4.3, 4.3, 0.8)` |
| 候选栅格姿态 / 相机 XY | `(28, 24, 0)` / `(4.9, 2.9)` |
| 六个 patch 的 bits | `[4, 4, 4, 4, 2, 2]` |
| cue→相机扇区 | `2` |
| 原逐 patch novel 数 | `0` |
| r0 cue 扇区缺口 | `1/3` |
| r0 O hypothesis_gain | `0.01666666666666667` |

为了只检查这一候选的数学分数，反例将已经由真实 raw pool 生成的那一行隔离后送入 `rank_context`。这不是它在完整 12 条短名单中最终被选择的证据，也不代表实际轨迹受到的影响大小。

还存在相关的支持集合问题：整体邻域缺口乘以可见比例，不等于可见 patch 的实际缺口。视野外或被遮挡 patch 的未观测方向可能给当前已充分观察的可见 patch 加奖。

建议的最小修正是利用原几何可见判定得到的 `visible_patches` 和 `novel_direction_patches`：

\[
d(c,a)=N_{\mathrm{visible,novel}}(c,a)/N_{\mathrm{visible}}(c,a).
\]

没有可见 patch 时收益为 0。若保留原 `visible_fraction=N_visible/N_local`，二者乘积就是 `N_visible,novel/N_local`，避免不可见缺口进入分子。cue→相机扇区只索引类别方向先验，不再查询 patch-relative 支持。描述符中的八方向均值可以继续作为 patch-relative 的统计摘要，但不应称为 cue 视角覆盖率。

该修正仍使用离散位图和二维遮挡近似；它只消除这一确定的计算错误，不会使剩余代理等同 Q、真实可观测面积或未来收益。

## 2. O 仅对 class_id 不敏感，并非纯 objectness

r0 第 121、148、170 行使用 `cue.confidence>=.6` 决定提案、可用支持及回退；第 188 行还用 confidence 乘收益。`VisibleSemanticMemoryV26` 的第 334–342 行明确将 confidence 定义为胜出类别的颜色票数 / 所有类别颜色票数。它是人工语义颜色的一致性，既不是物体存在概率，也不是校准网络置信度。

手工状态保持所有几何、cue 位置和 class_id 相同，仅 confidence 从 1 改为 .55，实际 raw pool 从 16 变为 8，O 从有 cue 排序变为完全 G 回退。因此原测试通过的“交换 2/3 后 O 不变”只证明固定其他检测输出时对类别 ID 不敏感，不能证明 O 去掉了全部语义信息。

本阶段可以不扩大实现：将 O 准确写成**固定 detector 位置及 confidence 下，去掉类别方向权重的条件消融**。O/S/X 共用同一检测输出，S−O 检验的是方向先验，而不是相对于零语义输入 G 的纯类别总增益。G 才是零 cue 输入的几何对照。若未来需要纯 objectness 对照，应独立定义不受类别票数比例影响的检测支持，再另外冻结实验；不要在当前运行后默换定义。

主线程说明本批 cue confidence 全为 1。这个事实使当前固定输出的 O/S 对比可解释，但不能把接口的普遍性质写成纯 objectness。本审查没有重新读取所有历史包来验证该批统计。

## 3. 候选、观测边界与安全的可接受范围

- `build_context` 输入为白名单观测状态和已见 cue，不读取世界、隐藏附件、GT owner、评价窗或 Q。描述符不使用 class_id。full 状态与原 256 样本及其他状态字段有一致性检查；未来 cue 被拒绝。
- 固定 cue 位置与 confidence 时，O/S/X 的 **raw pool** 相同，交换 class_id 不改变路线。短名单经过不同排序与实例保留后可以不同，不能说它们的最终 12 条候选完全相同。
- G 只取原 V26 G 结果；扩展池是原 G 的最终候选加观测 cue 环，并非所有可行几何候选的穷尽。G 与 O 不同池，G→O 同时引入位置提案和方向欠观测评分；只有 O→S 是该检测输出条件下的类先验消融。
- 路线使用已观测安全地图、完整往返动作成本和剩余预算。原测试实际核查安全栅格、动作数、预算与返回 anchor。静态可行不等同真实动态执行成功；本模块没有执行接口或收益反馈回写。
- 实例保留对 cue 对称，不直接指定 class 3 必须被选中。当前两 cue、容量 12 的范围合理；大量 cue 下容量和 cue_id 次序仍可能影响保留，不能推广成任意数量目标的公平保证。

## 4. 数学与因果解释的限制

两类方向先验在八扇区上的总质量均为 1，避免简单地给某类更大的总先验质量；但这不保证通过可见性、成本、候选容量过滤后的累计分数仍相等。固定尺度 .05、首次视线方向和邻域半径都是未学习的开发假设。

邻域含背景；patch 次数是付费帧的累计更新次数，不是独立视点数；残差是被截断并滑动更新的点到平面平方残差，不能称测量不确定性或精度误差。有效基线、真正实例边界、法向稳定性和校准不确定性当前输出 null，优于用 0 冒充已知。

修复后的新方向数仍不是新表面积，不包含隐藏表面发现，也不证明方向越多 Q 越高。18 个固定保存快照只能检查已有观测状态下的提案和排序行为，不能建立闭环收益、语义价值或泛化因果结论。改变 S/X 排名只证明类别先验进入计算。

## 5. 测试记录和必要补测

原文件 5/5 测试通过，0.480 秒，终端会话 `91119`。其后附带的第一次解析探针误在最终短名单中查找目标候选，触发 `StopIteration`；这是审查探针的索引位置错误，未作为模块测试失败。改为检查 raw pool 后，独立解析探针会话 `86267` 正常 exit 0，得到以上两个反例；没有修改算法或原测试。

建议修正版至少增加：

1. 同一候选相机已观测全部 patch，但 patch/cue 扇区不同：方向欠观测收益必须为 0。
2. 可见 patch 均已支持，缺口只在不可见 patch：收益必须为 0；反之可见的新方向应有正值。
3. 明确 O 的条件定义：交换 class_id 保持输出不变；改变语义 confidence 可以改变条件消融的结果，但不能继续将该行为标为纯 objectness。

以下是首反例的最小复现代码。只导入手工观测夹具，不实例化世界或 mapper；在本文所列 r0 源码上运行。`geometry_sha256` 来自现有手工测试夹具，不代表真实传感证据。

```python
from dataclasses import replace
import numpy as np
from test_observed_planner_v26 import state_for, cue_for
from nso.observed_state_v26 import GeometryPatchV26
from nso.observed_planner_v26 import ObservedPlannerV26, _sector, _xy
from nso.observed_debt_v28 import build_context, rank_context

state = state_for(budget=60, belief=np.zeros((43, 43), np.int8), patches=False)
cue, pose = cue_for(2), (28, 24, 0)
camera_xy = _xy(state, pose)
patches = []
for i, x in enumerate((4., 4.2, 4.4, 4.6, 5., 5.2)):
    point = (x, 4.3, .8)
    delta = camera_xy - np.asarray(point[:2])
    patches.append(GeometryPatchV26((i, 1, 1), point, (0., -1., 0.),
        10, 1 << _sector(delta), float(np.linalg.norm(delta)), 0.))
state = replace(state, patches=tuple(patches))
context = build_context(state, state, (cue,))
entry = next(row for row in context.pool if tuple(row['pose']) == pose)
_, direct = ObservedPlannerV26._geometry_gain(
    state, pose, ObservedPlannerV26._patch_arrays(state), {})
row = rank_context(replace(context, pool=(entry,)), 'O')['candidates'][0]
assert direct['visible_patches'] == 6
assert direct['novel_direction_patches'] == 0
print(row['semantic_gain'], row['semantic_evidence'])

low = build_context(state, state, (replace(cue, confidence=.55),), context.geometry_plan)
print(len(context.pool), len(low.pool),
      rank_context(context, 'O')['v28_audit']['exact_geometry_fallback'],
      rank_context(low, 'O')['v28_audit']['exact_geometry_fallback'])
```

复现环境需 `PYTHONPATH=tests/virtual3d`，使用 `.venv-3d/bin/python -B` 和单线程环境变量。以上代码是已执行解析检查的整理，不是新的保存历史运行授权。r1 应另留版本与测试证据，旧 r0 文件和产物不覆盖。

## 6. r1 修正后的独立复核

主线程另建 r1 后，本审查只读比较完整模块和测试 diff，并独立执行两个新增的回归用例；未重跑历史、未改源码。r0 模块及测试 SHA 仍与第 1 节一致。

| r1 文件 | SHA256 |
|---|---|
| `nso/observed_debt_v28_r1.py` | `2cd18825449822e266a51db4d05e1d45485bd885b75e59c8e6cbb4d615a4a11f` |
| `tests/virtual3d/test_observed_debt_v28_r1.py` | `0666ebd0597976c581544d54b08c32aaa1e547a61abed59aea6dbc6b16815f94` |
| `docs/research/V28_DESCRIPTOR_PREFLIGHT_R1_PROTOCOL_20260917.md` | `1c10300547eafde041290dfc481d0b0b2735f3f1cfc19082ea6dccdc9e5bd149` |

独立测试终端会话 `1048`，exit 0，2/2 通过，0.735 秒：

```text
test_identical_camera_cannot_earn_new_direction_from_cue_origin ... ok
test_invisible_patch_debt_cannot_reward_seen_visible_patches ... ok
Ran 2 tests in 0.735s
OK
```

两项都通过完整 `build_context` 的实际路线生成和可见性计算，再隔离真实 raw pool 行检查数学分数，不是手工填写 `novel_direction_patches` 后复述实现。首项六个可见 patch、零新方向，在 O/S/X 中收益均为 0；次项加一个不可见但有方向缺口的 patch，local=7、visible=6、novel=0，收益仍为 0。

r1 已把方向缺口换为候选下逐 patch 的 `novel/visible`，保留可见比例后等于 `novel/local`；cue 方向只作用于类先验。描述符字段也改为 `patch_relative_directional_support`，原通用 `directional_deficit` 为 null。零可见不会生成正证据。上述两个明确数学反例已修复。

O 的算法仍保留语义 detector 的 confidence，但代码和 r1 协议已明确其条件消融范围，因此不再将第 2 节的语义通道依赖隐藏为“纯 objectness”。这是解释纠正，不是新的独立 objectness 前端。

未发现这次有限修正引入新的阻断问题。结论只覆盖两个缺陷与观测边界：r1 的历史候选份额、语义排序变化、执行后 Q/J，以及自然语义识别价值均须按各自冻结协议单独报告；本审查不背书其性能。
