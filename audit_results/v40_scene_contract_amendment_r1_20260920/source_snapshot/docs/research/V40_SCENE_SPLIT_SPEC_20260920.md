# V40 P0 场景合同与开发/测试隔离规范

日期：2026-09-20。本文件对应纯 Python 合同、JSON 配置和准备脚本；本轮没有创建 World、测试几何、传感帧或策略/指标结果。配置是可执行的字段与隔离约束，**不是已经实现或冻结的场景资产**。

## 1. 样本与独立单位

|族|任务范围|预留测试父场景|独立开发父场景|
|---|---|---:|---:|
|A|机柜/机房通道，服务面和开口的互补观察|6 主测试|1|
|B|货架、工位及重复实例，层间与侧向遮挡|6 主测试|1|
|C|混合设施，异质结构及设施间预算分配|6 主测试|1|
|D|几何充分、低异质性的中性条件|6 主测试|1|
|E|规模、设施稀疏性和覆盖预算压力|6 边界测试|1|
|F|类别—结构关系失配或持续识别扰动|6 边界测试|1|

合计为 **24 主测试父 + 12 边界父 + 6 开发父**。旋转、镜像、传感噪声和同父结构条件只能是簇内变体，不能变成新增独立父。当前清单仅预留父身份，不生成这些变体。

开发、主测试、边界测试按 `layout / prototype / mesh` lineage 隔离。父清单中的 lineage 是将来生成时必须遵守的独立来源预留名：不是实际网格哈希，也不能证明已经完成内容去重。未来正式准入还需要真实资产内容、原始来源、父子变换链及去重审计；同源资产改名无法建立独立性。抽象结构词汇可以跨 split 共享，具体几何 prototype 和网格不可以跨 split 复用。

场景筛选只能使用预先任务/几何可行性约束，不可读取 S/G 胜负或质量分数。类别与隐藏结构分别建模；同一类别必须支持多个结构，一个结构也必须属于多个类别。P0 与实例信念账本采用同一支持域：每个类别—结构概率均须 **≥1e-9**，不得写 0 或接近 0 的排除概率；现有配置最小值为 0.1，准备器写出的六个开发骨架满足此约束。这里验证的是先验配置，不是尚未实现的几何生成器。识别噪声与真实类别—结构关系偏移是不同因素，不能合成一个“语义失效”开关后混淆归因。

## 2. 规范、API 与输入权限

- 配置：`configs/virtual3d/v40_scene_protocol_20260920.json`。
- 合同：`nso/scene_contract_v40.py`。只依赖标准库；不导入场景、仿真、网格或评分模块。
- 准备器：`scripts/prepare_scene_catalog_v40.py`。
- 测试：`tests/test_scene_contract_v40.py`。

API：

```python
load_json_strict(path) -> dict
validate_protocol(protocol) -> None
validate_scene_spec(scene_spec, expected_split=None, *, protocol=None,
                    budget_declarations=None) -> None
validate_public_spec(public_spec) -> None
validate_catalog(catalog, protocol, *, scene_specs=None,
                 budget_declarations=None) -> None
validate_budget_declarations(declarations, protocol, catalog=None) -> None
public_planner_spec(scene_spec, *, protocol=None, budget_declarations=None) -> dict
assert_formal_test_ready(scene_spec, *, catalog, protocol) -> None  # P0 清单必定拒绝准入
```

所有验证失败使用 `SceneContractError`。重复 JSON 键、NaN/Infinity、未知字段、布尔数值、非法分布和越界路径被拒绝。`public_planner_spec` 先验证完整合同，再返回白名单的独立深拷贝；不靠删除几个敏感键的黑名单来隔离数据。

单独的 `validate_public_spec`、不提供 `protocol` 的场景验证/提取只做结构验证，不能声称参数已经与协议绑定。集成必须传入协议，并用 `validate_catalog(..., scene_specs=[...])` 联合检查父身份、lineage、seed commitment 及完整公开参数。`prepare --validate-only` 强制加载六个开发 spec、逐个传入协议并做联合验证，不能只检查 catalog 元数据或文件哈希。

固定公开字典 schema 为 `v40.public_planner_spec.v1`，只包含：

|字段|许可内容|
|---|---|
|`task`|动作 0 开始、空强制前缀、预算、预算来源/档位、必须返回|
|`navigation`|所有方法共享粗拓扑先验的声明、分辨率、待实现公共图引用；禁止隐藏表面几何|
|`sensor`|相机尺寸、内参、量程及噪声参数|
|`motion`|移动/转向步长及相机高度|
|`pose_model`|精确或带噪里程计、噪声大小和轨迹相关性|
|`structure_prior`|抽象类别、抽象结构、多对多条件概率、仅开发集来源、校准状态|

公开提取结果不包含 parent/family/split、seed/commitment、实例列表、真实构型、ROI、GT、lineage、renderer 或 evaluator 引用。设施发现/关联必须来自随后付费观测；不能把评价用实例 ID 或仿真 owner ID 当作免费观测。

`renderer_private` 仅保留待实现的 world、真实实例和资产清单引用；`evaluation_private` 分开保留 GT、评价 ROI 和固定可观察外表面集合。当前这些引用全部是 `status=pending, relative_path=null, sha256=null`。公开粗导航图与开发先验训练清单也尚未实现。当 scene 为 `pending_geometry` 时，公开导航 `graph_asset` 也必须是 pending/null；不能在公开字段放入私有路径或任意声称 materialized 的图引用。

公共粗拓扑是导航先验，不能把完整已知可达图的实验写成无地图先验的未知环境探索。未来图只提供公开通行结构，不允许附带真实设施实例、实际遮挡表、未选视角的 RGB-D 或表面收益。合同验证只核字段；未来文件内容是否满足这些语义权限仍须单独审计。

## 3. 预算、姿态和评价口径

六个开发骨架暂用 160 动作、96×72 RGB-D、4 m 深度范围、精确位姿等设计初值，用于表达可配置接口，**不是试点测得参数，也不是正式预算冻结**。开发阶段可调整预算/噪声，之后须更新协议版本与 SHA。不得按隐藏最优路线或 S 的胜率给正式场景分配预算。

参数可配置是指在公共协议中统一声明，不能随意改某个 scene 的 `max_actions`、传感器、姿态、先验或导航分辨率。绑定验证逐项要求 scene 公开参数等于 `protocol.public_defaults`。

仅边界测试父可通过单独预声明 sidecar 使用预算档。其严格 schema 为 `v40.boundary_budget_declarations.v1`，包含原协议 `protocol_sha256` 和按父 ID 索引的 `assignments`；每项只允许 `budget_tier / max_actions`。standard 必须等于协议默认预算，short 必须为正且小于默认预算。开发父、主测试父、未知父或任何传感/先验覆盖均被拒绝。sidecar 必须在正式任务前另行封存；本轮只有验证器及合成测试样例，**没有为未来测试父执行或冻结任何预算例外**，原 catalog 和 seed 保持原样。

从首个实际观测之后自主选择第一动作；`start_action=0`、`forced_prefix_actions=[]`，不预付历史的 18 步探索前缀。所有移动、转向、观测和返航必须由后续运行器按同一成本合同记账。本合同本身不会执行任何动作。

评价定义预留为：所有任务设施中，在预定义可达候选与传感范围内可观察的外表面；包含水平与竖直表面，排除封闭内部；集合独立于各方法实际轨迹；漏检设施保留在分母。主表面分量为实例宏平均 F1@5cm。实际 GT 集合、可达性枚举与评估器尚未实现，本轮不输出虚构集合哈希或分数。

## 4. Seed commitment 与保管

36 个测试父各使用脚本内部产生的 256 bit 随机 seed，采用域分离 SHA256，对以下规范 JSON 字节承诺：`domain / protocol_id / parent_id / split / seed_hex`。公共清单只写 commitment，不写测试 seed；开发 seed 可见且由开发父标识确定，不涉及未来测试几何。

为使 commitment 将来可以兑现，真实测试 seed 保存在同目录 `private/test_seed_escrow.json`，目录权限 0700，文件权限 0600。准备器只在生成时写入，不打印 seed；公共复核器只检查该文件权限，不打开文件，连内容哈希也不重新读取。公共 manifest 中的 escrow SHA 在内存中形成并记录，不表示场景资产已生成。

这是可追溯的划分约束，**不是 OS 级盲法、独立第三方测试或对拥有相同账号的研究者不可访问的加密保密**。后续开发不得主动读取 escrow；正式测试由冻结后的专用生成/运行流程按协议使用。当前 seed 不承诺尚未实现的生成器版本，所以将来还要冻结生成器与真实资产；不能凭 seed commitment 声称整套测试已经冻结。

已有输出目录会导致准备器拒绝再次生成，避免悄悄换 seed。若写盘中断，保留已有 commitment/escrow，由显式恢复流程处理，不能删掉目录反复生成直到得到有利场景。验证测试使用独立的临时合成 seed，不读取真实 escrow。

## 5. 产物和操作

```bash
python3 -m unittest discover -s tests -p test_scene_contract_v40.py -v
python3 scripts/prepare_scene_catalog_v40.py
python3 scripts/prepare_scene_catalog_v40.py --validate-only
```

输出目录：`audit_results/v40_scene_contract_20260920/`。

- `catalog.json`：42 个父预留、36 个不可见测试 seed commitment、待实现资产状态。
- `development/DEV_A_00.json` 至 `DEV_F_00.json`：6 个开发配置骨架，不含几何。
- `public_development/DEV_A_00.json` 至 `DEV_F_00.json`：对应公开提取结果，可用于集成合同测试。
- `protocol.json`：本次准备所用配置的规范化副本。
- `manifest.json`：源文件、公共文件及生成时私有 escrow 的 SHA，零仿真/零观测/零评分计数。
- `private/test_seed_escrow.json`：权限受限的 seed 保管文件，不得提交或在正常开发中读取。
- `.gitignore`：排除本目录的 `private/`，防止普通批量 Git 提交携带测试 seed。

输出总量上限 5 MiB；实际准备器在写文件前检查所有计划输出的总字节数。不会读取或清理已有实验、下载资产、构造 World。集成入口优先使用 `development/DEV_A_00.json` 调用 `public_planner_spec(scene, protocol=protocol)`，不要把完整私有 scene spec 交给规划器。

## 6. 本轮能证明与不能证明的内容

纯合同测试覆盖：严格字段类型、重复键/非有限值、信息白名单、复制隔离、类别—结构多对多、跨 split lineage、变换父重复计数、seed 绑定与泄漏、空前缀、可配预算/姿态、pending 资产无假哈希、0600/0700 保管及公共复核不读取私有 seed。

这些测试不能证明未来场景的物理有效性、独立性、无泄漏渲染器、类别识别可靠性、任何规划效果或方法优势。`assert_formal_test_ready` 对当前 P0 预留格式无条件拒绝正式测试；后续必须实现真实资产和内容级 split 审计，并升级准入合同。单改 `frozen_ready` 字符串不能绕过这个门。

## 7. r1 合同修正与原封存保留

交叉审查后的 r1 修正统一了 ≥1e-9 支持域、pending 导航引用和协议参数绑定，增加显式边界预算声明校验。原 `audit_results/v40_scene_contract_20260920/` 的 17 个文件与 private seed 不作改写，不重新生成 seed。新源版本与旧公共数据复核单独记录于 `audit_results/v40_scene_contract_amendment_r1_20260920/`；新源 SHA 与旧 manifest 记录的源 SHA 分列，不能声称修改后的源码仍是原封存源码。公共复核不打开私有 escrow，原 seed 的内容哈希不重新计算。
