# V37 真实类别到配置先验的接入就绪审查

2026-09-18，独立只读代码审查。没有下载模型、训练、构造World、渲染新观测、运行TSDF或测量终点质量；主任务额度仍为35/36。两项小型数值检查只使用现有类别映射函数和从原文件AST提取的矩形更新函数，不加载检测网络。以下发现描述旧自然前端及与V35接口的差距，不能倒推它们导致了V35/V36实测收益；V35/V36没有经过这些旧检测／语义地图函数。

## 结论

现有类别词表不能直接支持V33的type_A/type_B到观察侧关联。V33的公共配置表声明type_A对应设备局部u轴负侧、type_B对应正侧，设备参考系为公共信息，且明确不声称自然网络或学习得到的映射。这里type是人为构造的配置类别；“冰箱”“椅子”等自然类别没有在项目数据中被验证为固定左／右服务侧。即使识别出设备型号，也必须先有该型号与配置的独立目录依据，再验证从RGB读取型号的可靠性。这是设备身份／目录先验，不能改名为自然大类泛化。

当前缺口是“可见识别证据→明确类别或型号→有来源的配置分布”的数据链。普通COCO检测器可提供物体大类和框，并不会自动得到设备型号、局部坐标系、服务侧、结构难度或追加视角的收益。实际任务图像与真实设备配置标注是否存在，以ROOT资产盘点为准；本审查未拿到可用于校准／留出评估的数据，故不报告自然识别准确率、拒识率或自然语义下游收益。

## 四种类别空间必须分开

| 来源 | 实际含义 | 能否直接给V35先验 |
| --- | --- | --- |
| COCO 80 | 通用物体大类；ID2=car，ID3=motorcycle | 不能；无配置关系数据 |
| 自定义52标签 | COCO子集重编号；ID2=bed，ID3=dining table | 不能；且pillow无检测来源 |
| V15 taxonomy | 人工资产open负、closed正；competition_v9与inspection_v4符号相反 | 不能；仅修复两个旧生成器的标签符号，不是自然类别映射 |
| V35人工RGB | 精确颜色ID2=type_A、ID3=type_B，16像素门、2个不同XY位置 | 仅适用于已冻结的受控二配置场景 |

不能按整数相等把这些空间拼接，也不能把自然检测结果改绘成V35人工颜色以声称自然前端接入成功。V15的source schema检查与禁止重复编码是值得保留的接口约束，但不包含COCO、自然型号或V33方向关联。

## 发现的实现问题

1. **自定义映射与检测框错位，已复现。** `semantic/class_mapping.py`的`map_yolo_to_custom`先丢弃不支持类别，`semantic_detector.py:114`却用返回列表产生掩码，再索引尚未过滤的原始框与分数。无模型检查输入COCO `[56,2]`（chair、car），映射变成`[0]`；长度1的掩码索引2行框触发`IndexError: boolean index did not match indexed array along dimension 0; dimension is 2 but corresponding boolean dimension is 1`。全支持／全不支持输入可能掩盖此问题。新适配器应先产生与原检测数相同的映射数组（未知保留-1）及同长度掩码，一次同步过滤框、分数、类别；最好保留拒识事件及原始标签供审计。冻结旧源不在本审查修改。

2. **旧语义地图会把一个类别写入全部类别通道。** `SemanticMap2D._draw_rect_in_map`在`semantic/semantic_map.py:78`对`sem_map_chw[:,...]`累加，而调用处传入完整类别张量，没有选`cls_id`切片。直接提取原方法AST，仅去除注解和装饰器后以3×8×8零数组调用，2×2区域、权重1得到各通道和`[4,4,4]`，三个通道完全相同。这不能保存类别身份。该方法也未使用检测框或深度定位物体：同帧所有物体投到机器人前方同一矩形，旋转在矩形绘制中被忽略。它不适合作为设备实例关联或服务侧先验的实现。

3. **旧图没有重复观测去重或配置置信度。** 每次回调累加全局计数；重复帧、静止重复观察会重复增益。分数先除以该帧分数总和，单个被前端接受的检测，无论原始分数0.2还是0.9，默认类别权重下都注入相同权重1。检测阈值只负责过滤，不能把这种计数叫后验或校准置信度。低置信、unknown、不同实例混合和冲突均未形成明确回退合同。

4. **RGB通道合同尚需安装源证据确认。** `env/habitat/exploration_env.py`直接使用`obs['rgb']`，转CHW；`main.py`把该张量交给`SemanticDetector`；后者转回HWC NumPy后直接传`model.predict`，未做颜色转换。旧OV-SDF的YOLO回退也将标为RGB的NumPy直接传入YOLO。就本次审查可访问环境，`.venv-3d`中`find_spec('ultralytics')=None`；在三个项目venv及可读的/root、/usr/lib、/usr/local/lib、/opt、/tmp中未找到其`engine/predictor.py`或`data/loaders.py`。因此本报告只确认项目调用侧RGB来源，不凭记忆声称此安装版本要求BGR；若ROOT找到或配置固定版本，必须记录其本地源hash并验证NumPy输入loader及predictor预处理，才能把此项升级为已确认问题。Detector还未严格校验uint8、3通道或值域，warm-up调用与正式转换也不完全相同。

5. **开放词汇缺失时的替代值不是识别。** `nso/clip_semantic_map.py`在CLIP不可用时返回零特征／固定相似度0.5，检测后端可为none或闭集YOLO。应在记录中显式输出后端与可用状态，禁止把固定相似度当自然类别置信度或开放词汇证据。

## V35已有保护与不能直接复用的部分

V35的人工RGB观察仅在单一颜色达到16像素且不存在另一颜色时给出类别；两个不同XY位置后加一次固定0.9语义先验，重复次数不继续放大。若两类都获得支持（含同帧两色各达门槛），已有先验归零；几何按精确完整位姿只纳入首次观测，且有log-odds上限。这些做法已经解决了受控实验的重复注入问题。

它仍是**单个全局二配置假设**，没有真实多物体实例ID、连续位姿距离门、跟踪失配处理或自然视觉置信度。0.9和几何预测0.99都是声明的假设概率，不是自然数据校准结果。类别冲突永久保留到该belief重建，不会按新实例隔离或随时间解除；空白RGB不会撤销已建立的历史先验。不能把“未知帧”简单说成V35每帧强制回到0.5，也不能把自然场景的两个不同设备类别误当成同一个设备的矛盾。

## 建议的新接口合同（待实现／验证，非已完成自然接入）

- 观察事件保留`frame_id`、时间、图像内容hash、明确`rgb8/bgr8`、模型版本hash、原标签与taxonomy、bbox、未校准score或经独立校准的类别概率、拒识原因。文件名／目录／GT配置／终点成绩不是输入特征。类型与分数检查在进入belief前完成。
- 实例关联由观测侧框、配准深度、已测相机姿态及跟踪产生，记录关联不确定性；无法可靠关联时不要累积配置先验。真实全局设备位姿或场景隐藏ID不得用来配对。公共设备坐标系来源单列，不能偷偷由测试GT补入。
- 配置registry明确`natural_category`、`equipment_model_identity`或`artificial_marker`三种作用域，以及资料出处、版本、适用型号、参考系与反例。只有经过事先冻结关系验证的条目才提供`P(configuration|label)`；当前普通COCO标签不应有随意填入的左右侧条目。未支持、低置信或冲突，当前事件输出共同先验／零语义增量并记原因；历史先验保留、衰减或撤销规则需另外写明。
- 检测置信度不等于配置关系概率。后者若获得依据，应对类别不确定性边缘化，例如`P(h|image)=sum_c P(h|c)P(c|image)`，unknown部分回到共同先验；无校准数据时只能记录假设分布和scope，不声称概率校准。
- 去重至少区分同一帧／内容重放、同一设备同视点、不同视点和不同设备；同一帧换ID不应变成独立证据。只在足够分离且关联一致的观测后生成一次有上限的语义先验，几何证据仍可纠正它。对于新设备另立状态，不能以全局颜色桶累计。
- G/S使用相同可见几何、模板、控制预算与返航能力，G仍可通过几何判断补看。自然语义的增量必须来自有证据的类别条件信息，不能来自删掉G补看能力或读取未付费观测。

下一步先交付无World的接口反例检查：混合已知／未知类别对齐、RGB/BGR显式转换、整数schema冲突拒绝、低置信／未知／关系缺失回退、同帧换ID重放不增益、不同设备隔离、错误配置先验可被几何纠正。真实前端验证再按V37已冻结计划采集至少两类×五独立设备×两视点，并按设备实例隔离校准与留出。接口检查通过不等于自然语义创新已实测，目录标签成功也不等于大类泛化。

## 当前自定义标签全表

以下来自工作区映射源，不是本轮模型成功识别的类别列表。52个自定义标签中51个有COCO入口，pillow无入口；所有标签目前均缺经验证的type_A/type_B关系。

| 自定义ID | 标签 | COCO ID |
| --- | --- | --- |
| 0 | chair | 56 |
| 1 | couch | 57 |
| 2 | bed | 59 |
| 3 | dining table | 60 |
| 4 | bench | 13 |
| 5 | cup | 41 |
| 6 | bottle | 39 |
| 7 | bowl | 45 |
| 8 | wine glass | 40 |
| 9 | fork | 42 |
| 10 | knife | 43 |
| 11 | spoon | 44 |
| 12 | pillow | 无入口 |
| 13 | tv | 62 |
| 14 | laptop | 63 |
| 15 | mouse | 64 |
| 16 | keyboard | 66 |
| 17 | cell phone | 67 |
| 18 | remote | 65 |
| 19 | microwave | 68 |
| 20 | oven | 69 |
| 21 | toaster | 70 |
| 22 | refrigerator | 72 |
| 23 | sink | 71 |
| 24 | book | 73 |
| 25 | clock | 74 |
| 26 | vase | 75 |
| 27 | potted plant | 58 |
| 28 | toilet | 61 |
| 29 | scissors | 76 |
| 30 | teddy bear | 77 |
| 31 | hair drier | 78 |
| 32 | toothbrush | 79 |
| 33 | backpack | 24 |
| 34 | umbrella | 25 |
| 35 | handbag | 26 |
| 36 | tie | 27 |
| 37 | suitcase | 28 |
| 38 | banana | 46 |
| 39 | apple | 47 |
| 40 | sandwich | 48 |
| 41 | orange | 49 |
| 42 | broccoli | 50 |
| 43 | carrot | 51 |
| 44 | hot dog | 52 |
| 45 | pizza | 53 |
| 46 | donut | 54 |
| 47 | cake | 55 |
| 48 | person | 0 |
| 49 | cat | 15 |
| 50 | dog | 16 |
| 51 | bird | 14 |

## 审查源SHA256

本审查未更改以下文件。后续适配器应新增版本化文件，避免重写V35/V36冻结证据。hash只表明审查对象，不能替代代码正确性或自然验证。

| 文件 | SHA256 |
| --- | --- |
| `semantic_detector.py` | `8987fe0e792ffd3e99ac97b549537fc9c8efe76387639d54e4434fd6532b4a56` |
| `semantic/class_mapping.py` | `9c613cb752388bd0e3bb718fb72a33d0dfc6f6ed0b9cbda414872321c383ab3c` |
| `semantic/semantic_map.py` | `8326b58bb7d250fc5f7c9cc1d70f644519ea737f95fe216966697511dc77b4cb` |
| `semantic/semantic_map_updater.py` | `6105d403e7c5d4e07cf9a3e3069e243fe65bc8ec4f94291a044facd394c38519` |
| `semantic/semantic_identifier.py` | `a22ef1ea1f0d1e8c725f9943e42a102e455a39973cbe367ffbb920a75c8d5f1e` |
| `nso/semantic_taxonomy_v15.py` | `a55b4524acb4da64508ca82170a5386d0c1f14942940bffecbe3199348171a3b` |
| `nso/observation_belief_v35.py` | `09a6a4d9d0ac392b179a0da3cdd6a88fcb1c05e5bc3d20d393337d3a20551653` |
| `nso/clip_semantic_map.py` | `a67ffb22e244aeea027dce62a60d74458c1f6e56fcda5dbca985d86c152193bf` |
| `configs/virtual3d/v33_direction_scene_r1_20260917.json` | `55bc23b57de6e13472970eb1b6670a6e6040db74d3b61e155fe84f9fa08cd7c9` |
| `env/habitat/exploration_env.py` | `c4853013fa6663735a1793a4eac88abd2d57b8a523da4db42c8ce2e7afd66ae8` |
| `main.py` | `b688e895fbde74da11bbb036a5ed456514011fb234bed1139153140c151d2a23` |
