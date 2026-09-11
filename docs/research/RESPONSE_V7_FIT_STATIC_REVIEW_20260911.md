# V7 拟合与标定发布流程独立静态审阅

审阅日期：2026-09-11。审阅对象为下表哈希对应的最终版本。本轮只审阅算法、数据流与发布门禁，没有在本审阅中拟合真实数据、读取 C outcomes 或调整参数。另行授权的训练产物数值复算不属于本报告的静态通过范围。

## 结论与实际放行边界

最终代码中，8 个父 context 分组交叉验证、history 等权与中心化、正则选择后容量门槛、五个强对照独立重训、缺失语义回退、四个 C 同批预测封存、两参数绝对面积标定与排名隔离，未发现尚未修正的实质性静态错误。审阅期间发现的发布顺序和产物绑定缺口已经落实为代码门禁，见下文。

这项结论不等于允许立即执行 C，更不证明语义增益或四模块完整系统优势。父代理另外发现：旧 V7 执行器沿 prefix 封存路径执行时，没有在每个动作前用最新地图重新检查可行性；T1、T4 的两类场景共 4 条失败分支受到该问题影响。详见 [碰撞诊断](/root/NSO/eval_results/response_v7_collision_diagnosis_20260911/summary.json)。C 当前暂停；新执行守卫及其对所有既有成功路线的影响需要独立核查。拟合入口保留旧 96 条分支及失败实际成本，不能将该训练版本称为带在线守卫的执行策略。入口 metadata 的 `next_stage` 是流程提示，不覆盖当前研究层面的暂停决定。

## 数学与数据合同核查

| 核查项 | 最终实现及判断 |
| --- | --- |
| 父 context 分组 | 必须是 T0–T7 共 8 个父 context、每个 2 个 family，共 16 个 history、96 个候选。交叉验证每次留出同一父 context 的两个 history，没有把配对 family 拆到训练与验证两侧。 |
| history 等权与中心化 | 每个 history 单独对特征、两列响应减均值，再除以候选数平方根。目标为各 history 平均平方误差之和加 ridge 正则；没有按全局候选数或历史长度给长 history 更高权重。 |
| 正则与容量 | 固定 alpha 网格为 0.1、1、10；交叉验证只用主要面积率响应选择 alpha，数值并列取更强正则。选择完成后全 T 重拟合，再检查有效自由度 `sum(s²/(s²+alpha)) <= 8/2`。容量失败保留失败 bank 与诊断并停止，没有搜索更大正则救场。 |
| 五个学习对照 | G、O、S、N、G_capacity 使用同一批 history 和两列 target 分别拟合，均有独立系数及父 context CV。没有用 S 的系数冒充 G/O 独立基线。 |
| 语义控制 | S/X 的 common 与 object 列一致，X 只替换条件列并使用冻结 S 系数；M 必须实际提供且等于 G，使用已训练 G 预测。G_shared/O_shared 是另外明确标注的冻结 S 参数机制对照。 |
| 物理尺度与方向支撑 | 输入是固定 12 列、4 common + 4 object + 4 conditional，不学习标准差缩放。结构门槛要求每个 history 至少两个有标识 patch 的方向角色；类别 front/side 支撑来自该类实际有标识 patch 的 descriptor，不能用无标识背景补足。不扩大声称每个 history 具备全部四方向。 |
| target 与失败 | 显式 outcome loader 读取全部固定 T 分支；失败、短执行及未返程分支不删除、不置零。面积率以已记录实际付费动作数为分母；计划成本单独保留。这些数值不能解释成完整成功往返路线的结果。训练所选失败统计标注为 in-sample 诊断。 |
| 预测排序 | 直接使用预测的主要相对面积率，成本与候选 ID 只做确定性并列裁决。没有再次除以成本，也没有调用绝对面积 calibrator 改排名。 |
| OOD 回退 | 仅用 T 的 G 前八列确定 min/max 加固定 25% 范围（至少 1e-6）。任一候选任一列越界，整个 history 的 guarded 结果统一用冻结 N 预测；raw 结果另存。该规则不提供概率覆盖保证。 |
| C 标定 | 必须全部四个 C、八个 history 完整执行并通过独立回放。每个独立训练 scorer 只拟合两个参数：`planned_cost * max(0, a * relative_rate + b)`；以 history 等权绝对面积 MAE 求解，固定 a/b 边界。按排序后的响应枚举截断区域并求线性规划，覆盖固定边界内的分段目标。 |
| 标定边界 | 标定使用预先已知的计划成本，实际付费成本另列。C 拟合改善是 C 内样本诊断，不是 E 确认；不回流任何排名、不重新拟合条件响应模型、不自动打开 E。零实际面积和无定义相关系数保留相应标记。 |

第二响应列 F1 增量沿用面积率选出的 alpha；本实现没有为 F1 另行选择正则，也不是概率不确定度模型。配对信息 oracle 在读取 T target 后独立计算，标注为知道物理 family 的事后上界，不参与门槛、拟合、候选池或 context 选择。

## 发布门禁与审阅修复

1. **四 C 先预测再执行已经形成实际门禁。** `predict` 必须同时验证四个完整 prepared context、T bank 与源文件兼容性，并写出全部 raw/guarded 预测及选择。新 `execute-calibration` 拒绝已经存在的目标 run；先在独立 sibling guard 目录写 `armed.json`，绑定全体 C release、bank、prepared 清单、执行源码及“目标不存在”状态，再调用既有执行器。成功 receipt 绑定最终 run artifact 清单。`calibrate` 要求 guard 成功、两份 receipt 相同且 release/bank/run 哈希匹配。不能对已存在 C run 事后补发这种凭据。
2. **receipt 与独立回放的文件清单兼容。** run 内副本改为 `analysis/execution_receipt.json`，原始回放允许该分析目录；不向冻结根目录加入未登记文件。外部 guard receipt 与其独立 artifact 清单均保留。
3. **bank 必须是接受的训练产物。** `load_bank` 同时要求完成的 fit metadata、artifact 清单中的 `model_bank.json` 哈希与 metadata 声明哈希完全一致，并核对 12×2 系数、4+4 列划分、history 中心化、完整 8 个父 context 和每个模型容量通过。
4. **类别方向支撑不再借用背景。** front/side 检查按实际 marked patch 的 sign 选择 descriptor；动作成本与角色支撑也限制为这些 patch 有非零 descriptor 的候选。
5. **目标读取计数表述准确。** 普通 `Inputs.read` 无条件拒绝 `outcome.json` / `outcomes.json`，只有显式 loader 可构造目标数组。`family_outcome_tables_parsed_at_seal` 只统计这些表；回放状态 JSON 自身包含部分响应数值，加载这些状态不等于整个进程对一切响应字段完全盲读。最终代码明确这些回放响应字段不进入预测。
6. **源兼容检查前移。** C 预测封存前已经要求 prepared 源清单是 bank 训练采集源清单的哈希匹配子集；执行 guard 再核当前采集源码，不能等执行完成才发现换版本。

上述门禁针对可审计的本地工作流程，提供文件与执行次序证据；它不是抵御操作者有意重写所有文件的密码学时间戳服务。新版本在线执行守卫若改变采集依赖或动作策略，必须新建协议/版本并另行登记，不能使用旧源码完全相等条件将新旧执行假装为同一实验。

## 已查看的版本与验证范围

| 文件 | SHA-256 |
| --- | --- |
| [fit_response_v7.py](/root/NSO/scripts/fit_response_v7.py) | `4aafe24d9890e60ad7407506ba8ea89b064fca9791ada70556a34b69467e3178` |
| [conditional_response_v7.py](/root/NSO/nso/conditional_response_v7.py) | `4f9caefddc0fdfacf0713135eb88a46c5325294b13f4e99f835bfbc0db7e35b4` |
| [response_v7_pipeline.json](/root/NSO/configs/virtual3d/response_v7_pipeline.json) | `b9bd557cab25791ab5bdd9b358c739e35bb621c01c923bf5b730753312ba0af8` |
| [response_v7_validation.json](/root/NSO/configs/virtual3d/response_v7_validation.json) | `cebec20a18273a6c70b17281f2dc84d79af1ec281e0228d5751bc063abac1702` |

作者另报告最终入口的 8 项合成门禁测试通过；本审阅没有重复运行该测试，也没有将作者测试当作独立真实数据重拟合。本报告核对了最终源码字节及关键修复。执行重放、T0 特征数学独立复算和后续真实训练 bank 独立 SVD 检查是范围不同的证据，不能互相替代。本轮没有修改模型、runner、冻结采集源或任何原始实验结果。
