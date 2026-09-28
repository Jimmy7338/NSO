# Exposure V3 固定48槽主实验冻结helper设计

2026-09-28。**仅设计，未创建helper代码、真实协议、源码包或World。** 本阶段没有已批准的机制gate格式；为避免将含糊的“有效语义作用”擅自转化为自动通过条件，先提交下面的可审阅合同。主线程可据V3完整结果决定是否采用；未通过不生成配置。

## 保持已经确定的矩阵

- 六个未见布局：AISLE/CELL/LOOP各T0/T1。
- 两个paid budget：120、160。
- 一个固定噪声实现：92811；同1%深度噪声，第二种子92812的另48槽仍延后。
- 四方法：G、B、S、NBV；共48槽。沿用已有Ground main helper的完整四方法块及词典序运行顺序，交替预算并轮换布局，不按开发或主实验收益重排。
- 复用已冻结V3执行闭包与archive，controller/mapper/metric/数值环境完全不变；替换为六个已存在、route-independent reference。不生成新场景、不接触主实验结果进行参数选择。

## 新helper准备接受的输入

建议新文件 `scripts/freeze_article_exposure_main_20260928.py`，通过 `--exposure-protocol`、`--mechanism-gate`、`--output` 读取固定输入。不修改Ground main helper或V3科学闭包。输出使用新fresh路径，phase仍为main，不得把主实验伪装成新development以绕过cap。

应在任何输出前完成：

1. 逐项核验V3 source closure、archive、数值运行环境和固定controller配置；保留原V1与Ground协议/源的既有封存证明。
2. 固定读取三个声明队列的全部36槽：V1开发12、Ground12、Exposure12。ledger与原协议、global registry逐run对应，全部终态；reserved、未知、缺少本地entry或重复run立即拒绝。失败必须保留，不要求36次都合格。
3. 每个终态有对应的独立review manifest与review.json，逐文件哈希有效，review的输入manifest/result与terminal ledger绑定，且`status=reviewed`和`all_checks_passed=true`。正常质量合格与失败已正确记账分别处理：review通过不能把`attempt_failed`或`qualified=false`变成成功。新helper应使用各review schema明确读取，不能假设三版本的失败文件布局完全相同。
4. 主阶段累计registry尚无main尝试；冻结这一首次固定48矩阵时不得静默跳过以前启动过的heldout样本。若已经有main状态，应停止，由主线程采用原协议继续而非重新冻结。
5. 核验6个已有reference manifest及内容、同一资产manifest、四实例分母和固定评价设置。禁止以开发结果为依据替换或删选其中布局。
6. 读取root明确编写的gate文件并核验其证据哈希，gate的决策时间必须在全部36份终态/review之后。只读输入检查失败不写半成品协议。

## 需要root确认的机制gate含义

建议gate明确区分三个布尔判断：`technical_pipeline_valid`、`actionable_shared_semantics_with_budget`、`release_main48`。只有三者均明确为true且`decision_owner=root`时，行政helper才可以输出协议。这些是root作出的科学判断；脚本只能检查结构、来源和数值一致性，不能由一个文本布尔值自动证明因果关系。

机制证据至少应绑定同版本S/B run对、scene与seed、相关paid step及原始记录哈希，并解释共同观测前缀、各自自身证据与类别、shared rho引起的后验差异，和该差异怎样进入实际候选/诊断决策。随后说明该时刻的剩余预算、所选宏动作/返航开销，是否仍有可执行的选择。已验证类别/几何证据到达得太晚、仅改变后验但候选面积恒零、或只有不可执行候选的分数变化，均不能仅凭“非零”写成可行动机制。

仍须主线程决定：何种证据等级足以满足此门——同状态直接候选排序变化、诊断宏动作选择变化，或必须观察到实际已执行宏动作差异。原预声明容许“非退化候选或动作作用”，当前任务使用“可行动S/B因果作用”；二者并未规定一条可机械判定的具体阈值。**本设计不擅自将某个等级升级为必须，也不把任意非零差值自动视为通过。** 推荐root在gate中写明所采用等级与理由，并绑定可独立复核的记录，而不是由helper猜测。

gate不读取或比较正负端点质量来决定释放，不要求S优于B，不允许丢弃负例，也不根据某布局获胜而删改矩阵。root可以阅读完整端点结果进行诚实解释，但机制门本身的通过依据应独立列出，不将“正F1”填入必要条件。

## 输出与纯测试

获批准后，helper写入与V3相同65源闭包/原archive、48槽、main输出目录、所有36份终态/review/gate证据pins、固定矩阵、预算与资源安排。移除消融专属`paired_baseline_protocol`、逐slot旧paired id和旧Ground终态门描述，换为明确的V3 predecessor与main gate绑定；不新增运行源。生成协议不启动runner。reviewer对main的支持另用新文件，不能改当前V3消融reviewer。

纯测试应覆盖：精确48因子矩阵与完整方法块；缺任何终态/review拒绝；失败已review允许保留；reserved/未知/篡改SHA拒绝；missing/false/未知gate等级拒绝；不允许正端点字段代替机制证据；source/archive/runtime/reference不一致拒绝；已有main reservation拒绝；fresh输出不可覆盖。所有测试使用合成ledger/review/reference，不创建World。

资源与时间门应在V3完成后采用实测完整生命周期耗时与原始记录字节数估算。保留1 GiB reserve及scratch，预算超时只停新启动并收尾已有任务；如时间不足，应由root不释放本夜main或如实保留未启动槽位，不为了完成48而缩短预算、更换seed或只选有利方法。
