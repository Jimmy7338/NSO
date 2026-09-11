# V7执行器独立静态审查

2026-09-11。审查对象为`eval_response_v7.py`、`response_v7_pipeline.json`，并沿调用链阅读前缀准备、V7候选／特征、状态恢复、面积可见性、重建评价及独立回放实现。没有修改环境、规划器、执行器、配置或已有实验数据。只读取T0的元数据、封存清单、文件结构和输出字段名称，**没有检查T0效果数值、选择优劣或任何其它上下文的实验效果**；没有生成场景、未来观测或候选分支。

## 1. 结论与范围

**未发现需要停止当前固定T前缀构造或T真实分支采集的GT泄漏、免费位姿移动、面积重复计数或成本二次除法。** 该判断是静态代码审阅与输入封存完整性核对，不代替全部分支的独立传感器／轨迹／融合／指标回放，更不证明语义效果。

当前必须分开的状态是：完成采集、回放通过、训练支持／容量通过、校准／未看检查通过。执行器的`status=complete`只表示所有预定路线已尝试完毕；它可以同时含有`failure`分支，且初始化的`replayed=False`不表示已通过复核。

进入C之前还有全C预测封存与实际模型来源绑定的流程缺口；结果发布前须处理旧类别字段别名及稀疏J-AUC说明。已有T原始数据不应因此改写。

## 2. 本次核对的固定版本

|文件|SHA256|
|---|---|
|scripts/eval_response_v7.py|`61b479a2adef4576bc19b4b66e59f35eb35d9f97cfec4b80c4d1004155c9308b`|
|configs/virtual3d/response_v7_pipeline.json|`b9bd557cab25791ab5bdd9b358c739e35bb621c01c923bf5b730753312ba0af8`|
|scripts/prepare_response_v7.py|`f294e829363dadbf6a79a558e4ecad6fe1820f4206f024e32aa048604b77f6d5`|
|nso/response_candidates_v7.py|`942c31ee9bd3ce6c27c3e5ee0d7c4e2af8ba3025338c68f94dea3eb35ae2cc50`|
|nso/response_features_v7.py|`3317f74c906b62434de3b7b46051e8edbdd4d7214eb88d0eb0580d31a2129a91`|
|env/virtual3d_response_v7.py|`d06f6b01877968cfa388262bcab1acf231944b40f3ca8ab743a9fbfe33b07860`|
|scripts/replay_response_v7.py（仅静态阅读）|`cc61f4edbda7d57d00d21d939fadb536318051aabd472479f87aba9d56a3c1fa`|

T0封存目录为`eval_results/response_v7_training_20260911/T0`。本次只做了以下不涉及效果的文件检查：57个源码成员与metadata中的SHA256逐项相符，ZIP成员集合完全一致；以上执行／准备／候选／特征／世界及pipeline当前源码与该归档一致；100项封存的前缀／候选／特征输入与原prepared清单中的hash一致，当前字节也一致。没有用输出收益选择检查文件。

## 3. 因果隔离：已核实的正确行为

### 3.1 物理世界与规划输入

前缀准备先生成固定20动作，规划器只使用其RGBDFrame、PlanarScan和由它们融合出的mapper。`candidate_routes(mapper, obs)`无world、GT、family或outcome形参；`response_features(mapper, routes)`只读取同一实测地图、保留历史帧和预定候选位姿。其`ray_scene`用于mapper提取网格的旧面遮挡，不是GT场景。

几何簇的接受条件、排序、法向定向以及候选分配没有按类别值选取；观察类别只注释已接纳簇，O与S共享marker对象性。缺失与交换测试在前缀实际重放的mapper上检查，不仅在最终向量上代数替换。共同G/O/N/M/G_capacity与S前8列逐项不变，S/X末4列才改变符号。真实标记RGB允许跨物理家族不同；同一家族的解释干预不改变实际RGB-D。

配对前缀比较深度、雷达所有字段、位姿、内参、时间戳、marker支撑及非marker RGB；候选完整states/actions/cost必须逐项相同。地图几何hash排除实际marker颜色，是合理的；不能要求两物理家族的TSDF顶点颜色也相等。准备阶段的quality hash使用`quality_evidence`返回范围，未包括全部原始quality键及范围外体素；T0独立前缀审计另核对了所有quality体素非label字段。后续各父回放可补全该更强不变量，但这里没有发现实际不一致。

### 3.2 评分／特征封存先于本次评价

执行器第87—121行先核对prepared完成及结构门槛、原artifact hash和原源码hash，归档执行源码与pipeline，再硬链接已封存的两族输入，写`pre_outcome_seal.json`。第133行才构造ReconstructionEvaluator。旧评分、V7特征及C阶段已有预测均来自prepared，不在得到该分支GT响应后重算选择。

训练T当前封存的是用于后续拟合的特征和旧对照预测；它不是一个已训练V7模型的前瞻效果测试。训练数据当然可含执行后的响应作为监督，不能把训练top1表现当未看验证。

新父context_id属于T0..T7/C0..C3。outer_seed仍为751/752，不声称独立未知seed。执行器与归档生成器没有自动打开753/754或301/401系列。世界构造器内部生成GT网格用于物理模拟本身，不等于规划器读取GT；实际边界依赖上述传感器接口与数据流。

## 4. 真实动作、状态恢复与成本

- 每条路线先从原21帧重新融合mapper，并用`state_restore`恢复原20步后的position、heading、step、moves、collisions。恢复后重新sense/scan，所有传感器字段必须和归档末帧精确相等；这不是瞬移到候选终点。
- 候选以已知安全图为基础，障碍／未知统一膨胀机器人半径。初态足迹若未知或受阻，直接返回无候选，不清空初始格。朝向图的前进和左右转均为单位成本；回程在转置有向图恢复，最终含恢复初始朝向。
- `execute_branch`只按`world.step(action)`执行states[1:]对应的动作；初始观测只在prefix融合，不免费重新领取收益。每步保存实际RGB-D/scan，更新地图并检查真实状态与原route.states，付费长度用实际执行的动作条数。
- 路线成本在特征中只除一次。主监督`area_per_action=A_new/paid`也是面积率；冻结模型排名直接用有符号预测，`select`中的成本只用于相同预测的固定tie-break，不再次除成本。原20步是所有分支共有的沉没前缀成本，不包含在短路线效率分母中，不能把该效率解释为完整任务从零启动的效率。
- 在同父世界、同绝对付费步，sensor_seed及噪声调用相同；每个候选重置step20后执行，家族未进入sensor RNG。不同候选若长度不同，其共同相对步共享标准噪声，短路线之后并未继续采集免费观测。

## 5. 指标核对

### 5.1 唯一外表面积

V7世界使用原有`union_surface_from_boxes`生成唯一实体并集外边界。评价器在该网格上按三角形面积抽取原始32000点，然后按固定可达位置格点做可观察性筛选。执行器权重是：

`w = unique_union_mesh_area / 32000`

此处**没有**除以过滤后的reference点数重新归一化。每条路线的新增量为：

`A_new = sum_i w * 1[reference_i在任何付费路线帧可见 且不在原prefix并集中]`。

去程、端点和回程分别取布尔并集，再按时间阶段从prefix累计去重；程序断言三个阶段增量之和等于总增量。`seen`没有被跨候选累积修改，每个候选均对同一原prefix评价。没有按物理类、marker、模型潜在权重或语义重要性加权。

可见性使用真实camera pose、GT首交、图像投影边界及相邻像素深度有效性，而不是规划预测。它是带有限图像／采样误差的“新增几何可见面积”估计，**不是**真实重建准确面积。固定可达格点筛选也不能保证包括连续可达位姿能看到的所有细缝；当前面积属于该预定observable reference。两家族的union网格不同，虽然抽样seed固定，其reference点未逐物理公共面一一配对，仍有蒙特卡洛误差。这些是原评价设计的限制，不是重复面积或类别加权缺陷。

### 5.2 F1、覆盖与联合量

F1@2/5cm使用重建网格均匀样本到GT的距离计算precision，用固定GT reference到重建网格的距离计算recall；允许末态F1低于prefix。覆盖用`belief != -1`在GT可达域的比例，面积增量乘同一可达格数与0.2²m²。J为C×F1。GT可达域只用于评价分母，不用于候选安全图。

J-AUC在动作0、arrival、terminal三个检查点间线性插值，并在短路线完成后保持到48动作；它是**共同窗口的稀疏分支近似AUC**。已保存逐步coverage不代表逐步F1被实际评价，因此回放同公式通过也不能称其为每动作实测J积分，更不代表余下时间继续探索的完整ANS价值。

## 6. 实际存在的缺口及处理优先级

|项|证据与影响|本轮处理边界|
|---|---|---|
|C阶段全体预测封存缺少机器门禁|`eval_response_v7.py:97`只检查当前prepared的bank_sha非空；不要求C0..C3全部prepared已封存，更不检查统一模型来自完整且回放通过的T0..T7|**阻断打开任何C outcomes之前的流程条件**，不阻断当前T采集。由上层先建立全C输入／预测manifest，绑定统一bank及T来源后再执行，不需要回写T原数据|
|模型bank只有hash，没有随prepared归档的本体或明确路径|`prepare_response_v7.py`从调用参数读取bank，保存bank_sha与预测，但未复制bank文件；执行器不会核验bank拟合数据角色／容量状态|进入C前保存实际bank文件及训练来源manifest，确保当前预测可复算，且每个C都绑定相同hash。hash存在不等于已证实T-only训练|
|旧类名别名仅声明，尚未应用于原输出|pipeline写了simple_recall→storage、complex_recall→ventilation；原schema仍为simple_recall_02cm/05cm、complex_recall_02cm/05cm，执行器没执行重命名|全局面积/P/R/F1/J不受影响。保留冻结原输出，在新派生表显式映射带阈值完整字段；不可沿用“简单/复杂类”解释，不改旧指标数据|
|采集完成不是失败门槛通过|分支failure会终止该路线，仍记录实际面积／实际paid率；meta最终complete可以包含failure|完整路线训练／决策主汇总必须保留失败并单列门槛，不把失败的短分母当完整巡检收益，不静默删除失败。这里没有读取或判断本次是否实际出现failure|
|稀疏J-AUC语义未附在每个输出字段上|branch_joint_auc字段由np.interp生成|后续报告／数据字典明确标“0/到达/终点插值，终点保持到48”；不把它改成当前面积率门槛的替代目标|
|协议的固定旧支撑误差尚未输出|`ReconstructionEvaluator.evaluate`的surface_error来自当前重建网格样本；没有对相同prefix支撑的单独误差量|不能把当前surface_error变化称为旧面精度改善。若本阶段必须交付旧支撑误差，需在保持原指标的派生诊断中补充，标明尚未完成|

此外，pipeline中的若干特征／拟合字段是记录性合同，prepare并非全部从该JSON读取参数。当前已核对默认代码与合同一致，但后续跨父上下文聚合必须要求相同pipeline及依赖hash，不能只凭各目录均complete认定同一实验版本。

## 7. alias、存储与失败留档

T0的100项准备输入和执行输入确认为同inode硬链接。它们节约空间，但不是独立物理副本。当前执行器只读这些文件，新增的reference、prefix_mesh、prefix_map及candidate目录与prepared已有路径没有重叠；没有发现当前写操作覆盖共享前缀或特征的alias错误。执行末尾同时核对output seal与prepared输入hash，能发现共享输入被改写。以后做清理／派生结果时必须仍把这些输入当不可变，不能原地改色、改语义或重存候选。

每步前检查400MiB余量；触发则异常退出并保留已有raw，不会跳到后面的好候选继续收集。动作记录和outcome在分支结束后一次写入，所以中途磁盘异常可能留下raw而无该分支完整actions/outcome；其状态只能标未完成，不能当零收益或完整样本。输出目录创建与硬链接阶段在主try之前，若那里I/O失败，metadata可能仍是running而不是failed_execution；这是恢复／状态表达缺口，发生于GT评价之前，不构成已完成数据的指标错误。

当前未执行任何网格删除。pipeline的“通过独立回放后可删网格”只是未来存储规则，不能凭status complete清理唯一raw。每父上下文的输入硬链接依赖原prepared目录作为可追溯来源；归档迁移时需保留原hash与解析路径，或建立有hash的替代来源清单。

## 8. 独立回放边界及剩余核验

已静态阅读`replay_response_v7.py`：在隔离进程使用源码snapshot，手动执行固定前缀；重建原reference，检查逐步真实轨迹、RGB-D/scan、prefix/arrival/terminal网格、每步coverage、加权面积及F1/J-AUC；验证prepared→执行seal和原artifact hash，并区分max_branches局部复核与full。C时还要求预测和choices文件在seal中。本文没有启动该回放，也不引用其运行结果。

该回放的特征检查调用同一冻结`response_features`实现，明确不是第二份独立描述量数学实现；旧面质量等共同依赖也来自归档。它能证明复现及输入隔离，不能单独排除共同算法性偏差。全父上下文还需检查真实执行完整性、失败率、共同候选充分性、所有T支撑／有效自由度、C全体封存、强G/O/N及容量对照、标签交换与缺失、独立校准与后续未开检查。原有48动作／12候选协议未完成项不会因新6候选流程而自动完成。

本次不会建议按T0收益修改设备几何、marker、候选或采样点，也没有判断当前模型能否通过语义门槛。

## 9. 固定前缀图解（审查后的独立交付）

审查完成后另生成[PNG](/root/NSO/docs/research/figures/response_v7_paired_prefix.png)与[PDF](/root/NSO/docs/research/figures/response_v7_paired_prefix.pdf)：预定T0、step20，两列采用相同原始观察时刻；顶行为归档RGB，次行为归档深度，底行为事后GT外表面在固定同角度下的透明图解。真值仅为此图裁剪使用，不传给规划器；没有读取候选效果或按收益挑视角。输入／脚本／输出hash见同目录response_v7_paired_prefix.json。图示证明受控输入条件，未评价隐藏结构的实际路线响应，更不证明自然语义或语义优势。
