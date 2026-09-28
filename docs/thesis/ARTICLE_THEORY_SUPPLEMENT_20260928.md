# 语义与几何联合主动建图：理论补充、实现边界与原论文核验

日期：2026-09-28。用途：供毕业论文方法章、讨论和相关工作并稿；本文不替换既有章节，不新增实测性能结论。式中的分析算例均为人工设定，不能填入实验结果表。文献核验日期同上；核验来源与算术检查见同目录 `ARTICLE_THEORY_REFERENCES_20260928.json`。

## 1. 研究问题、共同目标与信息权限

研究对象是工业设施布局调整后的主动建档：一次任务内设施保持静止，机器人在有限运动和观测预算下记录可达区域及设施表面。类别知识可提示设施的开口、凹入结构或服务侧，但实际构型仍需用观测确认。本文讨论主动观测与地图质量，不涉及移动障碍物的预测和动态避障。

令已支付代价的历史为 \(\mathcal H_t=(o_0,a_0,o_1,\ldots,a_{t-1},o_t)\)，公开先验为 \(\mathcal P\)，实际融合地图为 \(M_t=\operatorname{Fuse}(\mathcal H_t)\)。策略只能依赖它所获准使用的历史和公开先验。参考几何 \(G^\star\) 仅在轨迹与地图输出封存后进入独立评价。真实任务目标可以写成

\[
\max_{\pi}\;\mathbb E\!\left[J_\tau(M_T,G^\star)\right],\qquad
J_\tau=C_{\rm map}\,F_{1,\tau},\qquad
\sum_{t=0}^{T-1}c(a_t)\le B,\quad s_T=s_{\rm home}.
\tag{1}
\]

其中 \(C_{\rm map}\in[0,1]\) 是按共同参考区域归一化的二维覆盖，\(F_{1,\tau}\in[0,1]\) 是给定距离容差 \(\tau\) 下的表面精确率与召回率的调和平均。乘积避免仅凭区域覆盖或单一表面指标取得高联合分，但不能代替两个分量、返航状态和资源消耗的分别报告。同一参考面积下，使用覆盖面积或覆盖比例只相差常数；跨布局比较应保留归一化口径和布局分层。

式 (1) 是评价目标，不表示在线控制器能读取真实终点 F1。V35 双构型 CPU 实现比较的是公开模板上的理想地面曝光与潜在外竖面曝光的乘积 \(\widetilde J_h\)。后续有限宏动作实现比较的是预测新增表面与发现收益相对于完整计划代价的比值。这两种在线代理应分别注明版本，不能统称为“直接优化实测 F1”。实测深度可能缺失或有误，曝光面积也不等于融合后的精确率、召回率。因此，代理提高到式 (1) 提高之间仍需要共同后端上的实测验证。

**表 1　用于机制比较的信息契约。**

| 项目 | 几何对照 G | 语义方法 S | 评价器 |
|---|---|---|---|
| 已执行动作的几何观测、位姿与动作代价 | 可用 | 可用 | 可读取封存记录 |
| 相同公开模板、相同合法动作和候选生成规则 | 可用 | 可用 | 用于协议核对 |
| 已观测类别及类别—构型条件先验 | 不用于构型决策 | 按实验定义启用 | 用于分层统计 |
| 根据实际几何缺失或残差选择补看位置 | 允许 | 允许 | 检查依据是否来自已取得观测 |
| 未执行动作的真实传感结果、实际隐含构型身份 | 禁止 | 禁止 | 仅终点评价需要时使用 |
| 用模板表面填充实测 TSDF | 禁止 | 禁止 | 检查实际融合来源 |

表 1 适用于类别条件机制隔离。若某个外部方法以语义分割为必要输入，应单列其输入权限，不能将它改名为完全无语义方法。共同候选规则也不要求不同历史下产生完全相同的候选列表；在相同历史与相同随机种子下，信息消融不应暗中改变几何侧可获得的合法候选或传感代价。

没有类别信息的机器人仍能从缺口、遮挡边界和残差判断某些地方值得补看。可验证的语义作用是：在几何历史尚不能区分候选构型、而不同构型要求不同观察方向时，类别先验可能提前改变行动；并非禁止几何方法自主诊断。一次行动改变后的几何历史不同属于策略后果，不能据此要求整个后续轨迹始终一致。

预算按已执行的原子动作计费，包括转向、平移和显式观测。V35 原协议的共享前缀也计入总预算；后续宏动作的 `move + observe` 中显式观测代价为 1。仿真器在运动途中提供的观测必须对所有比较方法采用相同规则。规划 CPU 时间、峰值内存和动作预算应分别记录，动作可行不等于满足硬实时计算时限。

## 2. 双构型、两路线：语义收益的条件与诊断阈值

### 2.1 简化假设与直接决策

以下是说明机制的有限决策模型，不是实际 TSDF 的误差定理。

1. 隐含构型 \(H\in\{0,1\}\) 等概率；共享的初始几何历史对两构型具有相同分布。
2. 两条完整观察路线 \(a_0,a_1\) 均可执行并返回起点。路线 \(a_h\) 与实际构型匹配时效用为 \(U_+\)，不匹配时为 \(U_-\)，且 \(\Delta=U_+-U_->0\)。效用已包含同一评价口径，暂不另外改变二维覆盖。
3. 类别线索 \(S\) 的构型指示具有对称可靠性 \(\Pr(S=h\mid H=h)=q\)，\(q\in[1/2,1]\)。这里的 \(q\) 包含类别识别及类别—构型关联的有效性，不能用程序中设置的 0.9 先验强度代替其测量值。
4. 可选几何诊断 \(Y\) 为对称二元通道，正确率 \(r\in[1/2,1]\)，且给定 \(H\) 后与 \(S\) 条件独立。诊断后两条路线仍然预算可行。诊断消耗造成的共同机会损失为 \(\kappa\ge0\)，单位与效用一致。

假设 4 将诊断运动、观测和机会成本压缩为一个效用损失。若实际代价使某条后续路线不可行，应重新枚举可行动作，不能继续套用下面的闭式公式；也不能直接把 3 个动作写成 \(\kappa=3\) 与归一化 F1 相减。

没有额外诊断时，几何方法在平衡构型上最优期望效用为 \(U_-+\Delta/2\)，按类别线索选向为 \(U_-+q\Delta\)，因此

\[
V_{\rm semantic,direct}-V_{\rm geometry,direct}
=(q-1/2)\Delta.
\tag{2}
\]

式 (2) 同时要求线索有效且观察方向具有真实后果。如果两条路线都足以看到全部有效表面，\(\Delta\) 接近零；如果类别与构型独立，\(q=1/2\)。这两种情况均不产生该机制的期望优势。几何前缀已经辨清构型时，假设 1 不再成立，语义的剩余决策价值也应下降。

### 2.2 允许两个方法共同进行诊断

对任意诊断前构型权重 \(p=\Pr(H=0)\)，直接选向的正确概率为 \(m=\max(p,1-p)\)。取得二元诊断后，按各结果选择后验最优方向的期望正确概率是

\[
A(p,r)=\max\{pr,(1-p)(1-r)\}
       +\max\{p(1-r),(1-p)r\}
       =\max\{m,r\}.
\tag{3}
\]

前两个最大值分别对应两个诊断结果。最后一个等号依赖二构型、对称通道和相同误判损失；多构型、非对称代价或连续质量函数均不必满足此简化。因此，单次诊断的净决策价值为

\[
\operatorname{VOI}_{\rm net}(p)
=\Delta\max(0,r-m)-\kappa.
\tag{4}
\]

在前述对称类别模型下，G 与 S 分别从 \(m=1/2\) 和 \(m=q\) 出发，二者都允许选择“直接观察”或“先诊断再观察”，得到

\[
V_G^*=U_-+\max\{\Delta/2,r\Delta-\kappa\},\qquad
V_S^*=U_-+\max\{q\Delta,\max(q,r)\Delta-\kappa\}.
\tag{5}
\]

G 支付诊断的严格改进条件为 \(\kappa<(r-1/2)\Delta\)；S 支付诊断的严格改进条件为 \(\kappa<(r-q)_+\Delta\)。语义优势可以来自节省不再值得支付的诊断，而不必来自更高的最终构型识别率。若诊断非常便宜且 \(r\ge q\)，两者最优效用可能相同。

**表 2　纯解析例子：\(U_-=0.50,U_+=0.90,q=0.80,r=0.95\)。无实测数据参与。**

| 诊断机会损失 \(\kappa\) | G 的最优选择及效用 | S 的最优选择及效用 | 期望差 \(V_S^*-V_G^*\) |
|---|---|---|---|
| 0.02 | 诊断，0.86 | 诊断，0.86 | 0.00 |
| 0.08 | 诊断，0.80 | 直接按类别选向，0.82 | 0.02 |
| 0.20 | 直接选向，0.70 | 直接按类别选向，0.82 | 0.12 |

此例可用于说明为何“可靠类别、遮挡导致方向差异、诊断或绕行消耗明显”构成合理的研究场景。它同时给出便宜诊断、无方向差异及失效类别这三个边界条件。场景应先由工业观察任务和遮挡机制定义，再统一运行全部对照；不能从表 2 推断任一实际场景具有同样的增益。

### 2.3 与当前信息价值计算的对应

一般有限动作版本采用

\[
\operatorname{EVI}(d)=
\sum_y P(y\mid d,\mathcal H_t)
  \max_{a\in\mathcal A_d}\mathbb E[\widetilde U(a,H)\mid y,d,\mathcal H_t]
-\max_{a\in\mathcal A_d}\mathbb E[\widetilde U(a,H)\mid\mathcal H_t].
\tag{6}
\]

在一致的概率模型、相同的终端动作集 \(\mathcal A_d\) 和相同效用下，式 (6) 非负：忽略观测并保留原动作也是观测后的可选决策，再利用全期望公式即可得到不等式。这是标准决策论性质，不是本文的新理论。若诊断之后的预算比直接行动少，实际可行集合已改变；未扣成本的 EVI 非负不能证明诊断路线优于直接路线。

`discrete_observation_evi` 在每个预测结果下先跨实例求共同动作效用的和，再对动作取最大值；不能对每个物体各取最大值后相加，因为这些动作不一定能共同执行。该函数本身不处理运动与观测代价。`paid_diagnostic_value` 在其外部将最终视角代理收益除以“去诊断点、观测、去后续点、观测、完整返航”的总代价。它没有把两个视角的预测表面积直接相加，以免缺少集合重叠信息时重复计分。此比值评分是实现选择，不能称为式 (1) 的精确解。

## 3. 几何反馈何时能够覆盖错误语义先验

### 3.1 对数支持度阈值

先分析 V35 二构型实现。令错误类别先验支持构型 0，\(L_s=\operatorname{logit}(q_s)>0\)，\(q_s\in(1/2,1)\)，实际应支持构型 1；此处 \(q_s\) 表示算法设置的先验强度，与第 2 节的真实线索可靠性 \(q\) 不同。几何累计支持度初值为 \(L_{g,0}\in[-G,G]\)，更新为

\[
L_{g,t}=\operatorname{clip}_{[-G,G]}(L_{g,t-1}+\delta_t),\qquad
w_{1,t}=\sigma[-(L_s+L_{g,t})].
\tag{7}
\]

要求构型 1 的支持度至少为 \(\eta>1/2\)，等价于

\[
L_{g,t}\le -L_s-\operatorname{logit}(\eta).
\tag{8}
\]

若每个**合格且实际纳入**的反证满足 \(\delta_t\le-\epsilon<0\)，且在达到阈值前截断下界未阻挡更新，则一个充分数量为

\[
n\ge\max\left\{0,
\left\lceil\frac{L_{g,0}+\operatorname{logit}(q_s)
                         +\operatorname{logit}(\eta)}{\epsilon}\right\rceil\right\}.
\tag{9}
\]

截断允许达到该支持度的必要条件为 \(G\ge\operatorname{logit}(q_s)+\operatorname{logit}(\eta)\)。对于非单调证据序列，应使用实际累计状态检验式 (8)，不能忽略正向反复或直接对被截断前的原始残差求和。

V35 取 \(q_s=0.9,G=24\)，当 \(L_{g,0}=0\)、一个合格反证达到 \(\delta=-6\) 时，有 \(w_1\approx0.9782\)，超过 \(\eta=0.95\)。这是算法刻度的代数检验，不说明实测一次观测必定产生 \(-6\)，更不说明其真实正确概率为 97.82%。实际使用的是截断鲁棒残差形成的广义证据，并未校准为传感器的对数似然比。若反证不可观测、被门限拒绝、实例关联错误，或多次有偏测量被重复计数，式 (9) 的假设均可能不成立。

### 3.2 重复相关观测与证据资格

观测能够用于深度融合，不代表它应当作为新的独立类别或构型证据。实现应分别记录：测量包是否重复、实例关联是否可信、观察方向或有效残差是否提供新增诊断内容、几何支持点是否写入存储，以及信念实际是否更新。数据结构新增点数不能单独充当信息新颖性的定义。

V35 使用首次精确姿态才累计几何证据，类别先验在获得支持后只设置一次，类别冲突时清零。这是抑制明显重复的工程规则；相邻姿态仍可高度相关，所以不能据此宣称观测已经条件独立。后续局部实例实现还使用新点比例、最少新点数和证据帧数上限，其限制与 V35 不同，论文应分别标注实现版本。

**2026-09-28 的局部实现诊断。** `ObservedInstancesLocal` 先计算新颖支持与证据帧资格，再尝试写入支持点；若 `added == 0`，代码会将 `geometry_feedback_eligible` 置为 `False`。默认支持点上限为 4096。因此，支持集合已满时，即使观测通过实例接收、实际残差具有诊断性，也可能因无法新增存储点而不进入信念。这可解释一部分“取得了新观测，错误先验却未被纠正”的实现现象。这里没有证明其解释全部实验差异，也没有宣称修正已验证成功。

待验证的修改方向是把有限几何存储与证据资格账本分开：保留重复包拒绝、关联质量检查和相关重复抑制，但不让支持点容量决定一条新残差是否有资格参与推断。其必要检查应包括：满容量下新的有效视角是否仍能更新；重放同一包是否仍不重复更新；只有重复或无效深度时是否保持不变；更新是否来自相同的已付费输入。新增资格规则必须在新版本单独重跑验证，既有冻结结果不能由修正文稿替代。

### 3.3 跨实例可靠性共享的定位

后续 CPU 原型以两种先验模型 \(k\in\{\text{geometry},\text{class}\}\) 表达类别—构型关系是否值得依赖。对同类实例 \(j\)，其累计几何证据向量 \(L_j(h)\) 形成消息

\[
m_j(k)=\sum_h q_k(h\mid c)\exp L_j(h),\qquad
w_{-i}(k)\propto w_0(k)\prod_{j\ne i,\,c_j=c_i}m_j(k).
\tag{10}
\]

实例 \(i\) 使用排除自身后的混合先验，再恰好一次地施加自己的累计证据：

\[
b_i(h)\propto\left[\sum_k w_{-i}(k)q_k(h\mid c_i)\right]\exp L_i(h).
\tag{11}
\]

累计消息采用替换而非每帧重复追加，可避免同一份累计证据被多次相乘。式 (10)–(11) 是标准层次贝叶斯分解，不是新的推断算法；当前残差为广义证据，可靠性权重也不是经校准的真实类别准确率。跨实例共享还依赖实例关联和类别关系可迁移，错误关联或同类异构都可能带来负迁移。单个实例且没有同类历史时，该模型只退化为初始混合先验，不能仅靠重命名推断产生额外信息。

## 4. 宏动作返航可行性：可证明部分与边界

令 \(\mathcal G=(V,E)\) 为本轮规划期间固定的合法姿态图，边包含朝向状态与实际计费的原子动作，代价为正。设 \(d(s,s_{\rm home})\) 为图上最短返航代价；若不可达，则取无穷。定义剩余预算 \(b\) 下的可行性不变量

\[
\mathcal I(s,b):\quad d(s,s_{\rm home})\le b.
\tag{12}
\]

**命题：** 假设初始状态满足式 (12)，实际执行的边合法、代价与模型一致，且选取从 \(s\) 到 \(u\) 的宏动作 \(\sigma\) 时核验

\[
c(\sigma)+d(u,s_{\rm home})\le b,
\tag{13}
\]

则该宏动作的每个已执行前缀之后都仍存在预算内的返航路径。

**证明：** 执行前缀后位于 \(s_j\)，已花费 \(c_j\)。宏动作剩余后缀再接 \(u\) 到起点的最短路径，构成一条从 \(s_j\) 返航的可行路径，因此

\[
d(s_j,s_{\rm home})
\le c(\sigma)-c_j+d(u,s_{\rm home})
\le b-c_j.
\tag{14}
\]

宏动作末端尤其满足 \(\mathcal I(u,b-c(\sigma))\)。在每次重新规划时应用相同检查，可对宏动作次数归纳。显式观测虽然不改变位姿，也必须计入 \(c_j\)。如果没有可行的新观察宏动作，则执行保留的返航路径，可在预算内回到完整起始姿态。证毕。

上述命题只证明模型条件下“返航仍然可行”。如果控制器异常退出而不执行返航，或者真实碰撞、未建模代价、路径失效发生，命题不推出任务已成功返回。在线观测导致安全图删边时，也应重新验证返航路径；旧图上的归纳结论不能无条件跨越图变化。V35 另有两个模板的覆盖资格条件，返航可行不代表覆盖资格必然达成。论文可以报告预算不变量及其检查记录，不能将其扩大为现实场景无碰撞保证。

## 5. 有限候选与一次诊断前瞻的复杂度

### 5.1 V35 双模板动态规划

记姿态图规模为 \(N=|V|\)、\(E=|E|\)，最大出度为 \(d\)，两模板曝光位集长度为 \(m_0,m_1\)，剩余预算为 \(B\)。一次 `select` 内只有初始权重及一次诊断后的两个固定权重分支。忽略不可达状态剪枝，记忆化状态数有保守上界

\[
S\le 3N(B+1)2^{m_0+m_1}.
\tag{15}
\]

若一次位集更新和终端代理评价成本记为 \(T_{\rm mask}\)、\(T_{\rm terminal}\)，主要时间量级为 \(O(S(dT_{\rm mask}+T_{\rm terminal}))\)，加上反向 BFS 的 \(O(N+E)\)。位集不能在长度变化时无条件当作单位操作。状态存储随实际可达 \(S\) 及位集长度增长；式 (15) 对曝光位数呈指数，因此不能把整个算法称为多项式时间规划器。

当前状态上限为 1,500,000，时间阈值为 20 s，每 1024 个新增状态检查一次时间并在选择结束再次核验。它限制求解资源，但不是严格的实时最坏耗时保证。成功完成的有限代理动态规划与遭遇计算上限的失败应分别统计，不能仅报告成功任务的耗时。

### 5.2 后续有限宏动作控制器

设候选姿态数为 \(C\)，已观测实例数为 \(I\)，每实例最多 \(K\) 个结构假设，单个诊断有 \(Z\) 种离散结果，可用诊断点—实例组合数为 \(D\le IC\)。一个固定诊断的后续动作至多 \(C+1\) 个，含直接返航选项。若各分支的实例后验已经取得，跨实例共同动作评分为 \(O(ZICK)\)。将单实例后验计算成本记为 \(T_b(I,K)\)，全体诊断的成本可保守写成

\[
T_{\rm select}=
T_{\rm forecast}
+O\!\left(C^2T_{\rm route}
+D(Z+1)\{I T_b(I,K)+ICK\}\right),
\tag{16}
\]

其中 \(T_{\rm forecast}\) 包括几何预测与观测通道构造，\(T_{\rm route}\) 是一次精确图路线查询成本。当前选择函数只在本轮选择内缓存路线结果，独特源—目标对数可达 \(O(C^2)\)；不能因此假设已经为每个起点只做一次全图 BFS。当前 `PublicPrimitiveGraphV41.route` 对单位代价边仍使用堆式最短路，可取 \(T_{\rm route}=O((N+E)\log N)\)，并另计路线输出长度。若后续替换为单位边 BFS，才可将该项改为 \(O(N+E)\)。

当前共享先验的 `posterior` 会遍历同类实例；`instance_ids` 也会排序。因此在其直接实现下，可用 \(T_b(I,K)=O(IK+I\log I)\) 作保守描述，跨实例评分可能出现 \(I^2\) 项。若未来缓存充分统计量，应以新实现重新给出复杂度，不应把尚未实现的缓存写入当前结果。日志还保存预测分支和候选记录，实际内存不只包括当前最优动作。

一次前瞻的收益是使诊断的后续行动后果可以显式枚举；它没有搜索任意长度的信息树，也不是完整连续 POMDP 的最优解。独立限制候选数、每实例预测视角数及结果分支数，可控制单轮开销，但有可能遗漏有价值的视角，需要用实际耗时与候选覆盖率验证取舍。

### 5.3 不能直接继承的近似最优保证

覆盖与表面质量的乘积不自动满足子模性。一个简单反例是：动作 \(x\) 只使 \(C\) 从 0 变为 1，动作 \(y\) 只使 \(F\) 从 0 变为 1。则 \(J(\varnothing)=J(\{x\})=J(\{y\})=0\)，而 \(J(\{x,y\})=1\)。在已经选择 \(x\) 后加入 \(y\) 的边际收益比在空集加入 \(y\) 更大，违背收益递减。因此，未证明本问题满足相应条件时，不能从自适应子模规划文献直接借用贪心近似保证。该文献本身明确将保证建立在相应性质上；原作者修订版还对一项覆盖定理作了限定。[Golovin 与 Krause 原文及修订说明](https://arxiv.org/abs/1003.3967)

## 6. 原论文核验与研究定位

下面仅列本次直接核验的原论文、原作者页面或出版方页面。方法描述采用已读取版本；表中没有把公共 CPU 机制改写称为原完整系统复现。

| 文献 | 本次可核验的内容 | 与本文比较时的边界 |
|---|---|---|
| SWAP，Dharmadhikari 与 Alexis，ICRA 2023 | 原文 §III、§IV 将体积探索、语义网格补洞和满足分辨率／视角要求的表面检查组合；采用语义分割点云，未假设设施的形状、尺度与位置先验 | 其完整未知环境系统不同于本文公开模板任务。共同 CPU 实现仅能称 SWAP 启发的机制对照；不能据此宣称优于其原论文完整系统 |
| SPP，Dharmadhikari 与 Alexis，IEEE Transactions on Field Robotics，2025 | 原文用语义场景图发现空间重复模式并预测未观察区域，提供两种检查规划器；作者页面核实 DOI | “利用语义预测结构并规划”已有先行工作；本文需说明其类别条件结构权重、错误先验反馈与有限预算决策的具体差别 |
| VISTA，Nagami 等，本文核验 arXiv v1，2025 | §III–IV 使用可靠位姿与 RGB-D、在线语义 3DGS、体素观察方向记录及语义查询，引导滚动规划 | 其视角多样性与语义相关性组合不是本文原创。VISTA 启发的共同 CPU 几何—语义打分不含其原始在线语义 3DGS 完整系统 |
| Kaelbling、Littman 与 Cassandra，Artificial Intelligence，1998 | 原作者 PDF 的信念状态与部分可观测决策框架说明历史信息如何影响后续行动 | 贝叶斯信念、信息改变决策以及有限前瞻不是本文提出的新原理 |
| Hollinger 与 Sukhatme，IJRR，2014 | 出版方摘要与元数据核实其在资源预算下进行机器人信息采集的规划目标 | 本次未核读其全部证明，不把该文的算法性质转移到本项目 |
| Golovin 与 Krause，自适应子模性论文及修订版 | 原作者 arXiv 的性质条件与修订说明可核验 | 本项目没有据此获得子模性或贪心近似保证 |

来源对应如下，均于 **2026-09-28** 核验：

- **SWAP**：M. Dharmadhikari and K. Alexis, *Semantics-aware Exploration and Inspection Path Planning*. [原论文 HTML](https://arxiv.org/html/2303.07236v1)，[原论文元数据](https://arxiv.org/abs/2303.07236)。ICRA 2023 接收信息由作者提交页注明。
- **SPP**：M. Dharmadhikari and K. Alexis, *Semantics-aware Predictive Inspection Path Planning*. [原论文 HTML](https://arxiv.org/html/2506.06560v1)，[作者实验室出版目录](https://www.autonomousrobotslab.com/publications.html)，[出版 DOI](https://doi.org/10.1109/TFR.2025.3578402)。正式刊物及 DOI 经作者目录核实；本文不补填未经本次核验的卷页信息。
- **VISTA**：K. Nagami, T. Chen, J. Yu, O. Shorinwa, M. Adang, C. Dougherty, E. Cristofalo, and M. Schwager, *VISTA: Open-Vocabulary, Task-Relevant Robot Exploration with Online Semantic Gaussian Splatting*. [原论文 HTML v1](https://arxiv.org/html/2507.01125v1)，[作者提交页](https://arxiv.org/abs/2507.01125)。这里仅标注已核验的 2025 年预印本版本，不将其年份当作正式刊期判断。
- **POMDP**：L. P. Kaelbling, M. L. Littman, and A. R. Cassandra, *Planning and Acting in Partially Observable Stochastic Domains*, Artificial Intelligence, 101, 99–134, 1998。[MIT 原作者 PDF](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf)。
- **预算信息规划**：G. A. Hollinger and G. S. Sukhatme, *Sampling-based Robotic Information Gathering Algorithms*, The International Journal of Robotics Research, 2014。[出版方摘要与元数据](https://journals.sagepub.com/doi/10.1177/0278364914533443)。仅将实际核读的摘要内容用于本文定位。
- **自适应子模性**：D. Golovin and A. Krause, *Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization*。[原作者提交页，含 2017 年 v5 修订说明](https://arxiv.org/abs/1003.3967)，[修订全文](https://arxiv.org/html/1003.3967v5)。

可并入相关工作或贡献界定的中文段落：

> 本文采用已有的信念更新与有限信息价值框架，将类别条件的结构假设、实测几何纠错和带返航预算的观察决策连接起来。研究重点是工业设施建档中类别线索何时能够减少诊断代价，以及错误线索何时能够被已付费的观测覆盖。四模块接口组织该闭环，其名称不替代实现内容。本文通过共同传感与重建后端上的对照、信息消融和机制轨迹评价这一具体实现；贝叶斯推断、语义检查及信息采集的一般原理均归于既有工作。

## 7. 可直接进入实验设计的可检验预期

本节列出待验证预期，不将其写成已有结论。

| 因素 | 由简化模型得到的预期 | 应记录的证据 |
|---|---|---|
| 类别—构型关联变弱 | 其他条件一致时，提前选向优势下降；若反向关联未被识别，还可能出现负效应 | 类别关联分层、首次分歧动作、各构型配对终点结果 |
| 遮挡及方向互补性减弱 | \(\Delta\) 减小，语义信息的行动价值下降 | 相同视角的实际表面结果和同尺度遮挡示意 |
| 有用几何诊断变便宜 | G 更容易通过共同诊断能力弥补初始信息差 | 诊断行程、观测动作数、剩余预算及后续路线 |
| 错误类别先验且存在实测反证 | 只有反证确实取得、被接收并纳入信念时才可能纠错 | 原始残差、资格原因、实际信念增量、方向改变时间 |
| 支持点容量饱和 | 存储门限与反馈资格耦合可能压制后续纠错 | 满容量前后同一输入的资格记录；新旧实现独立版本比较 |
| 长预算或几何已充分辨识 | 语义边际优势可能缩小 | 全预算曲线和覆盖、F1 两个分量，而非单一最优终点 |

新增试验应先固定场景集合、预算、种子和成功判据，再统一比较完整方法与对照。对每个预设条件保留全部轨迹，包括超时、未返航和负增益。解析表 2 用来提出预期，不能作为支持真实系统效果的实验样本。现有冻结证据只支持其原始版本及测试条件；局部证据资格修正、新的共享可靠性设计和更广场景需分别取得新结果。

## 8. English text for manuscript integration

### Theory and scope paragraph

We consider budgeted active mapping of industrial facilities that remain static during each inspection episode. Semantic observations provide a prior over possible structural configurations, whereas measured geometry can revise this prior before subsequent view selection. An analytical two-configuration example shows that a semantic cue is useful when it predicts a consequential route choice and can avoid sufficiently costly geometric diagnosis. Both semantic and geometry-only policies retain access to the same diagnostic actions. This example characterizes a conditional decision advantage; it does not guarantee an improvement in measured TSDF accuracy. The online exposure and surface-gain scores are planning proxies, while map coverage and surface F1 are evaluated independently from fused measurements. We further derive the accumulated counter-evidence required to overturn an incorrect prior and a return-feasibility invariant for paid macro-actions on a fixed legal pose graph. These results require eligible, nonduplicated evidence and correctly modeled action costs. Bayesian belief updates and expected value of information are established tools rather than claimed theoretical innovations. The empirical contribution must therefore be assessed through controlled information ablations, observed decision changes, reconstruction outcomes, and clearly reported failure cases under a common sensing and reconstruction pipeline.

### Implementation limitation paragraph

The current local-instance implementation couples feedback eligibility to insertion of new support points. Once the support buffer reaches its capacity, an otherwise accepted observation may be excluded from belief updates because no additional point can be stored. This is an implementation diagnosis rather than a demonstrated explanation of every performance gap. Separating evidence eligibility from geometry storage requires independent validation of new observations, replay rejection, association quality, and correlated-view suppression. No performance improvement from this prospective correction is asserted here.

## 9. 并稿与溯源说明

建议将第 1 节的目标和信息契约并入问题定义；第 2 节保留核心式 (2)、(4)–(5) 与一张明确标注“解析例子”的表；第 3 节的阈值与实现限定并入几何纠错；第 4 节的条件命题可入方法附录；第 5 节入复杂度与局限；第 6 节入相关工作。第 3.2 节的具体未修问题及第 7 节待验证预期属于当前研究记录，只有取得相应新版本证据后才能转化为最终稿的实验结论。

本补充所核对的本地实现是 `nso/online_planner_v35.py`、`nso/observation_belief_v35.py`、`nso/controller_semantic_mechanism.py`、`nso/semantic_reliability.py`、`nso/observed_instances_local.py`、`nso/observed_instances_v41.py`、`nso/primitive_navigation_v41.py` 及 `docs/thesis/GRADUATION_METHOD_CHAPTER_20260928.md`。读取时文件 SHA-256 记录在配套 JSON 中；若代码后续修改，应据新版本复核实现对应，不能把本次只读检查当作未来代码的验证。
