# Appendix: observed multi-instance planning in ArticleV1

Prepared 28 September 2026. Frozen ArticleV1 method description; no new experimental result.

## A. Scope and module responsibilities

ArticleV1 studies documentation after industrial layout changes, with facilities static during each mission. Four facilities form two category pairs. Each has one of four structures, \(h\in\{\text{planar},\text{recessed},\text{louvered},\text{open-frame}\}\): \(H=4\) hypotheses and \(N=4\) evaluation targets. Online identities arise from observations; the controller does not receive their true poses or structures. Scene families comprise aisle, work-cell and loop layouts, separated into development and test layouts.

Inputs comprise a conservative navigation graph, RGB-D/planar scans, exact simulated poses, category–structure priors and nominal templates. Graph envelopes are category/structure independent. This is active documentation with a supplied navigation prior. Synthetic colored labels isolate category-conditioned decisions; natural recognition, localization uncertainty and moving-obstacle avoidance are outside scope.

| Interface | Executed ArticleV1 role |
|---|---|
| OV-SDF | Associate measured instance support; qualify observed categories; maintain four-structure marginals and optional shared category reliability. |
| STGHP | Forecast nominal unseen surfaces, compare direct and one-diagnostic macro routes, and select one executable target pose. |
| RPN-UQ | Validate paid primitive execution, observed collision constraints and complete-pose return cost; reject uncertain geometry prerequisites. |
| IGCR | Derive bounded structure evidence from current measured depth and update the observed-instance ledger. |

These CPU interfaces evaluate no learned uncertainty network. An independent marginal-variance penalty is disabled because shared reliability induces dependence. Open3D TSDF and occupancy consume measured packets only; templates never fill unobserved reconstructed surfaces.

| Aspect | Earlier local V35 evidence | ArticleV1 described here |
|---|---|---|
| Latent state | One binary configuration hypothesis | Four structure hypotheses per observed instance; optional same-class coupling |
| Geometric frame | Supplied facility frame and two templates | Measured label-plane estimates and common nominal shape family |
| Planning | Budget dynamic programming over exposure states; next atomic subgoal | Finite candidate macros; direct gain/cost versus one diagnostic and one final view |
| Common acquisition | Prescribed 18-action prefix | One common initial frame; bounded observation-driven initialization |
| Motion accounting | 1 m translation / 90° turn primitives | 0.25 m translation / 30° turn; each primitive and explicit observation costs one |
| Evaluation | Historical local coverage and saved public-ROI surface contract | Correct measured-free navigable coverage and all-four-instance macro surface F1, without prediction ROI cropping |

The binary posterior, fixed 0.99 diagnostic channel, exposure-state recursion and 80% template-coverage stopping gate from V35 do not define ArticleV1. Its old measurements remain a separate evidence set.

## B. Observation-qualified structure belief

Depth components and accumulated measured support determine association; category values do not select the associated region. A new instance requires visible marker support. A category is qualified only after at least four corresponding marker pixels in each of two eligible support frames, with no conflicting category and no current association ambiguity. Otherwise its category is withheld. Geometry-only controls retain the same marker geometry and association frontend.

Let \(L_i(h)\) be instance \(i\)'s cumulative generalized depth evidence, \(q_0(h)=1/4\), and \(q_1(h\mid c)\) the declared category prior. The two reliability models have initial weights \(w_0(k)=1/2\). For qualified same-class peers,

\[
m_j(k)=\sum_h q_k(h\mid c_i)e^{L_j(h)},\qquad
w_{-i}(k)=\frac{w_0(k)\prod_{j\ne i:c_j=c_i}m_j(k)}
{\sum_{\ell=0}^{1}w_0(\ell)\prod_{j\ne i:c_j=c_i}m_j(\ell)}.
\tag{A1}
\]

\[
\rho_i=w_{-i}(1),\qquad
b_i(h)=\frac{[(1-\rho_i)q_0(h)+\rho_iq_1(h\mid c_i)]e^{L_i(h)}}
{\sum_g[(1-\rho_i)q_0(g)+\rho_iq_1(g\mid c_i)]e^{L_i(g)}}.
\tag{A2}
\]

Cumulative messages replace previous messages; leave-one-out transfer applies own-instance evidence once. Without a qualified category, both models equal \(q_0\). This is standard hierarchical generalized Bayes; \(\rho_i\) is an uncalibrated model weight, not recognition accuracy.

| Policy | Category-conditioned belief | Cross-instance transfer | Paid diagnostic lookahead |
|---|---|---|---|
| NBV | No; geometry prior and measured feedback | No | No; direct nominal-area/cost only |
| G | No; geometry prior and measured feedback | No | Yes |
| B | Fixed mixture \((q_0+q_1)/2\), with own measured feedback | No | Yes |
| S | Equations (A1)–(A2) | Qualified same-class peers | Yes |

B updates from its own geometry. NBV is a mechanism comparator, not an external planner reproduction. All policies share repairs and acquisition limits. Ablations separately suppress measured belief feedback, anticipated information, or future changes to non-diagnosed instances; raw-depth fusion remains active.

With correct one-to-one association, each class offers only one peer. Its model-evidence ratio satisfies \(r\in[\min_h(q_1/q_0),\max_h(q_1/q_0)]\), and \(\rho=r/(1+r)\). Consequently, the S–B active-prior total variation is \(|\rho-1/2|\operatorname{TV}(q_1,q_0)\). For the declared cabinet prior \((0.4,0.3,0.2,0.1)\), this is at most \(0.0428572\). This bounds the transferred prior, not the posterior after own-instance evidence; it explains why weak priors can yield small or unchanged decisions.

## C. Bounded evidence and measured plane recovery

IGCR profiles scale, anchor and yaw nuisance candidates for each structure and scores truncated axial-depth residuals on current associated pixels. The best residual loss \(\ell_i(h)\) gives \(s_i(h)=-(\ell_i(h)-\min_g\ell_i(g))/0.05\), with loss truncation at 0.25 m. Each applied increment is shifted to maximum zero and capped below at −6; accumulated evidence is again shifted and capped below at −24. Invalid or absent depths supply no empty-space evidence.

The association cache retains 4096 support points. Full-cache rescue requires a unique nonduplicate packet, four novel voxels and 5% novel points, fewer than 32 successful feedback frames, and separation from every applied view by 0.25 m or 30°. Rescued scores must be informative; repeat feedback is excluded. These rules limit correlated reinforcement without establishing independence.

Successful single-frame fits remain valid. Otherwise an upright 0.32 × 0.30 m label is estimated from at least three paid views and 24 measured points, retaining eight views/1024 points per instance. Axial-noise weighted least squares fits yaw and offset with \(\sigma_z=\max(0.001,0.01z)\). Views require 0.15 m translation or 8° rotation from each retained view. Conditioning, noise, front-facing and public-size consistency are checked. Anchor ambiguity includes pixel footprints, linearized three-sigma perturbations and leave-one-view sensitivity, retaining 6 cm anchor and 12° yaw limits. Conflicts beyond 15 cm or 20° invalidate the plane. These are diagnostic envelopes, not calibrated guarantees; they affect planning and residual alignment, never TSDF input.

## D. Finite paid-macro selection

There are at most 32 candidate poses and eight forecasts per observed instance. Let \(A_{ih}(v)\) be new nominal exterior area at \(v\), averaged over scales 0.85, 1 and 1.15, using measured anchors, template self-visibility and accumulated support. Previously paid identical cameras and missing reliable planes yield zero surface forecasts. External occlusion, repeated-view precision improvement and pose-error distributions are unmodeled.

Let \(D(v)\) denote weighted unknown-cell area in a 2 m planar disk, set to zero at a previously visited XY node. With the declared unit area weights,

\[
g(v;b)=D(v)+\sum_{i\in\mathcal I_t}\sum_h b_i(h)A_{ih}(v),\qquad
U_{\rm dir}(v)=\frac{g(v;b)}{d(s,v)+1+d(v,s_{\rm home})}.
\tag{A3}
\]

Only affordable full-return routes enter the comparison. Here \(d\) counts graph primitives including heading changes, and each explicit observation contributes one additional action. Both quantities in the numerator are planning proxies; neither is measured coverage or F1.

For a diagnostic pose \(v\) on observed instance \(i\), a four-outcome prototype channel \(K_i(y\mid h,v)\) compares public nominal depth predictions. It does not render the actual future scene. Let \(b^{i,v,y}\) be the hypothetical branch from adding \(\log K_i(y\mid h,v)\) to the target's evidence and recomputing all applicable marginals. Then

\[
p(y\mid i,v)=\sum_h b_i(h)K_i(y\mid h,v),\qquad
c(v,u)=d(s,v)+1+d(v,u)+1+d(u,s_{\rm home}),
\tag{A4}
\]

\[
U_{\rm diag}(i,v)=\sum_y p(y\mid i,v)
\max\!\left\{0,\max_{u\ne v:\,c(v,u)\le b_t}
\frac{g(u;b^{i,v,y})}{c(v,u)}\right\}.
\tag{A5}
\]

Zero represents return without another view. Instance utilities are summed before maximizing over the shared final pose. First-view gain is omitted to avoid unknown cross-view overlap; movement-frame gains are also unpredicted, although real measurements are fused. This is expected final-view utility per complete cost, not EVI alone or an optimal POMDP solution. Each executable target retains its maximum score, with lexicographic ties. Only the first macro is committed; its paid observation triggers replanning from actual data.

## E. Initialization, execution and evaluation

All policies share geometry-only plane initialization: prioritize eligible instances with the fewest attempts, then smallest full-route cost; allow two distinct targets per instance and at most \(\lfloor B/5\rfloor\) executed initialization actions. Commitment consumes an attempt; executed actions are never refunded. Reliable-plane acquisition, ambiguity, conflict or loss of feasibility cancels the initialization macro. Safety and return feasibility are rechecked using current observations. The fixed-graph invariant \(d(s,s_{\rm home})\le b_t\) supports return feasibility, conditional on truthful graph edges and successful execution; it is not a guarantee under arbitrary new obstacles. A no-progress watchdog can latch return.

```text
Read public graph, shape family and frozen per-policy configuration.
Acquire the one common initial frame (action cost zero).
Repeat:
    Save and integrate the current measured packet exactly once.
    Validate the executed primitive and associate measured instances.
    Estimate/check label planes; apply only eligible depth residuals.
    Update view support, safety and cumulative instance belief messages.
    Complete or cancel the current macro when its execution rules require it.
    If no macro is active and return is not latched:
        choose shared initialization if eligible;
        otherwise compare affordable direct and diagnostic targets.
    Guard and execute one paid primitive, or stop/block.
Until termination or the declared action, time or storage limit.
Save the measured map; evaluate using the fixed private reference.
```

Every translation, turn and explicit observation costs one and obtains a saved sensor frame. Reference geometry is fixed before execution. The navigable denominator \(\mathcal D\) is the start-connected 0.1 m grid of conservative robot centers, using a 0.2 m footprint radius. Only cells measured free contribute:

\[
C_{\rm nav}=\frac{|\{x\in\mathcal D:M_T(x)=\text{free}\}|}{|\mathcal D|},\quad
Q=\frac14\sum_{i=1}^{4}\frac{2P_iR_i}{P_i+R_i},\quad
J_{\rm nav}=C_{\rm nav}Q.
\tag{A6}
\]

Zero denominators in individual F1 terms give zero. Recall uses each facility's fixed observable reference area; undiscovered facilities remain in the denominator. Prediction areas are evaluated on the entire mesh: observable target matches contribute true positives; false area is allocated to the nearest facility without a distance cutoff. Correct background and correct non-reference target surfaces are reported separately. The tolerance is 5 cm; reference and prediction area sampling use 0.3 m spacing with fixed seeds 4001 and 4002. Qualification requires controller stop, no collision, exact XY-and-yaw return, and one saved/fused frame per acquisition; there is no all-facilities-seen gate. Coverage, macro quality, per-instance terms and failures should accompany the product. These contracts permit conditional comparison within the declared facility family; they do not themselves establish semantic superiority or broad generalization.

---

# 中文并稿片段：ArticleV1 多实例方法及其适用范围

本节描述独立的多实例 CPU 实现，不改变旧 V35 实验的定义，也不预报本轮实验结果。任务为工业布局调整后、单次任务内环境静止的主动建档。每个场景有四个设施，分属两个重复类别；每个实例的隐含结构有平面、凹入、百叶和开放框架四种假设。在线实例身份由已取得的观测建立，控制器不获得实际构型或私有实例位置。已知保守导航图、准确模拟位姿、人工颜色标签与共同名义结构族构成研究条件，因此本实验验证的是这些条件下的语义决策作用。

OV-SDF负责几何实例关联、类别资格及构型信念；IGCR从当前实测深度形成受限残差证据；STGHP根据名义新增表面及一次诊断前瞻选择视角宏动作；RPN-UQ检查原子动作、已观测障碍和完整姿态返航预算。该CPU版本不包含新训练的感知或不确定性网络。二维占据栅格与三维TSDF仅融合实际取得的RGB-D和扫描，模板不用于补面。

类别资格需要同一类别在两个合格支持帧中各有至少四个标记像素，且不存在类别冲突或当前关联歧义。颜色可提供共同标记几何，类别值不参与支持区域关联。式（A1）先使用其他已观测同类实例的累计几何证据计算可靠性权重，再由式（A2）对当前实例施加自身证据一次。\(\rho_i\)表示类别先验模型相对于几何先验模型的权重，不是已校准的识别准确率；累计证据采用替换消息，避免反复乘入历史。G和NBV使用均匀几何先验，B使用固定的几何／类别混合先验并正常接受自身几何更新，S额外允许同类实例之间共享可靠性。G、B、S均有一次付费诊断前瞻；NBV只比较直接视角收益与总代价，因此G仍有自主判断补看的能力。

在正确的一一实例关联条件下，每类只有一个可共享同伴，其模型证据比位于\(\min(q_1/q_0)\)和\(\max(q_1/q_0)\)之间。由此，S相对B的当前先验总变差为\(|\rho-1/2|\operatorname{TV}(q_1,q_0)\)。例如公开cabinet先验为\((0.4,0.3,0.2,0.1)\)时，该差异至多约0.0428572。这个界只约束共享后的先验，不约束叠加自身几何证据后的后验；弱先验和少同伴可能使动作差异很小，不能仅凭有共享模块就推定收益。

证据缓存与推断资格分开。4096点支持缓存饱和后，只有非重复且唯一关联的当前包，满足至少四个新支持体素、5%新点比例、成功反馈少于32帧，并与每个已应用视角至少相隔0.25 m或30°，才允许进入补救资格；无信息残差不累计。该规则限制重复强化，不证明观测独立。单帧标记平面拟合成功时沿用原结果；否则在公开的0.32×0.30 m竖直标签假设下，以至少三视角、24测量点拟合平面，单实例最多保留八视角、1024点。点深度噪声与平面参数精度分别处理，结合线性化三倍标准差、标签中心歧义及留一视角敏感性，仍要求锚点不确定度不超过6 cm、偏航不超过12°。不满足条件则拒绝，不把观察方位替代为对象法向。

规划池最多32姿态，每个已观测实例最多预测八个候选视角。式（A3）将二维未知单元面积代理与后验加权的名义新增外表面积相加，再除以“到达—观测—返航”完整原子动作数。式（A4）—（A5）预测一次诊断后，在同一组可执行后续视角中选择最大期望收益；应先跨实例求和，再对共同后续视角取最大值，不能为每个物体分别选择互不兼容的最优动作。第一诊断视角的表面面积不重复加到最终视角，实际移动中取得的深度仍照常融合。模板通道未建模外部遮挡和位姿误差，名义表面收益也不是实测F1。控制器只承诺执行第一个视角宏动作，完成额外付费观测后再以实际证据重规划。

各方法共享一次零动作代价的初始观测，随后每次0.25 m平移、30°转向和显式观测均计一个动作。无可靠标记平面的实例可以进入共同几何初始化：按最少尝试次数与完整路线代价排序，每实例最多两个不同目标，全局初始化动作不超过\(\lfloor B/5\rfloor\)，不退还已执行代价。返航论证仅在固定合法图和成功执行条件下适用；新障碍可能导致停止或失败。

评价由式（A6）给出。\(C_{\rm nav}\)是固定、起点连通、考虑机器人尺寸的可通行中心格中，被实际测为自由的比例；\(Q\)为全部四实例表面F1的宏平均，未发现实例不能从分母剔除。表面精确率、召回率使用几何参考面积，而非类别重要性；完整预测网格不按有利区域裁剪。距离容差为5 cm，面积采样间距为0.3 m。任务资格要求合法停止、无碰撞、完整XY及朝向返航，以及取得、保存、融合帧一致，不要求先看全四个设施。联合指标、各分量、逐实例结果与失败状态应分别报告。

旧V35采用二构型、已给定设施坐标系、18步共同前缀和曝光状态动态规划；ArticleV1采用四构型多实例、实测标签平面和有限宏动作比较。旧二元闭式分析只解释信息价值的条件，不是本实现的执行方程或质量保证。层次贝叶斯、信息价值和返航约束均有既有理论基础；本附录的作用是明确这些原则在本系统中的组合及可核验边界。

## Integration notes

The frozen development protocol enables common cache-feedback repair, multi-view planes and measurement acquisition. Main and ablation experiments require their own frozen protocols; no completion or outcome is asserted here. Implementation bindings, unit-test checks and links to the earlier analytic supplement are recorded in `ARTICLE_MULTI_INSTANCE_METHOD_APPENDIX_20260928.sources.json`. This appendix should be integrated as a separately scoped method section, not substituted for the equations that generated historical V35 results.
