# Exposure v3 最终共享机制门：独立复核清单

本文件只准备核验步骤，不宣布通过、不启动主实验、不重跑分析快照。依据启动前的 `ARTICLE_EXPOSURE_MAIN_MECHANISM_GATE_20260928.md`；其SHA为 `5b4d61be3d3644a27224d596630e70227bea8d42f44bf283bb6a54e135a08af3`。本次读到的分析器SHA为 `65c3dfbdc9eea0af74893fd2d506bcb910f04ba47efded15ae0ce74e2c375690`。最终审查需使用根线程指定的完整12条V3终态快照及其manifest，不把本清单当作结果。

## 1. 先核验完整范围与输入版本

- 原开发12、Ground消融12、Exposure消融12均有终态及相应独立复核；原失败保留，不能以补测质量改写在线资格。新V3技术证据完整，不能只挑成功条目。
- 全12个V3槽位严格是三DEV布局×G/B/S/NBV，预算160、种子92801；源65闭包、协议SHA、原Ground配对、传感/标定、共同地面和曝光参数、评价版本及四实例分母均与封存协议一致。
- `main_gate_witnesses.json`含**六个**S/B位置：Ground三对和Exposure三对。最终V3门只审 `comparison=ExposureV3/S-B` 的三对；旧Ground不能替代新V3条件。每对 `side_status` 的protocol/artifact/review SHA必须与完整快照一致。
- `automatic_gate_pass=null`、`automatic_gate_approval=false`、`automatic_main_launch=false`均属预期。`evidence_ready_for_manual_review`只表示导出了待核验位置，不能直接判门通过。缺失/未开始/失败的条目保留原状态，不填零。

## 2. 对每个S/B候选见证核验这些字段

|层次|关键字段|必须成立的关系|
|---|---|---|
|实际共同输入|`common_actual_prefix_frames`、`first_observation_divergence_paid_step`、首动作步`t_a`、实际RGB-D/scan NPY|只去除RGB-D的run命名`frame_id`；深度、RGB、扫描、位姿、时间、标定及付费步均比较。决策步`t_d=t_a−1`必须在共同前缀内，两臂都实际执行`t_a`。|
|实例与证据|`observed_anchor_identity_pairs`、`planning_evidence.instances`、`recorded_peer_links`|按唯一已观测marker锚点匹配；不能用GT编号或类别本身强配身份。同类支持数≥2、无association/class冲突、实际同类peer存在且其合格几何证据有区分度。|
|反馈时序|本帧`geometry_feedback`、其`applied/instance`、`structure_belief`|先将已应用反馈更新进association快照，再读取共享信念；同一当前帧/观测SHA绑定，`geometry_log_scores=geometry_log_evidence`。标签可读不等于类别已合格。|
|共享差异|`peer_instance_ids`、`reliability_weights_leave_one_out`、`rho`、`active_structure_prior`、`structure_probabilities`|S/B共同输入下，共享合格同类证据造成实际权重或后验差；仅peer名单非空、未应用信息或预测分支中假想反馈不够。局部自身几何证据差异须解释，不能把不配对输入误称共享效应。|
|影响所选动作|`macro_commit_paid_step`、`macro_commit`、`current_decision`、原始`global_selection`|追溯当前macro的最后实际承诺步`t_c`，要求`t_c`也在共同前缀。检查完整direct/diagnostic选项、共同候选与面积、后验加权、最终唯一目标排序；必须影响最终所选可执行目标，不能只看未选分数或候选行顺序。|
|真实执行|`actual_executed_sensor_action`、`steps/{t_d}`、`packets/{t_a}_receipt`、`result.actions[t_a−1]`|控制器left/right与传感器turn_left/turn_right正确对应；实际姿态变化、付费帧连续及执行成功匹配，不能把不同名字的相同动作当分歧。|
|预算与安全|`remaining_budget`、`selected_route_candidate`、`committed_macro_budget_feasible`、`return_reserve_margin_after_action`、`decision_guard`|`remaining=budget−paid_step`；两臂目标路线均feasible、总到达+观察+返航成本≤remaining；`remaining−1−return_cost_after_action≥0`，实际安全门allowed且未改写为blocked/stop。|

`first_changed_posterior_paid_step`和`first_changed_score_paid_step`只是导航提示；它们不保证该后验属于最终胜出的候选，须直接核验`t_c`的对应实例、评分项与所选目标。考虑`select_unique_target`的1e−12平分容差和共同tie规则，排除纯显示/词典序差异。若选中diagnostic，核验完整付费两次观察与返航的followup成本、分支概率和实际第一macro；预声明允许diagnostic见证，不限定direct。

记录所选macro之后是否真正完成observe、在哪一步完成，或为何被终止；这是解释执行完整性的补充，不另加“必须先产生正质量/完整第二次观察”的门槛。原门要求可执行观察macro已传播成真实首次动作分歧。

## 3. 没有逐步return_latched时，如何证明在返航锁定前

分析器正确保留 `return_latched_recorded=null`，不会用终态`controller_final.return_latched`倒填历史。`first_recorded_return_step`及`return_reason_seen_before_or_at`只是已记录router行为：**router返回reason=return不等于watchdog锁定；未记录return也单独不能证明未锁定。**

可使用冻结实际执行分支的控制流不变量给出额外证明，并在最终报告标注为“根据封存源码推导”，不篡改原日志字段：

1. V3/Ground子类没有重写`choose()`；ArticleV1只调用SemanticMechanismController的`choose()`并添加`article_method`。后者覆盖旧base choose，不经过base的另一套逐步latch日志。
2. 初始化`_return_latched=False`。当前继承链里设置True的唯一实际choose分支，要求`_committed_target is None`且无进展计数达到阈值；它同时保持target=None。True后不会复位。
3. `_select_global()`唯一赋予非空承诺目标的分支受`not self._return_latched`保护。故`t_c`的 `global_replanned=true + selected非空 + macro_target与selected.target相同`，直接证明该次选择时尚未锁定。
4. 从`t_c`追踪至`t_d`，确认同一`macro_id/target`仍承诺、没有取消后遗留旧selection。其间若锁定，target必须为空且不能重建。因此`t_d`保存的非空`macro_target`及 `routing_reason=upstream_goal_with_paid_return_reserve`、匹配且feasible的唯一路由候选，结合该不变量证明`t_d`仍未锁定。两臂分别证明。
5. 再绑定`t_d`安全门与`t_a`实际动作。若目标为空、return/stop/blocked、承诺链断裂或来源无法核验，只能记“前返航证明不足”，不能以空latch字段或终态倒推通过。

本次已只读核对以下六个文件与V3协议65源SHA逐一相等：`controller_article_exposure_v3.py`、`controller_article_ground_v2.py`、`controller_article_v1.py`、`controller_semantic_mechanism.py`、`controller_v43.py`、`primitive_navigation_v41.py`。关键分支在 `controller_semantic_mechanism.py:414` 的choose、`:439` 的非锁定重规划和 `primitive_navigation_v41.py:236` 的带返航预算路由。实际最终审查仍需绑定源档案原字节，不能只信本清单中的行号。

## 4. 最终结论规则

三对全部呈现后，至少一对具备“共同实际输入→合格同类几何证据→共享权重/后验差→所选macro→前返航真实动作差”完整链，且全队列技术审查及资源条件满足，才可由根线程记录允许启动已预定48槽位主矩阵。无需F1/J正点估计；不得为通过添加新开发在线搜索。

若全部三对无实际动作差，或差异仅发生在传感历史分歧之后/返航阶段，或缺乏共享因果联系，则记“机制前提未满足，主矩阵未执行”。这不否定已经单独验证的共同前端修正或B/G类别作用，也不能将它们替代S/B共享门。未见布局的有效性仍需真正独立主比较，开发门通过本身不是效力证明。
