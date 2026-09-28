# V9 预注册分析器独立审核（2026-09-11）

本次静态逻辑及纯合成数值审核通过；初审提出的来源、指标及回放收据绑定问题均已修复。未构造任何未来世界，也未读取或执行任何真实后续分支。本报告不构成场景放行、回放通过或方法有效性的证据。

审核分析器 SHA：`173e3bbec13f00579f62b402c7094fd0898270b9053bbfd5d9559d1bd29f8432`；协议 SHA：`0cd8cdd4795ca7faaf5707e77fd266914d7858701c43eeccaa49dff781c3020a`。

23 项 progression 配置逐项对应 36 个运行门；46 项纯合成检查通过。检查包含正常全部通过、缺分支、安全/到达/返回失败、各均值阈值反例、每个 parent 的零增益、近零比较器绝对面积边界、交换类别、缺失标签回退、总评分对率评分、负失败样本保留与 undefined rate。提取出来的门逻辑在 AST 层面与绑定分析器原文完全一致。

## 主要判断

| 核查项 | 结论 |
|---|---|
| total and rate score families retain all six methods on same 48 actual routes | 已确认源码逻辑 |
| same real candidate outcome reused for all methods; no separate advantage-seeking rollouts | 已确认源码逻辑 |
| 8 histories form 4 blocks each exactly2 arrangements; overall arithmetic mean equals equal block mean | 已确认源码逻辑 |
| failed or negative outcomes are included; null rate gives undefined aggregate with count, not filtered or imputed | 已确认源码逻辑 |
| sparse interpolation includes paid prefix/arrival/final and holds to branch48; prefix150 counted separately, no free hidden observations | 已确认源码逻辑 |
| oracle reports complete six-route pools and is not used for policy choices or parent filtering | 已确认源码逻辑 |
| minimum four-block one-sided sign p0.0625 only; no significance or broad-distribution claim | 已确认源码逻辑 |

二维面积与 C×F1 使用同一完整 observable reference；新增可见面积保留出发、到达及返回的两对象和背景并集减去固定前缀，未按类别加权。V9 wrapper 对 covered_area 与 area×F1 公式额外复核，并汇总 F1@2cm、precision 和 recall。任何短路线终止后只保持地图到 48 动作；这不是完整剩余预算再规划。

## 已修复的初审问题

- freeze analyzer/helper before execution and bind actual source SHA
- exact8 history tuples and child checks, prep-to-execution source subset, actual guard receipt hashes
- recompute covered area and area-times-F1; aggregate secondary quality fields

修订前关于“缺二维面积与 area×F1 输出”的初述已经更正：执行器本来已在 before/after 输出这些字段，缺少的是分析器公式核对和方法均值汇总；修订已补齐。

## 收据版本复核

最终修订明确核对 verification.verifier_sha256 与预先冻结 reviewer 源码 SHA、verification.structural_review_sha256 与执行前放行收据 SHA。该项已修复；修复前审核与源码快照另存 before_final_receipt_identity_fix，以保留审核过程。

## 配置对应表

| 预注册 progression 键 | 实现位置或解释 |
|---|---|
| `meaning` | result explicitly denies whole-system/mainstream/natural-semantics/calibrated/generalization claims |
| `all_structural_source_integrity_and_independent_raw_replay_gates` | preflight exact8 histories, per-history checks, prep/source subset, bound independent release, guard receipts; replay_status bound full48 |
| `all_48_branches_collision_free` | all_routes_collision_free_target_and_return |
| `all_48_branches_reach_original_target_and_restore_original_pose_heading_within48` | same combined gate plus audit_outcome validates paid<=48, planned actual full-route cost, replay checks real heading/pose |
| `primary_total_comparators` | loops G/O/N, all six methods in total and rate |
| `min_S_relative_mean_new_area_gain_against_each_total_comparator` | S_total_area_over_{G,O,N} |
| `near_zero_comparator_area_m2` | cm < 1e-6 switches absolute rule |
| `near_zero_minimum_absolute_area_gain_m2` | absolute mean area difference >=1.0 |
| `min_S_minus_each_total_comparator_mean_global_F1_05m` | S_total_final_f1_05cm_over_{G,O,N} |
| `min_S_minus_each_total_comparator_mean_C_times_F1_05m` | S_total_final_joint_05cm_over_{G,O,N} |
| `each_parent_S_minus_each_total_comparator_new_area_strictly_positive` | all_parents_S_new_area_m2_over_{G,O,N} |
| `each_parent_S_minus_each_total_comparator_global_F1_strictly_positive` | all_parents_S_final_f1_05cm_over_{G,O,N} |
| `each_parent_S_minus_each_total_comparator_C_times_F1_strictly_positive` | all_parents_S_final_joint_05cm_over_{G,O,N} |
| `minimum_mean_S_minus_each_total_comparator_sparse_joint_auc` | S_total_branch_joint_auc_05cm_over_{G,O,N} >=-0.001 |
| `minimum_mean_S_total_minus_S_rate_new_area_m2` | total_vs_rate_new_area_m2 >=0 |
| `minimum_mean_S_total_minus_S_rate_global_F1` | total_vs_rate_final_f1_05cm >=0 |
| `minimum_mean_S_total_minus_S_rate_C_times_F1` | total_vs_rate_final_joint_05cm >=0 |
| `minimum_mean_S_total_minus_S_rate_sparse_joint_auc` | total_vs_rate_branch_joint_auc_05cm >=-0.001 |
| `mean_S_total_above_X_total_for_each_co_primary_metric` | correct_S_over_X_all_primary; strictly positive all3 |
| `each_parent_S_total_choice_changes_when_asset_assignment_exchanges` | {Q0,Q1,Q2,Q3}_semantic_choice_exchange |
| `M_all_scores_and_choices_exactly_equal_G_in_both_score_families` | missing_M_exact_G_both_families |
| `minimum_mean_S_minus_N_coverage_2d` | coverage_noninferiority_to_N >=-0.001 |
| `failure_action` | all48 outcomes retained; failed gates explicitly listed; no filters or mutation/training/new route APIs |

完整结构与来源检查发生在数值门之前，出错即中止；报告中的 integrity gate=True 是通过这些前置检查之后的结果，不能脱离完整调用单独理解。

若后续版本使用不同大小的配对候选池，需要重新冻结候选清单、总物理分支数及分析器。每个 parent 的两 arrangement 仍应等权，四个 parent 再等权；不要按候选数增加某 parent 的统计权重。旧 V9 的结构失败应继续保留。

验证入口：`check_numeric_gates.py`；运行收据：`numeric_gate_checks.json`；逐项配置映射：`review.json`。所有使用的源码快照与哈希保留在本目录。
