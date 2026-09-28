# Exposure v3 终态复核范围：启动前固定

复核源码：`scripts/review_article_exposure_episode_20260928.py`，SHA256 `fd77ee1a8d5bd2ac67d278bf96bc6911168093096c45068886a11cd56fa1dc75`。纯夹具：`tests/test_review_article_exposure_episode_20260928.py`，SHA256 `67b35203a78fbf62d7fc33a9f35b37eb070f07cc05a190ef9b62e9b4a0c68d9f`。14项通过，8.170秒，无World。本文与源码在首次Exposure在线复核前约定覆盖范围；不以实际成绩选择核验条目。

## 完整检查与固定几何抽查分开报告

每条终态记录完整检查以下内容：

- 新 `article.experiment_protocol.exposure_v3` schema、全12槽位一一配对Ground v2、同资产/方法/预算/种子/映射/评价/数值环境/资源上限；唯一配置增加为共同 `same_center_exposure_dedup=true`。方法专属override、未知基线、删选子矩阵均拒绝。
- 原62源与新执行闭包及zip逐文件SHA；共享旧源码不变，只允许被替换的Ground CLI退出新执行闭包，仍验证它在旧archive/live中保存。
- 原V1终态reader**实际调用**并保存嵌套报告，验证完整文件、原始传感包、运动、预算/返航/碰撞、全帧RGB-D融合、三项原始prediction seal、四实例评价分母和指标算术。不能仅根据兼容字段推断reader已经通过。
- 每个当前原始包重算Ground mask；原冻结测量支持/marker-plane组件独立读取实际accepted像素，恢复累积support、已付费K/T及平面来源。不会调用形状残差求解器、规划器choose、未来传感器或TSDF。
- 每个保存forecast检查版本/配置/source、公共原型、原包历史SHA、候选公共图位置与实际标定、post-feedback planning snapshot、严格字段、非负面积分区、结构尺度顺序/采样总量、结构均值、后验加权、完整相同K/T排除门和fallback。
- 网格处理检查原文件seal、原arrays SHA、唯一固定退化面阈值、全部删除索引、保序/保顶点，和不变的共同评价版本 `article.common_numeric_face_evaluation.v1`。不重测质量，不添加ROI或语义权重。

较贵的**独立原型几何重算**固定覆盖：每次重规划所选目标在全部实例forecast中的对应候选，以及本条首次出现至少一个非fallback可靠平面预测的重规划的所有非fallback候选。其他候选作上述完整来源和算术检查，**不声称全部进行了独立射线重算**。fallback按重建平面状态验证为零，不虚计重算数量。

几何重算沿用冻结原型bank/原面积积分，再由复核器自己构造历史同光心视锥并集，不调用新v3 `_areas` 验证其自身。保存 `exposure_candidate_checks.json`，每项明确 `independent_geometry_recomputed` 与sampling_reason。报告分别给 `all_candidates`、`independent_geometry_candidates`、首个全候选抽查步、总耗时 `elapsed_s`。首次真实review结束后按实际耗时决定复核并发，不扩大重复重算。

## 时序与数值规则

`association.instances` 是当帧反馈之前；有效 `geometry_feedback[*].instance` 是应用后的实例，再结合 `structure_belief` 的prior/probability/semantic得到实际规划快照。复核器严格验证该顺序及反馈只能改变指定信念字段，不能借此替换测量支持或类别身份。已有14项夹具包括该时序回归。

旧`unexcluded_expected_new_surface_area_m2`表示完全相同K/T排除门之前，**已经经过曝光去重**。真正的曝光前值来自component `new_surface_area_before_exposure_m2`，满足 before=after+excluded。故不会把“未过旧门”的值误当未去重面积。

离散mask、资格、字段、SHA和原始支持精确核对；普通有限浮点receipt以绝对1e-12检查，独立几何面积亦如此。请用原数值环境运行，避免已知BLAS差异造成平面拟合变化：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-3d/bin/python -B scripts/review_article_exposure_episode_20260928.py --episode <已终态episode目录> --protocol configs/virtual3d/article_exposure_ablation_v3_20260928.json
```

本命令只做终态记录复核。输出默认 `audit_results/article_stage_20260928/episode_reviews_exposure_v3/<run_id>/`，排他创建，不覆写。

## 缺失与失败

无预留返回unstarted，active reserved返回pending，不读取半写入文件，不生成失败报告。终态失败有原账本和failure SHA时保留 `failed_attempt_preserved`、qualified=false，不计算部分质量。未知或篡改输入为review_error。

Ground基线若失败且无完整controller seal，保留原失败并声明实际配对配置不可用；新臂仍独立按冻结共同协议检查，不能造出基线配置，也不能将其列成完整合格配对。所有尝试仍计入全局ablation24和开发12上限。

复核报告schema为 `article.saved_episode_review.exposure_v3`，manifest为 `article.exposure_episode_review_manifest.v3`。保留与Ground相同的status/qualified/metrics/metric_version/input_manifest_sha256/protocol_sha256/paired_baseline字段，供完整槽位分析读取。

首次真实V3集成尚未运行。14项纯夹具能证明schema、恶意receipt拒绝、采样范围、来源、面积和状态处理；不能替代真实episode兼容性与性能证据。
