# Exposure V3 执行适配交接

2026-09-28。本交接完成实现与纯夹具验证；未冻结真实配置、未创建真实 World、未启动任何在线 episode。启动权仍由主线程在 Ground12 完整终态及独立审查后决定。

新增执行链为 `nso/controller_article_exposure_v3.py`、`nso/article_experiment_exposure_v3.py`、`scripts/run_article_exposure_experiment_20260928.py`。批调度入口为 `scripts/run_article_exposure_batch_20260928.py`；冻结入口为 `scripts/freeze_article_exposure_ablation_20260928.py`。

控制器继承 GroundV2，首次接收任何数据之前以 `ArticleExposureViewPredictorV3(residual=self._residual, maximum_instances=...)` 替换预测器。ledger、residual 对象类型、paid driver、macro 规划与 mapper 接口不变，四方法均采用相同新预测器。controller 参数仅新增布尔 `same_center_exposure_dedup=True`；运行配置保留原 `article_version=article_ground_v2`，另加三个准确收据字段：`same_center_exposure_dedup`、`article_exposure_version=article.same_center_paid_exposure.v3`、`exposure_predictor_configuration_sha256`。关闭开关仅供纯组件检查，冻结 V3 协议要求共同开启。

新协议 schema 为 `article.experiment_protocol.exposure_v3`。消融的 `paired_baseline_protocol` 指向完整 GroundV2 协议，逐槽 `paired_baseline_run_id` 指向 `ground_...`；要求全部旧槽位一一配对，场景/方法/预算/种子、资产、reference、mapper、evaluation、数值构建环境、最大执行耗时、单集存储和 reserve 相同，仅允许增加共同曝光开关。未来 main 的 T0/T1 分割与原 96 ceiling 支持保留，但本交接不批准或生成 main。

`run_episode` 生命周期从 GroundV2 机械适配；AST 核对证明，除 started.scope 与 evaluation.schema 两个字符串外，函数体完全一致。传感、融合、评估和 ledger 等工具继续导入原模块，未 monkeypatch 任何冻结全局。新的 evaluation transport schema 为 `article.exposure_v3.canonical_evaluation.v1`，测量版本仍为 `article.common_numeric_face_evaluation.v1`；原始 prediction/seal、独立 preprocessing receipt、四实例 macro 分母和固定 surface evaluator 保持一致。

冻结 helper 固定绑定 Ground 协议 SHA `6621c39667e15a91f010d52c81f5a357049a447b6379246d07141953f631338a`、预声明 SHA `a008017abce88bcdc176f750c7df6ba87259c66375ac62255a7f4f9aeb295bc9` 和独立审查 predictor SHA `0843b90ddf7c71a01a16cd629d355ffd272a653049a4d0cd32d17a75a86126cf`。它要求 Ground 全部 12 槽终态、全局 ablation 恰已占用对应 12 槽，保留失败，拒绝 reserved/未知/缺槽/多占位。输出矩阵固定三个 DEV×四方法、budget160、seed92801，仍使用原共享 ablation cap24。会验证全部旧62源码及原 zip，再以动态导入闭包冻结新源码；当前新闭包为65，不在代码中硬编码数量。旧 Ground CLI 不在新执行闭包中，仍接受独立原文和归档校验。

批调度沿用原 scheduler 的独立私有模块实例，只替换该行政实例的协议读取与 runner 名，旧批处理进程不受影响。上限双 worker、资源预留、时间门、已有 reserved/unknown 不重试、失败占位、全局 phase caps 均沿用，子进程 Python 保留 `.venv-3d/bin/python` 原路径而不解析 symlink。

16 项夹具测试通过：控制器四方法、仅预测器与准确配置收据差异；显式布尔与主/消融分割；旧源哈希；完全同参配对；资源与数值环境阻断；构造/评价失败留痕且不可重试；原预测封存和同 metric；调度不重试/总 cap/资源/协议变更；冻结完整矩阵保留失败、未完成阻断和旧源/zip/预测器/环境匹配。测试只创建 tiny subprocess 和 mock sensor，不创建真实 World。

交付时执行源 SHA：

| 文件 | SHA256 |
|---|---|
| `nso/controller_article_exposure_v3.py` | `cc2dfbb1a2d27505cc32e12a3088d1012fd714118b877e4f88c5d27d4aa3e1a2` |
| `nso/article_experiment_exposure_v3.py` | `2447439f514454f6ada08895a4205632d9311db3cdce9706fccaa2ad3fe792a5` |
| `scripts/run_article_exposure_experiment_20260928.py` | `5614c98f95ebcd39d4348b8e6390c0a67851e98e65b206024f6a66a5df685ca8` |

建议由主线程先完成独立 V3 reviewer 对首个实际 episode 的兼容性检查，再依原固定矩阵推进。新的名义面积减重不是实测重建，也不能在未见真实轨迹与统一质量测量前写成收益。
