# 地面消融独立复核器

入口为 `scripts/review_article_ground_episode_20260928.py`。它只读已经终止的消融记录；没有预留的槽位返回 `unstarted`，`reserved` 槽位返回 `pending`，均不读取半写入实验文件，也不创建复核输出。默认输出目录为 `audit_results/article_stage_20260928/episode_reviews_ground_v2/<run_id>`，已存在输出不能覆盖。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-3d/bin/python -B scripts/review_article_ground_episode_20260928.py --episode audit_results/article_stage_20260928/<ground_phase>/episodes/<run_id>
```

若终止失败发生在归档protocol写入之前，可用 `--protocol` 指向已冻结协议；失败继续显示 `failed_attempt_preserved`，不补写实验分数。原失败基线仍保持失败状态。

复核器分两层。对具有完整artifact manifest的新记录，先实际调用**未修改的** `review_article_episode_20260928.py`，把兼容的付费动作、碰撞、返航、包SHA、映射整帧绑定、候选返航成本和指标算术结果保存为 `compatible_v1_receipts/`。随后检查新协议、前端、配对和派生评价。不能把代码阅读或单元测试称为真实新记录已通过旧读者；每条记录必须在终止后实际通过该嵌套审查。

新增检查包括：

- 严格 `article.experiment_protocol.ground_v2`、冻结状态、ablation阶段、三个DEV×G/B/S/NBV完整12格、160动作和92801种子；所有旧基线一一绑定，无槽位参数覆盖。
- 新旧公共资产、地图参数、评价参数及控制配置一致；控制器只新增地面开关，完整运行配置仅允许改变声明的ground字段和article_version。
- 新旧完整源码归档清单及SHA、当前源码、旧58源保留、新运行模块入封存、实际数值运行环境与协议一致。
- 同一个全局执行registry的development/main/ablation计数和上限；新版本不能重新获得开发12条额度。
- 每帧只从原付费RGB-D/K/T重新计算地面核验回执，精确比较原包SHA、排除掩码SHA、接受/回退理由与常数；标量仅容许1e-12舍入。
- 直接从封存原网格独立计算 `twice_area<=1e-12` 的面下标；逐项核验移除下标、数量、面积、最大边、原/派生数组哈希和保序保dtype约定。没有调用适配器本身去认证自身结果，且没有重算表面质量。
- 派生评价版本 `article.common_numeric_face_evaluation.v1`、评价设置、evaluator/adapter源码pin、原prediction seal及preprocessing receipt SHA。不能用仅数量一致、虚构“no ROI”字段或放宽阈值通过复核。

对原CELL_G失败基线，完整配置只能从独立 `common_evaluation_v1/<run_id>/prepared/` 终止后保全包读取。核验其manifest、输入快照、原始metadata字节、原failure ledger绑定和冻结构建计划，保留 `post_terminal_forensic_capture` 及原始 `attempt_failed`。其中已保存运动完成证明和旧prediction seal会带入报告；它们不被改称原管线端到端成功。

11项测试目前通过，覆盖正常矩阵/独立面计算以及错误基线、隐藏槽位参数、不完整有利子集、数值运行环境不同、ROI/类别裁剪、错误面下标/计数/哈希/微小面积、错误阈值、派生评价声明缺失、未开始与半写入预留记录。另明确验证了新入口依赖闭包可以不包含旧CLI，但不能移除或改变共同科学依赖；旧58源码仍由其原归档逐项核验。测试文件为 `tests/test_review_article_ground_episode_20260928.py`，全部只使用合成数组或临时metadata，没有World、TSDF或新质量测量。

复核脚本在科学执行源码封存之外，可以修复读者接口问题；每次复核输出记录自身与基础读者源码SHA，并以独立manifest封存所有嵌套报告。当前读者特意只审查本轮12条前端消融。未来main执行能力不等于本轮已授权main，也不自动套用该消融资格检查。
