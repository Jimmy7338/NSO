# GitHub 仓库续接说明（2026-09-11）

本次推送用于在另一台设备恢复代码、论文材料、关键实验证据和模型参数。研究入口依次为：

1. `docs/research/GOAL_PROMPT.md`
2. `docs/research/RESEARCH_GOAL_AND_STORY.md`
3. `docs/research/V11_NETWORK_TRAINING_AND_DATA_ADEQUACY_HANDOFF.md`
4. `docs/research/CURRENT_RESEARCH_STATE.json`
5. `docs/research/THESIS_CLAIM_EVIDENCE_LEDGER.md`

克隆时需要 Git LFS：

```bash
git lfs install
git clone https://github.com/Jimmy7338/NSO.git
cd NSO
git lfs pull
python3 -m venv .venv-3d
.venv-3d/bin/pip install -r requirements-3d.lock.txt
```

建议先运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-3d/bin/python -m unittest discover -s tests/virtual3d -p 'test_cpu*.py' -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-3d/bin/python -m unittest tests.virtual3d.test_semantic_gain_v11 -v
```

V11 训练复现入口是 `scripts/train_semantic_gain_v11.py`，独立数值复核入口是 `scripts/replay_semantic_gain_v11.py`。训练所需的 V9 决策输入与候选结果表已随仓库保留；冻结模型位于 `eval_results/semantic_gain_v11_development_20260911/frozen_model.json`。闭环入口是 `scripts/run_semantic_gain_v11_closed_loop.py`，当前正式完整批次尚未完成，应新建输出目录重新运行。

训练归档清单记录的是集成接口扩展前的模型源码哈希；当前源码重新训练后，`summary.json`、`folds.json` 和 `frozen_model.json` 与归档逐字节一致。连续性核验见 `audit_results/semantic_gain_v11_current_source_reproduction_20260911/verification.json`。

仓库保留了关键汇总、验证、配置、源码快照和不可替代模型参数。约 5.9 GiB 的逐帧 RGB-D、重复网格、候选执行缓存和未完成中间目录没有加入本次新提交；它们可由已提交脚本重新生成，也不是理解当前结论或继续 V11 所必需。历史提交中已经纳入 Git LFS 的数据集和神经网络权重仍保留，克隆后必须执行 `git lfs pull` 才能获得真实内容。

当前论文边界：V11 在已查看的 V9 开发域中通过整组留一训练和独立数值重训复核；完整闭环开发、未见 E 场景、自然语义、未知起点覆盖、外部主流基线和实车效果尚未完成。
