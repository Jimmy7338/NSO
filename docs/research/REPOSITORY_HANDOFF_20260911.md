# GitHub 仓库续接说明（2026-09-11）

本次推送用于在另一台设备恢复代码、论文材料、冻结协议、独立确认前缀和实验结果。新对话先读：

1. `docs/research/GOAL_PROMPT.md`
2. `docs/research/CONVERSATION_CONTEXT_HANDOFF_20260911.md`
3. `docs/research/RESEARCH_GOAL_AND_STORY.md`
4. `docs/research/V11_1_INDEPENDENT_CONFIRMATION_EFFICACY_RESULT.md`
5. `docs/research/CURRENT_RESEARCH_STATE.json`
6. `docs/research/THESIS_CLAIM_EVIDENCE_LEDGER.md`

## 环境恢复

仓库包含 Git LFS 资产，克隆后执行：

```bash
git lfs install
git clone https://github.com/Jimmy7338/NSO.git
cd NSO
git lfs pull
python3 -m venv .venv-3d
.venv-3d/bin/pip install -r requirements-3d.lock.txt
```

定向检查：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-3d/bin/python -m unittest tests.virtual3d.test_semantic_gain_v11 -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-3d/bin/python -m unittest discover -s tests/virtual3d -p 'test_cpu*.py' -v
```

## 当前冻结结果

V11.1 开发闭环在 56 条分支上通过，且 56/56 独立物理与指标重放通过。随后生成了两轮独立场景：

- E00--E11 首轮准备因覆盖压力层未与 efficacy 层分离，在未来动作和结果均未查看时停止；失败证据保留。
- F00--F11 使用种子 413--424，在世界构造和结果查看前冻结。24 个前缀结构门通过。
- F00--F07 efficacy 已运行完 96 条分支并通过全部预设门。学习语义相对同容量几何、固定语义和类别置换分别提高 29.13%、22.40% 和 65.75%；所有分支安全返航，低置信度精确回退到几何网络。
- F00--F07 的 96/96 后置独立物理与指标重放通过。
- F08--F11 覆盖压力层40条分支全部安全且40/40独立重放通过，但学习语义相对同容量几何为$-0.05\%$（0胜7平1负），与类别置换8/8相同。

正式结果路径：

```text
eval_results/semantic_gain_v11_1_confirmation2_efficacy_20260911
audit_results/semantic_gain_v11_1_confirmation2_analysis_20260911
```

## 复核与下一条命令

已通过的 96 条 efficacy 重放可用下列命令复现：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-3d/bin/python scripts/replay_semantic_gain_v11_1_confirmation.py \
  --run eval_results/semantic_gain_v11_1_confirmation2_efficacy_20260911 \
  --prepared eval_results/semantic_gain_v11_1_confirmation2_preparation_20260911 \
  --freeze audit_results/semantic_gain_v11_1_confirmation2_execution_freeze_20260911 \
  --output eval_results/semantic_gain_v11_1_confirmation2_efficacy_replay_20260911
```

覆盖压力层40条轨迹已用下列命令独立重放通过：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-3d/bin/python scripts/replay_semantic_gain_v11_1_confirmation_stress.py \
  --run eval_results/semantic_gain_v11_1_confirmation2_stress_20260911 \
  --prepared eval_results/semantic_gain_v11_1_confirmation2_preparation_20260911 \
  --freeze audit_results/semantic_gain_v11_1_confirmation2_stress_execution_freeze_20260911 \
  --output eval_results/semantic_gain_v11_1_confirmation2_stress_replay_20260911
```

不得根据 F00--F11 的已见结果修改模型或协议。任何方法改动必须形成新版本并使用新的未来确认集。

## 数据归档范围

本检查点纳入 V11.1 源码、配置、冻结清单、开发结果、两轮确认准备数据、F efficacy 原始轨迹结果、配对统计和论文材料。因此换设备后可直接运行独立重放和覆盖压力层，不需要重新生成已封存前缀。

工作区仍有约 5.9 GiB 的更早期逐帧 RGB-D、重复网格、候选缓存和诊断目录未加入本次提交；它们在上一检查点已明确为可再生中间数据，不是续接 V11.1 所需输入。历史 Git LFS 数据集和权重仍在仓库中，必须执行 `git lfs pull`。

论文当前只能声称冻结人工双设施机制的 efficacy 语义增量，以及 efficacy 96/96、压力层40/40物理指标重放。覆盖压力层没有语义增量；自然开放词汇、大规模未知起点覆盖、外部主流基线和实车效果仍待完成。
