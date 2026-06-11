# paper_fast 评估结果

> 完整分析报告见 [`docs/PAPER_FAST_EVAL_REPORT.md`](../../docs/PAPER_FAST_EVAL_REPORT.md)

| 项目 | 值 |
|------|-----|
| 权重 | `trained_models/paper_fast/model_best.{global,reach}` |
| 数据集 | Gibson `val` split |
| 回合数 | 20 |
| 每回合步数 | 500 |
| 模式 | `--paper_mode`（OV-SDF + STGHP + RPN-UQ + IGCR） |
| 评估时间 | 2026-06-11 |

## 主要指标

| 指标 | 结果 |
|------|------|
| 单回合最大覆盖率 | **79.55 ± 25.74%**（80 条场景-回合记录） |
| 探索面积（单回合最大） | **12,123 ± 581 m²** |
| 步曲线均值覆盖率末点 | **79.5%** |
| 评估耗时 | ~24 分钟 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `nso_eval_paper_fast.json` | 结构化指标摘要 |
| `explored_ratio.txt` | 每回合逐步覆盖率 |
| `explored_area.txt` | 每回合逐步探索面积 |
| `train.log` | main.py 评估日志 |
| `eval_console.log` | 完整终端输出 |
| `training_final_excerpt.txt` | 训练日志首尾摘录（完整训练日志在服务器云盘） |

## 复现命令

```bash
bash scripts/eval_paper_fast.sh 20
python scripts/summarize_eval_results.py \
  --log /mnt/nso_data/nso_runs/models/paper_fast_eval/train.log \
  --dump /mnt/nso_data/nso_runs/dump/paper_fast_eval \
  --tag paper_fast \
  --output eval_results/paper_fast/nso_eval_paper_fast.json
```
