# 已完成三维实验的可恢复存储清理（2026-09-11）

本次只处理 NSO 的临时 PDF 依赖与三组已完成、已全量独立回放的分支 TSDF 网格。原始 RGB-D、雷达扫描、参考表面、前缀网格、动作、候选、评分、指标、原始文件清单、冻结源码包与独立回放收据全部保留。V4 历史别名网格和 V7 既有清理均未改动。

已完成并最终核验：224 / 224 网格逐字节可重建，全部保留原始文件再次通过 SHA 检查。网格释放分配空间 740.49 MiB，临时包文件分配空间 67.93 MiB。并行工作区写入后，最终验证时剩余 1016.36 MiB，满足 400 MiB 余量。精确状态见 `summary.json` 及各 run 的 `compaction.json`。

## 固定范围

| 已完成实验 | 原分支数 | 删除后可重建的网格数 | 逻辑字节 |
|---|---:|---:|---:|
| counterfactual_view_development_first_20260910 | 72 | 144 | 520921499 |
| counterfactual_candidate_augmentation_v6_20260911 | 16 | 32 | 112665446 |
| competition_v8_1_execution_20260911 | 24 | 48 | 141821767 |

只允许每条实际完整分支的 `arrival_mesh.npz` 与 `final_mesh.npz`；保留全部 prefix mesh。删除前逐个使用该 run 冻结 `sources.zip` 内的 mapper 重放保留的实际传感器，重新压缩 NPZ，再要求**完整文件 SHA-256 与原始清单逐字节一致**。这次不是只抽样验证：224 个目标必须全部成功，才能逐 run 删除。初始试运行样本另外保留收据。

`derived_mesh_candidate_inventory.json` 逐文件记录 inode、链接数与分配字节。目标均无外部硬链接；实际已释放目标文件分配空间 776462336 字节。清理时的文件系统 free 差值单独保留，可能同时受工作区其他合法写入影响，不以逻辑大小冒充实际容量。

## 验证与恢复

工具：`/root/NSO/scripts/compact_completed_branch_meshes.py`。每组审计目录保留执行时工具原文 `compaction_tool.py`、全量 `raw_reconstruction.json` 与 `compaction.json`。收据绑定原文件清单、原回放、冻结源码包、metadata、每个重建数组与完整 NPZ 的 SHA，以及实际读取 raw/动作/候选输入的原始 SHA。

验证当前精简资产（把 run 名替换为表中任一项）：

```bash
.venv-3d/bin/python scripts/compact_completed_branch_meshes.py --run eval_results/competition_v8_1_execution_20260911 --validate-only
```

需要使用旧分析或旧完整回放时，先恢复它将读取的各 run 全部缺失 mesh：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-3d/bin/python scripts/compact_completed_branch_meshes.py --run eval_results/competition_v8_1_execution_20260911 --restore
```

恢复逐文件校验原始 SHA 后以不覆盖既有路径的原子硬链接发布，要求保留 400 MiB 余量。恢复全过程只读原始传感器、实际动作、候选和 config，不构造 world，不读参考/奖励，不重新导航。冻结工具原文用于核对与必要时恢复脚本版本，正常命令使用项目 scripts 路径。已实际恢复 augmentation 的一个 arrival mesh（2,974,288 字节），原子发布后再次验证原 SHA，并按既有完整见证再精简；实测见 `counterfactual_candidate_augmentation_v6_20260911/restoration_publication_probe.json`。

旧严格检查器仍会拒绝缺 mesh 的精简状态，已用 augmentation 的原 `check_manifest` 验证拒绝而不覆写原 `verification.json`；该行为是有意保留。新 `--validate-only` 只证明被完整见证的派生文件可以缺失且所有保留文件原样，不替代物理回放。部分清理、缺 raw、损坏 raw、损坏源码或收据、未列出文件均拒绝。

## 临时依赖

`/tmp/nso-plan-pdf` 为旧论文文档任务安装的 PyMuPDF 1.28.2 / pypdf 6.18.0，清理前未发现进程映射、打开文件或 cwd 引用。版本、重装命令、逐文件 SHA 与大小已归档于 `temporary_pdf_packages_inventory.json`；清理收据另存，约释放 68 MiB。

## 限制

这项工作恢复可用存储并增强资产可恢复性，不增加方法性能证据。已有实验结果和失败记录原样保留。9 项独立文件门测试见 `evidence_gate_tests.json`；真实 raw 到 NPZ 的验证由各 run 完整收据证明。
