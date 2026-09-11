# V9.1 网格可恢复精简（2026-09-11）

固定处理 `eval_results/competition_v9_1_execution_20260911` 的 44 条已完成路线：Q0/Q1/Q2 每个 arrangement 六条，Q3 每个 arrangement 四条；每条只处理 arrival/final 两个 TSDF mesh，共 88 个。全部 prefix mesh、原始 RGB-D/雷达、实际动作、参考表面、visibility、metrics、候选/预测、source archive、原始清单和独立回放保持原样。

已完成并最终核验：88 / 88 网格逐字节可重建，释放 262.30 MiB 分配空间。最终验证时可用 646.51 MiB，高于 400 MiB 余量。原始清单 5,596 项中，88 项为完整见证的精简网格，5,508 个保留文件再次通过原 SHA 检查，包括 5,072 个 raw 文件路径。精确收据见 `summary.json` 和子目录 `competition_v9_1_execution_20260911/compaction.json`。

## 新工具与前提

新文件 `/root/NSO/scripts/compact_competition_v9_1_meshes.py` 从已验证的三组历史清理工具单独派生，未修改旧工具、旧实验冻结源码或 V10 代码。新工具只允许这个 run 的固定 88 文件，不能把 Q3 扩展为六路线，也不能丢掉其他合法目标。

入口要求原 44 分支 / 1328 动作 full replay，且保留独立 3479 项指标检查的完整收据和输入绑定。指标审核的“通过”只证明数值核验一致，原两项 F1/C×F1 幅度失败门不被改写。清理范围事先固定为全部完整路线，不依赖哪条效果好。

每个目标由 `sources.zip` 内的冻结 `SemanticHistoryMapperV3` 与 V9 config，从保留的实际前缀和分支 RGB-D/scans 重新构造。重建 worker 不构造 world，不读取 reference 或 outcome 文件，也不调用规划/评分/评估。每个数组的 dtype/shape/SHA 与完整压缩 NPZ 的原始 SHA 均保存；88 个全部字节一致后才开始 unlink。管理入口仅使用已有指标收据的完成状态、计数及哈希，不用其中性能值选择删除范围。

`target_inventory.json` 记录删除前逐文件 bytes、allocated bytes、inode、nlink 和 SHA。88 文件全部 nlink=1，共 274724906 逻辑字节，275046400 分配字节（约 262.30 MiB）。文件系统 free 差另列，因为其他授权进程可能并行写入。

## 验证与恢复

核验精简后的完整保留资产及网格见证：

```bash
.venv-3d/bin/python scripts/compact_competition_v9_1_meshes.py --run eval_results/competition_v9_1_execution_20260911 --validate-only
```

旧独立回放、旧完整文件检查或需要 mesh 的分析，在运行前恢复：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-3d/bin/python scripts/compact_competition_v9_1_meshes.py --run eval_results/competition_v9_1_execution_20260911 --restore
```

恢复使用冻结源码和 raw，逐文件检查完整 SHA 后原子发布；不覆盖已存在路径，并保留 400 MiB 余量。当前工作脚本的执行时原文也保存在子目录 `compaction_tool.py`，供版本核对或必要时还原工具。旧严格核验器不会把缺 mesh 默认为通过；只有专门的 `--validate-only` 才识别完整的精简见证。

独立回放与 3479 项指标核验已经完成的原有收据仍有效，但完整重运行这些工具需要先恢复它们声明读取的所有资产。本次存储操作不增加任何方法性能证据。

## 检查范围

10 项文件保护测试覆盖完整 6/6/6/4 白名单、缺少目标、额外 Q3 目标、raw 丢失/变更、损坏收据或源码、未绑定输入、部分清理、路径越界与软链接拒绝。真实原始数据到 NPZ 的证明另见每组 `raw_reconstruction.json`，不能用单元测试代替。

实际恢复写回也已验证：`Q0/shelf_east/candidate_000/arrival_mesh.npz`（2868731 字节）原子发布后匹配原文件 SHA，再依据原完整见证重新精简。原回放、独立指标审核和原清单的 SHA 均保持不变。
