# V24：旧 V1 派生 PLY 逐字节无损归档（2026-09-15）

本批只处理已完成且原验证通过的 `eval_results/inspection_mechanism_v1_20260910` 中 464 个单硬链接、无活动写入的 `001.ply`、`004.ply`、`012.ply` 分支网格。归档不重新融合、不重算指标，原始传感、基线/真值/场景网格、模型、参考、源文件、旧 manifest、metrics 和 verification 均未改写。此前只读候选审计保留原样；本目录记录随后获得授权的实际执行。

## 结果与证据

- 原文件 229,322,076 字节，占用 230,240,256 字节；464 个 gzip 共 125,696,823 字节，占用 126,652,416 字节。压缩负载净释放 **103,587,840 字节（98.79 MiB）**；新审计文件与目录另有占用，实际空闲空间亦受并行工作影响。
- 每个 gzip 先以临时文件写入并 fsync，校验压缩 SHA、解压长度及原 SHA，再检查原文件 inode/属性未变、单链接且无打开的写 FD，随后移除未压缩路径。每件操作留有 `receipts.jsonl`；异常时原文件或已验证归档至少保留一份。
- `preserved_inventory_verification.json`：旧 inventory 的 **3,228/3,228** 项全部可核对，其中其余 **2,764** 项仍在原路径且 SHA 完全相同，**464** 项解压 SHA 与旧 inventory 相同；旧四份元数据文件 SHA 不变。
- `restore_roundtrip.json`：实际通过专用恢复入口将首件恢复到原路径，校验字节、权限、mtime，随后再归档；最终 464 件均处于归档状态。此检查不是仅在内存中解压。
- `manifest.json` 冻结候选清单、两个工具源码及旧输入 SHA；`target_inventory.json.gz` 保存每个原路径、SHA、原属性与压缩 SHA。`artifact_hashes.json` 覆盖本目录的工具副本、清单、审计和全部压缩文件。

唯一归档位置为：

`audit_results/inspection_mechanism_v1_ply_archive_v24_20260915/archives/<项目相对原路径>.gz`

归档数据是完整原文件字节的 gzip，不依赖 TSDF、传感重放或学习模型生成；恢复使用标准库。两个执行工具仍在项目 `scripts/`，本目录同时保留冻结源码副本 `archive_tool.py`、`restore_tool.py`。

## 原路径恢复与旧验证器

**旧的严格 replay/verification、原 inventory 逐路径检查，以及直接读取这些 PLY 的历史分析，必须先恢复其所需原路径。** 旧 manifest 未被改成接受缺失路径；不能直接把旧检查的缺文件失败称为数据损坏，也不能跳过其校验来报告通过。

在 `/root/NSO` 执行全量恢复（保留压缩副本）：

```bash
python3 -B scripts/restore_inspection_meshes_v24.py
```

全量恢复需额外容纳原文件约 **219.57 MiB**，工具另保留 **8 MiB** 空间；执行时可用空间不足 **238,628,864 字节**则拒绝，不会开始部分全量恢复。当前释放空间并不足以保证立即全量展开。只分析一个网格时可使用精确清单路径：

```bash
python3 -B scripts/restore_inspection_meshes_v24.py --only eval_results/inspection_mechanism_v1_20260910/box_101/fixed_01/far/001.ply
```

恢复到原路径后，工具再次匹配原 SHA；遇到现存且内容不同的路径会拒绝覆盖。原字节、mode、mtime 可还原，文件 inode 不要求与删除前相同。恢复入口与归档工具配套使用，均只依赖 Python 标准库；源码 SHA 必须与冻结 manifest 一致。

恢复后可以运行需要这些原路径的旧检查。若完成检查后需要重新收回展开空间，用以下命令再次验证已有 gzip 与原文件后归档，不会扩展至清单外文件：

```bash
python3 -B scripts/archive_inspection_meshes_v24.py --apply
```

日常检查归档完整性无需恢复：

```bash
python3 -B scripts/archive_inspection_meshes_v24.py --verify
```

验证旧 inventory 的所有原字节（包括归档项）可运行本批专用脚本。该检查与旧逐路径严格验证不同，结果明确记录 464 个归档项；它要求本批处于最终全部归档状态：

```bash
python3 -B audit_results/inspection_mechanism_v1_ply_archive_v24_20260915/verify_preserved_inventory.py
python3 -B scripts/archive_inspection_meshes_v24.py --verify
```

最后一条命令更新本目录的验证与产物哈希清单。原始实验元数据及旧验证结论始终保持原字节。本轮仅归档此批，不实施第二批清理。
