本次在 NSO 内完成一次有界、逐字节可恢复的旧派生产物归档，解决可用空间降至 0 的实际阻塞。没有新实验、传感采集、融合或重新评分；原始传感包、模型、参考缓存、旧指标和旧封存清单均保留。最终数量、分配块净释放量与当时可用空间见 [final_summary.json](/root/NSO/audit_results/completed_derived_archive_v24b_20260915/final_summary.json)。

| 已结束实验组 | 本次归档 | 数量 |
|---|---|---:|
| virtual3d_pilot_v1_20260910 | 单硬链接 PLY 检查点，排除真值和共享文件 | 121 |
| competition_v8_1_execution_20260911 | decisions.jsonl | 24 |
| competition_v9_1_execution_20260911 | decisions.jsonl | 44 |
| semantic_gain_v13_2_outbound_paired_20260914 | 单硬链接 modules/candidates/selection 派生日志 | 98 |
| semantic_gain_v13_paired_information_resumed_20260914 | 同上 | 88 |

375 份原文件共占 641,560,576 分配字节。采用 gzip 6，逐件先写磁盘、fsync、解压核对原 SHA-256，再检查 inode、链接数与活动写句柄后移除原路径。早期中断保留在本轮 receipts：根用户虽然看到 16 MiB 的 f_bfree 保留块，实际写入仍报 ENOSPC；V24c 额外检查实际可写 bavail。后来按授权增加旧派生日志归档，达到数百 MiB 的恢复目标。

零空间启动包含两次受限措施：25 个源文件存在、未追踪的 Python 缓存释放 303,104 B，源 SHA 前后不变；11 个已结束 shape 检查中完全重复的 NPZ 改为硬链接，释放 303,104 B，原 24 项产物的路径集合和 SHA 全部不变。后者各路径原 inode、mode、uid/gid、mtime 已记录在 bootstrap_exact_dedup.json；共享 inode 的 mtime 随锚文件，不等于每个路径原先各自的 mtime，禁止原地修改锚文件。若需恢复独立 mtime，须先复制为独立 inode，再按该清单恢复元数据。

清出的少量空间立即被持续写入消耗后，一个旧 V8 决策日志的压缩流及元数据先存入**本轮尚未封存 receipt 已分配磁盘块的空余**，fsync 并读回解压验证，之后才移除原日志。该压缩字节现在同时记录在 emergency_v8_bootstrap.json；并非只保存在 RAM。其余 23 个启动压缩包起初写在原文件旁，随后按 v8_archive_consolidation.json 无损迁入本归档，旧实验目录不再残留这些新旁路文件。原始失败记录没有删除。

[独立核验](/root/NSO/audit_results/completed_derived_archive_v24b_20260915/independent_verification.json)覆盖四个旧清单共 26,918 项：26,528 个仍在原路径的文件逐 SHA 一致，254 个本次日志匹配原清单，136 个原先已压缩网格匹配之前的归档记录；没有新的未映射缺失。早期 pilot 没有完整逐包原始哈希清单，因此如实核对 32 条终点、3,780 帧及对应 scan 数量、原 summary/metadata 和 8 个真值文件，未虚构历史包哈希。此前 464 PLY 归档的 475 项封存产物全部哈希一致，未改动该归档。

统一恢复清单是 restoration_manifest.json.gz，记录原路径、原字节 SHA、压缩位置及 SHA、标准元数据。351 份常规归档还记录扩展属性；24 份零空间 V8 启动日志记录 mode、uid/gid、atime/mtime 等标准元数据，没有额外的扩展属性快照。inode、ctime 只能作为历史记录，普通文件 API 不能恢复原 inode/ctime。三个真实原路径恢复检查覆盖 PLY、V13 日志和嵌入式 V8 压缩流：原字节、mode、mtime 全部通过，随后重新归档，压缩字节始终保留。

只读检查所有压缩内容及原字节哈希：

```bash
cd /root/NSO
python3 -B scripts/restore_completed_derived_v24b.py
```

按需恢复一个原路径（--only 可重复；回执必须使用归档目录之外的新路径）：

```bash
python3 -B scripts/restore_completed_derived_v24b.py --restore \
  --only eval_results/competition_v8_1_execution_20260911/P0/shelf_west/candidate_001/decisions.jsonl \
  --receipt /root/NSO/audit_results/my_restore_receipt.jsonl
```

省略 --only 可恢复全部 375 个原路径；保留压缩归档时，需要至少 **649,949,184 B（约 619.84 MiB）实际可用空间**。恢复工具遇到不一致的现有路径会拒绝覆盖，支持中断后按原清单继续。--rearchive 可在验证压缩包和已恢复文件字节一致后重新移除派生原路径，同样需要新的独立回执。

旧严格回放或读取本次 PLY/日志的分析器必须先恢复所需原路径，不能直接把“归档可恢复”冒充旧验证器已通过。V8/V9.1 原先的 48/88 份网格压缩仍遵循它们自己的恢复流程；本工具只负责本次 375 份文件。不要重新执行已经完成的 bootstrap/归档准备脚本。此前候选和失败、执行脚本均作为审计证据保留。
