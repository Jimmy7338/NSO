# 已结束 pilot 派生 PLY 无损归档

本目录只归档 `eval_results/virtual3d_pilot_v1_20260910` 中获批的 55 个独立 inode、133 个硬链接原路径。每个 inode 对应一个 `blobs/group_*.ply.gz`，逐文件落盘、fsync、解压 SHA 验证后才移除原别名。原始传感、NPZ、真值/scene/reference、模型及其他实验均不在执行清单；既有完成回执与旧归档未修改，cleanup 阶段的 ENOSPC 失败记录保留。

`restoration_inventory.json` 是原路径→压缩文件→原 SHA 的权威映射，包含每个原路径的 mode、uid/gid、mtime/atime、xattrs、原 inode/nlink。`restoration_integrity.json` 冻结映射与工具；`prepared_inventory.json` 和 `events.jsonl` 保留压缩与逐别名移除过程。inode 数值与 ctime 不能在恢复时复原，但原字节、路径、硬链接关系及已记录的常规元数据可以恢复。所有恢复出的别名仍须视为不可变证据，不能原地写其中一个而改变整组。

默认只读验证全部 55 个 gzip 的压缩 SHA、解压原 SHA 与仍存在的原路径：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/restore_pilot_linked_meshes_v25.py
```

只恢复一个原路径；必须提供一个尚不存在、位于本封存目录之外的回执路径：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/restore_pilot_linked_meshes_v25.py --restore --only eval_results/virtual3d_pilot_v1_20260910/clutter_seed82_aligned_coverage/meshes/00000.ply --receipt audit_results/my_pilot_restore.jsonl
```

同一命令增加 `--group` 会恢复该 inode 对应的全部原别名，并重建共享 inode 关系。省略 `--only` 则恢复本归档全部路径。原路径已有内容时仅接受相同字节，拒绝覆盖。空间门保留至少 64 MiB；全量恢复需在保留 gzip 的同时再容纳原始独立 inode 的占用，不能用压缩后大小估算恢复需求。

验证后重新归档恢复路径（保留现有 gzip，不重新编码）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/restore_pilot_linked_meshes_v25.py --rearchive --only eval_results/virtual3d_pilot_v1_20260910/clutter_seed82_aligned_coverage/meshes/00000.ply --group --receipt audit_results/my_pilot_rearchive.jsonl
```

严格旧回放、旧 inventory 检查和直接打开 PLY 的读者，必须先恢复它们需要的所有原路径。此前已存在的两个归档映射仍分别有效；本工具只处理本次 133 路径，不能自动替代此前恢复入口。不得为通过旧检查而改写旧 inventory 或评价结果。

一次实际恢复证明位于 `audit_results/virtual3d_pilot_linked_mesh_restore_proof_v25_20260915`：`group_053` 的 4 个原路径已恢复，SHA、共享 inode/nlink、mode、uid/gid、mtime/atime、xattrs 均通过，随后重新归档；gzip 全程保留。此证明只验证恢复工具，不是新的建图/策略实验。

异常恢复：若归档过程中失败而尚未形成最终 integrity，工具须显式使用 `--allow-incomplete`。它只接受 `events.jsonl` 中已持久化 `gzip_verified` 的组，并重新检查 gzip 与解压 SHA；未进入该阶段的原件不应被移除。不会把失败自动计为完成。
