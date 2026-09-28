# Article 实验的无损归档与可恢复精简方案（2026-09-28）

当前仅完成归档、计划和恢复验证，**没有删除任何真实实验 episode 中的原始文件**。G、B、S 三条已完成且独立审查通过的 AISLE 开发任务均保留完整目录，并分别产生无损 tar.xz 包。58 个冻结执行源码、协议、原科学 manifest 及独立审查器均未修改。

## 已完成的无损归档

| 方法 | 原文件字节 | 解压步骤后的逻辑字节 | tar.xz 字节 | 原文件／包体积 |
|---|---:|---:|---:|---:|
| G | 44,288,657 | 90,819,613 | 7,533,272 | 5.8791 |
| B | 44,300,692 | 90,830,884 | 7,537,956 | 5.8770 |
| S | 44,301,475 | 90,834,710 | 7,535,280 | 5.8792 |

三个包分别通过659个原文件的字节恢复校验，其中161个步骤gzip按原level=6、mtime=0及当前zlib 1.3重新生成，原长度和SHA全部一致。G另实际恢复到全新临时目录并通过原artifact manifest复核。该首次临时副本已在再次核对659个文件后清理，`tar.xz.restore_report.json`保留；清理不涉及真实实验目录。

包目录为[audit_results/article_stage_20260928/episode_archives_v1](../../audit_results/article_stage_20260928/episode_archives_v1)。每包旁的`.report.json`给出原字节、磁盘分配量、逻辑字节、包字节、验证数量和耗时。现有`.gitattributes`没有对`.tar.xz`配置LFS，三个包均小于单包100,000,000字节上限。

归档工具保持为[scripts/archive_article_episode_20260928.py](../../scripts/archive_article_episode_20260928.py)，其作用是字节保存及恢复，不运行策略、传感器、TSDF或表面评价。流式XZ解码限制128 MiB解码内存、总展开量和成员大小；路径越界、软硬链接、重复文件、未声明成员及截断XZ均拒绝。7项归档测试通过。

## 精简工作副本的具体方案

新增独立适配器[scripts/compact_article_episode_20260928.py](../../scripts/compact_article_episode_20260928.py)，不修改归档器或冻结执行源码。适配器提供三个明确分开的入口：

1. `plan`：核验完整原目录、已通过的独立review及无损包，生成外部计划与原阶段ledger快照，不删除文件。
2. `apply`：只有调用者提供精确的已审查计划SHA才执行；再次完整验证包、原目录、review、ledger目标条目和工具源码，先持久化包及外部操作意图，再移除允许的原始工作副本文件。
3. `materialize`：将完整原episode恢复到全新临时工作区，并提供原阶段ledger快照，使现有独立审查器可以直接处理；不恢复或运行策略。

删除范围固定为`packets/NNN_rgbd.npz`、`packets/NNN_scan.npz`、`steps/NNN.json.gz`、`prediction/mesh.npz`及`prediction/occupancy.npz`。其余文件全部保留，包括每帧执行receipt、协议、起始记录、终点结果、评价、controller最终状态、mapper元数据、原artifact manifest和预测封存。未知文件类型默认保留。原manifest继续描述完整可恢复episode，内容不被改写。

精简状态在episode外的独立receipt中说明，原目录仍占据原run ID，因此不会打开重新运行同一slot的入口。查看逐帧原始数据或重新审查时，通过`materialize`还原到新工作区。原review脚本继续严格要求完整原始库存，不在内部绕过缺失文件检查。

## 当前可审查计划

正式候选为[G durable_v2计划](../../audit_results/article_stage_20260928/compaction_plans_v1/dev_AISLE_G_b160_n92801_durable_v2/plan.json)，SHA256：

`000802e2e34826998a3a78ff48350f31ed2362868e70c15c90804d97dba1bea6`

该计划列出485个可移除文件和174个原样保留文件：

| 项目 | 字节 |
|---|---:|
| 当前原目录逻辑体积 | 44,288,657 |
| 可移除原始工作副本 | 41,895,562 |
| 保留原元数据 | 2,393,095 |
| 无损包 | 7,533,272 |
| 保留元数据＋包 | 9,926,367 |
| 预计释放已分配磁盘块 | 42,786,816 |

包已存在，因此将来执行该计划时，预计直接释放约40.8 MiB已分配磁盘空间。原目录与包的长期组合体积约9.47 MiB。仍未执行`apply`，真实原目录保持659个文件。

同目录早期未执行计划`dev_AISLE_G_b160_n92801`保留供审计；随后适配器补充了目录持久化和删除前归档同步，早期计划的源码pin不匹配当前适配器，不能执行。需要使用上述`durable_v2`候选。

恢复示例：

```bash
.venv/bin/python -B scripts/compact_article_episode_20260928.py materialize \
  --plan audit_results/article_stage_20260928/compaction_plans_v1/dev_AISLE_G_b160_n92801_durable_v2 \
  --workspace /tmp/article_review_workspace_NEW
```

返回的`review_command`指定完整恢复episode和新的review输出路径，可直接调用现有审查器。`apply`要求`--approve-plan-sha256`，本轮没有对真实episode调用该入口。

已用`durable_v2`计划实际恢复G的659个文件，恢复耗时7.29秒，随后不修改既有审查器直接复核，6433项检查全部通过；除恢复工作区路径外，审查结果与原review完全相同，包含全部指标、动作成本、来源pin和权限检查。证据保存在[compaction_validation_v1/dev_AISLE_G_b160_n92801_durable_v2](../../audit_results/article_stage_20260928/compaction_validation_v1/dev_AISLE_G_b160_n92801_durable_v2)。两次适配器测试生成的临时恢复副本均在重新核对完整库存及SHA后清理，恢复记录、独立review和清理回执已持久保存。最终复核G、B、S真实原目录各659个文件，仍与各自原manifest完全一致。

## 96条任务的存储估计

以当前G的实际体积作为单条示例，96条原目录约4,251,711,072字节，即3.96 GiB；采用“包＋保留元数据”约952,931,232字节，即0.89 GiB。按保留内容与Git对象各一份保守计算约1.77 GiB，另需预留公共场景、参考表面、源码归档、审查输出、计划和临时恢复空间。单条恢复约需44.3 MB；完成审查后可核验并清理该次临时副本。

上述为当前三条开发任务体积相近时的估计，不保证所有场景都有相同压缩率。执行器仍保留每条64 MiB存储上限和至少1 GiB空闲门；精简适配器不修改该门、96条主实验上限或累计开始registry。可优先在新任务间对已完成且已review的任务归档；是否精简原工作副本由后续具体操作决定。

## 失败任务如何保留

当前成功归档器和精简入口明确拒绝失败、未完成或未通过review的任务，6项精简测试包含该限制。失败不会被重新标记为合格，也不会为通过打包而修改ledger。

当前可行做法是保留失败目录原样，并在容量门触发时停止新增任务。若后续失败记录占用显著空间，可以新增独立的“失败保全包”格式：要求ledger已处于终态；逐字节保存全部现存文件、缺项列表、异常、原始状态及已有manifest；截断或不规范gzip按原字节保存；恢复后保持同样缺项和失败状态。该格式只能声明文件完整保存，不能声明episode合格、成功评价或原manifest完备。失败包需单独测试和审查，目前未实现，也未进行任何失败目录精简。

## 中断与安全验证

包和外部操作意图在首次删除前显式fsync，删除完成后同步目录并生成外部完成receipt。若中断，保留`prepared.json`和可写出的`incomplete.json`，不自动重试或继续删除；仍可从同一验证包完整恢复到新工作区。原科学manifest不受影响。

6项精简回归仅在自动清理的合成临时目录中执行删除，验证：计划不删除、显式批准后仅移除允许文件、保留元数据字节完全一致、归档损坏时拒绝、metadata无法加入删除名单、恢复路径无法越界、中途中断仍可完整恢复，以及失败状态不能进入精简。真实实验没有删除操作。
