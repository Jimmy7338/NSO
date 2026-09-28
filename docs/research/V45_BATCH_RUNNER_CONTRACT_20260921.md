# V45 开发批次编排约定（2026-09-21）

## 范围与入口

`nso.development_batch_v45.run_batch_v45(output_root, preflight_only=False)` 和
`scripts/run_development_batch_v45.py` 串联已冻结的 V44 单次运行、保存观测独立重算和离线评价。
五个槽固定为 R2_A_diagnostic、R3_A_G、R3_A_S、R4_C_G、R4_C_S；每个槽仍由 V44 入口写入
V43 的同一份 `audit_results/v43_development_batch_20260921/start_ledger.json`。
不得通过更换批次目录增加 World 次数、清空 ledger 或自动重试失败槽。

R2 必须结束为返航成功，且完整保存观测重算通过，才准许首个自主槽。
后续 G/S 正常结束却未完成任务（blocked、预算结束、未返航）的结果保留原身份；
只要完整性和独立重算通过，仍执行其余已声明槽，避免根据表现取消另一方法。
技术异常、不完整预测、来源变化、重算不一致则停止后续启动并列出未启动槽。
评价失败也停止，不能把缺少评价的槽当作有效低分/零分。

独立重算只读取已经保存的观测并重建 TSDF；它不是新 World 的轨迹重放，也不证明策略决策完全确定。
开发批次只有 A/C 各一个 G/S 对，不能作为独立测试集或显著性检验。

## 不可更改的输入

V43 原始协议按字节 SHA 记录且执行过程中复核。评价参考在任何对应实际轨迹之前已封存：

- DEV_A_00 manifest：`96d00c04384aefbecd55a6a81dcdfc16a432fae47e8c2761204388b5a4cd4326`。
- DEV_C_00 manifest：`a4c99ff26ef19fde5be5cd5c67fd00e6ea4152e2acc42e432f0c5e33ade8af4c`。

每个参考的 manifest 和四个成员文件均验证 SHA/字节数，逐槽复核。
编排源码、V44 原始入口/重算/评价源码以及其冻结依赖记录 SHA，并在每次编排 attempt 中保存副本。
每个完成 episode 的 manifest SHA、参考 SHA 和源码 SHA 先写入不可覆盖的 binding，再调用重算/评价。
两个子进程显式接受这些 SHA 参数，不能重新选择评价参考或为不同方法选择 ROI。

## 空间、超时与输出界限

实际 episode 输出的持久存储门仍为 `max(10 GiB, 2 × 384 MiB + 2 GiB)`。先检查所选输出路径；
仓库永久启动日志只存有界元数据，另需非 RAM 文件系统以及 64 MiB（32 MiB 上限加 32 MiB 余量）。
这两项检查用途不同；使用外部持久盘作为 episode 输出时，不要求仓库元数据分区也有 10 GiB。
门失败时不读取场景/参考、不创建输出、不写启动日志、不预留槽、不启动任何子进程。
`--preflight-only` 成功只验证参考/来源/现存 ledger，不生成文件、不启动 World。
每个新实际槽之前重新检查存储；V44 内部与传感器工厂仍各自保留原空间门。

每个子进程最大运行 900 秒；这是外层故障终止上限，不是规划实时性保证，也没有改变 V43 的
600 秒步骤边界检查。超时/输出超量会终止整个子进程组，保留已捕获日志，并永远禁止自动重启。
两个标准流各最多保留 256 KiB；日志后缀被截断时任务标为输出超限，不能当作完整结果。
V44 原始单文件/episode 界限保持 32/64 MiB；评价子进程单文件限制 256 KiB。
批次编排元数据总量最多 32 MiB（episode 不计其中且仍由 V44 独立限制），其中保留 256 KiB 给总结。
每个输出根目录最多八次编排 attempt，文件均独占创建并 fsync，不覆盖已有产物。
5 × 64 MiB 的 episode 加上 32 MiB 编排元数据仍在原 384 MiB 峰值预算内。
原先已经失败/中断的数据保留；本入口不会删除文件以满足界限。

## 永久尝试标记、恢复及故障身份

除 V43 ledger 外，在 V44 子进程启动之前写入
`audit_results/v45_development_batch_20260921/launches/<run_id>.json`。
该不可覆盖标记防止“子进程在 V43 reserve 前超时，于是换输出目录重启”的漏洞。
它不替代 V43 真实 World 计数；有 marker 无 reservation 表示启动尝试中断，必须审计，不能认定实际 World 为零。
全局非阻塞锁防止两个编排器同时发起同一槽。

已完成且绑定 ledger 的 episode 可恢复后处理；已封存的命令回执、stdout/stderr、JSON 结果按 SHA 复核并重用，
不重新调用子进程。存在 started 而缺少完整 command 回执的后处理任务视为中断，不自动重试。
对于 ledger 中 reserved、技术失败或孤立 episode 目录，停止并保留原数据。
已完成但历史上未经过 V45 的 V44 episode 可以只做后处理，无须再次启动 World。

目录约定：

- `output_root/episodes/<run_id>/`：原 V44 完整传感器/决策/预测文件。
- `output_root/reviews/<run_id>/binding.json`：episode、参考和编排来源绑定。
- `output_root/reviews/<run_id>/{episode_command,verification,evaluation}/`：不可覆盖命令与标准流。
- `output_root/orchestrator/attempt_NNN/`：本次来源快照、started 与 summary；恢复产生新的 attempt。

summary schema 为 `v45.development_batch.v1`。每个 runs 行提供 run_id、episode_root、episode_status、
world_created、orchestration_status、episode_manifest_sha256，以及 verification/evaluation 的绝对路径和 SHA。
若出现永久启动标记，同时返回 launch_receipt_path/sha256。尚未生成的证据字段保留 null。
启动标记存在但 ledger 缺失时 world_created=null，不能宣称没有创建过 World；
这类槽列于 launches_without_reservation，不列于 unattempted_runs。
外部汇总器必须复核原始文件和 ledger，不能直接信任 summary 的指标或成功标签。

## 测试与证据界限

单元测试使用临时目录和有限 subprocess fake，fake 只写有限 JSON，绝不调用 World 或模拟新的可探索场景。
真实子进程测试仅打印有限文本、触发输出界限或短暂 sleep，以验证硬停止和日志界限。
这类测试验证编排与故障处理，不是三维自主实验，不产生语义性能证据。
