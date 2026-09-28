# V45 批次汇总独立审查（2026-09-21）

审查对象为 `nso/batch_report_v45.py` 与 `scripts/report_development_batch_v45.py`，同时只读核对 V43 启动账本、V44 保存观测验证及离线评价、V45 编排器的证据字段。没有创建 World、执行传感动作或计算新性能结果。

## 首次审查发现

1. **历史摘要中的三个证据哈希未被核对。** 初版 `_verified_endpoint` 自行读取当前 episode manifest 哈希，然后将这个当前值传给 V44 加载器作为 expected SHA；同时没有检查批次行已经保存的 `episode_manifest_sha256`、`verification_result_sha256` 和 `evaluation_result_sha256`。它验证了文件内部身份字段及预测哈希，但不能发现同一身份下评价 JSON 的分数被改写（例如 Q 与 J 同时改为另一个符合乘法关系的值）。应先将当前三文件与历史汇总行的固定哈希比对，再进行语义身份检查；独立子进程 command/stdout 与 binding 也应保留关联。该问题不是要求重新运行评价，而是完整使用已经产生的证据固定值。

2. **预约前启动失败被归入未启动。** 初版只以 V43 账本判断启动状态。V45 永久 launch journal 已写入，而子进程在 V43 预约前超时或中断时，运行器禁止重试，但报告会显示 `not_started`。应读取 canonical launch journal，将这一情况单列为需要人工审计且不可自动重试的状态，并保留 `orchestration_status`。它与真正未尝试的槽位、已预约未完成的槽位都不同。

## 已确认的设计边界

- 报告遍历全部五个预声明槽位，不按分数或方法输赢过滤；技术失败与终点不成功仍应保留。
- 描述性 G/S 差值只可在两条终点保存观测重算与评价均通过时生成。公共 graph/spec/workspace 哈希一致包含动作预算一致，另核对参考、源代码和噪声种子。
- 合法的不成功终点可以参与描述性差值，但必须保留 task_success 标志；不能只比较成功样本而消除失败。
- 只允许 `C_nav`、`Q`、`J_nav` 的有限 [0,1] 值，且联合指标满足乘法关系；不插补未完成分数，不输出显著性检验或独立测试结论。
- 有限固定包、测试桩及空批次报告均不能作为实际语义性能证据。

## 最终复核（通过，适用边界如下）

首轮两项缺陷已修正：三份派生证据与批次固定哈希核对；子进程退出码、stdout/stderr 哈希及结果内容建立关联；验证/评价源码纳入批次源码闭包。永久 launch journal 已读入报告，预约前中断单列为 `launch_without_reservation_no_retry`。已启动但 World 是否创建尚未确认、以及已预约未终结两种状态均使用 `world_created: null`，不将未知伪装成未创建。

最终验证收据 `audit_results/v45_batch_validation_r1_20260921/result.json` 为 passed，**41 项测试通过（15 项汇总、26 项编排）**。这些是有限测试桩与子进程管理验证，不是物理自主轨迹或性能实验。首次验证目录保留；最后 null 修正的前后源码哈希记录于 `audit_results/v45_batch_report_revision_20260921/result.json`。

独立复核重新核对最终输出清单的 **43 个文件哈希/大小**、**34 组当前/归档源码 SHA-256**，全部一致。最终实际批次报告保留五个未启动槽位，预约 0、启动记录 0、自主验证终点 0、两组成对差值均为 null。没有运行 World、传感器或 TSDF。完整机器可读复核见 `audit_results/v45_batch_report_review_20260921/result.json`。

| 复核源码 | SHA-256 |
|---|---|
| `nso/batch_report_v45.py` | `788e5f8e9ce2a4a524e76c2765a16272c3c59d62738df4f11a7bf183b774e2bc` |
| `scripts/report_development_batch_v45.py` | `cd06ecaa353f922ee0424d092b5e3aea1555d8715a9998acb01d85b0c72fc3e6` |
| `tests/test_batch_report_v45.py` | `0964ff3b722c7c1cd51b8e23908375ed6161bc809444cb9497727515ec1b9ff6` |

没有尚未解决的阻断性汇总代码问题。报告仍依赖封存证据链的完整性；这些哈希用于发现错配或意外替换，不是抵抗协调改写全部记录的真实性签名。实际自主闭环与语义增益仍待持久存储门槛满足后验证。本复核不增加语义性能结论。
