# Exposure V3 运行与冻结链复核

2026-09-28。只读复核执行代码、旧封存源码和归档；没有修改执行源码或创建 World。复核时真实 V3 配置尚不存在。

**结论：未发现 V3 引入的源码绑定、配对配置、累计尝试上限或动作预算偏差。** 新闭包为65文件，Ground旧62文件全部维持原 SHA，原zip的文件集合与逐文件内容也匹配。新闭包只移出不再执行的 Ground CLI，加入 V3 controller、predictor、runner 和 CLI 四文件；移出的旧 CLI 仍被冻结 helper 单独验证。

| 检查 | 结果与边界 |
|---|---|
| 控制器信息权限 | Ground ledger、residual、规划与 mapper 接口不变，仅首次接帧前替换共同 predictor；四方法均启用曝光去重。 |
| 配对 | 完整 Ground 槽位一一覆盖；方法、场景、160预算、92801种子，以及资产、reference、mapper、metric、数值构建、最大耗时、单集空间和 reserve 相同。controller 只允许多一个共同曝光开关。 |
| 预测器配置绑定 | controller 收据保存新版 predictor 的实际 configuration SHA；新 predictor 原文另由新闭包和冻结 helper 的审查后固定 SHA 绑定。 |
| 世界与尝试 | 仍使用原 ArticleLedger。global reserve 先于本地 reserve，失败和未知占位不退款；phase 名与 stage 父目录保持相同。 |
| 总量 | 原共享 cap 保持 development12/main96/ablation24。当前冻结 helper 要求 Ground 恰占用12个终态消融槽，Exposure最多再12；新目录/版本不重置 cap。 |
| 物理预算 | 仍由原 driver/router 使用原 paid budget；本次所有槽160，返航、碰撞和融合计数资格不变。 |
| 规范评价 | adapter、原始 prediction seal、独立 preprocessing、四实例分母及固定 metric 版本相同。原封存 mesh 不被适配数组覆盖。 |
| 调度 | 每协议独占 flock、行政 launch claim、全局 ledger、最多双 worker 和累计 cap 均沿用。进程启动后失败不自动重试，后续不同预声明槽位仍可执行。 |
| 存储 | 64 MiB/episode 与1 GiB持久 reserve 保持；调度另加每活动进程余量和256 MiB scratch。两个不同协议并发各自调度可能竞争空间，当前 Ground完成后再启动V3可避免此情形。 |

需在运行说明中明确以下边界，而不能把它们写成已自动保证的条件：

1. **600秒不是全生命周期超时。** `maximum_elapsed_s=600` 传给 paid driver，参考加载、初始化开销及终态后的离线评价不受这个门强制结束。batch 的 `--max-wall-seconds` 只停止新启动，已经活动的子进程继续收尾。夜间吞吐和截止时间必须据实际完整耗时估算，不能简单按600秒做硬上界。
2. **当前V3冻结 helper检查 Ground终态，不检查独立review通过。** 预声明要求的独立审查仍是主线程的启动责任。main helper应将全36份终态与对应review纳入机器检查，失败episode可以被审查通过但仍不合格，二者不能混用。
3. **main机制门尚未在冻结 runner中自动执行。** runner保留未来main通道，只检查原T0/T1分割、合法配置及累计96上限，不理解人类“可行动语义作用”判断。未来应由新行政helper先验证明确批准的门文件，再生成固定48配置；不能仅因runner接受main字段就启动。
4. **声明文件与source archive的责任分层。** freeze helper校验预声明及旧archive，scheduler每次新启动校验新archive和live source；runner独立执行时校验live source闭包与协议，但不重新理解预声明内容或读取archive中的源码执行。正式启动应使用冻结配置与行政批入口。
5. **协议不可在启动期间编辑。** 与旧生命周期相同，runner先解析配置，加载reference后才记录协议文件SHA；理论上这段窗口内的文件改写可能造成解析内容与稍后保存内容不一致。固定配置、scheduler启动前绑定及独立review提供实际工作流保护。冻结后不改配置是必要操作约束；此继承窗口不是V3新增行为，本次未改被冻结源码。

本轮没有发现需要修改冻结科学实现的阻断问题。下一个主实验配置应以V3实际完整耗时重新核对时间/空间资源，并要求明确的root机制门；不能把共同去重改善直接解释成S对B的语义收益。
