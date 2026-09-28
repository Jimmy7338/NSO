# V44 已保存观测的独立重算契约（2026-09-21）

本实现回答一个有限问题：从已经保存的 RGB-D、单线扫描、公共参数重新创建映射器和原控制器，能否逐步得到相同的状态、决策及最终预测。它不创建 World、不查询传感器、不执行动作，也不能说明换一个策略会在真实环境走出怎样的新轨迹。V43 的五次启动上限、持久磁盘门槛、`maximum_replays_authorized_here=0` 均保持不变；该零上限不被转换为新增物理或反事实轨迹的许可。

## 输入与失败隔离

`nso.saved_replay_v44.load_saved_episode_v44` 在初始化映射器之前验证终端清单的所有大小与 SHA256；拒绝清单外文件、缺失文件、符号链接、不安全路径、重复 JSON 字段、非有限 JSON 数字以及未声明的 NPZ 字段。总归档限制为 64 MiB，单文件及 NPZ 解压后数组载荷限 32 MiB。数据采用 `allow_pickle=False` 读取。

原 V43 的 19 份源码和配置，或其增加 V44 writer/runner 后的 21 份闭包，必须同时匹配当前源码、归档副本和 started 清单。这是显式保存的项目源码闭包；它不是完整 Python/操作系统环境的归档。最终映射器快照还包含 Open3D 版本，重算按原值比较。

已保存包必须从 0 开始连续，每一步恰有一个 RGB-D、scan、执行回执及完整 step 记录。验证唯一 frame_id、paid_step、逐动作收费、先前位姿、0.25 m/30° 执行、扫描时间及姿态、前一步动作与下一包的因果对应；映射与控制器证据必须引用当前包 SHA。最终 mapper receipts 必须等于各步已保存映射回执，frames/融合次数及累计 belief SHA 必须与预测和终端结果一致。

V44 支持原始 `steps/NNN.json` 或确定性压缩的 `.json.gz`；一个逻辑步不得同时有两个别名。压缩字节由清单绑定，编码账本验证尺寸、级别和声明参数，解压读取严格限制在 32 MiB + 1 字节并拒绝溢出。压缩只改变存储，不改变字段或策略。

完整的 blocked、预算耗尽和未确认返航可读取，`task_success` 仅在 `controller_stop` 且确认 XY/朝向返航时为真。`episode_error`、输出上限、超时、预测保存失败不会截取成成功的轨迹；完整性检查后抛出 `FailedSavedEpisodeV44`。在最终清单生成前崩溃的尝试同样不能重算，可另审计原始失败文件与启动账本。

标记 `finite_fixture=True` 的测试必须声明 World 未创建，始终 `eligible_study_episode=False`。真实研究产物另须匹配原持久启动账本的 run_id、源码、结果 SHA、最终状态、World 标记和运行计数；其公共图、任务预算、工作空间也必须等于冻结公共 bundle。有限替身不能仅删除标签就升级为真实实验。

外部 `expected_manifest_sha256` 可固定清单内容，命令行和 API 均支持。未提供这个外部锚时，清单验证仅检测意外损坏，不能证明未发生清单与全部产物的协调替换。该区别在结果中显式记录。

## 重算与差异

`scripts/replay_episode_v44.py EPISODE --expected-manifest-sha256 SHA` 按保存顺序重新创建 V42 mapper 与 V43 S/G 或 diagnostic controller。每帧先融合一次，再 accept 当前证据，再 choose。完整 mapper 回执、控制器证据和完整决策逐项按 JSON 数值精确比较，因此动作、目标、候选效用、语义状态和不确定性诊断都在比较内。首次差异立即停止，不再喂入后续旧观测来伪装不同策略的有效闭环。

最后比较完整 mapper snapshot、belief/observed 数组及全量 mesh。网格比较重排顶点及三角形顺序后比较全部坐标和颜色，绝对容差 1e-9、相对容差 0；保留所有远处或孤立顶点，不按目标或发现对象裁剪。这里比较无向三角面几何和颜色，不单独证明有向面绕序或每条拓扑边编号一致。验证结束再检查源文件与全部产物保持原 SHA。

`--integrity-only` 只做完整性检查，不创建 mapper；其结果明确 `policy_determinism_verified=False`。CLI 将 JSON 输出到标准输出，不向被审计的 episode 目录写入新文件。

## 本轮验证范围

`tests/v44_episode_fixture.py` 提供两个明确有限夹具：预声明双帧与有限 mapper/controller 替身用于序列和故障测试；真实 CPU controller/mapper 的单初始帧、仅 home 节点、预算 1 用于独立重算，原控制器选择 stop，执行动作数为 0。二者都不是研究场景轨迹或新增性能数据。

16 项测试覆盖损坏、缺帧、多帧、重复身份、源码差异、失败终端、错动作、伪装 live、外部清单锚、压缩双别名、预测绑定、首次分歧终止、真实单帧重算和网格枚举顺序。依赖使用现有 `.venv-3d` 中 NumPy/Open3D；需要读取宿主内存盘依赖，无新增安装。完整自主 episode 尚为 0，语义性能优势证据未增加。
