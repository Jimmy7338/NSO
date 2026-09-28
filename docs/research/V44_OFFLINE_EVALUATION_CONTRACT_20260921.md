# V44 已保存完整预测的离线评价契约

本入口把 V43/V44 已保存预测接到 V40 表面评价器。它不创建 World、传感器、控制器或 TSDF，不重新运行轨迹，也不接收调用者自行填写的覆盖分数。当前仅是开发接口和有限解析单元测试；不构成新增的语义增益实验。

实现文件是 `nso/offline_evaluation_v44.py`，命令是 `scripts/evaluate_episode_v44.py`。V40–V43 文件保持不变。共同只读轨迹 loader 来自 `nso/saved_replay_v44.py`。

## 固定参考先于实验

`prepare-reference` 只接受 `DEV_[A-F]_00`，读 V40 已封存开发几何、全部设施库存、工作区和固定视点生成代码；逐项核对 P0/P1 manifest 中的 SHA256 与字节数。实例集合必须与完整 GT 三角形 owner 集合一致。不得按已检测、已访问或重建出的设施删减分母。

表面参考完全沿用 V40 静态规则：相对固定起点的 1.5 m 可达网格、四个朝向、机器人半径 0.2 m、连续边碰撞检查、0.3 m 面积采样、种子 4001，至多 50,000 个采样点。所有设施必须拥有至少一个外向、相机视锥内、深度范围内且无遮挡的参考采样点；否则准备失败，不能静默忽略设施。DEV_A 还必须重现之前静态集成参考的 fingerprint。

这些是稀疏候选视点与有限面积采样的近似，不能描述成精确可观测外表面；旧 V40 开发参考也不是正式测试集已冻结的证明。公开的 1 m 导航图与离线的 1.5 m 表面参考格是不同对象。后者用于固定评价定义，不发送给控制器。

准备输出包含完整七组 `ReferenceSurfaceV40` 数组、候选视点、二维分母掩码、参考描述和 manifest。根任务须在对应轨迹启动前封存 manifest SHA256；评价时显式提供该摘要。源码也绑定在参考中：修改生成器后不能用新代码静默加载旧参考。外部摘要保护的是已选定参考的内容，程序本身不能证明摘要选择发生在实验前；需要封存记录提供时间和流程证据。

```sh
.venv-3d/bin/python -B scripts/evaluate_episode_v44.py prepare-reference \
  --asset-id DEV_A_00 --output audit_results/<new-v44-reference-directory>
```

## 二维指标命名为 C_nav

`C_nav = 正确测为 free 的固定可通行格数 / 固定可通行格总数`。

分母采用整个声明工作区的 0.1 m 网格，行号随世界 y 增大，原点位于左下边界。离线使用全部设施的保守 AABB 和背景实体投影，沿 XY 各膨胀 0.2 m；保留房间内、机器人中心安全且与声明起点连通的格中心。连通边仍做连续碰撞检查，薄墙不能被网格步长跨越。每个保留中心以 0.01 m² 作为面积积分权重。

这是**保守可通行地面覆盖近似**。排除了保守障碍投影和起点不连通部分；并非全环境体素覆盖、SLAM 位姿精度、精确二维自由空间面积或精确传感器可见域。格中心属于离线保守可达域，实际覆盖仍必须来自已付费传感器产生的累计 free 证据；静态可达不等于已经观察。

分子只计完整已保存 `occupancy.belief == 0` 的分母格。`belief == -1` 未知和 `belief == 1` 误占都不得给覆盖分；另报触碰比例和分母域内误占格数。V42 `occupancy.observed` 仅表示最后一帧触碰范围，**不能作为累计覆盖**，此入口只校验其类型和一致性，不将它用于指标。

由于指标新增了明确的二维任务域，联合开发指标记为 `J_nav = C_nav × Q`。不将其不加说明地写成原始泛称 `C_map × Q`，也不回写旧实验的分数。正式对比之前仍须预声明这种定义，不能据结果选择不同分母。

## 三维质量和完整预测

`Q` 复用冻结 V40 的全部设施实例宏平均 F1，距离阈值 0.05 m、预测面积采样 0.3 m、种子 4002。评价读取原始保存的 `prediction/mesh.npz`，只接受 vertices、triangles、vertex_colors 三组；不接受 ROI/owner 替代文件、不删除外围预测、不按语义加权、不用检测库存决定实例数。

缺失设施保留在宏平均分母并得到零分。错误外围表面计入 false-positive 面积；正确背景和固定参考域外的正确表面分别报告。V40 固定评价器自身会去除几何完全重复三角形以避免复制 TP 使精度上升，该规则对所有方法一致。空完整网格得到全部设施零质量。

## 只读校验和成功界限

评价入口要求轨迹 manifest 的外部 SHA256，以及参考 manifest 的预封存 SHA256。共享 loader 校验所有保存文件、源码与归档一致、付费步骤连续、动作/包/receipt 绑定、公共图、最终预测和失败状态；入口再核对预测网格的坐标系、网格尺寸、分辨率、占据摘要、frame 数以及参考工作区。参考自身核对完整表面 fingerprint 和分母掩码摘要。

严重失败或缺少完整预测的运行由 loader 拒绝，不把成功前缀当成完整运行。保存的 finite fixture 在读取开发参考前直接拒评，错误文本为 `finite fixture cannot be scored as a development episode`，避免把解析帧配上 DEV 场景 GT。有完整保存但未返回或阻塞的开发运行可以得到 `diagnostic_non_success_scored`，但 `eligible_successful_development_endpoint` 为 false。只有确认返回、任务类型为自主控制器、确实创建了开发 World、非 finite fixture，且能绑定唯一持久启动账本、结果摘要和源码/slot 的运行才可能成为成功开发端点。固定动作诊断即使正常停止也不能进入该类别。所有失败启动仍应在整体实验成功率和失败率中保留，不能从汇总分母剔除。

外部摘要是在信任封存流程的前提下防止更换内容，不是对抗任意自造整包证据的密码学签名。本入口不重建 TSDF，不声称完成了独立回放验证；回放是另一条需要单独记录结果的检查。单次成功开发端点也不能证明统计显著优势，输出始终标记 `formal_performance_evidence=false`、`semantic_performance_claim=false`。

```sh
.venv-3d/bin/python -B scripts/evaluate_episode_v44.py evaluate \
  --episode <saved-episode-directory> --reference <sealed-reference-directory> \
  --episode-manifest-sha256 <externally-recorded-episode-sha256> \
  --reference-manifest-sha256 <presealed-reference-sha256> \
  --output <new-evaluation-json>
```

评价输出必须在原轨迹和原参考目录之外，不能覆盖已有输出。准备和解析评价只需要既有 NumPy，不需要 Open3D 后端；完整轨迹 loader 读取既有传感器契约，仍不会构造 World。实际独立 TSDF 回放由其他入口负责。

## 有限测试范围

`tests/test_offline_evaluation_v44.py` 验证固定分母、未知为零、free 与 touched 区分、薄墙连通性、所有设施占据、起点不能替换、掩码/网格摘要、外部摘要必需、完整网格和 `J_nav` 接线、缺失设施零分、外围错误面积、空网格以及未返回状态。用于接线测试的 loader/reference 是显式解析替身，不能冒充 DEV_A 轨迹或真实设施重建结果；测试没有 World、扫描、轨迹或 TSDF 积分。
