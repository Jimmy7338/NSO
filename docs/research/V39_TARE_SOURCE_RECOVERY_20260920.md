# V39：从新克隆恢复 TARE 依赖与提交范围

本页补充已冻结的 L2 协议，不修改任何已运行策略、桥接源码、运行时恢复脚本或实验结果。

## 冷克隆恢复顺序

仓库的 `third_party/official_baselines/` 被忽略；运行库位于易失 RAM。新克隆需要先取回 Git LFS 保存的证据压缩包，再运行两个独立恢复命令：

```bash
git lfs pull
python3 -B scripts/restore_tare_sources_v39.py
python3 -B scripts/restore_tare_runtime_v39.py
```

第一条 Python 命令恢复官方源、`garage.yaml`、原捆绑 OR-Tools 9.8.3296 及 SONAME 软链接。第二条恢复固定的 57 个 Focal/Noetic 运行包与**已归档**的完整 TARE ELF，核对依赖/导入并启动零传感节点。两者都不执行新的世界、轨迹或质量实验；CPU Open3D 仿真环境仍按项目原环境配置恢复。

源码恢复固定官方 commit `44500592b86138257273e0cab264e6a847ccefc7`，下载地址与全部 2,846 个文件哈希来自已跟踪的 `docs/research/official_baseline_source_manifest.json`；脚本还固定该 manifest 自身 SHA。官方压缩包为 38,270,178 B，SHA-256 `a3adfcab09a669a2a1b9bf0aaec734171cc46ce0fa5cc16a3b54fc51fdb36024`。恢复后的 `libortools.so.9.8.3296` 为 44,054,264 B，SHA-256 `581d11fdda03d9b28d31edba61a2e57f192ba3bf595df459aa08d6b69b08676a`；`garage.yaml` SHA-256 为 `664d236e625a6381e2bb1d3c313c964f15ce3e0e8c12eec7c96574e541d851ff`。

已有目录时只核对全部已记录文件与两个必要 SONAME 链接，不重新下载、删除或覆盖。冷恢复默认下载/解包至专属 `/dev/shm/nso_v39_tare_sources`，再在仓库预期位置创建软链接；重启后可重新运行。需要持久源目录时使用 `--storage repository`。下载前分别预留已知归档大小与 160 MiB 解包上限，操作保持根分区至少 64 MiB、tmpfs 至少 512 MiB、系统可用内存至少 2 GiB。拒绝覆盖未知旧目录或非本任务的悬空链接。上游指向作者 `/opt/ros/.../toplevel.cmake` 的外部 workspace 链接按原 manifest 明确跳过，不写入宿主 `/opt/ros`。

实际验证没有重复下载现有的 38 MB 归档：2,846 个文件、136,587,253 B 内容和必要软链接全部校验通过。4 项小型提取检查验证正常文件/内部链接、已知外部链接跳过，以及路径越界和非法链接拒绝。收据在 `audit_results/v39_tare_source_bootstrap_20260920/`。冷下载分支本轮未重新访问归档；已有官方归档哈希源自此前真实下载，当前实际树与全部文件哈希一致。运行时脚本另已完成从空 RAM 目录重建的实际测试，见 V39 runtime preflight。

## 必须归档和可下载恢复的内容

| 资产 | 提交或恢复方式 | 原因 |
|---|---|---|
| 原完整 `tare_planner_node` | 显式归档 `audit_results/tare_native_node_attempt_20260911/` 全部 16 个文件 | 运行时脚本固定该 ELF SHA；只有源码不能保证重新编译得到逐字节相同的二进制 |
| 官方 TARE 源、garage、OR-Tools | 官方固定归档恢复；不必将 136 MB 树再次加入 Git | 上游 URL、归档 SHA、逐文件 SHA 已在版本库，新增 bootstrap 可执行 |
| Noetic/Focal `.deb` | 由 runtime bootstrap 按官方签名链和冻结包 SHA 下载 | 不提交 57 个系统包或 RAM 安装树 |
| 两个 bootstrap 脚本 | 显式提交 `restore_tare_sources_v39.py` 与 `restore_tare_runtime_v39.py` | 源恢复和运行库恢复分别可调用，不改已冻结算法 |
| runtime recipe/验签/冷恢复证据 | 提交 `audit_results/v39_tare_runtime_preflight_20260920/` 完整小证据包 | `installed_packages.json` 和 `ros.asc` 自身哈希固定在恢复脚本；清单还引用日志与验证收据 |
| 来源及许可元数据 | 提交 `audit_results/v39_tare_source_bootstrap_20260920/` | 保留固定快照原 `package.xml` 与两个 README，不丢弃出处 |
| L1/L2 代码、协议、原始结果与失败 | 按各次冻结 manifest / seal 显式提交 | 新克隆可以核对冻结输入；保留旧失败，不能仅提交汇总数字 |

TARE 固定快照在 `package.xml` 中声明 `BSD`，但该快照没有独立 LICENSE、COPYING 或 NOTICE 文件；本记录不擅自补成某个 BSD 条款版本。OR-Tools 原 README 提及所捆绑 SCIP 的独立条款。来源收据原样保留上述文件及哈希，不将第三方资料改标为本项目许可证。大型第三方源码/库优先由原官方固定 URL 恢复。

## 显式暂存检查点

普通 `git add` 会跳过 `*.log`。原 ELF 构建目录中的构建日志/节点日志，以及 runtime 证据目录的三个 `node-*.log`，都被已有清单引用，应显式强制纳入；不要对整个 workspace 执行无差别强制添加。

L1/L2 两份 manifest 合计引用 75 个不同路径；本次审计时其中 55 个已经跟踪，19 个为新的项目文件，另一个 `garage.yaml` 由上述官方 bootstrap 恢复。既有 V33 几何、V34 信息模板和 V36 配置都已跟踪，不需要重新生成。新输入 `audit_results/v39_saved_prefix_preflight_r1_20260920/` 四个文件必须一起提交，包含 LFS `sources.zip`；初版已通过的 preflight 检查记录同样保留，r1更新针对日志空间上限及封存准备，不是算法失败重试。

另需提交没有直接出现在冻结 source closure 中但支撑复现的 `scripts/verify_external_preflight_v39.py`、`scripts/build_external_comparison_v39.py`、`tests/test_tare_adapter_v39.py`、两个恢复脚本，以及新 V39 文档/图表与实验收据。恢复脚本的成功不替代 LFS 上传：`*.zip` 受现有 Git LFS 属性管理，推送后应确认远端对应对象存在。
