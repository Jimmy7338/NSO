# 2026-09-28 GitHub完整项目备份

**后续发布变更：** 本文记录清理前的备份操作。之后的[独立项目迁移](STANDALONE_RELEASE_20260928.md)按维护者要求裁剪了退役源码、部分源码归档及旧主分支历史；下文“全部冻结源码”和“不重写历史”仅描述此前备份，不表示当前发布树仍完整保留这些文件。

目标仓库：<https://github.com/Jimmy7338/NSO>，沿用 `main`。本次按用户要求尽可能保存重要文件与数据，不创建PR、不强推、不重写既有历史。

## 备份内容

- 毕业论文五章、中英文摘要、参考文献、24页全文PDF及可编辑源稿；核心两章、中英文文章、图表CSV/PDF/PNG/SVG、构建记录。
- 当前源码、场景配置、测试、环境依赖清单、研究进度、原审查与阶段决策记录。
- 所有可见的 `eval_results/` 和 `audit_results/` 实验组，包括原始传感包、逐步决策、地图、终点网格、冻结源码、输入清单、复核与回放、中断/失败和负结果。
- 旧网格压缩归档、恢复索引及恢复脚本；这些可能承载已移除原路径的唯一可恢复数据，不作为普通缓存省略。
- 40份既有构建/测试/维护日志；排版环境复制收据已移入文档，避免链接指向被忽略的临时目录。

本轮起始清点为285,118个未跟踪文件、约9.46 GiB逻辑体积。实验目录其中284,976个文件、约9.45 GiB。数字表示备份前的路径内容量，不是Git传输量；大量硬链接、相同文件和本机Git已有对象会影响实际存储。后续补入日志、说明和索引会使提交文件数略有变化。

[目录与类型清点](github_backup_20260928/inventory.json)及[本机内容去重统计](github_backup_20260928/blob_inventory.json)保留原始统计。去重统计只回答本机对象库是否已有字节，不代表这些对象已经上传到GitHub；是否备份由远端提交及其可达树决定。

## 排除范围

运行环境、编译器/TeX包缓存、Python依赖本体、构建中间产物和 `.pyc` 继续按原规则忽略。四个零字节运行锁显式忽略。`audit_results/v40_scene_contract_20260920/private/` 的留出种子继续沿用既有私有忽略规则，不因备份发布到公开仓库。

`storage-offload`此前下载的两份归档属于其他SDK的调试符号，不包含NSO实验，因此未据此省略历史实验。原始数据未删除，也未通过重新生成实验替代备份。

## 字节一致性与恢复

`.gitattributes`对实验记录和论文目录禁用文本换行转换，保持SHA绑定文件（包括CRLF CSV）的原始字节。已有ZIP/模型继续使用原Git LFS规则，不迁移或重写旧历史。Git会保存软链接及其内容；跨路径硬链接通常在检出后成为独立文件，因此恢复磁盘需求可能高于本机当前物理占用。

恢复时克隆仓库并按需要执行 `git lfs pull`。先阅读[毕设交付说明](../thesis/GRADUATION_DELIVERY_GUIDE_20260928.md)和[当前研究状态](CURRENT_RESEARCH_STATE.json)，不要自动恢复冻结实验。普通Git文件包含CPU实验结果；LFS用于既有模型及ZIP，下载依赖GitHub账户的可用存储/带宽。

科学运行环境依据根目录 `requirements-cpu.lock.txt`、`requirements-3d.lock.txt` 等清单按用途恢复；不应把全部历史模型或GPU依赖作为阅读论文的前提。PDF可直接阅读。重新编译使用Pandoc、Tectonic 0.17.0、Noto CJK/Liberation字体及pypdfium2/Pillow；当前本地运行路径和核验记录见[环境收据](../thesis/GRADUATION_ENVIRONMENT_COPY_20260928.json)，依赖本体不在Git中。

GitHub的[单次推送上限为2 GiB](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit)。[流式测量](github_backup_20260928/pack_estimate.json)得到仅现有待传数据对象的包约4.88 GiB，因此本次按新增内容体积分4批快进推送；这是同一次备份操作，不筛选有利实验。实际上传完成以远端 `main` 与最终本地提交一致为准。
