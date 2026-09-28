# SGAM 独立快照与原 NSO 历史资料恢复

本次按维护者要求分开保存两个用途的仓库，未启动新的科学实验。

- **[Semantic-Geometric-Active-Mapping](https://github.com/Jimmy7338/Semantic-Geometric-Active-Mapping)** 保存清理后的完整快照 `95553562010426a1ec2f98dbb87346a9ee49adb5`，根树为 `410d2900bb5034e8c8bbd4739250992629101918`，包含310,862条文件路径。该快照不继承旧作者或Cursor的提交历史。现有文献引用及外部依赖来源仍如实保留。
- **[NSO](https://github.com/Jimmy7338/NSO)** 在上述快照之后追加恢复提交，重新保存历史源码、依赖、资产及许可。原先的当前CPU实现、论文与实测结果继续保留；旧的47次提交仍由本机恢复引用保存，不通过此次恢复重新接回远程提交历史。

## 恢复范围

恢复2,740条被删除的非缓存路径，其中包括131个原始源码ZIP、1,811条历史源码快照，以及自研旧脚本/模块、旧环境、数据和第三方源码。另恢复原`env/__init__.py`，将旧依赖文件保存为`requirements-legacy-ans.txt`，并在根MIT许可中同时保留原作者与2026 NSO contributors的版权行。205条解释器/编辑器缓存继续排除。原库另外补回6项旧环境/评价/接口检查，保留清理版新增的2项直接冻结测试；这些旧接口检查不加入SGAM快照。

当前`requirements.txt`继续对应CPU工作流程。旧训练链需按`requirements-legacy-ans.txt`另行配置，不应覆盖当前科学环境。必要的旧模型LFS规则已补回，缓存及Cursor编辑器排除规则保留。

## 完整性与可复现范围

所有恢复内容按清理前Git对象、SHA256及文件模式核对。131份混合源码ZIP保持原字节，不抽取删改其中内容。旧59源保护清单、当前43源清单和6项输入清单均重新通过校验；既有实验结果和原始校验记录未改写。当前CPU回归的命令、计数及日志摘要见[验证记录](repository_split_20260928/validation.json)。论文源文件、图表和四份PDF保持原交付哈希。

为节省磁盘，135个已核验的LFS文件与现存本地缓存采用硬链接，75个重复旧数据文件也采用硬链接；这些归档/数据按不可变材料使用，不能就地改写或chmod。需要派生数据时先复制到新路径。源程序与历史源码快照使用独立文件，不互相硬链接。

以下旧外部资源在恢复前本机已经不完整，此次不把“指针或链接恢复”写成“运行环境全部恢复”：

- 6个LFS文件本机仅恢复指针：`pointnav_gibson_v1.zip`、`pretrained_models/model_best.slam`、以及stage1/stage2的4份旧训练权重。已通过原NSO的LFS下载接口和HEAD核验，6份远程本体均可下载且尺寸匹配；本轮未重复下载约0.7 GiB文件。[远程可用性记录](repository_split_20260928/legacy_lfs_remote_availability.json)不冒充重新下载后的字节哈希校验。
- `env/habitat/habitat_api`、`habitat-api`、`habitat-sim`只恢复Gitlink，未下载子模块。
- 3个`data/scene_datasets/`符号链接的外部目标目前缺失。

因此，当前CPU分析和历史源码校验已恢复，完整旧Habitat训练运行仍需要其外部资源与独立环境。没有补写历史实验成绩或宣称旧训练系统已运行成功。

## 后续使用

当前本地`origin`继续指向NSO；`sgam`指向新的独立仓库。带历史依赖的原库main不能作为SGAM的后续发布源。原[独立发布说明](STANDALONE_RELEASE_20260928.md)描述清理快照发布时的状态；本记录描述后续原库恢复，二者不互相覆盖历史事实。

[清理快照远程核验](repository_split_20260928/clean_repository_verification.json) · [逐文件恢复回执](repository_split_20260928/restoration_receipt.json) · [缓存排除清单](repository_split_20260928/excluded_cache_files.json)
