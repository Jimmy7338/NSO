# NSO 提交与备份说明

本目录的 `origin` 指向 `Jimmy7338/NSO`，`main`包含恢复后的历史材料；`sgam`指向新的独立项目仓库。SGAM发布应从独立的干净分支进行，不要将原库恢复后的main推入SGAM。两个仓库的对应关系见[迁移与恢复说明](docs/research/REPOSITORY_SPLIT_AND_RECOVERY_20260928.md)。独立项目发布时执行的一次历史迁移见[迁移说明](docs/research/STANDALONE_RELEASE_20260928.md)，不应在后续日常提交中反复重建历史。

## 常规操作

```bash
git status --short
git diff --check
git add <本次已检查的路径>
git commit -m "Describe the concrete project change"
git push origin main
```

先核对实验数据、文稿和代码的实际改动，再分批提交。默认不创建PR，不把本地恢复引用、虚拟环境、临时构建目录或编辑器配置推到远程。

## 大文件和科学材料

已有ZIP与可选检测器权重采用`.gitattributes`的LFS规则；论文、地图和逐步记录保持原始字节。按需使用`git lfs pull`恢复当前树的LFS资源，避免不必要地抓取全部历史。对大型新增数据先检查单文件与推送包大小，不通过删除失败实验或改写结果缩减数据。

历史结果、源码摘要和实际归档的可用范围不同。独立SGAM快照裁剪了旧实现，原NSO随后恢复历史源码及归档；两边均保留原receipt历史值，不得把重新生成的数据或当前文件哈希填回旧记录。[论文入口](docs/thesis/GRADUATION_DELIVERY_GUIDE_20260928.md)与[发布清单](docs/research/standalone_cleanup_20260928/removed_files.json)说明当前交付范围。

推送前检查密钥、凭据及无关个人文件，确认输出目录和待推分支。提交与推送不等于投稿或创建合并请求。
