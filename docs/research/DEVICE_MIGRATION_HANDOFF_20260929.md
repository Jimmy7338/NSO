# NSO 换设备总交接

日期：2026-09-29。工作仓库：<https://github.com/Jimmy7338/NSO>，分支 `main`。本轮基线为 `3453d9c797491935ebd07f4613ec1416a354672b`，本次迁移提交在其后；以远端分支与本次用户回执中的最终提交号核对。SGAM 是此前的独立发布快照，本轮不改动那个仓库。

## 先看这些成果

| 内容 | 当前入口 |
|---|---|
| 最新小论文 | [main.pdf](../thesis/submission_20260929_revised/main.pdf)，22 页、8 图、5 表；[完整材料 ZIP](../thesis/submission_20260929_revised.zip) |
| 小论文可编辑源 | [Markdown](../thesis/ARTICLE_SUBMISSION_EN_REVISED_20260929.md)、[main.tex](../thesis/submission_20260929_revised/main.tex)、[修订说明](MAIN_REVISION_20260929.md) |
| 毕业论文 | [57 页 PDF](../thesis/GRADUATION_THESIS_20260928.pdf)、[章节与构建索引](../thesis/GRADUATION_DELIVERY_GUIDE_20260928.md) |
| 完整英文证据 | [36 页 PDF](../thesis/VIRTUAL_PAPER_EN_20260928.pdf)、[源稿](../thesis/VIRTUAL_PAPER_EN_20260928.md) |
| 答辩与展示 | [PPTX](../thesis/defense_20260929/semantic_geometric_mapping_defense.pptx)、[PDF](../thesis/defense_20260929/semantic_geometric_mapping_defense.pdf)、[讲稿](../thesis/defense_20260929/speaker_notes.md)、[离线交互演示](../thesis/demos/v40_replay/index.html) |
| 代码与实验依据 | [复现索引](FINAL_METHOD_REPRODUCTION_INDEX_20260929.md)、`nso/`、`env/`、`scripts/`、`tests/`、`audit_results/`、`eval_results/` |
| 环境恢复 | [CPU、pdfLaTeX、绘图与旧构建环境说明](DEVICE_MIGRATION_ENV_20260929.md) |

旧的 `submission_20260929/` 是修订前的 20 页主稿，保留作冻结快照。继续写作以 `submission_20260929_revised/` 和 [CURRENT_MAIN](../thesis/CURRENT_MAIN.md) 为准。此前“固定交付完成”指既定工作包完成，不表示期刊已录用或证明全部场景普遍优越。

## 本轮保存与核对

- 保存最新修订主文、编译源码、PDF、LFS ZIP、全部包内依赖及编译收据。
- 保存 [22 页审阅预览](../thesis/build_review_revision_20260929/)、[编辑过程片段](main_revision_edit_sources_20260929/)；最新版核验脚本不再依赖旧设备 `tmp/` 中的页面记录。
- 保存三个此前未提交的图表版本目录及其来源记录，共 21 文件；它们是历史版本，当前论文引用的正式图仍按稿件路径使用。
- 保存历史批次启动记录、发布清点及本次资产核验；不改变实验指标、原始失败和源码封存。
- 小车两张原始照片与用户抠图早已保存于 `docs/thesis/assets/robot_20260928/`。

[压缩归档完整性核验](../../audit_results/device_migration_20260929/expanded_archive_verification.json)已将 **17,909 个展开文件、1,118,397,984 字节**逐一与归档成员对比，缺失、差异和未归档独有文件均为 **0**。对应 **36 个归档、327,911,656 字节**均与原 `HEAD` 中普通 Git blob 相同，因此这些原始实验归档不依赖 Git LFS。35 包通过完整解码/重编码的字节恢复验证；1 包为失败过程普通 tar.xz，单独核对全部成员。

为避免再次重复提交，`.gitignore` 仅增加这批已验证旧 episode 的 97 个具体展开子目录；未删除本地文件，也没有忽略实验的新结果目录。每个被排除的展开文件都有 SHA256 和压缩包成员映射，压缩原件继续由 Git 跟踪。

LFS 的本地内容清点与远端检查见同目录 `lfs_head_inventory_before_commit.json`、`remote_lfs_verification.json`。当前 **225 个不同 LFS 对象全部获得服务端下载信息**；最新源码 ZIP 已实际重新下载 **14,870,257 字节**，SHA256 与本地一致。六个旧数据/权重路径在本机仅有指针，其对象也在远端下载清单中；本次没有完整下载这些旧模型，因此不把服务端可用性核对称作旧模型的完整内容复核。

## 新设备最短开始方式

安装 Git 与 Git LFS 后：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Jimmy7338/NSO.git
cd NSO
git lfs install --local
git lfs pull
git log -1 --oneline
python3 -B scripts/review_main_revision_20260929.py
```

最后一步只用 Python 标准库检查保存的修订包与旧包，不启动实验。应通过 240 项文稿一致性检查。仓库较大，读论文可先下载最新 main PDF 或完整 15 MB 左右的源码 ZIP；完整克隆及原始数据按需恢复。

35 个标准 episode 归档按需恢复到**新的空路径**：

```bash
python3 -B scripts/archive_article_episode_20260928.py restore \
  --archive audit_results/article_stage_20260928/episode_archives_v1/dev_AISLE_B_b160_n92801.tar.xz \
  --output tmp/restored-dev_AISLE_B
```

`dev_CELL_G_b160_n92801_original_failed.tar.xz` 是普通 tar.xz，不能传给上述标准恢复子命令，应在新的空目录用常规 tar 解压；原失败身份保留。不要在迁移时批量展开全部归档，也不要覆盖受校验的原结果目录。

PDF 可直接阅读；最新投稿包 `main.tex` 使用相对资源路径，可以在其他工作目录编译。原 Markdown 多处使用 `/root/NSO`，从 Markdown 构建宜保留该路径或在独立副本中适配。Python 环境、TeX 缓存和个人登录凭据不随仓库复制，具体恢复步骤见环境说明。

## 给下一位协作者的任务状态

用户计划先发表小论文，再完善毕业论文；学校没有硬性发表要求，学校/学位/专业字段按要求留空。保留四模块接口和层次结构，CPU 仿真为主，实车是后续补充。

当前 main 已补齐方法、完整规划递推和算法。原确认的平均联合指标增益为 6.1638%；新增背景测试为 2.9818%，收益集中在改变观察方向的条件，另外三个条件保持相同结果。预算敏感性和共享语义的未建立增益仍保留，表述优化没有更改实测证据。

后续优先：导师审阅新版 main；把确认采用的方法表述同步到毕业稿；若继续实验，先限定一个明确问题和预算，再检验观察方向歧义、观察成本与剩余预算的关系。当前不要因换设备自动重跑冻结批次、重新开启此前完成的 Goal，或覆盖原数据。发布代码可以按用户授权提交推送；没有明确要求时不创建 PR/MR，不代用户投稿或联系编辑。
