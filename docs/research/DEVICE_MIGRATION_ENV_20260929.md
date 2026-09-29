# 换设备环境与恢复说明（2026-09-29）

本说明根据离开旧设备前的文件和运行收据清点编写。只记录恢复条件，不代表已在新设备安装或验证。最新小论文是 `docs/thesis/ARTICLE_SUBMISSION_EN_REVISED_20260929.md` 及 `docs/thesis/submission_20260929_revised/main.pdf`（22 页）；原 `submission_20260929/` 是保留的旧版本。是否已推送及最终提交号以本轮总交接与远端核验为准。

## 1. 最短恢复：先读成果和检查文件

推荐使用 Linux/WSL 的大小写敏感文件系统。旧设备是 Ubuntu 24.04.2、Python 3.12.3。阅读 PDF、检查仓库和直接编译期刊包的 `main.tex` 不要求 `/root/NSO` 路径；但旧 Markdown 文稿含大量 `/root/NSO` 的绝对资源链接。从 Markdown 重新构建时，最省事的是仍检出到 `/root/NSO`，或者先在单独的修订副本中适配资源路径。不能只把构建器的输出参数换成新目录，就认为输入资源路径已自动迁移。

以下以 Git、Git LFS 和 Python 已安装为前提。克隆先跳过 LFS 自动下载，随后明确恢复当前版本的 LFS 对象：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Jimmy7338/NSO.git
cd NSO
git lfs install --local
git lfs pull
git status --short
python3 -B scripts/verify_final_reproduction_index_20260929.py
```

最后一个脚本已存在，只用 Python 标准库，核验其冻结索引内的 542 项绑定关系，不启动实验。它核验的是原固定交付索引，不覆盖整个历史仓库，也不能单独代替本次新增 revised 包的校验。最新主文的输入、编译器、PDF 和包内文件哈希另见 `docs/thesis/submission_20260929_revised/build_receipt.json`。

无需恢复科学环境即可阅读 PDF、Markdown、图表和离线演示。入口包括：

- 小论文：`docs/thesis/submission_20260929_revised/main.pdf` 与同级 `main.tex`，以及 `docs/thesis/submission_20260929_revised.zip`。
- 大论文：`docs/thesis/GRADUATION_THESIS_20260928.pdf`；分章来源见 `docs/thesis/GRADUATION_DELIVERY_GUIDE_20260928.md`。
- 答辩：`docs/thesis/defense_20260929/` 中的 PPTX、PDF、讲稿和问答。
- 复现入口：`docs/research/FINAL_METHOD_REPRODUCTION_INDEX_20260929.md`。
- 当前状态：`docs/research/CURRENT_RESEARCH_STATE.json`；主文修改不等于新增实验结果。

## 2. CPU 科学环境

已提交的最小依赖是 `requirements-cpu.txt`、`requirements-3d.txt`，完整历史环境记录是 `requirements-cpu.lock.txt`、`requirements-3d.lock.txt`。最小 3D 文件固定 NumPy 1.26.4、SciPy 1.11.4、Matplotlib 3.8.4、`open3d-cpu` 0.19.0。安装分发名为 `open3d-cpu`，导入名为 `open3d`。当前局部 CPU 实验不要求 PyTorch、Habitat、旧网络权重或 GPU。

在新设备安装 Python 3.12 及其 venv 支持后，从仓库根目录执行：

```bash
python3.12 -m venv .venv-3d
.venv-3d/bin/python -m pip install -r requirements-3d.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-3d/bin/python -B -c 'import numpy, scipy, matplotlib, open3d; print(numpy.__version__, scipy.__version__, matplotlib.__version__, open3d.__version__)'
```

这些是恢复建议，未在全新环境执行。历史 lock 文件是已有环境快照，不是已验证的跨平台安装器；先恢复最小依赖，再按实际需要补充。替换 Python、Open3D、BLAS、CPU 架构或库版本后，不能预先保证融合网格的位级一致。

Ubuntu 上本机已安装的相关系统包包括 `python3.12-venv`、`libegl1`、`libgl1`、`libgomp1`。如果 Open3D 导入报告对应动态库缺失，再由新设备包管理器安装。不要将显示窗口支持当作 CPU 离线融合的必要工作流。

保存包重放的脚本与命令已经列在 `FINAL_METHOD_REPRODUCTION_INDEX_20260929.md` 第 4 节；需要重放时使用新输出目录，保留旧收据。迁移本身不要求重跑冻结实验、消耗未使用额度或重算已保存指标。

## 3. 最新 main 的排版恢复

最新 revised 主文实际使用 **pdfLaTeX / TeX Live 2023** 编译，收据中 `actual_pdflatex_validation=true`，不依赖旧设备 `tmp/` 下的 Tectonic 缓存。旧设备 Pandoc 为 3.1.3；系统已安装 `texlive-base`、`texlive-latex-base`、`texlive-latex-recommended`。Springer 模板与本项目所需的额外样式文件已保存在提交包和 `docs/thesis/journal_template_20260929/springer/`。

仅重编译保存的 TeX 时，复制包到新工作目录，避免覆盖交付版本：

```bash
mkdir -p tmp/device-migration
cp -a docs/thesis/submission_20260929_revised tmp/device-migration/journal-copy
cd tmp/device-migration/journal-copy
for pass in 1 2 3; do
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex || break
done
```

执行前需要新设备已有 `pdflatex`；上述循环和新设备编译未在本轮执行。保持包内 `Fig*.pdf`、类文件和样式文件在同一目录。编译后检查 `main.log`，不要仅凭生成了 PDF 就认定布局合格。

从新版 Markdown 重新生成投稿包的现有入口是 `scripts/build_revised_journal_20260929.py`，它调用现有 `scripts/build_final_journal_20260929.py`。新设备需先安装 Pandoc 和 pdfLaTeX，并满足前述 `/root/NSO` 资源路径条件。从仓库根目录执行以下命令可生成新路径的包：

```bash
python3 -B scripts/build_revised_journal_20260929.py \
  --build --engine pdflatex --export \
  --work tmp/device-migration/journal-build \
  --output docs/thesis/submission_device_migration_check
```

这段命令使用脚本实际支持的参数；输出目录必须尚不存在或为空。未加 `--render` 时不需要 Python PDF 预览依赖。排版程序会拒绝覆盖已有正式输出，并核验输入在构建期间没有改变。编译器或字体更新可能改变页码及 PDF 字节，原构建哈希描述原 PDF，不要求新环境重新编译的文件具有相同哈希。

## 4. 绘图、PDF 预览及答辩文件

旧环境中确认存在的附加依赖版本为：Pillow 12.3.0、pypdfium2 4.30.0、python-pptx 1.0.2、lxml 6.1.3、XlsxWriter 3.2.9；科学绘图使用 Matplotlib 3.8.4。原本部分依赖安装在被忽略的 `tmp/final-writing-python/` 和 `tmp/thesis-python_20260928/site-packages/`，这些路径不会随普通 Git 克隆恢复。

可在新设备建立独立写作环境：

```bash
python3.12 -m venv .venv-local
.venv-local/bin/python -m pip install -r requirements-cpu.txt \
  'Pillow==12.3.0' 'pypdfium2==4.30.0' 'python-pptx==1.0.2' \
  'lxml==6.1.3' 'XlsxWriter==3.2.9'
```

以上版本来自本地安装元数据，不代表包索引未来始终提供所有平台的轮子。本次没有重新下载安装。这里选用现有忽略规则中的 `.venv-local/` 作为写作环境；解释器目录只保留本机使用。

答辩构建入口 `scripts/build_final_defense_20260929.py` 已存在，可以用上述环境代替旧 `PYTHONPATH` 路径；指定新的 `--output`。脚本使用 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` 和 `NotoSansCJK-Bold.ttc`。旧设备对应 Ubuntu 字体包为 `fonts-noto-cjk`；通用毕业论文另使用 `fonts-liberation` / `fonts-liberation2`。Windows、macOS 或其他 Linux 的字体路径可能不同，需要显式适配后检查页面。

毕业论文与完整英文证据稿的旧构建脚本 `scripts/build_writing_deliverables_20260928.py` 仍调用 `tmp/thesis-tex-runtime_20260928/bin/tectonic`，并使用 `--only-cached`。Git 不包含该 0.17.0 编译器及缓存本体，不能在新机器直接照旧命令保证成功。准确来源、编译器 SHA、bundle 地址和复制校验记录见 `docs/thesis/GRADUATION_ENVIRONMENT_COPY_20260928.json`。需要重新排版这些通用稿时，再安装相应 Tectonic 版本并恢复/获取 TeX 依赖，或单独适配系统 XeLaTeX；当前没有已验证的一键环境恢复脚本。保存的 PDF 与源文稿不受这一限制。

## 5. 临时目录与外部素材核对

| 本地内容 | 迁移判断 |
|---|---|
| `tmp/final-journal_20260929_revised/prepared/` 顶层 47 文件 | 本轮逐文件 SHA256 比较，与正式 `docs/thesis/submission_20260929_revised/` 全部一致；没有仅存于该临时副本的正式包文件。 |
| 同级 `visual/page_*.png`、contact sheets、`pages.json` | 已额外复制至 [持久审阅目录](../thesis/build_review_revision_20260929/)：26 个 PNG 与 `pages.json`。新版 review 脚本优先读取持久目录，换设备核验不再依赖临时预览。它们仍是 PDF 派生材料，不是独立科学数据。 |
| `tmp/main_revision_20260929/revise_main.py` 与 `method_replacement.md` | 已额外保存在 [编辑过程归档](main_revision_edit_sources_20260929/) 并附 README。正式修改结果在新版 Markdown、TeX 和包中；归档脚本用于追溯一次性编辑，不是常规构建入口。 |
| `.venv/`、`.venv-cpu/`、`.venv-3d/`、TeX/Python 下载缓存 | 机器运行依赖，按既有规则忽略；不适合代替跨设备安装清单。 |
| `/dev/shm/` 与 `tmp/v38-tex` 软链接 | 旧设备易失运行路径，不应作为新设备成果或依赖的唯一来源。 |
| `docs/thesis/assets/robot_20260928/20260928-142859.jpg`、`20260928-142915.jpg`、`robot_cutout_user.png` | 三个用户图像均已在 Git 跟踪；无需再从 `/root/.codex/attachments/` 找回。相关系统图与平台图的 manifest 位于各自 `docs/thesis/figures/` 子目录。 |

本说明未扫描用户凭据目录，未复制 SSH 密钥、GitHub 登录信息、代理凭据或个人工具配置。新设备如需推送，由用户在新设备重新登录 GitHub；公开仓库的普通下载不依赖旧设备凭据。

## 6. 不能由克隆自动恢复的内容

- Git LFS 需单独下载，且受远端对象是否可用及账户带宽限制；只看到指针文件不等于已拿到原始 ZIP/模型。默认 `git lfs pull` 恢复当前检出版本，历史 LFS 版本按需另取。
- 克隆不会保留原目录之间的硬链接关系，完整检出可能比旧机器实际占用更大。先确保磁盘有足够余量；旧机器清点时只余约 0.5 GiB，不是新机器最低容量要求。
- 被既有规则排除的私有留出种子不因迁移而公开；已经完成且提交的实验数据、配置和收据应以具体清单核验，不能把未执行预留任务写成已完成结果。
- 作者/单位、学校模板、最终投稿和真实小车闭环验证不由环境迁移自动完成。当前学校、学位和专业按用户要求留空。

新设备的默认任务应是继续修订最新 main 与毕业稿、利用现有证据完善表述。确认有新的研究问题和有限实验计划后再运行新增实验，避免因换机把已完成工作全部重做。
