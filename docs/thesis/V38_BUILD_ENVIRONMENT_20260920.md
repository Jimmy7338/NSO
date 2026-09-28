# V38 论文编译与 PDF 检查环境

2026-09-20。已完成固定版本 Tectonic 安装、中文最小稿编译、离线缓存回编和 PDF 页面检查。此记录只证明编译工具链可用，不代表完整论文已经编译或审稿通过。安装、源稿、日志及最小 PDF 保存在 `audit_results/v38_build_environment_20260920/`，目录中 `artifact_hashes.json` 对 13 个产物给出 SHA256。

## 工具与下载核对

Tectonic 采用官方 GitHub 发布的 `0.17.0` Linux x86_64 musl 版本：[固定发布包](https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz)。下载包为 10,151,914 B，SHA256 为 `8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7`，与本次指定的发布哈希一致；解包二进制为 26,401,904 B，SHA256 为 `a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da`。二进制放在 `/root/NSO/tmp/v38-tex/bin/tectonic`，验证成功后删除下载压缩包。

PDF 渲染器为 `.venv-3d` 中的 `pypdfium2==4.30.0`。固定 manylinux wheel 为 2,812,008 B，SHA256 `f1f78d2189e0ddf9ac2b7a9b9bd4f0c66f54d1389ff6c17e9fd9dc034d06eb3f`。通过 `pip --no-deps --no-cache-dir --no-compile` 安装，随后删除 wheel；没有下载其依赖或创建 pip 缓存。

## 中文与本地字体

最小稿使用 `\documentclass[UTF8,fontset=none]{ctexart}`，本地字体设置如下：

```tex
\setmainfont{Liberation Serif}
\setsansfont{Liberation Sans}
\setmonofont{Liberation Mono}
\setCJKmainfont{Noto Serif CJK SC}
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans Mono CJK SC}
```

实际读取的 14 个字体文件路径和 SHA256 记录在 `font_inventory.json`。未复制或额外下载系统 CJK 字体。最小稿还验证了 `amsmath`、`graphicx` 和 `booktabs`。

## 实际编译命令

工作目录为 `/root/NSO`。所有缓存都限定在任务目录，不修改 `HOME`：

```bash
TECTONIC_CACHE_DIR=/root/NSO/tmp/v38-tex/cache \
XDG_CACHE_HOME=/root/NSO/tmp/v38-tex/cache \
XDG_CONFIG_HOME=/root/NSO/tmp/v38-tex/config \
/root/NSO/tmp/v38-tex/bin/tectonic \
  --keep-logs --keep-intermediates --print \
  --outdir /root/NSO/tmp/v38-tex/smoke \
  /root/NSO/tmp/v38-tex/smoke/minimal.tex
```

首次编译由已保存的 `run_guarded.py` 包装：每 0.05 秒检查共享磁盘，低于 80 MiB 则终止进程组，为 64 MiB 保留额外缓冲；单次上限 240 秒。首次编译用时 154.20 秒，exit 0，保护门未触发，结束时可用 92,504,064 B。Tectonic 按需下载格式和宏包，未下载整套 TeX Live 或完整 bundle。缓存关联默认 bundle v33，标识为 `6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c`。

确认缺失宏包已缓存后，实际离线回编命令为：

```bash
TECTONIC_CACHE_DIR=/root/NSO/tmp/v38-tex/cache \
XDG_CACHE_HOME=/root/NSO/tmp/v38-tex/cache \
XDG_CONFIG_HOME=/root/NSO/tmp/v38-tex/config \
/root/NSO/tmp/v38-tex/bin/tectonic \
  --only-cached --keep-logs \
  --outdir /root/NSO/tmp/v38-tex/smoke_cached \
  /root/NSO/tmp/v38-tex/smoke/minimal.tex
```

记录了 1.57 秒、exit 0 的离线回编；此前一次同命令也产生了 PDF，但未保存进程退出收据，不将其单独列为一项成功验证。完整稿新增宏包时，应先使用相同缓存按需补全并继续监控容量，再以 `--only-cached` 回编。

## PDF 检查结果与剩余限制

最小 PDF 共 1 页、29,212 B，SHA256 为 `8832f6aa480102c1deb7b37c227e42ef4bb92124b15bf9314aab30e02b389e03`。使用 pypdfium2 渲染为 918×1188 像素并实际查看，中文、英文字体、公式和三线表可读，无缺字方框或重叠；提取文本包含“语义”“覆盖”“方法”。页面图片与提取文本均封存。

```python
import pypdfium2 as pdfium
pdf = pdfium.PdfDocument("paper.pdf")
for index in range(len(pdf)):
    page = pdf[index]
    page.render(scale=1.5).to_pil().save(f"page-{index+1:02d}.png")
    text = page.get_textpage().get_text_bounded()
```

最小稿日志有两类非致命提示：读取系统绝对字体路径，其他机器需安装相同字体并核对清单；默认 ctex 字号使数学扩展字体发生最高 0.41063 pt 的字号替代。未出现缺失字符或 TeX 错误。后续主稿应结合最终字号和数学字体检查这些提示，并逐页检查浮动体、引用、图例与越界；本轮没有修改主稿或实验图片。
