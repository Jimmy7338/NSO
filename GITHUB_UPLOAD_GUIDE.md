# GitHub 上传指南

> **作者：** 李兆宇  
> **仓库：** NSO（基于 ANS 扩展的主动覆盖探索项目）

---

## 一、检查状态

```bash
cd ~/NSO
git status
git remote -v
git log --oneline -5
```

---

## 二、大文件与 Git LFS

本仓库含预训练权重、场景配置等较大文件，已配置 `.gitattributes` 使用 **Git LFS**。

```bash
git lfs install
git lfs track
git lfs ls-files
```

推送时需同时推送 LFS 对象：

```bash
git push origin main
git lfs push origin main --all
```

若单包过大（>400 MB）导致 TLS 超时，可：

1. **增大缓冲区：**
   ```bash
   git -c http.postBuffer=524288000 push origin main
   ```

2. **分批推送 commit：**
   ```bash
   git push origin <commit_sha>:main
   ```

3. **Bundle 中转（服务器无法直连 GitHub 时）：**
   ```bash
   git bundle create nso_push.bundle origin/main..HEAD
   # 在可访问 GitHub 的机器：
   git clone <repo_url> NSO-local && cd NSO-local
   git pull /path/to/nso_push.bundle main
   git push origin main
   git lfs push origin main --all
   ```

---

## 三、不应提交的内容

- `.env`、API 密钥、个人 token
- `/mnt/nso_data/` 上的完整 3D 网格（仓库内仅为软链接）
- 训练中间产物：`periodic_*`、`train.log`、可视化帧缓存
- `nso_push_*.bundle`（中转文件，勿纳入版本库）

`trained_models/` 仅纳入 `model_best.*` 最终权重；stage3/4 最新 checkpoint 见 [trained_models/README.md](trained_models/README.md)。

---

## 四、首次推送新仓库

```bash
# 若 remote 仍指向 ANS 官方仓库，更换为自己的
git remote remove origin
git remote add origin git@github.com:<your_user>/NSO.git

git push -u origin main
git lfs push origin main --all
```

---

## 五、提交信息建议

与论文主线一致时使用清晰前缀，例如：

- `paper: 更新 OV-SDF/STGHP 文档`
- `feat: 添加 --paper_mode 评估脚本`
- `data: LFS 纳入 stage2 checkpoint`

---

## 六、常见问题

| 问题 | 处理 |
|------|------|
| `TLS disconnect` | 增大 `http.postBuffer` 或 bundle 中转 |
| LFS 对象缺失 | `git lfs fetch --all` 后重推 |
| 误提交大文件 | `git lfs migrate import` 或从历史中移除后 force（慎用） |

更多运行期问题见 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。
