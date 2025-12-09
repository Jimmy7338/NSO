# GitHub 上传指南

## 步骤 1：准备 Git 仓库

### 1.1 检查当前状态

```bash
cd /home/ubuntu/lzy/ANS/Neural-SLAM
git status
```

### 1.2 查看当前远程仓库

```bash
git remote -v
```

如果显示的是原作者的仓库（devendrachaplot/Neural-SLAM），需要移除并添加你自己的仓库。

---

## 步骤 2：在 GitHub 上创建新仓库

1. 登录 GitHub (https://github.com)
2. 点击右上角的 "+" 号，选择 "New repository"
3. 填写仓库信息：
   - **Repository name**: `Neural-SLAM` 或你喜欢的名称
   - **Description**: `Enhanced Active Neural SLAM with semantic detection and structural awareness`
   - **Visibility**: 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"（因为本地已有）
4. 点击 "Create repository"

---

## 步骤 3：更新远程仓库配置

### 3.1 移除旧的远程仓库（如果存在）

```bash
git remote remove origin
```

### 3.2 添加你的 GitHub 仓库

**方式 1：使用 SSH（推荐）**

```bash
# 替换 YOUR_USERNAME 和 YOUR_REPO_NAME 为你的实际值
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
```

**方式 2：使用 HTTPS**

```bash
# 替换 YOUR_USERNAME 和 YOUR_REPO_NAME 为你的实际值
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### 3.3 验证远程仓库配置

```bash
git remote -v
```

应该显示你新添加的仓库地址。

---

## 步骤 4：添加和提交更改

### 4.1 查看所有更改

```bash
git status
```

### 4.2 添加所有更改的文件

```bash
# 添加所有修改和新文件
git add .

# 或者只添加特定文件
git add README.md
git add docs/
git add *.py
# ... 等等
```

### 4.3 提交更改

```bash
git commit -m "Enhanced Neural SLAM with semantic detection and structural awareness

- Added YOLOv8 semantic object detection
- Implemented structural reward (door frames, narrow passages, open areas)
- Added frontier reward for visible but unvisited areas
- Integrated semantic loop detection with NetVLAD
- Enhanced reward mechanism with multi-level rewards
- Updated documentation and project structure"
```

**提示：** 可以根据实际情况修改提交信息。

---

## 步骤 5：推送到 GitHub

### 5.1 首次推送（如果远程仓库是空的）

```bash
# 推送主分支并设置上游
git push -u origin main
```

如果默认分支是 `master` 而不是 `main`：

```bash
# 先重命名本地分支（如果需要）
git branch -M main

# 然后推送
git push -u origin main
```

### 5.2 如果远程仓库已有内容（冲突处理）

如果 GitHub 仓库初始化时创建了 README 等文件，需要先拉取：

```bash
# 拉取远程内容并允许不相关历史合并
git pull origin main --allow-unrelated-histories

# 解决可能的冲突后，再推送
git push -u origin main
```

---

## 步骤 6：验证上传

1. 访问你的 GitHub 仓库页面
2. 检查文件是否都已上传
3. 查看 README.md 是否正确显示

---

## 常见问题

### Q1: 推送时提示需要认证

**SSH 方式：**
- 确保已配置 SSH 密钥：`ssh -T git@github.com`
- 如果未配置，参考：https://docs.github.com/en/authentication/connecting-to-github-with-ssh

**HTTPS 方式：**
- 使用 Personal Access Token 代替密码
- 创建 Token：GitHub Settings → Developer settings → Personal access tokens

### Q2: 文件太大无法推送

如果某些文件（如模型文件、数据集）太大：

```bash
# 使用 Git LFS（Large File Storage）
git lfs install
git lfs track "*.pt"
git lfs track "*.glb"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

或者将这些大文件添加到 `.gitignore` 中。

### Q3: 想保留原仓库作为上游

如果你想保留原仓库作为参考：

```bash
# 添加原仓库作为上游
git remote add upstream git@github.com:devendrachaplot/Neural-SLAM.git

# 你的仓库作为 origin
git remote set-url origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Q4: 只想推送部分文件

```bash
# 创建 .git/info/exclude 文件来忽略特定文件（不提交到仓库）
# 或者使用 git add 只添加需要的文件
```

---

## 后续更新

以后有新的更改时：

```bash
# 1. 查看更改
git status

# 2. 添加更改
git add .

# 3. 提交
git commit -m "描述你的更改"

# 4. 推送
git push
```

---

## 快速命令总结

```bash
# 1. 移除旧远程（如果需要）
git remote remove origin

# 2. 添加新远程（替换为你的仓库地址）
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git

# 3. 添加所有文件
git add .

# 4. 提交
git commit -m "Initial commit: Enhanced Neural SLAM"

# 5. 推送
git push -u origin main
```

---

**注意：** 
- 确保 `.gitignore` 已正确配置，避免上传大文件或不必要的文件
- 检查 `data/`、`pretrained_models/` 等目录中的大文件是否应该上传
- 建议使用 SSH 方式，更安全且方便

