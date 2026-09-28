# SSCNav 语义补全地图集成指南

> **⚠️ 归档文档（2025-12）**  
> 作者：李兆宇。SSC 不作为 NSO 论文创新点；代码仍保留于 `semantic/ssc_completer.py` 供消融参考。  
> **当前推荐：** `--paper_mode` + OV-SDF，见 [完整项目指南](./COMPLETE_PROJECT_GUIDE.md)。

---

## 概述

本文档说明如何将 SSCNav 的语义场景补全（Semantic Scene Completion, SSC）功能集成到项目中，以提高地图覆盖率和探索效率。

## SSCNav 原理

SSCNav 通过语义场景补全模块，利用部分观测数据预测未观测区域的语义信息，从而：
1. **提高地图覆盖率**：预测被遮挡区域的语义内容
2. **优化探索策略**：基于补全的语义信息引导探索
3. **增强导航能力**：提供更完整的环境表示

## 集成方案

### 方案 1：使用现成的 SSC 模型（推荐）

#### 选项 A：使用 SSCNet / TS3D
- **SSCNet**: 经典的语义场景补全网络
- **TS3D**: 更先进的时序语义场景补全
- **优点**: 有现成的 PyTorch 实现，可直接使用

#### 选项 B：使用轻量级 SSC 模型
- **MonoScene**: 单目图像的语义场景补全
- **优点**: 只需要 RGB 图像，不需要深度图

### 方案 2：简化实现（快速原型）

实现一个基于规则的语义补全模块：
- 使用几何推理（如房间结构、物体分布规律）
- 结合已有的语义检测结果进行插值
- **优点**: 实现简单，无需额外模型

## 实现步骤

### 步骤 1：安装依赖

```bash
# 如果需要使用 SSCNet
pip install torch torchvision
# 其他依赖根据选择的方案而定
```

### 步骤 2：创建 SSC 模块

创建 `semantic/ssc_completer.py` 实现语义场景补全功能。

### 步骤 3：集成到地图更新流程

在 `exploration_env.py` 的地图更新逻辑中调用 SSC 模块。

### 步骤 4：更新语义地图

将补全后的语义信息融合到 `SemanticMap2D` 中。

## 使用方式

### 基础使用（基于规则的补全）

```bash
python main.py \
  --use_semantic \
  --use_ssc_completion \
  --ssc_confidence_thresh 0.5 \
  --ssc_update_interval 10 \
  --ssc_max_distance 10 \
  --ssc_use_structural_prior
```

### 完整训练（推荐）

使用提供的脚本：
```bash
bash scripts/train_with_ssc.sh
```

### 使用深度学习模型（如果可用）

```bash
python main.py \
  --use_semantic \
  --use_ssc_completion \
  --ssc_model_path path/to/ssc_model.pth \
  --ssc_confidence_thresh 0.5 \
  --ssc_update_interval 10
```

## 参数说明

- `--use_ssc_completion`: 启用语义场景补全
- `--ssc_confidence_thresh`: 补全结果的置信度阈值
- `--ssc_update_interval`: SSC 更新的间隔步数
- `--ssc_model_path`: SSC 模型路径（如果使用预训练模型）

## 性能优化

1. **延迟更新**：不是每步都进行补全，而是定期更新
2. **置信度过滤**：只保留高置信度的补全结果
3. **区域限制**：只对关键区域（如未探索区域）进行补全

## 参考资源

- SSCNav 论文: https://arxiv.org/abs/2012.04512
- SSCNav 官网: https://sscnav.cs.columbia.edu/
- SSCNet 实现: 可在 GitHub 搜索 "SSCNet pytorch"

