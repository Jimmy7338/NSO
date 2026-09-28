# 系统整体架构图：小车融合版

定稿：[PNG](system_architecture.png) · [PDF](system_architecture.pdf)。用于毕设图1及中英文文章图2。

采用内置 `image_gen.imagegen` 生成与修订。工具不暴露具体模型名称或模型选择参数，因此不声称使用了某个指定型号。定稿原生尺寸为 **1672 × 941 px**；未进行伪4K放大。PDF无损嵌入相同RGB像素，属于栅格插图，非矢量图。

版式以用户小车为平台入口，将语义证据和几何反馈汇聚为构型信念，连接预算规划与动作执行；实测重建和离线评价独立展示。小车经过生成式合成，地图、网格及传感缩略图均为机制示意。实际实验网格继续使用论文中的独立结果图。

原始小车抠图保存在 `docs/thesis/assets/robot_20260928/robot_cutout_user.png`，与用户附件逐字节相同。旧架构图保留在 `virtual_method_20260928`，实验代码与数据未因本次制图更改。

生成过程依次采用以下提示词；最后一次输出为论文定稿。

1. [整体设计](generation_prompt.txt)
2. [主要数据流修订](revision_prompt.txt)
3. [先验与观测支路修订](final_revision_prompt.txt)
4. [共同信念节点与底部连线](wiring_revision_prompt.txt)
5. [清除冗余连线](cleanup_revision_prompt.txt)
6. [闭环连线定稿](bridge_revision_prompt.txt)

`drafts/`仅保留生成过程，包含尚未修正的连线，不用于论文。所有输入、提示词、各轮输出和导出记录的校验值见 [provenance.json](provenance.json)。

重新导出PDF：

```bash
python3 -B scripts/export_generated_architecture_20260928.py
```

导出器仅进行无损格式封装，不修图、不重画。论文源稿的图注已注明生成式合成和示意属性。
