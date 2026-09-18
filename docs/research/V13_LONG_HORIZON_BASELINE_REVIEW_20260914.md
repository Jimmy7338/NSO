# 长期观察价值与外部对照的补充核对

核对日期：2026-09-14。补充已有 TARE、SCOUT、Active3D 等文献研究。本页用于限定设计依据与对照范围，没有复现这些论文的性能。

## GLEAM：长期可达目标不等于物体类别创新

GLEAM 的 ICCV 2025 论文使用以机器人为中心的地图和长期可达目标；其地图四类状态是占据、空闲、未知、前沿。它强调场景多样性及跨场景验证，方法中的“semantic”不能直接解释为货架／设备类别先验。[论文 §4.2–4.3](https://arxiv.org/html/2505.20294v1)、[CVF 正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_GLEAM_Learning_Generalizable_Exploration_Policy_for_Active_Mapping_in_Complex_ICCV_2025_paper.pdf)。

官方实现使用 Isaac Gym／CUDA，并列出 RTX 3090/4090 环境，当前仓库未执行其 CPU 移植。[官方实现与环境说明](https://github.com/zjwzcx/GLEAM)。

本项目的设计判断：保留全局目标和局部可达性检查有依据，但必须把目标选择的时间范围和实际执行方式统一。V13.2 的去程后重规划是对本项目执行定义的单项检验，不是 GLEAM 复现。类别结构增量仍需相同几何条件下的正确／错误／缺失语义对照，不能从名称相似推导优势。

## MACARONS：几何覆盖预测本身也是强对照

MACARONS（CVPR 2023）从 RGB 构建占据预测并学习下一视点的覆盖增益，采用在线自监督机制。[CVF 论文与摘要](https://openaccess.thecvf.com/content/CVPR2023/html/Guedon_MACARONS_Mapping_and_Coverage_Anticipation_With_RGB_Online_Self-Supervision_CVPR_2023_paper.html)。官方代码区分深度、占据、表面覆盖增益模块，并提供不同初始化路线。[官方实现](https://github.com/Anttwo/MACARONS)。

本项目的设计判断：不能只让语义模型学习、几何对照一直使用固定启发式，再把所有收益归于类别知识。同容量的几何观察收益预测是必须保留的内部对照；其作用与官方 MACARONS 的完整复现需分别标注。当前双目深度与移动小车设定也不能直接套用该论文的性能数字。

## GenNBV：表示、动作空间、语义三种贡献要拆开

GenNBV（CVPR 2024）以强化学习预测自由空间中的下一视点，结合几何、语义和动作表征，论文实验面向无人机采集设定。[原始论文](https://arxiv.org/abs/2402.16174)、[CVF 正式论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_GenNBV_Generalizable_Next-Best-View_Policy_for_Active_3D_Reconstruction_CVPR_2024_paper.pdf)。本次只核对摘要、任务范围及官方仓库，不声称完整实现或全部消融复核。

本项目的设计判断：候选池扩大、目标执行范围变化和语义条件化必须分别对照。V13/V13.2 的同候选比较可以检验执行范围带来的变化；只有正确类别改变选择且获得稳定净收益，才推进类别创新主张。外部算法的感知、动作和预算条件需实际对齐后才能比较。

## 当前执行含义

继续保留 ANS 层次及四模块。先完成 V13.2 固定预算信息筛查；若通过，按 [训练隔离与语义安慰剂约束](V13_TRAINING_ISOLATION_AND_PLACEBO_CONTRACT.md)设计未来未见验证。若失败，不能仅靠增加训练规模或重新命名“语义地图”完成目标。

以上对本项目的建议是结合文献与本地代码作出的设计判断，不是文献已经证明本项目有效。TARE 的现有编译／接口检查、这些新核对论文的代码阅读，以及本机 CPU 原型结果继续分别记账。
