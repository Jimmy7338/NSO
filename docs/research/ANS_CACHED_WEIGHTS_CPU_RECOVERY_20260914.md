# ANS 缓存权重与 CPU 兼容性复核

2026-09-14：本地 8 份 ANS 检查点缓存均通过 SHA256、仅权重 CPU 反序列化、当前结构严格加载和合成输入前向检查。工作树中的相应文件仍是 Git LFS 指针；此前“工作树指针不可直接加载”的判断成立，但“本机没有可加载权重”的宽泛表述需修正。

## 已验证资产

| 位置 | 模块 | 缓存读取 | 严格加载与单次推理 |
|---|---|---|---|
| pretrained_models | global、local、slam | 3/3 | 3/3 |
| trained_models/paper_fast | global | 1/1 | 1/1 |
| trained_models/stage1_slam_local | local、slam | 2/2 | 2/2 |
| trained_models/stage2_paper_global | local、slam | 2/2 | 2/2 |

原始缓存哈希、逐张量形状、有限性与大小见 [资产清单](../../audit_results/model_lfs_cache_20260914/verification.json)；当前源码哈希、环境版本、严格加载及输出形状见 [CPU 兼容性检查](../../audit_results/model_lfs_cpu_compatibility_v1_1_20260914/verification.json)。所有权重留在原缓存，没有重复展开占用磁盘，也没有改动运行中的 V13 规划源码。

CPU 环境原有 PyTorch 2.6.0+cpu，补装了 torchvision 0.21.0+cpu；版本配对依据 [PyTorch 官方历史版本安装表](https://pytorch.org/get-started/previous-versions/)，下载记录见 [安装清单](../../audit_results/model_lfs_cache_20260914/torchvision_install.json)。没有升级 PyTorch，也没有安装 CUDA。

## 检查内容与边界

使用 128×128 RGB、240×240 全局策略输入和建图网格、64×64 局部建图视域，以及当前代码的隐藏维度。局部策略使用一维 episode mask，与 main.py 的实际调用一致；SLAM 分支包含 build_maps=True 的地图合并路径。全部使用合成零输入、CPU、eval 模式、固定种子和有限值检查。

构造网络时临时关闭 ResNet 初始权重下载，随后 strict=True 加载完整检查点，逐个替换所有参数及持久化缓冲；不以随机初始化的网络输出冒充检查点推理。该操作只在检查脚本进程内生效。

初次检查脚本错误地给局部策略传入二维 mask，导致 3 个局部检查失败；失败记录保存在 [初次输出](../../audit_results/model_lfs_cpu_compatibility_20260914/verification.json)。修正的是检查脚本输入，原网络没有修改。首次运行还发现 torchvision 缺失，补齐后才完成上述检查。

当前 PyTorch 对 grid_sample/affine_grid 使用现代默认对齐方式，旧源码会发出历史默认行为变更提示。此次只检查当前组合能运行；没有验证它与训练时旧版本数值等价，不能静默改对齐方式后称原模型复现。

允许表述：原 ANS 全局、局部、建图模块的缓存资产可用，在本机 CPU 上与当前结构兼容并能完成合成前向。不能表述：训练来源或训练成绩已核验、视觉输入归一化与原训练一致、真实 RGB-D 闭环已接入、RPN-UQ／开放词汇语义网络已经恢复、规划有效性或实车效果成立。

后续先用实际保存的 RGB-D 序列核对预处理、坐标和动作接口，再比较已知真值下的建图误差及路径结果；该验证与当前语义信息价值实验分别记账。
