# Active Neural SLAM (Enhanced Version)

本项目是基于 ICLR 2020 论文《Learning To Explore Using Active Neural SLAM》的开源实现进行改进和扩展的版本。

**参考论文：** [Learning To Explore Using Active Neural SLAM](https://openreview.net/pdf?id=HklXn1BKDH)

### 主要改进：

- ✅ **语义增强探索**：集成 YOLOv8 进行实时语义对象检测，基于语义信息优化探索策略
- ✅ **结构感知奖励**：识别门框、狭窄通道、开阔区域等结构特征，引导智能体优先探索关键区域
- ✅ **前沿区域奖励**：鼓励探索可见但未访问的区域，提高探索效率
- ✅ **语义回环检测**：基于 NetVLAD 和语义信息的回环检测，提高定位精度
- ✅ **增强的奖励机制**：多层次的奖励函数，结合语义、结构和前沿信息

### Overview:

本项目包含三个核心模块：Global Policy（全局策略）、Local Policy（局部策略）和 Neural SLAM Module（神经 SLAM 模块）。

Neural SLAM 模块从 RGB 图像预测地图和智能体位姿。全局策略基于地图状态输出长期目标，通过路径规划器转换为短期目标。局部策略负责导航到短期目标。

![overview](./docs/overview.png)


## Installing Dependencies
We use earlier versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-api](https://github.com/facebookresearch/habitat-api). The specific commits are mentioned below.

Installing habitat-sim:
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout 9575dcd45fe6f55d2a44043833af08972a7895a9; 
pip install -r requirements.txt; 
python setup.py install --headless
python setup.py install # (for Mac OS)

```

Installing habitat-api:
```
git clone https://github.com/facebookresearch/habitat-api.git
cd habitat-api; git checkout b5f2b00a25627ecb52b43b13ea96b05998d9a121; 
pip install -e .
```

Install pytorch from https://pytorch.org/ according to your system configuration. The code is tested on pytorch v1.2.0. If you are using conda:
```
conda install pytorch==1.2.0 torchvision cudatoolkit=10.0 -c pytorch #(Linux with GPU)
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch #(Mac OS)
```

## Setup

### 安装依赖

安装项目依赖：
```bash
pip install -r requirements.txt
```

The code requires datasets in a `data` folder in the following format (same as habitat-api):
```
Neural-SLAM/
  data/
    scene_datasets/
      gibson/
        Adrian.glb
        Adrian.navmesh
        ...
    datasets/
      pointnav/
        gibson/
          v1/
            train/
            val/
            ...
```
Please download the data using the instructions here: https://github.com/facebookresearch/habitat-api#data

To verify that dependencies are correctly installed and data is setup correctly, run:
```
python main.py -n1 --auto_gpu_config 0 --split val
```


## Usage

### 基础训练

训练完整的模型（所有模块）：
```bash
python main.py
```

### 使用语义信息训练（推荐）

启用语义检测和增强的奖励机制：
```bash
python main.py \
  --use_semantic \
  --semantic_use_all_classes \
  --semantic_conf_thresh 0.1 \
  --semantic_interval 1 \
  --semantic_reward_coeff 0.12 \
  --structural_reward_coeff 0.12 \
  --frontier_reward_coeff 0.15 \
  --w_struct_door 2.0 \
  --door_boost_distance 5.0 \
  --room_exploration_boost 1.5 \
  --exp_name training_with_semantic
```

或使用提供的脚本：
```bash
bash scripts/train_with_semantic.sh
```

### 评估

评估预训练模型（需要先下载原项目的预训练模型）：
```bash
python main.py \
  --split val \
  --eval 1 \
  --train_global 0 \
  --train_local 0 \
  --train_slam 0 \
  --load_global pretrained_models/model_best.global \
  --load_local pretrained_models/model_best.local \
  --load_slam pretrained_models/model_best.slam \
  -v 1  # 启用可视化
```

### 详细文档

更多详细说明请参考：
- [完整项目指南](./docs/COMPLETE_PROJECT_GUIDE.md) - 全面的项目介绍和使用指南
- [语义训练指南](./docs/TRAINING_WITH_SEMANTIC.md) - 语义功能使用说明
- [语义奖励说明](./docs/SEMANTIC_REWARD_EXPLANATION.md) - 语义奖励计算详解


## 参考论文

本项目基于以下论文的实现：

> Chaplot, D.S., Gandhi, D., Gupta, S., Gupta, A. and Salakhutdinov, R., 2020. Learning To Explore Using Active Neural SLAM. In International Conference on Learning Representations (ICLR). ([PDF](https://openreview.net/pdf?id=HklXn1BKDH))

### Bibtex:
```
@inproceedings{chaplot2020learning,
  title={Learning To Explore Using Active Neural SLAM},
  author={Chaplot, Devendra Singh and Gandhi, Dhiraj and Gupta, Saurabh and Gupta, Abhinav and Salakhutdinov, Ruslan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

## 致谢

本项目基于以下开源项目：
- **Habitat API** (https://github.com/facebookresearch/habitat-api) - 仿真环境
- **PPO 实现** (https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/) - 强化学习算法
- **原项目** (https://github.com/devendrachaplot/Neural-SLAM) - 基础实现

感谢原项目作者和相关开源社区的支持。
