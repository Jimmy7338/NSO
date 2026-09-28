#!/bin/bash
# 使用语义信息进行训练的脚本

cd "$(dirname "$0")/.." || exit 1

python main.py \
  --split train \
  --eval 0 \
  --train_global 1 \
  --train_local 1 \
  --train_slam 1 \
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
  --num_processes 4 \
  --num_mini_batch 2 \
  --exp_name training_with_semantic

