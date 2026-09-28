#!/usr/bin/env python3
"""
Matterport3D安装验证脚本
检查场景文件、数据集配置和任务配置是否正确
"""

import os
import sys
import gzip
import json

def check_scene_files():
    """检查场景文件"""
    print("\n[1] 检查场景文件...")
    scene_dir = "data/scene_datasets/mp3d"
    
    if not os.path.exists(scene_dir):
        print(f"  ✗ 场景目录不存在: {scene_dir}")
        print(f"    请创建目录并下载场景文件")
        return False
    
    glb_files = [f for f in os.listdir(scene_dir) if f.endswith('.glb')]
    navmesh_files = [f for f in os.listdir(scene_dir) if f.endswith('.navmesh')]
    
    if len(glb_files) == 0:
        print(f"  ✗ 未找到场景文件 (.glb)")
        print(f"    请下载Matterport3D场景文件到: {scene_dir}/")
        return False
    
    print(f"  ✓ 找到 {len(glb_files)} 个场景文件")
    print(f"  ✓ 找到 {len(navmesh_files)} 个导航网格文件")
    
    # 检查是否有匹配的navmesh
    missing_navmesh = []
    for glb in glb_files:
        navmesh = glb.replace('.glb', '.navmesh')
        if navmesh not in navmesh_files:
            missing_navmesh.append(glb)
    
    if missing_navmesh:
        print(f"  ⚠ 警告: {len(missing_navmesh)} 个场景缺少导航网格文件")
        print(f"    示例: {missing_navmesh[0]}")
    else:
        print(f"  ✓ 所有场景都有对应的导航网格文件")
    
    print(f"  示例场景: {glb_files[0]}")
    return True

def check_dataset_config():
    """检查数据集配置"""
    print("\n[2] 检查数据集配置...")
    dataset_file = "data/datasets/pointnav/mp3d/v1/val/val.json.gz"
    
    if not os.path.exists(dataset_file):
        print(f"  ✗ 数据集配置文件不存在: {dataset_file}")
        print(f"    运行: bash scripts/download_mp3d_config.sh")
        return False
    
    print(f"  ✓ 数据集配置文件存在")
    
    try:
        with gzip.open(dataset_file, 'rt') as f:
            data = json.load(f)
            scenes_in_dataset = set([ep['scene_id'].split('/')[-1].split('.')[0] 
                                   for ep in data['episodes']])
            print(f"  ✓ 验证集中有 {len(scenes_in_dataset)} 个场景")
            print(f"  示例场景: {sorted(list(scenes_in_dataset))[:3]}")
            return scenes_in_dataset
    except Exception as e:
        print(f"  ✗ 无法读取数据集文件: {e}")
        return False

def check_task_config():
    """检查任务配置"""
    print("\n[3] 检查任务配置...")
    task_config = "env/habitat/habitat_api/configs/tasks/pointnav_mp3d.yaml"
    
    if not os.path.exists(task_config):
        print(f"  ✗ 任务配置文件不存在: {task_config}")
        return False
    
    print(f"  ✓ 任务配置文件存在")
    return True

def check_scene_match(scenes_in_dataset):
    """检查场景文件与数据集配置是否匹配"""
    print("\n[4] 检查场景匹配...")
    
    scene_dir = "data/scene_datasets/mp3d"
    if not os.path.exists(scene_dir):
        print("  ⚠ 跳过（场景目录不存在）")
        return
    
    glb_files = [f.replace('.glb', '') for f in os.listdir(scene_dir) if f.endswith('.glb')]
    scenes_in_files = set(glb_files)
    
    if scenes_in_dataset:
        missing_scenes = scenes_in_dataset - scenes_in_files
        if missing_scenes:
            print(f"  ⚠ 警告: 数据集配置中有 {len(missing_scenes)} 个场景未下载")
            print(f"    缺失场景示例: {sorted(list(missing_scenes))[:3]}")
        else:
            print(f"  ✓ 所有数据集中的场景都已下载")
        
        extra_scenes = scenes_in_files - scenes_in_dataset
        if extra_scenes:
            print(f"  ℹ 信息: 有 {len(extra_scenes)} 个额外场景文件（不在验证集中）")

def main():
    print("=" * 60)
    print("Matterport3D安装验证")
    print("=" * 60)
    
    results = {
        'scenes': check_scene_files(),
        'dataset': check_dataset_config(),
        'task': check_task_config()
    }
    
    if isinstance(results['dataset'], set):
        check_scene_match(results['dataset'])
    
    print("\n" + "=" * 60)
    print("验证总结:")
    print("=" * 60)
    
    if results['scenes'] and results['dataset'] and results['task']:
        print("✓ 所有检查通过！可以运行Matterport3D场景")
        print("\n运行命令:")
        print("  python main.py \\")
        print("    --task_config tasks/pointnav_mp3d.yaml \\")
        print("    --split val \\")
        print("    --eval 1 \\")
        print("    --num_processes 1 \\")
        print("    --auto_gpu_config 0")
    else:
        print("✗ 部分检查未通过，请根据上述提示修复问题")
        if not results['scenes']:
            print("  - 需要下载场景文件")
        if not results['dataset']:
            print("  - 需要下载数据集配置（运行: bash scripts/download_mp3d_config.sh）")
        if not results['task']:
            print("  - 任务配置文件缺失")
    
    print("=" * 60)

if __name__ == "__main__":
    main()


