#!/bin/bash
# Matterport3D数据集配置下载脚本

set -e

echo "=========================================="
echo "下载Matterport3D数据集配置"
echo "=========================================="

PROJECT_DIR="/home/ubuntu/lzy/ANS/Neural-SLAM"
cd "$PROJECT_DIR"

# 创建目录结构
echo "创建目录结构..."
mkdir -p data/datasets/pointnav/mp3d/v1/train
mkdir -p data/datasets/pointnav/mp3d/v1/val
mkdir -p data/datasets/pointnav/mp3d/v1/test

# 下载数据集配置
echo "从habitat-api下载数据集配置..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# 克隆habitat-api（只获取数据集配置部分）
echo "克隆habitat-api仓库..."
git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/habitat-api.git
cd habitat-api
git sparse-checkout init --cone
git sparse-checkout set data/datasets/pointnav/mp3d

# 复制配置文件
if [ -d "data/datasets/pointnav/mp3d" ]; then
    echo "复制数据集配置文件..."
    cp -r data/datasets/pointnav/mp3d/* "$PROJECT_DIR/data/datasets/pointnav/mp3d/"
    echo "✓ 数据集配置下载完成！"
else
    echo "✗ 未找到数据集配置文件，尝试完整克隆..."
    cd "$TEMP_DIR"
    rm -rf habitat-api
    git clone --depth 1 https://github.com/facebookresearch/habitat-api.git
    if [ -d "habitat-api/data/datasets/pointnav/mp3d" ]; then
        cp -r habitat-api/data/datasets/pointnav/mp3d/* "$PROJECT_DIR/data/datasets/pointnav/mp3d/"
        echo "✓ 数据集配置下载完成！"
    else
        echo "✗ 仍然未找到配置文件，请手动下载"
        exit 1
    fi
fi

# 清理
cd "$PROJECT_DIR"
rm -rf "$TEMP_DIR"

# 验证
echo ""
echo "验证下载结果..."
if [ -f "data/datasets/pointnav/mp3d/v1/val/val.json.gz" ]; then
    echo "✓ 验证集配置文件存在"
    python3 -c "
import gzip, json
f = gzip.open('data/datasets/pointnav/mp3d/v1/val/val.json.gz', 'rt')
data = json.load(f)
scenes = set([ep['scene_id'].split('/')[-1].split('.')[0] for ep in data['episodes']])
print(f'✓ 验证集中有 {len(scenes)} 个场景')
print(f'  场景列表: {sorted(list(scenes))[:5]}...')
f.close()
" 2>/dev/null || echo "  注意: 无法读取JSON文件（可能需要先下载场景）"
else
    echo "✗ 验证集配置文件不存在"
fi

echo ""
echo "=========================================="
echo "下一步："
echo "1. 访问 https://niessner.github.io/Matterport/ 申请访问权限"
echo "2. 下载场景文件到: data/scene_datasets/mp3d/"
echo "3. 运行验证脚本: python verify_mp3d_setup.py"
echo "=========================================="


