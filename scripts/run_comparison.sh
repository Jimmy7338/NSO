#!/bin/bash
# 快速运行对比实验的脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "对比实验运行脚本"
echo "=========================================="

# 默认参数
SCENE="${1:-Cantwell}"
EPISODES="${2:-1}"
TIMEOUT="${3:-3600}"

echo "场景: $SCENE"
echo "Episodes: $EPISODES"
echo "超时时间: ${TIMEOUT}秒"
echo ""

cd "$PROJECT_DIR"

# 运行对比实验
python scripts/compare_experiments.py \
  --scene "$SCENE" \
  --episodes "$EPISODES" \
  --timeout "$TIMEOUT"

echo ""
echo "=========================================="
echo "实验完成！查看结果："
echo "  ls -la comparison_results/results_*/"
echo "  cat comparison_results/results_*/comparison_report.md"
echo "=========================================="

