#!/usr/bin/env bash
# 安装 NSO 的 Habitat 2 环境（nso_h2）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-nso_h2}"
TP="$PROJECT_DIR/third_party/habitat-lab"

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [[ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]]; then
  echo "未找到 conda: $CONDA_DIR" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  log "创建 conda 环境 ${ENV_NAME} (python=3.9)..."
  conda create -n "$ENV_NAME" python=3.9 cmake=3.14 -y
fi
conda activate "$ENV_NAME"

log "安装 habitat-sim 0.2.4..."
conda install -y habitat-sim=0.2.4 headless withbullet -c conda-forge -c aihabitat

if [[ ! -d "$TP" ]]; then
  log "未找到 habitat-lab 源码，请先运行: bash scripts/fetch_habitat2_source.sh"
  log "（需本机下载 v0.2.4.tar.gz 并 SFTP 到 third_party/）"
else
  log "安装 habitat-lab (editable)..."
  if [[ -d "$TP/habitat-lab" ]]; then
    pip install -e "$TP/habitat-lab"
  elif [[ -f "$TP/setup.py" ]] || [[ -d "$TP/habitat" ]]; then
    pip install -e "$TP"
  else
    echo "目录结构异常: $TP" >&2
    exit 1
  fi
fi

log "安装 PyTorch CUDA..."
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

log "安装 NSO Python 依赖（opencv/faiss 等用 conda，避免 pip 下载超时）..."
conda install -y -c conda-forge opencv faiss-cpu scikit-fmm seaborn -q 2>/dev/null || true
pip install --default-timeout=600 \
  'gym>=0.22.0,<0.23.1' numpy scipy scikit-image scikit-learn matplotlib \
  tensorboard scikit-fmm pillow pyyacs pyyaml

if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
  pip install --default-timeout=600 -r "$PROJECT_DIR/requirements.txt" || true
fi

log "========== 状态检查 =========="
python -c "import habitat_sim; print('habitat_sim', habitat_sim.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
if python -c "import habitat" 2>/dev/null; then
  python -c "import habitat; print('habitat-lab OK')"
else
  echo "habitat-lab 未安装 — 完成 fetch_habitat2_source.sh 后重新运行本脚本"
fi

log "完成。使用: conda activate ${ENV_NAME}"
