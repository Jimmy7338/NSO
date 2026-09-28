#!/usr/bin/env bash
# NSO 服务器一键安装脚本（Ubuntu 22.04+，CPU/GPU 均可）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-nso}"
GHFAST="${GHFAST:-https://ghfast.top/https://github.com}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "项目目录: $PROJECT_DIR"

# 1. 系统依赖
if command -v apt-get >/dev/null 2>&1; then
  log "安装系统依赖..."
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential cmake git wget curl pkg-config \
    libegl1-mesa-dev libgles2-mesa-dev libgl1-mesa-dev \
    libjpeg-dev libpng-dev python3-dev
fi

# 2. Miniconda
if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
  log "安装 Miniconda..."
  curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "创建 conda 环境 $ENV_NAME (Python 3.8)..."
  conda create -n "$ENV_NAME" python=3.8 -y
fi
conda activate "$ENV_NAME"

# 3. habitat-sim（与项目 README 版本接近的 0.1.7 headless）
log "安装 habitat-sim..."
conda install -y -c conda-forge -c aihabitat habitat-sim=0.1.7 headless

# 4. habitat-api 子模块（GitHub 直连不稳定时用镜像 tarball）
HABITAT_API_DIR="$PROJECT_DIR/env/habitat/habitat_api"
if [[ ! -f "$HABITAT_API_DIR/setup.py" ]]; then
  log "下载 habitat-api 子模块..."
  mkdir -p "$PROJECT_DIR/env/habitat"
  curl -fsSL -o /tmp/habitat-api-dev.tar.gz \
    "${GHFAST}/devendrachaplot/habitat-api/archive/8ac1b16aa0554b10748a925f8a937954c77c1563.tar.gz"
  tar -xzf /tmp/habitat-api-dev.tar.gz -C /tmp
  rm -rf "$HABITAT_API_DIR"
  mv /tmp/habitat-api-8ac1b16aa0554b10748a925f8a937954c77c1563 "$HABITAT_API_DIR"
fi

log "安装 habitat (editable)..."
pip install -e "$HABITAT_API_DIR" --no-deps
pip install gym==0.10.9 yacs numpy-quaternion attrs opencv-python imageio imageio-ffmpeg

# 5. PyTorch（有 NVIDIA 时用 CUDA 11.7 版，与 torch 1.13.1 匹配）
if python -c "import torch" 2>/dev/null; then
  log "PyTorch 已安装，跳过"
elif command -v nvidia-smi >/dev/null 2>&1; then
  log "安装 PyTorch (CUDA 11.7)..."
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
else
  log "安装 PyTorch (CPU)..."
  pip install torch==1.13.1 torchvision==0.14.1 \
    --index-url https://download.pytorch.org/whl/cpu
fi

# 6. 其余 Python 依赖
log "安装 NSO 依赖..."
conda install -y scikit-image scikit-learn matplotlib tensorboard scipy
pip install --default-timeout=600 -i "$PYPI_MIRROR" scikit-fmm
pip install --default-timeout=600 -i "$PYPI_MIRROR" \
  -r "$PROJECT_DIR/requirements.txt"

# 7. 数据目录
mkdir -p "$PROJECT_DIR/data/scene_datasets" \
         "$PROJECT_DIR/data/datasets/pointnav/gibson/v1/val" \
         "$PROJECT_DIR/pretrained_models" \
         "$PROJECT_DIR/tmp"

log "安装完成。验证导入..."
cd "$PROJECT_DIR"
python - <<'PY'
import torch, habitat_sim, habitat, skfmm
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("habitat_sim", habitat_sim.__version__)
print("habitat", habitat.__version__)
PY

log "运行 python verify_mp3d_setup.py 检查数据集；下载数据后执行 bash scripts/run_nso.sh"
log "无显示器服务器请保持 MPLBACKEND=Agg（run_nso.sh 已默认设置）"
