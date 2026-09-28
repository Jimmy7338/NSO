#!/usr/bin/env bash
# 修复 headless habitat-sim 在部分 GPU 服务器上 RGB 全黑（ASTC 纹理解码）
# 安装带 CUDA 的 habitat-sim，用 GPU 做 EGL 渲染
set -euo pipefail

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-nso}"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "当前 habitat-sim:"
conda list habitat-sim | head -5

echo "切换为 display 版 habitat-sim（GPU 渲染，替代 headless）..."
conda install -y --force-reinstall -c aihabitat -c conda-forge \
  "habitat-sim=0.1.7=py3.8_linux_856d4b08c1a2632626bf0d205bf46471a99502b7" \
  "habitat-sim-mutex=1.0=display_nobullet"

echo "验证 RGB 通道（需 NVIDIA OpenGL）..."
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
xvfb-run -a python - <<'PY'
import numpy as np, habitat_sim
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
sim_cfg.gpu_device_id = 0
rgb_cfg = habitat_sim.SensorSpec()
rgb_cfg.uuid = "rgb"
rgb_cfg.sensor_type = habitat_sim.SensorType.COLOR
rgb_cfg.resolution = [256, 256]
rgb_cfg.position = [0.0, 1.25, 0.0]
agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb_cfg]
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
sim.initialize_agent(0)
o = sim.get_sensor_observations()["rgb"]
print("rgb123 max", o[:, :, :3].max(), "mean", float(o[:, :, :3].mean()))
sim.close()
assert o[:, :, :3].max() > 0, "RGB 仍全黑，请检查 nvidia-smi 与驱动"
PY
echo "habitat-sim 渲染修复完成。"
