#!/usr/bin/env python3
"""habitat-lab 0.2.x 冒烟：PointNav 测试集 reset + step。"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

try:
    import habitat
    from habitat.config.default import get_config
except ImportError:
    print("请先安装 habitat-lab: bash scripts/fetch_habitat2_source.sh && bash scripts/install_habitat2.sh",
          file=sys.stderr)
    sys.exit(1)


def main():
    from habitat.config import read_write

    cfg_path = "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    config = get_config(cfg_path)
    with read_write(config):
        config.habitat.dataset.split = "val"

    env = habitat.Env(config=config)
    obs = env.reset()
    rgb_key = "rgb" if "rgb" in obs else list(obs.keys())[0]
    print("habitat-lab Env OK, obs keys:", list(obs.keys()))
    print("rgb shape", obs[rgb_key].shape if hasattr(obs[rgb_key], "shape") else type(obs[rgb_key]))
    out = env.step(0)
    print("step OK, keys:", list(out.keys()) if isinstance(out, dict) else type(out))
    env.close()
    print("PASS")


if __name__ == "__main__":
    main()
