"""
Habitat 2 版 Exploration_Env：复用 H0.1 逻辑（配置/动作空间兼容已在基类处理）。
"""
from env.habitat.exploration_env import Exploration_Env as _ExplorationEnvH1

from . import compat


class Exploration_Env(_ExplorationEnvH1):
    """Habitat 2 环境；观测键名与 H0.1 一致时无需额外包装。"""

    pass
