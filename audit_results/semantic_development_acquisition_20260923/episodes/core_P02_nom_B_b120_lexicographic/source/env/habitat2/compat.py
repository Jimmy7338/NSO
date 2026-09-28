"""Habitat 2 与 NSO (H0.1) 之间的兼容辅助。"""
from __future__ import annotations


def normalize_observations(obs):
    """将 H2 观测 dict 规范为含 rgb / depth 的 dict。"""
    if obs is None:
        return obs
    if "rgb" in obs and "depth" in obs:
        return obs
    out = dict(obs)
    for k, v in obs.items():
        lk = k.lower()
        if "rgb" in lk or lk == "color_sensor":
            out.setdefault("rgb", v)
        if "depth" in lk:
            out.setdefault("depth", v)
    return out


def get_scene_id(env_wrapper):
    """Exploration_Env / RLEnv 上的场景路径。"""
    if hasattr(env_wrapper, "habitat_env"):
        sim = env_wrapper.habitat_env.sim
        if hasattr(sim, "config") and hasattr(sim.config, "SCENE"):
            return sim.config.SCENE
        if hasattr(sim, "curr_scene_name"):
            return sim.curr_scene_name
    env = getattr(env_wrapper, "_env", env_wrapper)
    if hasattr(env, "current_episode") and env.current_episode is not None:
        return env.current_episode.scene_id
    if hasattr(env, "sim") and hasattr(env.sim, "curr_scene_name"):
        return env.sim.curr_scene_name
    return None


def get_agent_state(env_wrapper, agent_id=0):
    if hasattr(env_wrapper, "habitat_env"):
        return env_wrapper.habitat_env.sim.get_agent_state(agent_id)
    env = getattr(env_wrapper, "_env", env_wrapper)
    return env.sim.get_agent_state(agent_id)
