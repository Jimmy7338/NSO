"""将 habitat-lab 0.2.x VectorEnv 适配为 NSO 使用的 H0.1 接口。"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union

import numpy as np

_OBS_KEY = "obs"


def _normalize_obs(obs: Any) -> np.ndarray:
    """从 EnvObsDictWrapper / Exploration_Env 输出提取 (C,H,W) float32 数组。"""
    if isinstance(obs, dict):
        if _OBS_KEY in obs:
            obs = obs[_OBS_KEY]
        elif len(obs) == 1:
            obs = next(iter(obs.values()))
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32, copy=False)
    return np.asarray(obs, dtype=np.float32)


def _split_reset_result(result: Any) -> Tuple[np.ndarray, dict]:
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return _normalize_obs(obs), info if isinstance(info, dict) else {}
    return _normalize_obs(result), {}


def _unwrap_obs_and_info(obs: Any, info: dict) -> Tuple[np.ndarray, dict]:
    """处理 auto_reset 后 obs 被赋值为 reset() 完整返回 (obs, info) 的情况。"""
    if isinstance(obs, tuple) and len(obs) == 2:
        inner_obs, inner_info = obs
        if isinstance(inner_info, dict):
            merged = dict(info)
            merged.update(inner_info)
            info = merged
        obs = inner_obs
    return _normalize_obs(obs), info


def _split_step_result(result: Any) -> Tuple[np.ndarray, float, bool, dict]:
    if isinstance(result, tuple) and len(result) == 4:
        obs, rew, done, info = result
        info = info if isinstance(info, dict) else {}
        obs, info = _unwrap_obs_and_info(obs, info)
        return obs, float(rew), bool(done), info
    raise TypeError(f"unexpected step result: {type(result)}")


class VectorEnvCompat:
    """补齐 observation_space，并将 reset/step 转为 (obs, info) 批格式。"""

    def __init__(self, venv: Any) -> None:
        self.venv = venv
        self.num_envs = venv.num_envs
        if hasattr(venv, "observation_space"):
            self.observation_space = venv.observation_space
        else:
            self.observation_space = venv.observation_spaces[0]
        if hasattr(venv, "action_space"):
            self.action_space = venv.action_space
        else:
            self.action_space = venv.action_spaces[0]

    def reset(self):
        raw = self.venv.reset()
        if not isinstance(raw, list):
            raw = [raw]
        obs_list, infos = [], []
        for item in raw:
            o, i = _split_reset_result(item)
            obs_list.append(o)
            infos.append(i)
        return np.stack(obs_list), tuple(infos)

    def step_async(self, actions: Sequence[Any]) -> None:
        if hasattr(self.venv, "step_async"):
            self.venv.step_async(actions)
        else:
            self.venv.async_step(actions)

    def step_wait(self):
        if hasattr(self.venv, "step_wait"):
            raw = self.venv.step_wait()
        else:
            raw = self.venv.wait_step()
        if not isinstance(raw, list):
            raw = [raw]
        obs, rews, dones, infos = [], [], [], []
        for item in raw:
            o, r, d, i = _split_step_result(item)
            obs.append(o)
            rews.append(r)
            dones.append(d)
            infos.append(i)
        return np.stack(obs), np.stack(rews), np.stack(dones), tuple(infos)

    def step(self, actions: Sequence[Any]):
        self.step_async(actions)
        return self.step_wait()

    def get_short_term_goal(self, inputs):
        if hasattr(self.venv, "get_short_term_goal"):
            return self.venv.get_short_term_goal(inputs)
        results = []
        for i, inp in zip(range(self.num_envs), inputs):
            results.append(
                self.venv.call_at(i, "get_short_term_goal", {"inputs": inp})
            )
        return np.stack(results)

    def get_rewards(self, inputs):
        if hasattr(self.venv, "get_rewards"):
            return self.venv.get_rewards(inputs)
        results = []
        for i, inp in zip(range(self.num_envs), inputs):
            results.append(self.venv.call_at(i, "get_rewards", {"inputs": inp}))
        return np.stack(results)

    def get_reachability_supervision(self, inputs):
        if hasattr(self.venv, "get_reachability_supervision"):
            return self.venv.get_reachability_supervision(inputs)
        maps, labels = [], []
        for i, inp in zip(range(self.num_envs), inputs):
            m, lab = self.venv.call_at(
                i, "get_reachability_supervision", {"inputs": inp})
            maps.append(m)
            labels.append(lab)
        return np.stack(maps), np.asarray(labels, dtype=np.float32)

    def close(self):
        return self.venv.close()
