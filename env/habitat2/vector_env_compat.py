"""将 habitat-lab 0.2.x VectorEnv 适配为 NSO 使用的 H0.1 接口。"""
from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np


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

    def _split_reset_result(self, result):
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def _split_step_result(self, result):
        if isinstance(result, tuple) and len(result) == 4:
            return result
        raise TypeError(f"unexpected step result: {type(result)}")

    def reset(self):
        raw = self.venv.reset()
        if not isinstance(raw, list):
            raw = [raw]
        obs_list, infos = [], []
        for item in raw:
            o, i = self._split_reset_result(item)
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
            o, r, d, i = self._split_step_result(item)
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
