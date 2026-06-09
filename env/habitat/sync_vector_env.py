# 单进程主线程 VectorEnv：matplotlib Tk 可视化必须在主线程创建窗口
from typing import Any, Callable, List, Sequence, Tuple, Union

import numpy as np

from habitat.core.env import Env, RLEnv


class SyncVectorEnv:
    """与 habitat VectorEnv 接口兼容，环境在主线程同步运行（供 -v 1 可视化）。"""

    def __init__(
        self,
        make_env_fn: Callable[..., Union[Env, RLEnv]],
        env_fn_args: Sequence[Tuple],
        auto_reset_done: bool = True,
    ) -> None:
        del auto_reset_done  # 与 VectorEnv 一致在 done 时 reset
        self._auto_reset_done = True
        self._envs = [make_env_fn(*args) for args in env_fn_args]
        self._num_envs = len(self._envs)
        self.envs = self._envs
        self.observation_spaces = [e.observation_space for e in self._envs]
        self.action_spaces = [e.action_space for e in self._envs]
        self.observation_space = self.observation_spaces[0]
        self.action_space = self.action_spaces[0]
        self._is_closed = False
        self._is_waiting = False
        self._paused = []

    @property
    def num_envs(self):
        return self._num_envs - len(self._paused)

    def reset(self):
        obs_list = []
        infos = []
        for env in self._envs:
            obs, info = env.reset()
            obs_list.append(obs)
            infos.append(info)
        return np.stack(obs_list), tuple(infos)

    def step_async(self, actions: List[int]) -> None:
        self._pending_actions = actions
        self._is_waiting = True

    def step_wait(self):
        results = []
        for env, action in zip(self._envs, self._pending_actions):
            results.append(env.step(action))
        self._is_waiting = False
        obs, rews, dones, infos = zip(*results)
        if self._auto_reset_done:
            obs_list = list(obs)
            infos_list = list(infos)
            for i, done in enumerate(dones):
                if done:
                    old_info = infos_list[i]
                    obs_list[i], infos_list[i] = self._envs[i].reset()
                    if isinstance(old_info, dict) and 'exp_reward' in old_info:
                        infos_list[i]['exp_reward'] = old_info['exp_reward']
                        infos_list[i]['exp_ratio'] = old_info['exp_ratio']
            obs = np.stack(obs_list)
            infos = tuple(infos_list)
        else:
            obs = np.stack(obs)
        return obs, np.stack(rews), np.stack(dones), infos

    def step(self, actions: List[int]):
        self.step_async(actions)
        return self.step_wait()

    def get_short_term_goal(self, inputs):
        results = [env.get_short_term_goal(inp) for env, inp in zip(self._envs, inputs)]
        return np.stack(results)

    def get_reachability_supervision(self, inputs):
        maps, labels = [], []
        for env, inp in zip(self._envs, inputs):
            m, lab = env.get_reachability_supervision(inp)
            maps.append(m)
            labels.append(lab)
        return np.stack(maps), np.asarray(labels, dtype=np.float32)

    def close(self) -> None:
        if getattr(self, "_is_closed", True):
            return
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()
        self._is_closed = True

    def __del__(self):
        if hasattr(self, "_is_closed"):
            self.close()
