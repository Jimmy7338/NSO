import torch


def make_vec_envs(args):
    version = int(getattr(args, "habitat_version", 0) or 0)
    if version == 0:
        import os
        version = int(os.environ.get("NSO_HABITAT_VERSION", "1"))
    if version == 2:
        from .habitat2 import construct_envs
    else:
        from .habitat import construct_envs
    envs = construct_envs(args)
    envs = VecPyTorch(envs, args.device)
    return envs


# Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch():

    def __init__(self, venv, device):
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
        self.device = device

    def reset(self):
        obs, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        actions = actions.cpu().numpy()
        obs, reward, done, info = self.venv.step(actions)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def get_rewards(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float()
        return reward

    def get_short_term_goal(self, inputs):
        stg = self.venv.get_short_term_goal(inputs)
        stg = torch.from_numpy(stg).float()
        return stg

    def get_reachability_supervision(self, inputs):
        if not hasattr(self.venv, "get_reachability_supervision"):
            raise NotImplementedError("当前 VectorEnv 未实现 get_reachability_supervision")
        return self.venv.get_reachability_supervision(inputs)

    def close(self):
        return self.venv.close()
