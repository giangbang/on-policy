import numpy as np
from gymnasium import spaces
import magent2.environments
import copy
from magent2.environments.magent_env import magent_parallel_env as Env


def transpose_space(space):
    shape = space.shape
    assert len(shape) == 3, shape
    shape = (shape[2], shape[0], shape[1])
    low = space.low.reshape(shape)
    high = space.high.reshape(shape)
    transposed = spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    return transposed


class MAgentEnv:
    def __init__(self, env_id, seed, map_size: int = None, **kwargs):
        import importlib

        importlib.import_module(f"magent2.environments.{env_id}")
        self.env: Env = eval(f"magent2.environments.{env_id}").env(seed=seed, **kwargs)
        # change this to debug
        self.env.set_random_enemy(True)
        self.n_agents = self.env.n_agents

        self.observation_space = spaces.Tuple(
            [
                transpose_space(self.env.agent_observation_space)
                for _ in range(self.n_agents)
            ]
        )
        self.action_space = spaces.Tuple(
            [self.env.agent_action_space for _ in range(self.n_agents)]
        )
        self.share_observation_space = spaces.Tuple(
            [transpose_space(self.env.state_space) for _ in range(self.n_agents)]
        )
        self.cum_rw = None
        self.report_sum_rw = [
            {"cumulative_rewards": np.nan} for _ in range(self.n_agents)
        ]

    def seed(self, seed=None):
        self.env.seed(seed=seed)

    def reset(self):
        obses, _ = self.env.gym_reset()
        state = self.env.state()
        assert len(state.shape) == 3
        assert len(obses.shape) == 4
        state = np.transpose(state, [2, 0, 1])
        obses = np.transpose(obses, [0, 3, 1, 2])

        if self.cum_rw is not None:
            for rp, cw in zip(self.report_sum_rw, self.cum_rw):
                rp["cumulative_rewards"] = cw

        self.cum_rw = np.zeros(self.n_agents, dtype=np.float32)
        return obses, [state] * self.n_agents, None

    def step(self, actions):
        actions = actions.astype(np.int32)
        next_obses, rewards, dones, info = self.env.gym_step(actions)

        next_obses = np.transpose(next_obses, [0, 3, 1, 2])
        state = self.env.state()
        assert len(state.shape) == 3
        state = np.transpose(state, [2, 0, 1])

        assert np.prod(self.cum_rw.shape) == np.prod(rewards.shape)
        self.cum_rw += rewards.reshape(self.cum_rw.shape)

        if len(dones.shape) == 2:
            dones = dones.squeeze(-1)

        infos = [copy.deepcopy(info) for _ in range(self.n_agents)]
        for rp, inf in zip(self.report_sum_rw, infos):
            inf.update(rp)

        return (
            next_obses,
            [state] * self.n_agents,  # this is copy by reference
            rewards,
            dones,
            infos,
            None,  # available actions
        )

    def get_avail_actions(self):
        return None

    def close(self):
        self.env.close()
