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

        map_size = 22

        importlib.import_module(f"magent2.environments.{env_id}")
        self.env: Env = eval(f"magent2.environments.{env_id}").env(
            seed=seed, map_size=map_size, **kwargs
        )
        # change this to debug
        self.env.set_random_enemy(True)
        # self.env.enemy_dont_move = True
        self.n_agents = self.env.n_agents
        self._seed = seed
        print("seed:", self._seed)

        self.observation_space = spaces.Tuple(
            [
                transpose_space(self.env.agent_observation_space)
                for _ in range(self.n_agents)
            ]
        )
        self.action_space = spaces.Tuple(
            [self.env.agent_action_space for _ in range(self.n_agents)]
        )
        # self.share_observation_space = spaces.Tuple(
        #     [transpose_space(self.env.state_space) for _ in range(self.n_agents)]
        # )
        self.share_observation_space = self.observation_space
        self.cum_rw = None
        self.report_rw = [
            {
                "cumulative_rewards": np.nan,
                "attack_cnt": np.nan,
                "kill_cnt": np.nan,
                "attacked_cnt": np.nan,
                "dead_cnt": np.nan,
            }
            for _ in range(self.n_agents)
        ]

        self.attack_cnt = 0
        self.kill_count = 0
        self.attacked_cnt = 0
        self.dead_cnt = 0

    def seed(self, seed=None):
        if seed is not None:
            self._seed = seed
        self.env.seed(seed=seed)

    def reset(self):
        obses, _ = self.env.gym_reset(self._seed)
        state = self.env.state()
        assert len(state.shape) == 3
        assert len(obses.shape) == 4
        state = np.transpose(state, [2, 0, 1])
        obses = np.transpose(obses, [0, 3, 1, 2])

        if self.cum_rw is not None:
            for rp, cw, kil, atk, atked, dead in zip(
                self.report_rw,
                self.cum_rw,
                self.kill_count,
                self.attack_cnt,
                self.attacked_cnt,
                self.dead_cnt,
            ):
                rp["cumulative_rewards"] = cw
                rp["attack_cnt"] = atk
                rp["kill_cnt"] = kil
                rp["attacked_cnt"] = atked
                rp["dead_cnt"] = dead

        self.cum_rw = np.zeros(self.n_agents, dtype=np.float32)
        self.attack_cnt = np.zeros(self.n_agents, dtype=np.int32)
        self.attacked_cnt = np.zeros(self.n_agents, dtype=np.int32)
        self.kill_count = np.zeros(self.n_agents, dtype=np.int32)
        self.dead_cnt = np.zeros(self.n_agents, dtype=np.int32)
        # return obses, [state] * self.n_agents, None
        return obses, obses, None

    def step(self, actions):
        actions = actions.astype(np.int32)
        next_obses, rewards, dones, info = self.env.gym_step(actions)

        # test sum rw
        # rewards = np.reshape([np.sum(rewards)] * self.n_agents, rewards.shape)

        self.kill_count += np.sum(rewards == 5)
        self.attack_cnt += np.sum(rewards > 0.01)
        self.attacked_cnt += np.sum(rewards < -0.075)
        self.dead_cnt += np.sum(rewards < -1)

        next_obses = np.transpose(next_obses, [0, 3, 1, 2])
        state = self.env.state()
        assert len(state.shape) == 3
        state = np.transpose(state, [2, 0, 1])

        assert np.prod(self.cum_rw.shape) == np.prod(rewards.shape)
        self.cum_rw += rewards.reshape(self.cum_rw.shape)

        if len(dones.shape) == 2:
            dones = dones.squeeze(-1)

        infos = [copy.deepcopy(info) for _ in range(self.n_agents)]
        for rp, inf in zip(self.report_rw, infos):
            inf.update(rp)

        # return (
        #     next_obses,
        #     [state] * self.n_agents,  # this is copy by reference
        #     rewards,
        #     dones,
        #     infos,
        #     None,  # available actions
        # )
        return next_obses, next_obses, rewards, dones, infos, None

    def get_avail_actions(self):
        return None

    def close(self):
        self.env.close()
