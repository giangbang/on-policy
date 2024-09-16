from gym import spaces


class MAgentEnv:
    def __init__(self, env_id, seed, map_size: int = None, **kwargs):
        import importlib

        importlib.import_module(f"magent2.environments.{env_id}")
        self.env = eval(f"magent2.environments.{env_id}").env(
            seed=seed, map_size=map_size, **kwargs
        )
        self.n_agents = self.env.n_agents
        self.observation_space = self.env.agent_action_space
        self.action_space = self.env.agent_action_space
        self.share_observation_space = spaces.Tuple(
            [self.env.state_space for _ in range(self.n_agents)]
        )

    def reset(self):
        obses, _ = self.env.gym_reset()
        state = self.env.state()
        return obses, [state] * self.n_agents, None

    def step(self, actions):
        next_obses, rewards, dones, info = self.env.gym_step(actions)
        state = self.env.state()

        return (
            next_obses,
            [state] * self.n_agents,
            rewards,
            dones,
            [info] * self.n_agents,
            None,  # available actions
        )

    def get_avail_actions(self):
        return None

    def close(self):
        self.env.close()
