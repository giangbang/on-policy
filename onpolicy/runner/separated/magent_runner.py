import time
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class MAgentRunner(Runner):
    def __init__(self, config):
        super(MAgentRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        for episode in range(1, episodes + 1):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = (
                    self.collect(step)
                )

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, avail_actions = self.envs.step(
                    actions
                )

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    avail_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.env_id,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                if self.env_name == "MAgent2":
                    idv_rews = []
                    for agent_id in range(self.num_agents):
                        rews = []
                        for info in infos:
                            if "cumulative_rewards" in info[agent_id].keys():
                                rews.append(info[agent_id]["cumulative_rewards"])
                        idv_rews.append(np.mean(rews))
                        train_infos[agent_id].update(
                            {"average_episode_rewards": idv_rews[agent_id]}
                        )
                        print(
                            "average episode rewards of agent{} is {}".format(
                                agent_id,
                                train_infos[agent_id]["average_episode_rewards"],
                            )
                        )

                    avg_rw_all_agents = np.mean(idv_rews)
                    print("Avg rewards all agents:", avg_rw_all_agents)
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
                self.buffer[agent_id].available_actions[step],
            )
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            avail_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        # rewards.shape is n_thread x n_agent x 1
        rewards = np.tile(rewards, (1, self.num_agents, 1)).reshape(
            rewards.shape[0], self.num_agents, self.num_agents
        )

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                *self.buffer[0].rnn_states_critic.shape[2:],
            ),
            dtype=np.float32,
        )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        active_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        bad_masks = np.array(
            [
                [
                    [0.0] if info[agent_id]["bad_transition"] else [1.0]
                    for agent_id in range(self.num_agents)
                ]
                for info in infos
            ]
        )

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):

            self.buffer[agent_id].insert(
                share_obs[:, agent_id],
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
                bad_masks[:, agent_id],
                active_masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=self.all_args.deterministic_eval,
                )

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if (
                    self.eval_envs.action_space[agent_id].__class__.__name__
                    == "MultiDiscrete"
                ):
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(
                            self.eval_envs.action_space[agent_id].high[i] + 1
                        )[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate(
                                (eval_action_env, eval_uc_action_env), axis=1
                            )
                elif (
                    self.eval_envs.action_space[agent_id].__class__.__name__
                    == "Discrete"
                ):
                    eval_action_env = np.squeeze(
                        np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1
                    )
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions_env
            )
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        eval_means = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(
                np.sum(eval_episode_rewards[:, :, agent_id], axis=0)
            )
            eval_train_infos.append(
                {"eval_average_episode_rewards": eval_average_episode_rewards}
            )
            print(
                "eval average episode rewards of agent%i: " % agent_id
                + str(eval_average_episode_rewards)
            )
            self.writter.add_scalar(
                f"eval_episode_rewards_agent{agent_id}",
                eval_average_episode_rewards,
                total_num_steps,
            )
            eval_means.append(eval_average_episode_rewards)
        eval_means = np.mean(eval_means)
        self.writter.add_scalar("eval_episode_rewards", eval_means, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render("rgb_array")[0][0]
                all_frames.append(image)

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(list(obs[:, agent_id])),
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True,
                    )

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if (
                        self.envs.action_space[agent_id].__class__.__name__
                        == "MultiDiscrete"
                    ):
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(
                                self.envs.action_space[agent_id].high[i] + 1
                            )[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate(
                                    (action_env, uc_action_env), axis=1
                                )
                    elif (
                        self.envs.action_space[agent_id].__class__.__name__
                        == "Discrete"
                    ):
                        action_env = np.squeeze(
                            np.eye(self.envs.action_space[agent_id].n)[action], 1
                        )
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

                if self.all_args.save_gifs:
                    image = self.envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(
                    np.sum(episode_rewards[:, :, agent_id], axis=0)
                )
                print(
                    "eval average episode rewards of agent%i: " % agent_id
                    + str(average_episode_rewards)
                )

        if self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/render.gif",
                all_frames,
                duration=self.all_args.ifi,
            )
