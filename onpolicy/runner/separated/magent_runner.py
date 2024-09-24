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
        print(f"Number of agents: {self.num_agents}.")
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

                # print(set(rewards.flatten().tolist()))

                data = (
                    obs,
                    share_obs,
                    rewards,  # n_threads x n_agent x 1
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
                    sum_rews = []
                    for agent_id in range(self.num_agents):
                        rews = []  # cumulative rewards of agents id on all threads
                        for info in infos:  # iterate over n threads
                            if "cumulative_rewards" in info[agent_id].keys():
                                rews.append(info[agent_id]["cumulative_rewards"])

                        idv_rews.append(np.nanmean(rews))  # mean rewards over threads
                        sum_rews.append(np.nanmean(rews))

                        train_infos[agent_id].update(
                            {"average_episode_rewards": idv_rews[agent_id]}
                        )
                        print(
                            "average episode rewards of agent{} is {}".format(
                                agent_id,
                                train_infos[agent_id]["average_episode_rewards"],
                            )
                        )

                    avg_rw_all_agents = np.nanmean(idv_rews)  # mean rewards over agents
                    print("Avg rewards all agents:", avg_rw_all_agents)

                    avg_sum_rw_all_agents = np.nansum(
                        sum_rews
                    )  # sum rewards over agents
                    print("Sum rewards all agents:", avg_sum_rw_all_agents)

                    self.writter.add_scalar(
                        "average_episode_rewards", avg_rw_all_agents, total_num_steps
                    )
                    self.writter.add_scalar(
                        "sum_episode_rewards", avg_sum_rw_all_agents, total_num_steps
                    )

                    # other loggings for mostly debuging purposes
                    def avg_stats(infos, stat_key):
                        alls = []
                        agent_id = 0
                        for info in infos:
                            alls.append(info[agent_id][stat_key])
                        return np.nanmean(alls)

                    print(
                        "avg kill count (of all agents)", avg_stats(infos, "kill_cnt")
                    )
                    print(
                        "avg correct attack count (of all agents)",
                        avg_stats(infos, "attack_cnt"),
                    )
                    print(
                        "avg miss attack count (of all agents)",
                        avg_stats(infos, "attacked_cnt"),
                    )
                    print(
                        "avg dead count (of all agents)", avg_stats(infos, "dead_cnt")
                    )

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

            # video logging
            if episode % (self.eval_interval * 5) == 0:
                self.render(total_num_steps=total_num_steps)

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

        # each agent simultaneously predict rewards of all other agents
        # but sometime the share_obs of one agent is not enough to correctly regress values
        # of other agents, so here we take the predicted values as the prediction of the value owners
        values = values[:, np.arange(self.num_agents), np.arange(self.num_agents)]
        values = np.reshape(values, self.buffer[0].value_preds.shape[1:])

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
                values,
                rewards[:, agent_id],
                masks[:, agent_id],
                bad_masks[:, agent_id],
                active_masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

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

        while True:
            eval_actions_collector = []
            eval_rnn_states_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = self.trainer[agent_id].policy.act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=False,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            # Obser reward and next obs
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.all_args.n_eval_rollout_threads, self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(
                        np.sum(one_episode_rewards[eval_i], axis=0)
                    )
                    one_episode_rewards[eval_i] = []

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_env_infos = {"eval_average_episode_rewards": eval_episode_rewards}
                self.log_env(eval_env_infos, total_num_steps)

                break

    @torch.no_grad()
    def render(self, total_num_steps):
        assert self.vis_envs is not None
        render_battles_won = 0
        render_episode = 0

        render_episode_rewards = []
        one_episode_rewards = []
        for render_i in range(1):
            one_episode_rewards.append([])
            render_episode_rewards.append([])

        render_obs, render_share_obs, _ = self.vis_envs.reset()

        frames = [self.vis_envs.env.render()]

        render_obs = render_obs[None, ...]

        render_rnn_states = np.zeros(
            (
                1,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        render_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

        while True:
            render_actions_collector = []
            render_rnn_states_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                render_actions, temp_rnn_state = self.trainer[agent_id].policy.act(
                    render_obs[:, agent_id],
                    render_rnn_states[:, agent_id],
                    render_masks[:, agent_id],
                    deterministic=False,
                )
                render_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                render_actions_collector.append(_t2n(render_actions))

            render_actions = np.array(render_actions_collector).transpose(1, 0, 2)

            # Obser reward and next obs
            (
                render_obs,
                render_share_obs,
                render_rewards,
                render_dones,
                render_infos,
                render_available_actions,
            ) = self.vis_envs.step(render_actions[0])
            render_obs = render_obs[None, ...]
            render_rewards = render_rewards[None, ...]
            render_dones = render_dones[None, ...]

            frames.append(self.vis_envs.env.render())

            for render_i in range(1):
                one_episode_rewards[render_i].append(render_rewards[render_i])

            render_dones_env = np.all(render_dones, axis=1)

            render_rnn_states[render_dones_env == True] = np.zeros(
                (
                    (render_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )

            render_masks = np.ones(
                (1, self.num_agents, 1),
                dtype=np.float32,
            )
            render_masks[render_dones_env == True] = np.zeros(
                ((render_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for render_i in range(1):
                if render_dones_env[render_i]:
                    render_episode += 1
                    render_episode_rewards[render_i].append(
                        np.sum(one_episode_rewards[render_i], axis=0)
                    )
                    one_episode_rewards[render_i] = []

            if render_episode >= 1:
                render_episode_rewards = np.concatenate(render_episode_rewards)
                break

        import cv2

        height, width, _ = frames[0].shape
        fps = 20
        vid_dir = self.run_dir

        out = cv2.VideoWriter(
            os.path.join(vid_dir, f"record_at_step_{total_num_steps}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()

        print("Done recording video")
