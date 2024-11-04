from typing import Union
import numpy as np
from torch.autograd import Variable
import torch
import math
import torch.nn as nn
from onpolicy.algorithms.r_mappo_mgda.algorithm.rDGN_MAPPOPolicy import RDGN_MAPPOPolicy
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

from onpolicy.algorithms.r_mappo_mgda.algorithm.rMAPPOPolicy import R_MAPPOPolicy


class R_MAPPO_MultHead:
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(
        self,
        args,
        policy: Union[R_MAPPOPolicy, RDGN_MAPPOPolicy],
        device=torch.device("cpu"),
        agent_id: int = None,
    ):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.args = args
        self.agent_id = int(agent_id) if agent_id is not None else None

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert (
            self._use_popart and self._use_valuenorm
        ) == False, "self._use_popart and self._use_valuenorm can not be set True simultaneously"

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(self.policy.num_agents, device=device)
        else:
            self.value_normalizer = None

    def cal_value_loss(
        self,
        a: Union[torch.Tensor, int],
        values: torch.Tensor,
        value_preds_batch: torch.Tensor,
        return_batch: torch.Tensor,
        active_masks_batch: torch.Tensor,
    ):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        assert (
            values.shape == value_preds_batch.shape
        ), f"{values.shape}, {value_preds_batch.shape}"
        assert (
            return_batch.shape == values.shape
        ), f"{return_batch.shape} {values.shape}"

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = (
                self.value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if isinstance(a, torch.Tensor) or isinstance(a, np.ndarray):
            a = a.squeeze()
            error_clipped = error_clipped[torch.arange(len(a), device=self.device), a]
            error_original = error_original[torch.arange(len(a), device=self.device), a]
        elif isinstance(a, int):
            error_clipped = error_clipped[..., a]
            error_original = error_original[..., a]
        else:
            raise TypeError("what")

        error_clipped = error_clipped.unsqueeze(-1)
        error_original = error_original.unsqueeze(-1)

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            assert (
                value_loss.shape == active_masks_batch.shape
            ), f"{value_loss.shape} {active_masks_batch.shape}"
            assert len(value_loss.shape) == 2
            assert value_loss.shape[-1] == 1
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_loss(self, a, imp_weights, adv_targ, active_masks_batch, dist_entropy):
        # assert imp_weights.shape == adv_targ[..., a].shape, "{} {}".format(imp_weights.shape, adv_targ[..., a].shape)
        # print(adv_targ.shape) # torch.Size([10000, 1]) mappo, torch.Size([10000, 2]) rmappo

        if self.agent_id is not None:
            assert isinstance(a, int)
            adv_targ = adv_targ[..., a].unsqueeze(-1)
        else:
            assert adv_targ.shape[0] == a.shape[0]
            assert len(adv_targ.shape) == 2 and len(a.shape) == 2
            a = torch.from_numpy(a).to(self.device)
            adv_targ = torch.gather(adv_targ, -1, a)
        assert (
            adv_targ.shape == imp_weights.shape
        ), f"{adv_targ.shape} {imp_weights.shape}"
        assert imp_weights.shape == adv_targ.shape
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss - dist_entropy * self.entropy_coef

        if self._use_max_grad_norm:
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)

        return policy_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            agent_id,
        ) = sample
        assert agent_id is not None or self.agent_id is not None

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        n_agents = values.shape[-1]
        # actor update
        assert action_log_probs.shape == old_action_log_probs_batch.shape
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        self.policy.actor_optimizer.zero_grad()

        # surr1 = imp_weights * adv_targ[..., a]
        # surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        # if self._use_policy_active_masks:
        #     policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
        #                                     dim=-1,
        #                                     keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        # else:
        #     policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # policy_loss = policy_action_loss

        # self.policy.actor_optimizer.zero_grad()

        # if update_actor:
        #     (policy_loss - dist_entropy * self.entropy_coef).backward(retain_graph=True)

        # if self._use_max_grad_norm:
        #     actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)

        # Only optimize the agent id

        if self.agent_id is not None:
            policy_loss = self.ppo_loss(
                self.agent_id, imp_weights, adv_targ, active_masks_batch, dist_entropy
            )
        else:
            policy_loss = self.ppo_loss(
                agent_id, imp_weights, adv_targ, active_masks_batch, dist_entropy
            )
        policy_loss.backward()

        # self.policy.actor_optimizer.zero_grad()
        import math

        actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        # if math.isnan(actor_grad_norm):
        # print("nan, loss", policy_loss)
        # print(agent_id, imp_weights, adv_targ, active_masks_batch, dist_entropy)
        self.policy.actor_optimizer.step()

        # critic update
        agent_id = agent_id if agent_id is not None else self.agent_id
        value_loss = self.cal_value_loss(
            agent_id, values, value_preds_batch, return_batch, active_masks_batch
        )

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
        )

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1]
            )
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()

        active_masks = (buffer.active_masks[:-1] == 0.0).repeat(
            advantages_copy.shape[-1], axis=-1
        )

        advantages_copy[active_masks] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info["value_loss"] = 0
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["critic_grad_norm"] = 0
        train_info["ratio"] = 0

        if math.isnan(mean_advantages) and math.isnan(std_advantages):
            # the agent is dead the whole episode
            # this can happen, for example when the number of Monte Carlo step
            # is too short compared to the actual episode length
            # Concretely, if the actual episode has length of 1k,
            # the agent dies at step 100, and the mappo sample step between
            # each update is 10, then after the agent die, in the next 10 steps
            # it still dead because the actual episode is yet to reset.
            # in such case we just ignore these agents

            return train_info

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = buffer.feed_forward_generator(
                    advantages, self.num_mini_batch
                )

            for sample in data_generator:

                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                ) = self.ppo_update(sample, update_actor)

                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["critic_grad_norm"] += critic_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
