import torch
import numpy as np
import torch.nn as nn
from onpolicy.algorithms.utils.util import check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.utils.util import get_shape_from_obs_space

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: (
    autograd.Variable(*args, **kwargs).cuda()
    if USE_CUDA
    else autograd.Variable(*args, **kwargs)
)


class CNNLayer(nn.Module):
    def __init__(self, obs_shape):
        super(CNNLayer, self).__init__()

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, input_channel * 2, 2, 2),
            nn.ReLU(),
            nn.Conv2d(input_channel * 2, input_channel * 2, 2, 2),
            nn.ReLU(),
            nn.Conv2d(input_channel * 2, input_channel * 2, 2, 2),
            nn.ReLU(),
        )

        dummy_input = torch.randn(obs_shape)
        dummy_output = self.cnn(dummy_input)
        self.flatten_dim = dummy_output.view(-1).shape[0]

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return x


class Encoder(nn.Module):
    def __init__(self, observation_shape=32, hidden_dim=128):
        super(Encoder, self).__init__()
        # print("observation_shape", observation_shape)

        self.img_obs = len(observation_shape) >= 3
        if self.img_obs:
            self.cnn = CNNLayer(observation_shape)

        din = self.cnn.flatten_dim if self.img_obs else observation_shape[0]
        self.fc = nn.Sequential(
            nn.Linear(din, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        if self.img_obs:
            x = self.cnn(x)
        embedding = self.fc(x)
        return embedding


class AttModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self, x, mask):

        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        if mask is not None:
            h = torch.clamp(torch.mul(torch.bmm(q, k), mask), 0, 9e13) - 9e15 * (
                1 - mask
            )
        else:
            h = torch.clamp(torch.bmm(q, k), 0, 9e13)
        att = F.softmax(h, dim=2)

        out = torch.bmm(att, v)
        # out = torch.add(out,v)
        # out = F.relu(self.fcout(out))
        return out


class Q_Net(nn.Module):
    def __init__(self, hidden_dim, dout):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self, x):
        q = self.fc(x)
        return q


class DGN(nn.Module):
    def __init__(self, n_agent, observation_space, hidden_dim, num_actions):
        super(DGN, self).__init__()

        self.encoder = Encoder(observation_space.shape, hidden_dim)
        self.att_1 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.att_2 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.q_net = Q_Net(hidden_dim * 2, num_actions)

    def forward(self, x, mask):
        h1 = self.encoder(x)
        h2 = self.att_1(h1, mask)
        # h3 = self.att_2(h2, mask)

        h = torch.cat([h1, h2], dim=-1)

        q = self.q_net(h)
        return q


class RDGN_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(
        self,
        args,
        obs_space,
        num_agents,
        action_space,
        agent_id: int,
        device=torch.device("cpu"),
    ):
        super(RDGN_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_agents = num_agents
        self.agent_id = agent_id
        assert isinstance(self.agent_id, int), type(self.agent_id)

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        self.att_1 = AttModel(
            num_agents, args.hidden_size, args.hidden_size, args.hidden_size
        )
        self.att_2 = AttModel(
            num_agents, args.hidden_size, args.hidden_size, args.hidden_size
        )
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        self.act = ACTLayer(
            action_space, self.hidden_size * 2, self._use_orthogonal, self._gain, args
        )

        self.to(device)
        self.algo = args.algorithm_name

    def forward(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
        adj_matrix=None,
    ):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Convolutional graph RL
        actor_features_2 = self.att_1(actor_features, adj_matrix)
        actor_features = torch.cat([actor_features, actor_features_2], dim=-1)

        actor_features = actor_features[:, self.agent_id]

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        adj_matrix=None,
    ):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Convolutional graph RL
        actor_features_2 = self.att_1(actor_features, adj_matrix)
        actor_features = torch.cat([actor_features, actor_features_2], dim=-1)

        actor_features = actor_features[:, self.agent_id]

        if self.algo == "hatrpo":
            action_log_probs, dist_entropy, action_mu, action_std, all_probs = (
                self.act.evaluate_actions_trpo(
                    actor_features,
                    action,
                    available_actions,
                    active_masks=(
                        active_masks if self._use_policy_active_masks else None
                    ),
                )
            )

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=active_masks if self._use_policy_active_masks else None,
            )

        return action_log_probs, dist_entropy
