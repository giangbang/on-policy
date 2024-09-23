from utils.separated_buffer import SeparatedReplayBuffer
from utils.shared_buffer import SharedReplayBuffer
import numpy as np
import lingam

from typing import List, Union


def find_causal_w(buffers: Union[List[SeparatedReplayBuffer], SharedReplayBuffer]):
    if isinstance(buffers, list) and isinstance(buffers[0], SeparatedReplayBuffer):
        returns = []
        actions = []
        for i, b in enumerate(buffers):
            returns.append(b.returns[..., i][..., None])
            actions.append(b.actions)
        X = np.stack([actions, returns], axis=-1)
        X = np.reshape(-1, X.shape[-1])
        returns_dim = actions_dim = len(buffers)

    elif isinstance(buffers, SharedReplayBuffer):
        actions = buffers.actions
        returns = buffers.returns[
            :, :, np.arange(buffers.num_agents), np.arange(buffers.num_agents)
        ]
        returns = np.reshape(returns, actions.shape)
        X = np.stack([actions, returns], axis=-1)
        returns_dim = actions_dim = buffers.num_agents
    else:
        raise NotImplementedError("Unknown input.")

    prior_knowledge = -1 * np.ones((X.shape[-1], X.shape[-1]), dtype=int)

    prior_knowledge[:actions_dim, :actions_dim] = 0
    prior_knowledge[actions_dim:, actions_dim:] = 0

    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(X)
    return model.adjacency_matrix_[actions_dim:][:actions_dim]
