import torch.nn as nn
from .util import init
import torch

"""CNN Modules and utils."""


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    def __init__(
        self, obs_shape, hidden_size, use_orthogonal, use_ReLU, kernel_size=3, stride=1
    ):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        if input_height < 15 and input_width < 15:
            self.cnn = nn.Sequential(
                nn.Conv2d(input_channel, input_channel, 3),
                nn.ReLU(),
                nn.Conv2d(input_channel, input_channel, 3),
                nn.ReLU(),
            )
        else:
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
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, hidden_size),
        )

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return self.network(x)


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        self.cnn = CNNLayer(
            obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU
        )

    def forward(self, x):
        x = self.cnn(x)
        return x
