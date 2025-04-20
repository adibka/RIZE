"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np

def softmax(x):
    return F.softmax(x, dim=-1)


class QuantileMlp(nn.Module):

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            embedding_size=64,
            num_quantiles=32,
            dropout=False,
            drop_rate=0.0,
            layer_norm=True,
            **kwargs,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        # hidden_sizes[:-2] MLP base
        # hidden_sizes[-2] before merge
        # hidden_sizes[-1] before output
        if dropout:
            assert drop_rate > 0, "Dropout rate should be set!"

        self.base_fc = []
        last_size = input_size
        for next_size in hidden_sizes[:-1]:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.Dropout(drop_rate) if dropout else nn.Identity(),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)
        self.num_quantiles = num_quantiles
        self.embedding_size = embedding_size
        self.tau_fc = nn.Sequential(
            nn.Linear(embedding_size, last_size),
            nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
            nn.Sigmoid(),
        )
        self.merge_fc = nn.Sequential(
            nn.Linear(last_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)
        self.const_vec = ptu.from_numpy(np.arange(1, 1 + self.embedding_size))

    def forward(self, state, action, tau):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """
        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)  # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * self.const_vec * np.pi)  # (N, T, E)
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        h = self.merge_fc(h)  # (N, T, C)
        output = self.last_fc(h).squeeze(-1)  # (N, T)
        return output

    @torch.no_grad()
    def get_values(self, state, action, tau):
        return eval_np(self, state, action, tau)


class SingleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super(SingleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Q architecture
        self.Q = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action, xyz):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q = self.Q(obs_action)
        return q

    @torch.no_grad()
    def get_values(self, state, action, xyz):
        return eval_np(self, state, action, xyz)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def orthogonal_init_(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


# class Critic(nn.Module):
#     def __init__(
#             self,
#             hidden_sizes,
#             output_size,
#             input_size,
#             embedding_size=64,
#             num_quantiles=32,
#             layer_norm=True,
#             n_nets=5,
#             **kwargs,
#     ):
#         super().__init__()
#         self.nets = []
#         self.n_nets = n_nets
#         for i in range(n_nets):
#             net = QuantileMlp(
#                 hidden_sizes,
#                 output_size,
#                 input_size,
#                 embedding_size,
#                 num_quantiles,
#                 layer_norm,
#                 **kwargs,
#             )
#             self.add_module(f'zf{i}', net)
#             self.nets.append(net)

#     def forward(self, state, action, tau):
#         quantiles = torch.stack(tuple(net(state, action, tau) for net in self.nets), dim=1)
#         return quantiles
            
# class DoubleQCritic(nn.Module):
#     def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
#         super(DoubleQCritic, self).__init__()
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
        
#         # Q1 architecture
#         self.Q1 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

#         # Q2 architecture
#         self.Q2 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

#         self.apply(orthogonal_init_)

#     def forward(self, obs, action, both=False):
#         assert obs.size(0) == action.size(0)

#         obs_action = torch.cat([obs, action], dim=-1)
#         q1 = self.Q1(obs_action)
#         q2 = self.Q2(obs_action)

#         if both:
#             return q1, q2
#         else:
#             return torch.min(q1, q2)