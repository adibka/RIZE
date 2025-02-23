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


class Critic(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            embedding_size=64,
            num_quantiles=32,
            layer_norm=True,
            n_nets=5,
            **kwargs,
    ):
        super().__init__()
        self.nets = []
        self.n_nets = n_nets
        for i in range(n_nets):
            net = QuantileMlp(
                hidden_sizes,
                output_size,
                input_size,
                embedding_size,
                num_quantiles,
                layer_norm,
                **kwargs,
            )
            self.add_module(f'zf{i}', net)
            self.nets.append(net)

    def forward(self, state, action, tau):
        quantiles = torch.stack(tuple(net(state, action, tau) for net in self.nets), dim=1)
        return quantiles
