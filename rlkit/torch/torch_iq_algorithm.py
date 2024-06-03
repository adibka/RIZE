import abc
from collections import OrderedDict

from typing import Iterable
from torch import nn as nn

# from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
# from rlkit.core.online_rl_algorithm import OnlineRLAlgorithm
# from rlkit.core.vec_online_rl_algorithm import VecOnlineRLAlgorithm
from rlkit.core.vec_online_iq_algorithm import VecOnlineIQAlgorithm
from rlkit.core.trainer import IQTrainer


# class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):

#     def to(self, device):
#         for net in self.trainer.networks:
#             net.to(device)

#     def training_mode(self, mode):
#         for net in self.trainer.networks:
#             net.train(mode)


# class TorchBatchRLAlgorithm(BatchRLAlgorithm):

#     def to(self, device):
#         for net in self.trainer.networks:
#             net.to(device)

#     def training_mode(self, mode):
#         for net in self.trainer.networks:
#             net.train(mode)


# class TorchVecOnlineRLAlgorithm(VecOnlineRLAlgorithm):

#     def to(self, device):
#         for net in self.trainer.networks:
#             net.to(device)

#     def training_mode(self, mode):
#         for net in self.trainer.networks:
#             net.train(mode)

class TorchVecOnlineIQAlgorithm(VecOnlineIQAlgorithm):

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchIQTrainer(IQTrainer, metaclass=abc.ABCMeta):

    def __init__(self):
        self._num_train_steps = 0

    def train(self, policy_batch, expert_batch):
        self._num_train_steps += 1
        self.train_from_torch(policy_batch, expert_batch)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass
