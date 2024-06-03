import gtimer as gt
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_iq_algorithm import TorchIQTrainer
from rlkit.torch.idsac.iq_loss import iq_distr_loss

from .risk import distortion_de
from .utils import LinearSchedule


class TruncIDSACTrainer(TorchIQTrainer):

    def __init__(
            self,
            args,
            env,
            policy,
            zf,
            target_zf,
            fp=None,
            target_fp=None,
            discount=0.99,
            reward_scale=1.0,
            alpha=1.0,
            policy_lr=3e-4,
            zf_lr=3e-4,
            tau_type='iqn',
            fp_lr=1e-5,
            num_quantiles=32,
            risk_type='neutral',
            risk_param=0.,
            risk_param_final=None,
            risk_schedule_timesteps=1,
            optimizer_class=optim.Adam,
            soft_target_tau=5e-3,
            target_update_period=1,
            clip_norm=0.,
            use_automatic_entropy_tuning=False,
            target_entropy=None,
    ):
        super().__init__()
        self.args = args
        self.env = env
        self.device = args['device']
        self.policy = policy
        self.zf = zf
        self.target_zf = target_zf

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.tau_type = tau_type
        self.num_quantiles = num_quantiles

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        else:
            self.alpha = alpha

        self.zf_criterion = trunc_quantile_regression_loss
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.zf_optimizer = optimizer_class(
            self.zf.parameters(),
            lr=zf_lr,
        )
        
        self.fp = fp
        self.target_fp = target_fp
        if self.tau_type == 'fqf':
            self.fp_optimizer = optimizer_class(
                self.fp.parameters(),
                lr=fp_lr,
            )

        self.discount = discount
        self.reward_scale = reward_scale
        self.clip_norm = clip_norm

        self.risk_type = risk_type
        self.risk_schedule = LinearSchedule(risk_schedule_timesteps, risk_param,
                                            risk_param if risk_param_final is None else risk_param_final)
        # TQC 
        self.quantiles_to_drop = args['n_nets'] * args['quantiles_to_drop_per_net']
        self.total_quantiles = args['n_nets'] * num_quantiles

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def get_tau(self, obs, actions, fp=None):
        if self.tau_type == 'fix':
            presum_tau = ptu.zeros(len(actions), self.num_quantiles) + 1. / self.num_quantiles
        elif self.tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            presum_tau = ptu.rand(len(actions), self.num_quantiles) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
        elif self.tau_type == 'fqf':
            if fp is None:
                fp = self.fp
            presum_tau = fp(obs, actions)
        tau = torch.cumsum(presum_tau, dim=1)  # (B, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau

    def getZ(self, obs, actions, tau_hat):
        z_pred = self.zf(obs, actions, tau_hat)  # (B, N, T)
        return z_pred

    def get_targetZ(self, next_obs, terminals, alpha, next_tau_hat):
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs,
            reparameterize=True,
            return_log_prob=True,
        )
        target_z = self.target_zf(next_obs, new_next_actions, next_tau_hat)
        sorted_target_z, _ = torch.sort(target_z.reshape(next_obs.shape[0], -1))
        trunc_target_z = sorted_target_z[:, self.quantiles_to_drop:]
        target_z_values = (1. - terminals) * self.discount * (trunc_target_z - alpha * new_log_pi)  # (B, N * T)
        return target_z_values

    def train_from_torch(self, policy_batch, expert_batch):
        # Policy batch samples
        policy_rewards = policy_batch['rewards']
        policy_terminals = policy_batch['terminals']
        policy_obs = policy_batch['observations']
        policy_actions = policy_batch['actions']
        policy_next_obs = policy_batch['next_observations']
        
        # Expert batch samples
        expert_rewards = expert_batch['rewards']
        expert_terminals = expert_batch['terminals']
        expert_obs = expert_batch['observations']
        expert_actions = expert_batch['actions']
        expert_next_obs = expert_batch['next_observations']
        gt.stamp('preback_start', unique=False)
        """
        Update Alpha
        """
        new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            policy_obs,
            reparameterize=True,
            return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha
        gt.stamp('preback_alpha', unique=False)
        """
        Update ZF
        """
        # TODO: when using FQF, these setting won't work! obs and actions should be handled properly!
        with torch.no_grad():
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(policy_obs, policy_actions, fp=self.target_fp)
            policy_z_target = self.get_targetZ(policy_next_obs, policy_terminals, alpha, next_tau_hat)
            expert_z_target = self.get_targetZ(expert_next_obs, expert_terminals, alpha, next_tau_hat)

        tau, tau_hat, presum_tau = self.get_tau(policy_obs, policy_actions, fp=self.fp)
        policy_z_pred = self.getZ(policy_obs, policy_actions, tau_hat)
        expert_z_pred = self.getZ(expert_obs, expert_actions, tau_hat)

        zf_loss, zf_loss_dict = self.zf_criterion(
            expert_z_pred, 
            expert_z_target,
            policy_z_pred,
            policy_z_target,
            log_pi,
            alpha,
            next_tau_hat,
            presum_tau,
            self.args['iq_kwargs'],
        )
        gt.stamp('preback_zf', unique=False)

        self.zf_optimizer.zero_grad()
        zf_loss.backward(retain_graph=True)
        self.zf_optimizer.step()
        gt.stamp('backward_zf', unique=False)
        """
        Update Policy
        """
        risk_param = self.risk_schedule(self._n_train_steps_total)

        if self.risk_type == 'VaR':
            tau_ = ptu.ones_like(policy_rewards) * risk_param
            q_new_actions = self.zf(policy_obs, new_actions, tau_).mean(dim=1)    # (B, T)
        else:
            with torch.no_grad():
                new_tau, new_tau_hat, new_presum_tau = self.get_tau(policy_obs, new_actions, fp=self.fp)
            z_new_actions = self.zf(policy_obs, new_actions, new_tau_hat).mean(dim=1)    # (B, T)
            if self.risk_type in ['neutral', 'std']:
                q_new_actions = torch.sum(new_presum_tau * z_new_actions, dim=1, keepdims=True)
                if self.risk_type == 'std':
                    q_std = new_presum_tau * (z_new_actions - q_new_actions).pow(2)
                    q_new_actions -= risk_param * q_std.sum(dim=1, keepdims=True).sqrt()
            else:
                with torch.no_grad():
                    risk_weights = distortion_de(new_tau_hat, self.risk_type, risk_param)
                q_new_actions = torch.sum(risk_weights * new_presum_tau * z_new_actions, dim=1, keepdims=True)

        policy_loss = (alpha * log_pi - q_new_actions).mean()
        gt.stamp('preback_policy', unique=False)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad = ptu.fast_clip_grad_norm(self.policy.parameters(), self.clip_norm)
        self.policy_optimizer.step()
        gt.stamp('backward_policy', unique=False)
        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.zf, self.target_zf, self.soft_target_tau)
            if self.tau_type == 'fqf':
                ptu.soft_update_from_to(self.fp, self.target_fp, self.soft_target_tau)
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            # policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['ZF Loss'] = zf_loss.item()
            self.eval_statistics['ZF RHO Term'] = zf_loss_dict['rho']
            self.eval_statistics['ZF CHI2 Term'] = zf_loss_dict['chi2_loss']
            self.eval_statistics['Policy Loss'] = policy_loss.item()
            
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Expert Predictions',
                ptu.get_numpy(expert_z_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Policy Predictions',
                ptu.get_numpy(policy_z_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Expert Targets',
                ptu.get_numpy(expert_z_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Policy Targets',
                ptu.get_numpy(policy_z_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
                exclude_max_min=True
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
                exclude_max_min=True
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
                exclude_max_min=True
            ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        networks = [
            self.policy,
            self.zf,
            self.target_zf,
        ]
        if self.tau_type == 'fqf':
            networks += [
                self.fp,
                self.target_fp,
            ]
        return networks

    def get_snapshot(self):
        snapshot = dict(
            policy=self.policy.state_dict(),
            zf=self.zf.state_dict(),
            target_zf=self.target_zf.state_dict(),
        )
        if self.tau_type == 'fqf':
            snapshot['fp'] = self.fp.state_dict()
            snapshot['target_fp'] = self.target_fp.state_dict()
        return snapshot

    """For CQL style Training"""
    def _get_tensor_values(self, obs, actions, tau):
        actions_batch_size = actions.shape[0]
        obs_batch_size = obs.shape[0]
        num_repeat = int(actions_batch_size / obs_batch_size)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        tau_temp = tau.unsqueeze(1).repeat(1, num_repeat, 1).view(tau.shape[0] * num_repeat, tau.shape[1])
    
        z1_pred = self.zf1(obs_temp, actions, tau_temp)
        z2_pred = self.zf2(obs_temp, actions, tau_temp)
        z1_pred = z1_pred.view(obs.shape[0], num_repeat, -1)
        z2_pred = z2_pred.view(obs.shape[0], num_repeat, -1)
        return z1_pred, z2_pred
    
    """For CQL style Training"""
    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_actions, _, _, log_pi, *_ = network(obs_temp, reparameterize=True, return_log_prob=True,)
        return new_actions.detach(), log_pi.view(obs.shape[0], num_actions, 1).detach()
