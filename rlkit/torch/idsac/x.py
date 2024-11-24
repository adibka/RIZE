import gtimer as gt
from collections import OrderedDict
import csv
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_iq_algorithm import TorchIQTrainer
from rlkit.torch.idsac.iq_loss import iq_loss
from .risk import distortion_de
from .utils import LinearSchedule
from torch.cuda.amp import GradScaler, autocast

class IDSACTrainer(TorchIQTrainer):

    def __init__(
            self,
            args,
            env,
            policy,
            target_policy,
            zf1,
            zf2,
            target_zf1,
            target_zf2,
            fp=None,
            target_fp=None,
            discount=0.99,
            reward_scale=1.0,
            alpha=1.0,
            policy_lr=3e-4,
            zf_lr=3e-4,
            tau_type='iqn',
            fp_lr=1e-5,
            expert_lambda=10.0,
            expert_lambda_lr=1e-3,
            tune_expert_lambda=True,
            policy_lambda=10.0,
            policy_lambda_lr=1e-3,
            tune_policy_lambda=True,
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
            dropout=False,
            drop_rate=0.01,
    ):
        super().__init__()
        self.args = args
        self.env = env
        self.device = args['device']
        self.policy = policy
        self.target_policy = target_policy
        self.zf1 = zf1
        self.zf2 = zf2
        self.target_zf1 = target_zf1
        self.target_zf2 = target_zf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.tau_type = tau_type
        self.num_quantiles = num_quantiles
        self.tune_expert_lambda = tune_expert_lambda
        self.tune_policy_lambda = tune_policy_lambda
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

        if self.tune_expert_lambda:
            self.expert_lambda = ptu.tensor(expert_lambda, dtype=torch.float, requires_grad=True)
            self.expert_lambda_optimizer = optimizer_class(
                [self.expert_lambda],
                lr=expert_lambda_lr,
            )
        else:
            self.expert_lambda = expert_lambda

        if self.tune_policy_lambda:
            self.policy_lambda = ptu.tensor(policy_lambda, dtype=torch.float, requires_grad=True)
            self.policy_lambda_optimizer = optimizer_class(
                [self.policy_lambda],
                lr=policy_lambda_lr,
            )
        else:
            self.policy_lambda = policy_lambda
            
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()
            self.log_alpha = ptu.tensor(np.log(alpha), dtype=torch.float, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        else:
            self.alpha = alpha
        
        self.zf_criterion = iq_loss
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.zf1_optimizer = optimizer_class(
            self.zf1.parameters(),
            lr=zf_lr,
        )
        self.zf2_optimizer = optimizer_class(
            self.zf2.parameters(),
            lr=zf_lr,
        )
        self.scaler_pi = GradScaler()
        self.scaler_zf1 = GradScaler()
        self.scaler_zf2 = GradScaler()
        
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
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.EPS = 1e-6

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
        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau

    def getZ(self, obs, actions, tau_hat, presum_tau):
        z1_pred = self.zf1(obs, actions, tau_hat)
        z2_pred = self.zf2(obs, actions, tau_hat)
        if self.args['expectation_z']:
            q1_pred = torch.sum(presum_tau * z1_pred, dim=1, keepdims=True)
            q2_pred = torch.sum(presum_tau * z2_pred, dim=1, keepdims=True)
            return q1_pred, q2_pred
        else:
            return z1_pred, z2_pred

    def get_targetZ(self, next_obs, next_actions, next_tau_hat, next_presum_tau):
        target_z1 = self.target_zf1(next_obs, next_actions, next_tau_hat)
        target_z2 = self.target_zf2(next_obs, next_actions, next_tau_hat)
        target_z = torch.min(target_z1, target_z2)
        if self.args['expectation_z']:
            target_q = torch.sum(next_presum_tau * target_z, dim=1, keepdims=True)
            return target_q
        else:
            return target_z

    def getV(self, obs):
        actions, _, _, log_pi, *_ = self.policy(
            obs,
            reparameterize=True,
            return_log_prob=True,
        )
        tau, tau_hat, presum_tau = self.get_tau(obs, actions, fp=self.fp)
        current_z1, current_z2 = self.getZ(obs, actions, tau_hat, presum_tau)
        current_v1 = current_z1 - self.alpha * log_pi
        current_v2 = current_z2 - self.alpha * log_pi
        return current_v1, current_v2

    def get_targetV(self, next_obs):
        next_actions, _, _, next_log_pi, *_ = self.target_policy(
            next_obs,
            reparameterize=True,
            return_log_prob=True,
        )
        next_tau, next_tau_hat, next_presum_tau = self.get_tau(next_obs, next_actions, fp=self.target_fp)
        target_v = self.get_targetZ(next_obs, next_actions, next_tau_hat, next_presum_tau) - self.alpha * next_log_pi
        return target_v

    def train_from_torch(self, policy_batch, expert_batch):
        # Policy batch samples
        policy_rewards = policy_batch['rewards']
        policy_terminals = policy_batch['terminals']
        policy_obs = policy_batch['observations']
        policy_actions = policy_batch['actions']
        policy_next_obs = policy_batch['next_observations']
        # Expert batch samples
        expert_terminals = expert_batch['terminals']
        expert_obs = expert_batch['observations']
        expert_actions = expert_batch['actions']
        expert_next_obs = expert_batch['next_observations']
        gt.stamp('preback_start', unique=False)
        """
        Update ZF
        """
        self.zf1_optimizer.zero_grad()
        self.zf2_optimizer.zero_grad()
        
        with torch.no_grad():
            policy_v_target = self.get_targetV(policy_next_obs)
            policy_target = (1. - policy_terminals) * self.discount * policy_v_target
            expert_v_target = self.get_targetV(expert_next_obs)
            expert_target = (1. - expert_terminals) * self.discount * expert_v_target

        with autocast():
            tau, tau_hat, presum_tau = self.get_tau(policy_obs, policy_actions, fp=self.fp)
            policy_z1_pred, policy_z2_pred = self.getZ(policy_obs, policy_actions, tau_hat, presum_tau)
            expert_z1_pred, expert_z2_pred = self.getZ(expert_obs, expert_actions, tau_hat, presum_tau)
            
            _, _, _, log_pi, *_ = self.policy(
                policy_obs,
                reparameterize=True,
                return_log_prob=True,
            )
            _, _, _, expert_log_pi, *_ = self.policy(
                expert_obs,
                reparameterize=True,
                return_log_prob=True,
            )
            if self.args["iq_kwargs"]["loss"] == "v0":
                v0_1, v0_2 = self.getV(expert_obs)
            else:
                v0_1, v0_2 = None, None

            zf1_loss, zf1_loss_dict = self.zf_criterion(
                expert_z1_pred, 
                expert_target,
                expert_log_pi,
                policy_z1_pred, 
                policy_target,
                policy_rewards,
                log_pi,
                v0_1,
                self.expert_lambda,
                self.policy_lambda,
                self.alpha,
                self.args,
            )
            zf2_loss, zf2_loss_dict = self.zf_criterion(
                expert_z2_pred, 
                expert_target,
                expert_log_pi,
                policy_z2_pred, 
                policy_target,
                policy_rewards,
                log_pi,
                v0_2,
                self.expert_lambda,
                self.policy_lambda,
                self.alpha,
                self.args,
            )
            
        gt.stamp('preback_zf', unique=False)
        self.scaler_zf1.scale(zf1_loss).backward(retain_graph=True)
        self.scaler_zf1.step(self.zf1_optimizer)
        self.scaler_zf1.update()
        gt.stamp('backward_zf1', unique=False)
        self.scaler_zf2.scale(zf2_loss).backward(retain_graph=True)
        self.scaler_zf2.step(self.zf2_optimizer)
        self.scaler_zf2.update()
        gt.stamp('backward_zf2', unique=False)
        """
        Update expert_lambda
        """
        if self.tune_expert_lambda:
            implicit_reward = (expert_z1_pred + expert_z2_pred)/2 - expert_target
            expert_lambda_loss = 0.5 * ((implicit_reward.detach() - self.expert_lambda)**2).mean()
            self.expert_lambda_optimizer.zero_grad()
            expert_lambda_loss.backward()
            self.expert_lambda_optimizer.step()
        else:
            expert_lambda_loss = 0.
        """
        Update policy_lambda
        """
        if self.tune_policy_lambda:
            implicit_reward = (policy_z1_pred + policy_z2_pred)/2 - policy_target
            policy_lambda_loss = 0.5 * ((implicit_reward.detach() - self.policy_lambda)**2).mean()
            self.policy_lambda_optimizer.zero_grad()
            policy_lambda_loss.backward()
            self.policy_lambda_optimizer.step()
        else:
            policy_lambda_loss = 0.
        """
        Update FP
        """
        if self.tau_type == 'fqf':
            with torch.no_grad():
                dWdtau = 0.5 * (2 * self.zf1(obs, actions, tau[:, :-1]) - z1_pred[:, :-1] - z1_pred[:, 1:] +
                                2 * self.zf2(obs, actions, tau[:, :-1]) - z2_pred[:, :-1] - z2_pred[:, 1:])
                dWdtau /= dWdtau.shape[0]  # (N, T-1)
            gt.stamp('preback_fp', unique=False)

            self.fp_optimizer.zero_grad()
            tau[:, :-1].backward(gradient=dWdtau)
            self.fp_optimizer.step()
            gt.stamp('backward_fp', unique=False)
        """
        Update Policy
        """
        if self.args['use_policy_expert_obs']:
            obs = torch.cat([policy_obs, expert_obs], dim=0)
        else:
            obs = policy_obs
        
        new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs,
            reparameterize=True,
            return_log_prob=True,
        )
        risk_param = self.risk_schedule(self._n_train_steps_total)
        if self.risk_type == 'VaR':
            tau_ = ptu.ones_like(policy_rewards) * risk_param
            q1_new_actions = self.zf1(obs, new_actions, tau_)
            q2_new_actions = self.zf2(obs, new_actions, tau_)
        else:
            with torch.no_grad():
                new_tau, new_tau_hat, new_presum_tau = self.get_tau(obs, new_actions, fp=self.fp)
            z1_new_actions = self.zf1(obs, new_actions, new_tau_hat)
            z2_new_actions = self.zf2(obs, new_actions, new_tau_hat)
            if self.risk_type in ['neutral', 'std']:
                q1_new_actions = torch.sum(new_presum_tau * z1_new_actions, dim=1, keepdims=True)
                q2_new_actions = torch.sum(new_presum_tau * z2_new_actions, dim=1, keepdims=True)
                if self.risk_type == 'std':
                    q1_std = new_presum_tau * (z1_new_actions - q1_new_actions).pow(2)
                    q2_std = new_presum_tau * (z2_new_actions - q2_new_actions).pow(2)
                    q1_new_actions -= risk_param * q1_std.sum(dim=1, keepdims=True).sqrt()
                    q2_new_actions -= risk_param * q2_std.sum(dim=1, keepdims=True).sqrt()
            else:
                with torch.no_grad():
                    risk_weights = distortion_de(new_tau_hat, self.risk_type, risk_param)
                q1_new_actions = torch.sum(risk_weights * new_presum_tau * z1_new_actions, dim=1, keepdims=True)
                q2_new_actions = torch.sum(risk_weights * new_presum_tau * z2_new_actions, dim=1, keepdims=True)    
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)
        
        policy_loss = (self.alpha * log_pi - q_new_actions).mean()
        gt.stamp('preback_policy', unique=False)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad = ptu.fast_clip_grad_norm(self.policy.parameters(), self.clip_norm)
        self.policy_optimizer.step()
        gt.stamp('backward_policy', unique=False)
        """
        Update Alpha
        """
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = 0
        gt.stamp('preback_alpha', unique=False)
        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.soft_target_tau)
            ptu.soft_update_from_to(self.zf1, self.target_zf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.zf2, self.target_zf2, self.soft_target_tau)
            if self.tau_type == 'fqf':
                ptu.soft_update_from_to(self.fp, self.target_fp, self.soft_target_tau)
        """
        Monitor weights
        """
        policy_param_norm = ptu.param_norm(self.policy.parameters())
        zf1_param_norm = ptu.param_norm(self.zf1.parameters())
        zf2_param_norm = ptu.param_norm(self.zf2.parameters())

        policy_grad_norm = ptu.grad_norm(self.policy.parameters())
        zf1_grad_norm = ptu.grad_norm(self.zf1.parameters())
        zf2_grad_norm = ptu.grad_norm(self.zf2.parameters())
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['ZF1 Loss'] = zf1_loss.item()
            self.eval_statistics['ZF2 Loss'] = zf2_loss.item()
            self.eval_statistics['ZF Expert Reward'] = \
                        (zf1_loss_dict['expert_reward'] + zf2_loss_dict['expert_reward']) / 2
            self.eval_statistics['ZF Policy Reward'] = \
                        (zf1_loss_dict['policy_reward'] + zf2_loss_dict['policy_reward']) / 2
            self.eval_statistics['ZF CHI2 Term'] = \
                        (zf1_loss_dict['chi2_loss'] + zf2_loss_dict['chi2_loss']) / 2
            self.eval_statistics['Policy Loss'] = policy_loss.item()
            if self.tune_expert_lambda:
                self.eval_statistics['expert_lambda Loss'] = expert_lambda_loss.item()
                self.eval_statistics['expert_lambda Value'] = self.expert_lambda.item()
            else:
                self.eval_statistics['expert_lambda Value'] = self.expert_lambda
            if self.tune_policy_lambda:
                self.eval_statistics['policy_lambda Loss'] = policy_lambda_loss.item()
                self.eval_statistics['policy_lambda Value'] = self.policy_lambda.item()
            else:
                self.eval_statistics['policy_lambda Value'] = self.policy_lambda
            self.eval_statistics['Policy Grad Norm'] = policy_grad_norm
            self.eval_statistics['Policy Param Norm'] = policy_param_norm
            self.eval_statistics['Zf1 Grad Norm'] = zf1_grad_norm
            self.eval_statistics['Zf1 Param Norm'] = zf1_param_norm
            self.eval_statistics['Zf2 Grad Norm'] = zf2_grad_norm
            self.eval_statistics['Zf2 Param Norm'] = zf2_param_norm
            # self.eval_statistics['Policy Grad'] = policy_grad
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Expert Predictions',
                ptu.get_numpy((expert_z1_pred + expert_z2_pred) / 2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Policy Predictions',
                ptu.get_numpy((policy_z1_pred + policy_z2_pred) / 2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Expert Targets',
                ptu.get_numpy(expert_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Policy Targets',
                ptu.get_numpy(policy_target),
            ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Q1 OOD',
            #     ptu.get_numpy(q1_ood),
            #     exclude_max_min=True,
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Q2 OOD',
            #     ptu.get_numpy(q2_ood),
            #     exclude_max_min=True,
            # ))
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
                self.eval_statistics['Alpha'] = self.alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
            else:
                self.eval_statistics['Alpha'] = self.alpha
        self._n_train_steps_total += 1
        # self.alpha_decay(self._n_train_steps_total)
        
    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        networks = [
            self.policy,
            self.target_policy,
            self.zf1,
            self.zf2,
            self.target_zf1,
            self.target_zf2,
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
            target_policy=self.target_policy.state_dict(),
            zf1=self.zf1.state_dict(),
            zf2=self.zf2.state_dict(),
            target_zf1=self.target_zf1.state_dict(),
            target_zf2=self.target_zf2.state_dict(),
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