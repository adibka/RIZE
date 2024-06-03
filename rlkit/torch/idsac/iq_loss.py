"""
Copyright 2022 Div Garg. All rights reserved.
Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""
import torch
import torch.nn.functional as F


def iq_loss(
    expert_pred,
    expert_target,
    policy_pred,
    policy_target,
    policy_r,
    log_pi,
    alpha,
    args
):
    expert_reward = (expert_pred - expert_target)
    policy_reward = (policy_pred - alpha * log_pi - policy_target)
    loss = - expert_reward.mean() + policy_reward.mean()
    
    if args['regularize']:
        chi2_loss = 1/(4 * args['alpha']) * \
                        (torch.cat([expert_reward - 10., (policy_reward + alpha * log_pi) - policy_r], dim=-1)**2).mean()
        loss += chi2_loss
        
    loss_dict = {
        'expert_reward': expert_reward.mean().item(),
        'policy_reward': policy_reward.mean().item(),
        'chi2_loss': chi2_loss.item() if args['regularize'] else 0,
    }
    return loss, loss_dict


def iq_distr_loss(
    input_expert, 
    target_expert,
    input_policy,
    target_policy,
    policy_reward,
    log_pi,
    alpha,
    tau,
    weight,
    args
):
    expert_reward = input_expert - target_expert
    policy_reward = input_policy - alpha * log_pi - target_policy
    loss = - expert_reward.mean() + policy_reward.mean()

    if args['regularize']:
        expert_rho = quantile_regression_loss(input_expert, 10 + target_expert, tau, weight)
        policy_rho = quantile_regression_loss(input_policy, policy_reward + target_policy, tau, weight)
        chi2_loss = 1/(4 * args['alpha']) * torch.cat([expert_rho, policy_rho], dim=-1).mean()
        loss += chi2_loss
    
    loss_dict = {
        'expert_reward': expert_reward.mean().item(),
        'policy_reward': policy_reward.mean().item(),
        'chi2_loss': chi2_loss.item() if args['regularize'] else 0,
    }
    return loss, loss_dict


def quantile_regression_loss(inputs, targets, tau, weight):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    """
    inputs = inputs.unsqueeze(-1)
    targets = targets.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)
    # weight = weight.detach().unsqueeze(-2)
    expanded_inputs, expanded_targets = torch.broadcast_tensors(inputs, targets)
    L = (expanded_inputs - expanded_targets)**2
    sign = torch.sign(expanded_inputs - expanded_targets) / 2. + 0.5
    rho = torch.abs(tau - sign) * L
    return rho.sum(dim=-1)
