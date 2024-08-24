"""
Copyright 2022 Div Garg. All rights reserved.
Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""
import torch
import torch.nn.functional as F


def iq_loss(
    expert_z_pred, 
    expert_target,
    expert_log_pi,
    policy_z_pred, 
    policy_target,
    policy_r,
    policy_log_pi,
    v0,
    bias,
    alpha,
    args,
    ):
    
    discount = args['trainer_kwargs']['discount']
    iq_args = args['iq_kwargs']
    expert_reward = expert_z_pred - expert_target
    policy_reward = policy_z_pred - policy_target
    expert_v_pred = expert_z_pred - alpha * expert_log_pi
    policy_v_pred = policy_z_pred - alpha * policy_log_pi
    
    expert_loss = - expert_reward.mean()
    
    if iq_args['loss'] == "value_policy":
        # sample using only policy states
        # E_(ρ)[V(s) - γV(s')]
        value_loss = (policy_v_pred - policy_target).mean()
    elif iq_args['loss'] == "value":
        # sample using expert and policy states (works online)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = torch.cat([(expert_v_pred - expert_target), (policy_v_pred - policy_target)], dim=-1).mean()
    elif iq_args['loss'] == "value_expert":
        # sample using only expert states (works offline)
        # E_(ρ)[V(s) - γV(s')]  
        value_loss = (expert_v_pred - expert_target).mean()
    elif iq_args['loss'] == "v0":
        # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
        # (1-γ)E_(ρ0)[V(s0)]
        value_loss = (1 - discount) * v0.mean()

    if iq_args['regularize']:
        td_expert = expert_reward - bias
        td_policy = policy_reward - policy_r
        chi2_loss = iq_args['chi'] * (torch.cat([td_expert, td_policy], dim=-1)**2).mean()
        
    loss = expert_loss + value_loss + chi2_loss
    loss_dict = {
        'expert_reward': expert_reward.mean().item(),
        'policy_reward': policy_reward.mean().item(),
        'value_loss': value_loss.item(),
        'chi2_loss': chi2_loss.item() if iq_args['regularize'] else 0,
    }
    return loss, loss_dict
