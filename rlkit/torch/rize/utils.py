import rlkit.torch.pytorch_util as ptu
import torch

class LinearSchedule(object):

    def __init__(self, schedule_timesteps, initial=1., final=0.):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final = final
        self.initial = initial

    def __call__(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial + fraction * (self.final - self.initial)

def get_tau(obs, actions, tau_type='iqn', num_quantiles=24, fp=None):
    if tau_type == 'fix':
        presum_tau = ptu.zeros(len(actions), num_quantiles) + 1. / num_quantiles
    elif tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
        presum_tau = ptu.rand(len(actions), num_quantiles) + 0.1
        presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
    elif tau_type == 'fqf':
        if fp is None:
            fp = self.fp
        presum_tau = fp(obs, actions)
    tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
    with torch.no_grad():
        tau_hat = ptu.zeros_like(tau)
        tau_hat[:, 0:1] = tau[:, 0:1] / 2.
        tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
    return ptu.get_numpy(tau), ptu.get_numpy(tau_hat), ptu.get_numpy(presum_tau)