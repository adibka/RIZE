"""
imports
"""
import argparse
import yaml
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.torch_replay_buffer import TorchReplayBuffer
from rlkit.envs import make_env
from rlkit.envs.vecenv import SubprocVectorEnv, VectorEnv
from rlkit.launchers.launcher_util import set_seed, setup_logger
from rlkit.samplers.data_collector import (VecMdpPathCollector, VecMdpStepCollector)
from rlkit.torch.idsac.idsac import IDSACTrainer
from rlkit.torch.idsac.networks import QuantileMlp, Critic, softmax
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.torch_iq_algorithm import TorchVecOnlineIQAlgorithm
torch.set_num_threads(4)
torch.set_num_interop_threads(4)


"""
Define experiment
"""
def experiment(variant):
    dummy_env = make_env(variant['env'])
    obs_dim = dummy_env.observation_space.low.size
    action_dim = dummy_env.action_space.low.size
    expl_env = VectorEnv([lambda: make_env(variant['env']) for _ in range(variant['expl_env_num'])])
    expl_env.seed(variant["seed"])
    expl_env.action_space.seed(variant["seed"])
    eval_env = SubprocVectorEnv([lambda: make_env(variant['env']) for _ in range(variant['eval_env_num'])])
    eval_env.seed(variant["seed"])

    M = variant["layer_size"]
    num_quantiles = variant["num_quantiles"]
    tau_type = variant["trainer_kwargs"]["tau_type"]
    
    zf1 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M],
    )
    zf2 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M],
    )
    target_zf1 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M],
    )
    target_zf2 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M],
    )
    eval_policy = MakeDeterministic(policy)
    target_policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M],
    )
    # fraction proposal network
    fp = target_fp = None
    if variant['trainer_kwargs'].get('tau_type') == 'fqf':
        fp = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=num_quantiles,
            hidden_sizes=[M // 2, M // 2],
            output_activation=softmax,
        )
        target_fp = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=num_quantiles,
            hidden_sizes=[M // 2, M // 2],
            output_activation=softmax,
        )
    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
        zf1,
        tau_type,
    )
    expl_path_collector = VecMdpStepCollector(
        expl_env,
        policy,
    )
    replay_buffer = TorchReplayBuffer(
        variant['replay_buffer_size'],
        dummy_env,
    )
    expert_buffer = TorchReplayBuffer(
        variant['replay_buffer_size'] // 10,
        dummy_env,
    )
    iq_args = variant['iq_kwargs']
    expert_buffer.load(iq_args['expert_path'], iq_args['demos'], 
                       iq_args['subsample_freq'], variant['seed']
                      )
    trainer = IDSACTrainer(
        args=variant,
        env=dummy_env,
        policy=policy,
        target_policy=target_policy,
        zf1=zf1,
        zf2=zf2,
        target_zf1=target_zf1,
        target_zf2=target_zf2,
        fp=fp,
        target_fp=target_fp,
        num_quantiles=num_quantiles,
        **variant['trainer_kwargs'],
    )
    algorithm = TorchVecOnlineIQAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        expert_buffer=expert_buffer,
        **variant['algorithm_kwargs'],
    )
    algorithm.to(ptu.device)
    algorithm.train()


"""
Main
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--env", type=str)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--reward_type", type=str)
    parser.add_argument("--sparse_type", type=str)
    parser.add_argument("--noise_std", type=float)
    parser.add_argument("--sparse_prob", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    with open(f'configs/{args.env}.yaml', 'r', encoding="utf-8") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
        
    variant["trainer_kwargs"]["alpha"] = args.alpha
    variant["seed"] = args.seed
    variant["iq_kwargs"]["loss"] = args.loss
    variant["iq_kwargs"]["reward_type"] = args.reward_type
    variant["iq_kwargs"]["sparse_type"] = args.sparse_type
    variant["iq_kwargs"]["noise_std"] = args.noise_std
    variant["iq_kwargs"]["sparse_prob"] = args.sparse_prob
    
    if torch.cuda.is_available():
        ptu.set_gpu_mode(True, 0)
        
    set_seed(args.seed)
    log_prefix = variant["env"][:-3].lower()
    setup_logger(log_prefix, variant=variant, seed=args.seed)
    variant["device"] = ptu.device
    
    experiment(variant)
    