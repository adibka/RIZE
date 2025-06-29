from collections import OrderedDict, deque

import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.data_management.path_builder import PathBuilder
from rlkit.envs.vecenv import BaseVectorEnv
from rlkit.samplers.data_collector.base import DataCollector
from rlkit.torch.rize.utils import get_tau

class VecMdpPathCollector(DataCollector):

    def __init__(
            self,
            env: BaseVectorEnv,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._env_num = self._env.env_num
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._obs = None  # cache variable

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._obs = None

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )

    def collect_new_paths(
            self,
            max_path_length,
            num_paths,
            discard_incomplete_paths,
    ):
        if self._obs is None:
            self._start_new_rollout()

        num_paths_collected = 0
        while num_paths_collected < num_paths:

            actions = self._policy.get_actions(self._obs)
            next_obs, rewards, terminals, env_infos = self._env.step(actions)
            
            if self._render:
                self._env.render(**self._render_kwargs)

            # unzip vectorized data
            for env_idx, (
                    path_builder,
                    next_ob,
                    action,
                    reward,
                    terminal,
                    env_info,
            ) in enumerate(zip(
                    self._current_path_builders,
                    next_obs,
                    actions,
                    rewards,
                    terminals,
                    env_infos,
            )):
                obs = self._obs[env_idx].copy()
                terminal = np.array([terminal])
                reward = np.array([reward])
                # store path obs
                path_builder.add_all(
                    observations=obs,
                    actions=action,
                    rewards=reward,
                    next_observations=next_ob,
                    terminals=terminal,
                    agent_infos={},
                    env_infos=env_info,
                )
                self._obs[env_idx] = next_ob
                if terminal or len(path_builder) >= max_path_length:
                    self._handle_rollout_ending(path_builder, max_path_length, discard_incomplete_paths)
                    self._start_new_rollout(env_idx)
                    num_paths_collected += 1

    def _start_new_rollout(self, env_idx=None):
        if env_idx is None:
            self._current_path_builders = [PathBuilder() for _ in range(self._env_num)]
            self._obs = self._env.reset()
        else:

            self._current_path_builders[env_idx] = PathBuilder()
            self._obs[env_idx] = self._env.reset(env_idx)[env_idx]

    def _handle_rollout_ending(self, path_builder, max_path_length, discard_incomplete_paths):
        if len(path_builder) > 0:
            path = path_builder.get_all_stacked()
            path_len = len(path['actions'])
            if (path_len != max_path_length and not path['terminals'][-1] and discard_incomplete_paths):
                return
            self._epoch_paths.append(path)
            self._num_paths_total += 1
            self._num_steps_total += path_len
