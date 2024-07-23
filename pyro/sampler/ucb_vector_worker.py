"""A worker class for environments that receive a vector of actions and return a
 vector of states. Not to be confused with vec_worker, which handles a vector of
 environments."""
from collections import defaultdict

from pyro import EpisodeBatch
from garage.sampler.default_worker import DefaultWorker

from pyro.policies import AdaptiveTanhGaussianPolicy
from pyro.q_functions.adaptive_mlp_q_function import AdaptiveMLPQFunction
from pyro.q_functions.adaptive_lstm_q_function import AdaptiveLSTMQFunction

import torch
import torch.nn as nn
import numpy as np


class UCBVectorWorker(DefaultWorker):
    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_episode_length,
            worker_number, lstm=False, dropout=0., layer_normalization=False):
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)
        self._n_parallel = None
        self._prev_mask = None
        self._masks = []
        self._last_masks = []
        self._rewards = []
        self._actions = []
        self._terminals = []
        self._env_infos = defaultdict(list)

        self._lstm = lstm
        self._dropout = dropout
        self._layer_normalization = layer_normalization

    def update_env(self, env_update):
        super().update_env(env_update)
        self._n_parallel = self.env.n_parallel

    def pad_observation(self, obs):
        pad_shape = list(obs.shape)
        pad_shape[1] = self._max_episode_length - pad_shape[1]
        pad = torch.zeros(pad_shape)
        padded_obs = torch.cat([obs, pad], dim=1)
        mask = torch.cat(
            [torch.ones_like(obs, dtype=torch.bool),
             torch.zeros_like(pad, dtype=torch.bool)], dim=1)[..., :1]
        return padded_obs, mask


    def start_rollout(self):
        """Begin a new rollout."""
        self._path_length = 0
        self._prev_obs, _ = self.env.reset(n_parallel=self._n_parallel)
        self._prev_obs, self._prev_mask = self.pad_observation(self._prev_obs)

        op = []
        oq = []

        layer_size = 128

        for i in range(len(self.agent)):
            p = AdaptiveTanhGaussianPolicy(
                    env_spec=self.env.spec,
                    encoder_sizes=[layer_size, layer_size],
                    encoder_nonlinearity=nn.ReLU,
                    encoder_output_nonlinearity=None,
                    emitter_sizes=[layer_size, layer_size],
                    emitter_nonlinearity=nn.ReLU,
                    emitter_output_nonlinearity=None,
                    encoding_dim=layer_size//2,
                    init_std=np.sqrt(1 / 3),
                    min_std=np.exp(-20.),
                    max_std=np.exp(0.),
                )
            if isinstance(self.agent[i], dict):
                p.load_state_dict(self.agent[i])
            else:
                p = self.agent[i]
            p.reset()
            op.append(p)
        
        for i in range(len(self.qf)):
            if self._lstm == True:
                q = AdaptiveLSTMQFunction(
                        env_spec=self.env.spec,
                        encoder_sizes=[layer_size, layer_size],
                        encoder_output_nonlinearity=None,
                        emitter_sizes=[layer_size, layer_size],
                        emitter_output_nonlinearity=None,
                        encoding_dim=layer_size//2
                    )
            else:
                q = AdaptiveMLPQFunction(
                            env_spec=self.env.spec,
                            encoder_sizes=[layer_size, layer_size],
                            encoder_nonlinearity=nn.ReLU,
                            encoder_output_nonlinearity=None,
                            emitter_sizes=[layer_size, layer_size],
                            emitter_nonlinearity=nn.ReLU,
                            emitter_output_nonlinearity=None,
                            encoding_dim=layer_size//2,
                            dropout=self._dropout,
                            layer_normalization=self._layer_normalization
                        )
            
            if isinstance(self.qf[i], dict):
                q.load_state_dict(self.qf[i])
            else:
                q = self.qf[i]
            oq.append(q)

        self.agent = op
        self.qf = oq

    def sample_action(self, policies, qfs):
        if isinstance(policies, list):
            for policy in policies:
                policy.eval()
        else:
            policies.eval()
            policies = [policies]
        
        if isinstance(qfs, list):
            for qf in qfs:
                qf.eval()
        else:
            qfs.eval()
            qfs = [qfs]

        # a = argmax_a Q_mean(s, a) + \lambda * Q_std(s, a)
        with torch.no_grad():
            # Generate action from each actor
            act_candidates = [policy.get_actions(self._prev_obs, self._prev_mask)[0] for policy in policies]

            # Get actor information
            act_candidate_infos = [policy.get_actions(self._prev_obs, self._prev_mask)[1] for policy in policies]

            # Evaluate each action on each critic separately and stack their Q-values
            q_vals_list = []
            for act_candidate in act_candidates:
                q_vals = torch.stack([q(self._prev_obs, act_candidate, self._prev_mask) for q in qfs], dim=0)
                q_vals_list.append(q_vals)

            # Stack the Q-values along a new dimension to keep track of action candidates
            q_vals_stacked = torch.stack(q_vals_list, dim=0)
            #print(policies[0].get_actions(self._prev_obs, self._prev_mask)[0].shape)
            #print(act_candidates[0].shape)
            #print(act_candidates[1].shape)
            #print(q_vals_list[0].shape)
            #print(q_vals_list[1].shape)
            #print(q_vals_stacked.shape)

            # Compute mean and standard deviation over the critic axis to get the UCB term
            mean_q_vals = q_vals_stacked.mean(0)
            std_q_vals = q_vals_stacked.std(0)

            # Use 1 as exploration coefficient
            ucb_val = mean_q_vals + 1 * std_q_vals

            # Argmax over action axis to select the action with the highest UCB value
            #print(ucb_val.shape)
            argmax_ucb_val = torch.argmax(ucb_val, dim=0)
            argmax_ucb_val = torch.mode(argmax_ucb_val, dim=0).values.item()
            #print(argmax_ucb_val.shape)
            act = act_candidates[argmax_ucb_val]
            actinfo = act_candidate_infos[argmax_ucb_val]
        
        if isinstance(policies, list):
            for policy in policies:
                policy.train()
        else:
            policies.train()
        
        if isinstance(qfs, list):
            for qf in qfs:
                qf.train()
        else:
            qfs.train()

        return act, actinfo

    def step_rollout(self, deterministic):
        """Take a vector of time-steps in the current rollout

        Returns:
            bool: True iff the path is done, either due to the environment
            indicating termination or due to reaching `max_episode_length`.
        """
        if self._path_length < self._max_episode_length:
            #print("agent", self.agent, "qf", self.qf)
            #print(self.agent)
            a, agent_info = self.sample_action(self.agent, self.qf)

            if deterministic and 'mean' in agent_info:
                a = agent_info['mean']

            a_shape = (self._n_parallel,) + self.env.action_space.shape[1:]
            env_step = self.env.step(a.reshape(a_shape))
            next_o, r = env_step.observation, env_step.reward
            d, env_info = env_step.terminal, env_step.env_info
            self._observations.append(self._prev_obs)
            self._rewards.append(r)
            self._actions.append(a)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            for k, v in env_info.items():
                self._env_infos[k].append(v)
            self._masks.append(self._prev_mask)
            self._path_length += 1
            # TODO: make sure we want to use step_Type and not simply booleans
            self._terminals.append(d * torch.ones_like(r))
            if not env_step.terminal:
                next_o, next_mask = self.pad_observation(next_o)
                self._prev_obs = next_o
                self._prev_mask = next_mask
                return False
        self._lengths = self._path_length * torch.ones(self._n_parallel,
                                                       dtype=torch.int)
        self._last_observations.append(self._prev_obs)
        self._last_masks.append(self._prev_mask)
        return True

    def collect_rollout(self):
        """Collect the current rollout of vectors, convert it to a vector of
        rollouts, and clear the internal buffer

        Returns:
            garage.EpisodeBatch: A batch of the episodes completed since
                the last call to collect_rollout().
        """
        observations = torch.cat(
            torch.split(torch.stack(self._observations, dim=1), 1),
            dim=1
        ).squeeze(0)
        self._observations = []
        last_observations = torch.cat(self._last_observations)
        self._last_observations = []
        masks = torch.cat(
            torch.split(torch.stack(self._masks, dim=1), 1),
            dim=1
        ).squeeze(0)
        self._masks = []
        last_masks = torch.cat(self._last_masks)
        self._last_masks = []
        actions = torch.cat(
            torch.split(torch.stack(self._actions, dim=1), 1),
            dim=1
        ).squeeze(0)
        self._actions = []
        rewards = torch.cat(
            torch.split(torch.stack(self._rewards, dim=1), 1),
            dim=1
        ).squeeze(0)
        self._rewards = []
        terminals = torch.cat(
            torch.split(torch.stack(self._terminals, dim=1), 1),
            dim=1
        ).squeeze(0)
        self._terminals = []
        env_infos = self._env_infos
        self._env_infos = defaultdict(list)
        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        for k, v in agent_infos.items():
            agent_infos[k] = torch.cat(
                torch.split(torch.stack(v, dim=1), 1),
                dim=1
            ).squeeze(0)
        zs = torch.zeros((self._n_parallel, self._lengths[0]))
        for k, v in env_infos.items():
            if torch.is_tensor(v[0]):
                env_infos[k] = torch.cat(
                    torch.split(torch.stack(v, dim=1), 1),
                    dim=1
                ).squeeze(0)
            else:
                env_infos[k] = torch.cat(
                    torch.split(torch.as_tensor(v).float() + zs, 1),
                    dim=1
                ).squeeze(0)
        lengths = self._lengths
        self._lengths = []
        episode_infos = dict()
        return EpisodeBatch(self.env.spec, episode_infos, observations,
                            last_observations, masks, last_masks, actions,
                            rewards, dict(env_infos), dict(agent_infos),
                            terminals, lengths)

    def rollout(self, deterministic=False):
        """Sample a single vectorised rollout of the agent in the environment.

        Returns:
            garage.EpisodeBath: The collected trajectory.

        """
        self.start_rollout()
        while not self.step_rollout(deterministic):
            pass
        return self.collect_rollout()
