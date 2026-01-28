import torch
import numpy as np
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.utils.trans_tools import _flatten, _sa_cast


class OnPolicyCriticBufferEP:
    """On-policy buffer for critic that uses Environment-Provided (EP) state."""

    def __init__(self, args, share_obs_size):
        self.episode_length = args.episode_length
        self.hidden_sizes = args.hidden_sizes
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.use_gae = args.use_gae
        self.use_proper_time_limits = args.use_proper_time_limits
        self.rnn_hidden_size = args.critic_hidden_sizes

        self.share_obs_shape = share_obs_size

        # Buffer for share observations
        self.share_obs = np.zeros(
            (self.episode_length + 1,self.share_obs_shape),
            dtype=np.float32,
        )

        self.rnn_states_critic = np.zeros(
            (
                self.episode_length + 1,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # Buffer for value predictions made by this critic
        self.value_preds = np.zeros(
            (self.episode_length + 1, 1), dtype=np.float32
        )

        # Buffer for returns calculated at each timestep
        self.returns = np.zeros(
            (self.episode_length + 1, 1), dtype=np.float32
        )

        # Buffer for rewards received by agents at each timestep
        self.rewards = np.zeros(
            (self.episode_length,  1), dtype=np.float32
        )

        # Buffer for masks indicating whether an episode is done at each timestep
        self.masks = np.ones(
            (self.episode_length + 1, 1), dtype=np.float32
        )

        self.bad_masks = np.ones_like(self.masks)

        self.masks[-1] = 0
        self.bad_masks[-1] = 0

        self.step = 0

    def insert(
        self, rewards
    ):
        """Insert data into buffer."""
        self.rewards[self.step] = rewards.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def get_mean_rewards(self):
        """Get mean rewards for logging."""
        return np.mean(self.rewards)

    def reset_rnn_states(self):
        self.rnn_states_critic = np.zeros(
            (self.episode_length + 1, self.rnn_hidden_size),
            dtype=np.float32
        )
        return self.rnn_states_critic[0]

    def compute_returns(self, next_value, value_normalizer=None):
        """Compute returns either as discounted sum of rewards, or using GAE.
        Args:
            next_value: (np.ndarray) value predictions for the step after the last episode step.
            value_normalizer: (ValueNorm) If not None, ValueNorm value normalizer instance.
        """
        if (
            self.use_proper_time_limits
        ):  # consider the difference between truncation and termination
            if self.use_gae:  # use GAE
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = self.bad_masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = self.bad_masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:  # do not use GAE
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.value_preds[
                            step
                        ]
        else:  # do not consider the difference between truncation and termination, i.e. all done episodes are terminated
            if self.use_gae:  # use GAE
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + self.value_preds[step]
            else:  # do not use GAE
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.masks[step + 1]
                        + self.rewards[step]
                    )

    def naive_recurrent_generator_critic(self, critic_num_mini_batch=1):
        T = self.episode_length

        share_obs_batch = self.share_obs[:-1].reshape(T, -1)
        value_preds_batch = self.value_preds[:-1].reshape(T, 1)
        return_batch = self.returns[:-1].reshape(T, 1)

        rnn_states_critic_batch = self.rnn_states_critic[0]

        yield share_obs_batch, rnn_states_critic_batch, value_preds_batch, return_batch
