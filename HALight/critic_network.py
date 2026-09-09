"""Critic."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)
from harl.utils.envs_tools import check
from models.rnn_layer import RNNLayer

class Critic_network(nn.Module):
    def __init__(self, share_state_size,hidden_size1=128, hidden_size2=128):
        super().__init__()
        self.share_state_size = share_state_size
        self.layer1 = nn.Linear(share_state_size, hidden_size1)
        self.value_normalizer1 = nn.LayerNorm(hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.value_normalizer2 = nn.LayerNorm(hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, 1)
        self.rnn = RNNLayer(
            hidden_size2,
            hidden_size2,
        )
        nn.init.orthogonal_(self.layer1.weight, gain=0.01)
        nn.init.zeros_(self.layer1.bias)
        nn.init.orthogonal_(self.layer2.weight, gain=0.01)
        nn.init.zeros_(self.layer2.bias)
        nn.init.orthogonal_(self.layer3.weight, gain=0.01)
        nn.init.zeros_(self.layer3.bias)

    def forward(self, state,rnn_states):
        x = self.layer1(state)
        x = F.relu(x)
        x = self.value_normalizer1(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.value_normalizer2(x)
        critic_features,rnn_states = self.rnn(x, rnn_states)
        values = self.layer3(critic_features)
        return values, rnn_states


class Critic:
    """V Critic.
    Critic that learns a V-function.
    """
    def __init__(self, args, share_states_size):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_param = args.clip_param
        self.critic_epoch = args.critic_epoch
        self.critic_num_mini_batch = args.critic_num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self.use_max_grad_norm = args.use_max_grad_norm
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.use_huber_loss = args.use_huber_loss

        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps

        self.share_obs_space = share_states_size
        self.critic = Critic_network(self.share_obs_space,hidden_size1=args.critic_hidden_sizes,hidden_size2=args.critic_hidden_sizes).to(self.device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
        )

    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_values(self, cent_obs,rnn_states_critic):
        """Get value function predictions.
        Args:
            cent_obs: (np.ndarray) centralized input to the critic.
            rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        cent_obs = check(cent_obs).to(self.device)
        rnn_states_critic = check(rnn_states_critic).to(self.device)
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic)
        return values, rnn_states_critic

    def cal_value_loss(
        self, values, value_preds_batch, return_batch, value_normalizer=None
    ):
        """Calculate value function loss.
        Args:
            values: (torch.Tensor) value function predictions.
            value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
            return_batch: (torch.Tensor) reward to go returns.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if value_normalizer is not None:
            value_normalizer.update(return_batch)
            error_clipped = (
                value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()


        return value_loss

    def update(self, sample, value_normalizer=None):
        """Update critic network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
            critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        """
        (
            share_obs_batch,
            rnn_states_critic_batch,
            value_preds_batch,
            return_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(self.device)
        return_batch = check(return_batch).to(self.device)

        values,_ = self.get_values(share_obs_batch, rnn_states_critic_batch)

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, value_normalizer=value_normalizer
        )

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm

    def train(self, critic_buffer, value_normalizer=None):
        """Perform a training update using minibatch GD.
        Args:
            critic_buffer: (OnPolicyCriticBufferEP or OnPolicyCriticBufferFP) buffer containing training data related to critic.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        for _ in range(self.critic_epoch):
            data_generator = critic_buffer.naive_recurrent_generator_critic(self.critic_num_mini_batch)

            for sample in data_generator:
                self.update(sample, value_normalizer=value_normalizer)

    def prep_training(self):
        """Prepare for training."""
        self.critic.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.critic.eval()
