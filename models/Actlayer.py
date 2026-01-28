import torch
import torch.nn as nn

class FixedCategorical(torch.distributions.Categorical):
    """Modify standard PyTorch Categorical."""

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class ACTLayer(nn.Module):
    """MLP Module to compute actions."""
    def __init__(self, action_dim, inputs_dim):
        super(ACTLayer, self).__init__()
        self.action_dim = action_dim
        self.fc = nn.Linear(inputs_dim, action_dim)
        nn.init.orthogonal_(self.fc.weight, gain=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x,deterministic=False):
        x = self.fc(x)
        action_distribution = FixedCategorical(logits=x)
        actions = (
            action_distribution.mode()
            if deterministic
            else action_distribution.sample()
        )
        action_log_probs = action_distribution.log_probs(actions)
        return actions, action_log_probs

    def get_logits(self, x):
        x = self.fc(x)
        action_distribution = FixedCategorical(logits=x)
        action_logits = action_distribution.logits
        return action_logits

    def evaluate_actions(self, x, action):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            x: (torch.Tensor) input to network.
            action: (torch.Tensor) actions whose entropy and log probability to evaluate.
            available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        x = self.fc(x)
        action_distribution = FixedCategorical(logits=x)
        action_log_probs = action_distribution.log_probs(action)
        dist_entropy = action_distribution.entropy().mean()
        return action_log_probs, dist_entropy, action_distribution

