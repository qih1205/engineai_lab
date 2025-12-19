from collections.abc import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """Simple multi-layer perceptron used by shared backbones."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(activation())
            last_dim = dim
        self.model = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ActorCriticNetwork(nn.Module):
    """Actor-critic backbone with a shared feature encoder."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()
        self.shared = MLP(obs_dim, hidden_dims, activation)
        tail_dim = hidden_dims[-1] if hidden_dims else obs_dim
        self.actor_head = nn.Linear(tail_dim, action_dim)
        self.critic_head = nn.Linear(tail_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(observations)
        actor_logits = self.actor_head(features)
        critic_value = self.critic_head(features).squeeze(-1)
        return actor_logits, critic_value
