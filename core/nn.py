import torch
import torch.nn as nn


class ActorMLP(nn.Module):
    """A simple feedforward neural network for the actor."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dims: list[int]):
        super(ActorMLP, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            for out_dim in output_dims
        ])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.backbone(state)
        action_logits = [head(features) for head in self.action_heads]
        return action_logits
