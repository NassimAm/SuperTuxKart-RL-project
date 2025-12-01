import torch
import torch.nn as nn
from .action import Action


class ActorMLP(nn.Module):
    """A simple feedforward neural network for the actor."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super(ActorMLP, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        output_types = Action.get_types()
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid() if (out_type == 'continuous-sigmoid') or (out_type == 'discrete') else nn.Tanh()
            )
            for out_type in output_types
        ])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.backbone(state)
        action = [head(features).squeeze(-1) for head in self.action_heads]
        return action
