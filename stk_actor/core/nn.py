import torch
import torch.nn as nn
from .action import Action
from typing import Optional


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
        self.output_types = Action.get_types()
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid() if (out_type == 'continuous-sigmoid') or (out_type == 'discrete') else nn.Tanh()
            )
            for action_name, out_type in self.output_types
        ])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.backbone(state)
        action = [head(features).squeeze(-1) for head in self.action_heads]
        return action


class ActorResMLP(nn.Module):
    """A simple feedforward neural network for the actor with an additional steer-drift residual connection."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super(ActorMLP, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_types = Action.get_types()
        self.steer_index = next(i for i, (name, _) in enumerate(self.output_types) if name == 'steer')
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + (1 if action_name == 'drift' else 0), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid() if (out_type == 'continuous-sigmoid') or (out_type == 'discrete') else nn.Tanh()
            )
            for action_name, out_type in self.output_types
        ])

    def forward(self, state: torch.Tensor, steer_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Check if model is in training mode and steer_values are provided
        if self.training and (steer_values is None):
            raise ValueError("steer_values must be provided during training for residual connection.")
        # Infer features
        features = self.backbone(state)
        # Get steer predictions
        steer_preds = self.action_heads[self.steer_index](features).squeeze(-1)
        # Get all action predictions
        action = []
        for (action_name, _), head in zip(self.output_types, self.action_heads):
            if action_name == 'drift':
                # Concatenate steer predictions or provided steer values for drift prediction
                steer_input = steer_values if self.training else steer_preds
                drift_input = torch.cat([features, steer_input.unsqueeze(-1)], dim=-1)
                action.append(head(drift_input).squeeze(-1))
            elif action_name == 'steer':
                # Use the steer predictions
                action.append(steer_preds)
            else:
                # Otherwise predict normally
                action.append(head(features).squeeze(-1))
        # Return all action predictions
        return action


class CriticMLP(nn.Module):
    """A simple feedforward neural network for the actor."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super(CriticMLP, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)
