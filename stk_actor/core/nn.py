import torch
import torch.nn as nn
from .action import Action
from typing import Optional
from torch.distributions import Normal, Bernoulli

EPS = 1e-6


def logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(EPS, 1 - EPS)
    return torch.log(x) - torch.log(1 - x)


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


class StochasticActorMLP(nn.Module):
    """A stochastic adaptation of the ActorMLP."""

    def __init__(self, input_dim: int, hidden_dim: int, pretrained_mlp: Optional[ActorMLP] = None):
        super(StochasticActorMLP, self).__init__()
        self.deterministic_actor = ActorMLP(input_dim, hidden_dim) if pretrained_mlp is None else pretrained_mlp
        # Add std neurons for continuous actions
        self.acceleration_std_layer = nn.Linear(hidden_dim, 1)
        self.steer_std_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.deterministic_actor(state)

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        PPO-ready sampler.

        Returns:
            actions: list[Tensor] each shape [B] (same ordering as Action.get_types())
            log_prob: Tensor shape [B]
            entropy: Tensor shape [B]
            info: dict with means/stds (optional)
        """
        features = self.deterministic_actor.backbone(state)

        actions = []
        logp_sum = torch.zeros(state.shape[0], device=state.device)
        ent_sum = torch.zeros(state.shape[0], device=state.device)

        means_out = []
        stds_out = []

        for (action_name, out_type), head in zip(
            self.deterministic_actor.output_types,
            self.deterministic_actor.action_heads,
        ):
            mean_bounded = head(features).squeeze(-1)   # [B] in [-1,1] or [0,1] or prob for discrete
            means_out.append(mean_bounded)

            # --- Continuous actions ---
            if (action_name == 'steer') and (out_type == "continuous-tanh"):
                # mean in [-1,1] -> mean in R
                mu = torch.atanh(mean_bounded)

                log_std = self.steer_std_layer(features).squeeze(-1)
                log_std = log_std.clamp(-5.0, 2.0)
                std = torch.exp(log_std)
                stds_out.append(std)

                dist = Normal(mu, std)

                if deterministic:
                    raw = mu
                else:
                    raw = dist.rsample()

                a = torch.tanh(raw)

                # squash correction (tanh)
                logp = dist.log_prob(raw) - torch.log(1 - a.pow(2) + EPS)
                # entropy (approx: base gaussian entropy; common in practice)
                ent = dist.entropy()

                actions.append(a)
                logp_sum += logp
                ent_sum += ent

            elif (action_name == 'acceleration') and (out_type == "continuous-sigmoid"):
                # mean in [0,1] -> mean in R
                mu = logit(mean_bounded)

                log_std = self.acceleration_std_layer(features).squeeze(-1)
                log_std = log_std.clamp(-5.0, 2.0)
                std = torch.exp(log_std)
                stds_out.append(std)

                dist = Normal(mu, std)

                if deterministic:
                    raw = mu
                else:
                    raw = dist.rsample()

                a = torch.sigmoid(raw)

                # squash correction (sigmoid): derivative = a*(1-a)
                logp = dist.log_prob(raw) - torch.log(a * (1 - a) + EPS)
                ent = dist.entropy()

                actions.append(a)
                logp_sum += logp
                ent_sum += ent

            # --- Discrete actions (assumed Bernoulli) ---
            elif out_type == "discrete":
                p = mean_bounded.clamp(EPS, 1 - EPS)
                dist = Bernoulli(probs=p)

                if deterministic:
                    a = (p >= 0.5).float()
                else:
                    a = dist.sample()

                logp = dist.log_prob(a)
                ent = dist.entropy()

                actions.append(a)
                logp_sum += logp
                ent_sum += ent
                stds_out.append(torch.zeros_like(p))

            else:
                raise ValueError(f"Unknown out_type={out_type} for action={action_name}")

        info = {
            "means": means_out,
            "stds": stds_out,
        }
        return actions, logp_sum, ent_sum, info


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
