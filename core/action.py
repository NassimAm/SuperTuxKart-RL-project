from typing import TypedDict
import numpy as np
import torch


class Action(TypedDict):
    """Base class for actions in the environment."""

    acceleration: float = 0.0
    """Acceleration component of the action between 0.0 and 1.0."""
    brake: bool = False
    """Brake component of the action."""
    drift: bool = False
    """Drift component of the action."""
    fire: bool = False
    """Fire component of the action."""
    nitro: bool = False
    """Nitro boost component of the action."""
    rescue: bool = False
    """Rescue component of the action."""
    steer: float = 0.0
    """Steering component of the action between -1.0 and 1.0 (positive -> right, negative -> left)."""
    look_back: bool = False  # TODO: see how to integrate this in the action space
    """Look back component of the action."""

    CONTINUOUS_ACTIONS = ['acceleration', 'steer']
    DISCRETE_ACTIOSNS = ['brake', 'drift', 'fire', 'nitro', 'rescue']

    def from_env(input_action: dict[str, np.ndarray]) -> 'Action':
        """Create an Action from a list representation."""
        return Action(
            acceleration=input_action['acceleration'],
            brake=bool(input_action['brake']),
            drift=bool(input_action['drift']),
            fire=bool(input_action['fire']),
            nitro=bool(input_action['nitro']),
            rescue=bool(input_action['rescue']),
            steer=input_action['steer'],
            look_back=False
        )

    def to_env(action: 'Action') -> np.ndarray:
        """Convert the action to the environment's expected format."""
        return {
            'acceleration': action['acceleration'],
            'brake': int(action['brake']),
            'drift': int(action['drift']),
            'fire': int(action['fire']),
            'nitro': int(action['nitro']),
            'rescue': int(action['rescue']),
            'steer': action['steer']
        }

    def to_model(action: 'Action') -> list[torch.Tensor]:
        """Convert the action to the model's expected format."""
        return torch.cat([
            torch.tensor([action['acceleration']]),
            torch.tensor([float(action['brake'])]),
            torch.tensor([float(action['drift'])]),
            torch.tensor([float(action['fire'])]),
            torch.tensor([float(action['nitro'])]),
            torch.tensor([float(action['rescue'])]),
            torch.tensor([action['steer']]),
        ])

    def get_dims() -> list[int]:
        """Get the dimensions of each action component."""
        return [
            1,  # acceleration
            1,  # brake
            1,  # drift
            1,  # fire
            1,  # nitro
            1,  # rescue
            1,  # steer
        ]

    def acceleration_to_continuous(acceleration_discrete: int, acceleration_steps: int = 5) -> float:
        """Convert discrete acceleration to continuous value between 0 and 1."""
        assert acceleration_steps > 1, "acceleration_steps must be greater than 1"
        if isinstance(acceleration_discrete, np.ndarray):
            acceleration_discrete = acceleration_discrete.item()
        return (acceleration_discrete / (acceleration_steps - 1))

    def acceleration_to_discrete(acceleration_continuous: float, acceleration_steps: int = 5) -> int:
        """Convert continuous acceleration value between 0 and 1 to discrete."""
        assert acceleration_steps > 1, "acceleration_steps must be greater than 1"
        if isinstance(acceleration_continuous, np.ndarray):
            acceleration_continuous = acceleration_continuous.item()
        clamped_value = max(0.0, min(1.0, acceleration_continuous))
        return int(round(clamped_value * (acceleration_steps - 1)))

    def steer_to_continuous(steer_discrete: int, steer_steps: int = 7) -> float:
        """Convert discrete steer to continuous value between -1 and 1."""
        assert (steer_steps > 1) and (steer_steps % 2 == 1), "steer_steps must be an odd number greater than 1"
        if isinstance(steer_discrete, np.ndarray):
            steer_discrete = steer_discrete.item()
        return (steer_discrete / (steer_steps - 1)) * 2.0 - 1.0

    def steer_to_discrete(steer_continuous: float, steer_steps: int = 7) -> int:
        """Convert continuous steer value between -1 and 1 to discrete."""
        assert (steer_steps > 1) and (steer_steps % 2 == 1), "steer_steps must be an odd number greater than 1"
        if isinstance(steer_continuous, np.ndarray):
            steer_continuous = steer_continuous.item()
        clamped_value = max(-1.0, min(1.0, steer_continuous))
        return int(round(((clamped_value + 1.0) / 2.0) * (steer_steps - 1)))
